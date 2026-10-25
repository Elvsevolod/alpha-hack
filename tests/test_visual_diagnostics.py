import base64
from dataclasses import replace
import hashlib
import json

import httpx
import numpy as np
import pandas as pd
import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import DiagnosticReportArgs, ToolCommand
from automl_agent.controller import Controller, SYSTEM
from automl_agent.qwen import QwenClient, QwenOutcomeUnknown
from automl_agent.tool_results import payload
from automl_agent.util import canonical, file_hash
from automl_agent.visual_diagnostics import (DiagnosticTools, confusion_report, distribution_report,
    render_report, residual_report)
from context_compressor import ContextError
from context_compressor.contracts import text_context
from conftest import write_task


REAL_CRITIQUE = QwenClient.critique


def test_distributions_counts_quantiles_and_missing_data():
    frame = pd.DataFrame({"value": [0., 0., 1., 2., 10., np.nan, np.inf],
                          "category": ["a", "b", "a", None, "a", "b", "a"], "empty": [np.nan] * 7})
    report = distribution_report(frame)
    value = report["columns"]["value"]
    assert value["missing"] == value["nonfinite"] == 1 and value["finite_rows"] == 5
    assert value["quantiles"]["50"] == 1 and value["mean"] == 2.6
    assert value["skew"] > 1 and sum(value["histogram"]["counts"]) == 5
    assert report["columns"]["category"]["categories"] == [{"label": "a", "count": 4}, {"label": "b", "count": 2}]
    assert report["columns"]["empty"]["reason"] == "no_finite_values"
    json.dumps(report, allow_nan=False)


def test_numeric_category_uses_declared_role():
    report = distribution_report(pd.DataFrame({"code": [1, 1, 2]}), ["code"])
    assert report["columns"]["code"]["type"] == "categorical"
    assert report["columns"]["code"]["categories"] == [{"label": "1", "count": 2}, {"label": "2", "count": 1}]


def test_confusion_matrix_and_per_class_metrics_use_all_predictions(tmp_path):
    actual = np.array(["b"] * 60 + ["a"] * 60)
    predicted = np.array(["b"] * 50 + ["a"] * 10 + ["b"] * 20 + ["a"] * 40)
    report = confusion_report(actual, predicted, ["b", "a"])
    assert report["matrix"] == [[50, 10], [20, 40]]
    assert report["normalized"][0] == pytest.approx([5 / 6, 1 / 6])
    assert report["classes"][0]["precision"] == pytest.approx(50 / 70)
    assert report["classes"][0]["recall"] == pytest.approx(50 / 60)
    assert report["classes"][0]["f1"] == pytest.approx(100 / 130)
    assert report["top_confusions"][0] == {"actual": "a", "predicted": "b", "count": 20}
    report["scope"] = "fit_inner_holdout"
    assert render_report(report, tmp_path / "matrix.png") is None
    assert (tmp_path / "matrix.png").stat().st_size <= 65536


def test_sparse_classes_keep_full_matrix_with_uncertainty_and_image(tmp_path):
    report = confusion_report(["a"] * 50 + ["b"] * 2, ["a"] * 52, ["a", "b"])
    assert report["status"] == "available" and report["matrix"] == [[50, 0], [2, 0]]
    assert report["classes"][1]["uncertain"] and report["classes"][1]["support"] == 2
    report["scope"] = "fit_inner_holdout"
    assert render_report(report, tmp_path / "rare.png") is None
    assert (tmp_path / "rare.png").is_file()


def test_residual_numbers_and_binned_plot_do_not_return_rows():
    predicted = np.linspace(0, 10, 120)
    actual = predicted + 2
    report = residual_report(actual, predicted)
    assert report["rmse"] == report["mae"] == report["mean"] == report["median"] == 2
    assert report["top_10_percent_squared_error_fraction"] == pytest.approx(.1)
    assert sum(report["histogram"]["counts"]) == 120
    assert all(item["rows"] >= 30 and item["residual_mean"] == 2 for item in report["prediction_intervals"])
    assert not {"row_keys", "y_true", "y_pred", "predictions"} & set(report)
    with pytest.raises(ValueError, match="finite"):
        residual_report(np.array([np.inf]), np.array([0.]))


@pytest.mark.parametrize("arguments", [
    {"kind": "residuals"}, {"kind": "distributions", "experiment_id": "x"},
    {"kind": "distributions", "columns": ["x", "x"]},
    {"kind": "distributions", "columns": ["a", "b", "c", "d"]},
])
def test_diagnostic_contract_rejects_ambiguous_requests(arguments):
    with pytest.raises(ValueError):
        DiagnosticReportArgs(**arguments)


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-visual-fixture")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fixture.invalid/v1")
    monkeypatch.delenv("OPENAI_EXTRA_BODY", raising=False)
    path = write_task(tmp_path)
    frame = pd.read_csv(tmp_path / "train.csv", dtype={"row": str})
    frame = pd.concat([frame] * 4, ignore_index=True)
    frame["row"] = [f"training_{i}" for i in range(len(frame))]
    frame.to_csv(tmp_path / "train.csv", index=False)
    c = Controller(path, Limits(seconds=600, cpu=1), run_id="visual_test")
    c.prepare()
    yield c
    c.close()


def diagnose(c, operation="diagnose", **arguments):
    return c.dispatch(ToolCommand(operation_id=operation, tool="get_diagnostic_report",
        arguments={"kind": "distributions", "columns": ["value"], "knowledge_gap": "Inspect the distribution or error mechanism", **arguments}))


def report_source(c, raw):
    args = dict(raw["read"]["arguments"]["source"])
    parts = []
    while True:
        fragment = c.context_memory.store.read_source(**args)
        parts.append(fragment["content"])
        if fragment["complete"]:
            return json.loads("".join(parts))
        args["offset"] = fragment["range"]["end"]


def stream_response(body, operation="next_read", diagnostic=False):
    arguments = {"kind": "distributions", "columns": ["value"]} if diagnostic else {"report": "missing"}
    tool = "get_diagnostic_report" if diagnostic else "get_dataset_report"
    if "tools" in body:
        delta = {"tool_calls": [{"index": 0, "id": "call_visual", "type": "function", "function": {
            "name": tool, "arguments": canonical({"operation_id": operation, **arguments})}}]}
        finish = "tool_calls"
    else:
        delta = {"content": canonical({"operation_id": operation, "tool": tool, "arguments": arguments})}
        finish = "stop"
    events = [{"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
              {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
               "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}]
    content = "".join("data: " + canonical(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)


def publish_via_client(c, mode):
    c.qwen.transport = mode
    c.qwen.client.close()
    c.qwen.client = httpx.Client(transport=httpx.MockTransport(lambda request:
        stream_response(json.loads(request.content), operation="make_diagnostic", diagnostic=True)))
    command, _ = c.qwen.next_command(SYSTEM, c.context_memory.snapshot(SYSTEM))
    result = c.dispatch(command)
    assert result["status"] == "success", result
    return payload(result)


def test_distribution_worker_publication_cache_and_closed_control(agent, monkeypatch):
    c = agent
    before_csv = c.publisher.output.read_bytes()
    before_leader = c.journal.get("leader_id")
    raw = payload(diagnose(c))
    assert raw["artifacts"] and not raw["cached"]
    report = report_source(c, raw)
    assert report["rows"] == len(c.plan.fit) and report["columns"]["value"]["rows"] == len(c.plan.fit)
    assert raw["read"]["tool"] == "read_run_records"
    monkeypatch.setattr(c.executor, "run", lambda *args: pytest.fail("A cached diagnostic must not run another worker"))
    again = payload(diagnose(c, operation="same_report"))
    assert again["cached"] and again["artifacts"] == raw["artifacts"]
    assert c.publisher.output.read_bytes() == before_csv and c.journal.get("leader_id") == before_leader
    assert c.journal.get("control_opened") is not True
    rejected = payload(diagnose(c, operation="forbidden_target", scope="validation_features", columns=["answer"], include_target=True))
    assert rejected["status"] == "rejected"


def test_residual_worker_uses_only_staged_fit_rows_and_keeps_csv(agent, monkeypatch):
    c = agent
    baseline = c.journal.get("baseline_id")
    _, source_result = c.lookup(baseline)
    original = c.executor.run
    staged = []
    def inspect(directory, request, seconds):
        frame = json.loads((directory / "fit_y.json").read_text())
        staged.extend(frame["index"])
        return original(directory, request, seconds)
    monkeypatch.setattr(c.executor, "run", inspect)
    result = diagnose(c, kind="residuals", columns=[], experiment_id=baseline,
                      budget_seconds=max(30, min(200, source_result.seconds * 2 + 15)))
    assert result["status"] == "success", result
    raw = payload(result)
    report = report_source(c, raw)
    assert set(staged) == {c.dataset.row_keys[i] for i in c.plan.fit}
    assert not set(staged) & {c.dataset.row_keys[i] for i in c.plan.validation + c.plan.control}
    assert report["evaluation_scope"] == "fit_inner_holdout" and report["exploratory"]
    assert report["fit_keys_hash"] != report["evaluation_keys_hash"]
    assert raw["artifacts"] and report["rmse"] >= 0


def test_diagnostic_admission_preserves_refit_reserve(agent, monkeypatch):
    monkeypatch.setattr(agent.budget, "final_reserve", lambda costs: agent.budget.remaining - 1)
    monkeypatch.setattr(agent.executor, "run", lambda *args: pytest.fail("No worker may cross the final reserve"))
    result = diagnose(agent)
    assert result["status"] == "budget_too_small" and not payload(result)["artifacts"]


def test_confirmed_worker_failure_is_not_retried_or_called_unknown(agent, monkeypatch):
    calls = []
    def timeout(*args):
        calls.append(args)
        return {"status": "timeout", "seconds": 1}
    monkeypatch.setattr(agent.executor, "run", timeout)
    first = diagnose(agent)
    second = diagnose(agent, operation="known_timeout")
    assert first["status"] == second["status"] == "technical_failure"
    assert payload(second)["cached"] and payload(second)["error_code"] == "timeout" and len(calls) == 1


@pytest.mark.parametrize("mode", ["native", "json"])
def test_diagnostic_image_reaches_real_compressed_client_and_audits(agent, mode):
    raw = publish_via_client(agent, mode)
    snapshot = agent.context_memory.snapshot(SYSTEM)
    assert json.loads(snapshot.images_json), agent.context_memory.store.pending_fragments()
    sent = []
    def handle(request):
        body = json.loads(request.content)
        blocks = body["messages"][1]["content"]
        assert isinstance(blocks, list) and len(blocks) == 2
        png = base64.b64decode(blocks[1]["image_url"]["url"].split(",", 1)[1])
        assert hashlib.sha256(png).hexdigest() == raw["artifacts"][0]["sha256"]
        context = text_context(blocks)
        assert context["visual_diagnostics"]["image_included"]
        sent.append(body)
        return stream_response(body)
    agent.qwen.client.close()
    agent.qwen.client = httpx.Client(transport=httpx.MockTransport(handle))
    command, _ = agent.qwen.next_command(SYSTEM, snapshot)
    assert command.tool == "get_dataset_report" and len(sent) == 1
    audit = json.loads(sorted((agent.workdir / "knowledge_audits").glob("*.json"), key=lambda p: p.stat().st_mtime)[-1].read_text())
    assert audit["send_allowed"] and audit["multimodal_text_blocks_verified"]
    agent.dispatch(command)
    assert json.loads(agent.context_memory.snapshot(SYSTEM).images_json) == []


@pytest.mark.parametrize("mode", ["native", "json"])
def test_chart_hypothesis_is_revised_then_approved_before_real_training(agent, monkeypatch, mode):
    c = agent
    from automl_agent.completion import CompletionCosts
    # The critic now gates measured expensive work, independently of its cap.
    monkeypatch.setattr('automl_agent.completion.candidate_costs', lambda *a, **k:
        CompletionCosts([], {'candidate_fit_seconds': 120., 'seconds': 17.}))
    raw = publish_via_client(c, mode)
    observation = c.journal.operation("make_diagnostic").result["evidence_ref"]
    csv_before = c.publisher.output.read_bytes()
    records_before = len(c.records())
    calls_before = c.budget.counter("llm_calls")
    monkeypatch.setattr(c.qwen, "critique", REAL_CRITIQUE.__get__(c.qwen))
    research_calls, critic_calls = [], []

    def handle(request):
        body = json.loads(request.content)
        blocks = body["messages"][1]["content"]
        context = text_context(blocks)
        if "review_id" in context:
            critic_calls.append(context)
            assert isinstance(blocks, str) and "tools" not in body and "core" not in context
            assert context["candidate"]["hypothesis"]["evidence_refs"] == [observation]
            assert context["validation"]["comparison_scope"] == "research_validation_only"
            revise = len(critic_calls) == 1
            message = {"content": canonical({"review_id": context["review_id"],
                "verdict": "revise" if revise else "approve", "evaluation_scope": "research_validation",
                "comparator_id": "baseline", "evidence_assessment": "The fit distribution is exploratory evidence.",
                "comparison_assessment": "Keep the same validation protocol and baseline comparator.",
                "leakage_assessment": "Fit-only diagnostics and learning; final control stays closed.",
                "cost_assessment": "A bounded separate model-family comparison.",
                "falsification_assessment": "Retain the baseline if validation RMSE does not improve.",
                "revision_requests": ["State the single model-family change explicitly."] if revise else []})}
            finish = "stop"
        else:
            research_calls.append(context)
            if len(research_calls) == 1:
                assert isinstance(blocks, list) and context["visual_diagnostics"]["image_included"]
                png = base64.b64decode(blocks[1]["image_url"]["url"].split(",", 1)[1])
                assert hashlib.sha256(png).hexdigest() == raw["artifacts"][0]["sha256"]
            else:
                assert "hypothesis_revision_required" in canonical(context)
            args = {"pipeline": {"model": "extra_trees", "hyperparameters": {"n_estimators": 5, "max_depth": 4}},
                "budget_seconds": 120, "hypothesis": {"observation": "Inspect the published fit distribution.",
                    "changed_component": "model family" if len(research_calls) > 1 else "bounded estimator",
                    "expected_effect": "Compare nonlinear capacity against the unchanged baseline.",
                    "acceptance": "Lower validation RMSE on the unchanged protocol.", "evidence_refs": [observation]}}
            operation = "chart_hypothesis_" + str(len(research_calls))
            if body.get("tools"):
                message = {"tool_calls": [{"id": "call_" + operation, "type": "function", "function": {
                    "name": "train_model", "arguments": canonical({"operation_id": operation, **args})}}]}
                finish = "tool_calls"
            else:
                message = {"content": canonical({"operation_id": operation, "tool": "train_model", "arguments": args})}
                finish = "stop"
        return httpx.Response(200, json={"choices": [{"finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}})

    c.qwen.client.close()
    c.qwen.client = httpx.Client(transport=httpx.MockTransport(handle))
    command, _ = c.qwen.next_command(SYSTEM, c.context_memory.snapshot(SYSTEM))
    rejected = payload(c.dispatch(command))
    assert rejected["status"] == "hypothesis_revision_required" and not rejected["experiment_started"]
    assert len(c.records()) == records_before and c.budget.counter("candidate_count") == 0
    assert c.publisher.output.read_bytes() == csv_before
    command, _ = c.qwen.next_command(SYSTEM, c.context_memory.snapshot(SYSTEM))
    accepted = payload(c.dispatch(command))
    assert accepted["critic"]["status"] == "approved" and accepted["result"]["status"] == "completed"
    assert len(research_calls) == len(critic_calls) == 2
    assert c.budget.counter("llm_calls") == calls_before + 4 and c.budget.counter("candidate_count") == 1
    assert not c.journal.get("control_opened")


@pytest.mark.parametrize("mode", ["native", "json"])
def test_image_budget_compaction_and_forgery_rejection(agent, mode):
    publish_via_client(agent, mode)
    snapshot = agent.context_memory.snapshot(SYSTEM)
    comp = agent.context_memory.compressor
    prepared = comp.fit(snapshot, mode, budget=262144)
    body = json.loads(prepared.wire_json)
    assert isinstance(body["messages"][1]["content"], list)
    body["messages"][1]["content"][1]["image_url"]["url"] = "https://unapproved.invalid/chart.png"
    wire = canonical(body)
    forged = replace(prepared, wire_json=wire, request_hash=hashlib.sha256(wire.encode()).hexdigest(),
                     wire_bytes=len(wire.encode()), conservative_size=len(wire.encode()) + prepared.response_reserve)
    with pytest.raises(ContextError, match="inline PNG"):
        comp.store.register_request(forged, json.loads(snapshot.observations_json))
    compact = comp.prepare(snapshot, mode, budget=131072, minimum_stage=2)
    compact_body = json.loads(compact.wire_json)
    assert isinstance(compact_body["messages"][1]["content"], str)
    assert not text_context(compact_body["messages"][1]["content"])["visual_diagnostics"]["image_included"]
    assert compact.conservative_size + 8192 <= 131072


@pytest.mark.parametrize("mode", ["native", "json"])
def test_explicit_unsupported_image_uses_numeric_fallback(agent, mode):
    publish_via_client(agent, mode)
    snapshot = agent.context_memory.snapshot(SYSTEM)
    calls = []
    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(400, json={"error": "This model does not support image input"})
        assert isinstance(body["messages"][1]["content"], str)
        return stream_response(body)
    agent.qwen.client.close()
    agent.qwen.client = httpx.Client(transport=httpx.MockTransport(handle))
    command, _ = agent.qwen.next_command(SYSTEM, snapshot)
    assert len(calls) == 2 and agent.journal.get("vision_disabled")
    assert command.tool == "get_dataset_report"


def test_unknown_image_request_outcome_does_not_retry_as_text(agent):
    publish_via_client(agent, "native")
    snapshot = agent.context_memory.snapshot(SYSTEM)
    calls = []
    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("Synthetic unknown request outcome")
    agent.qwen.client.close()
    agent.qwen.client = httpx.Client(transport=httpx.MockTransport(handle))
    with pytest.raises(QwenOutcomeUnknown):
        agent.qwen.next_command(SYSTEM, snapshot)
    assert len(calls) == 1 and not agent.journal.get("vision_disabled")


def test_classification_worker_and_selected_optuna_recipe(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-visual-fixture")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fixture.invalid/v1")
    path = write_task(tmp_path, kind="classification")
    frame = pd.concat([pd.read_csv(tmp_path / "train.csv", dtype={"row": str})] * 4, ignore_index=True)
    frame["row"] = [f"training_{i}" for i in range(len(frame))]
    frame.to_csv(tmp_path / "train.csv", index=False)
    c = Controller(path, Limits(seconds=600, cpu=1), run_id="matrix_test")
    try:
        c.prepare()
        trained = c.dispatch(ToolCommand(operation_id="fit_logistic", tool="train_model", arguments={
            "pipeline": {"model": "logistic"}, "budget_seconds": 30,
            "hypothesis": {"observation": "Binary numeric feature signal", "expected_effect": "Separate classes",
                "changed_component": "logistic baseline", "acceptance": "Completed valid classifier"}}))
        assert trained["status"] == "success", trained
        experiment = payload(trained)["result"]["experiment_id"]
        count = c.budget.counter("candidate_count")
        csv = c.publisher.output.read_bytes()
        measured = diagnose(c, kind="confusion_matrix", columns=[], experiment_id=experiment)
        assert measured["status"] == "success", measured
        raw = payload(measured)
        report = report_source(c, raw)
        assert raw["artifacts"] and report["labels"] == c.task.classes["answer"]
        assert sum(map(sum, report["matrix"])) == report["evaluation_rows"]
        assert report["macro_f1"] > .8 and c.budget.counter("candidate_count") == count
        assert c.publisher.output.read_bytes() == csv
        original_spec, result = c.lookup(experiment)
        search_spec = original_spec.model_copy(update={"pipeline": original_spec.pipeline.model_copy(update={
            "engine": "optuna", "search_owner": "optuna", "prediction_policy": "class_offsets", "postprocess_fit_policy": "inner_holdout"})})
        manifest = c.workdir / "selected_fixture.json"
        frozen = original_spec.pipeline.model_copy(update={"hyperparameters": {"C": .25}, "class_offsets": [.25, -.25],
            "prediction_policy": "class_offsets", "postprocess_fit_policy": "inner_holdout"})
        manifest.write_text(canonical({"frozen_pipeline": frozen.model_dump(mode="json")}))
        result = result.model_copy(update={"artifacts": {"worker_manifest": str(manifest), "worker_manifest_sha256": file_hash(manifest)}})
        monkeypatch.setattr(c, "lookup", lambda _: (search_spec, result))
        _, _, replay = DiagnosticTools(c).validate(DiagnosticReportArgs(kind="confusion_matrix", experiment_id=experiment))
        assert replay.engine == "direct" and replay.class_offsets is None
        assert replay.hyperparameters == {"C": .25} and replay.prediction_policy == "class_offsets"
        assert replay.postprocess_fit_policy == "inner_holdout" and replay.search_owner == "none"
    finally:
        c.close()


@pytest.mark.parametrize("strategy", ["group", "temporal", "temporal_group"])
def test_diagnostic_inner_fit_preserves_group_and_time_boundaries(tmp_path, tabular, monkeypatch, strategy):
    from automl_agent.api import prediction_bundle
    from automl_agent.executor import stage_frame
    from automl_agent.validation import inner_split
    from automl_agent.visual_diagnostics import run_worker
    X, y, task = tabular
    X = X.copy()
    X["when"] = np.repeat(np.arange(12), 15)
    X["group"] = np.arange(len(X)) // 4
    task = task.model_copy(update={"time_column": "when", "group_column": "group", "feature_columns": list(X)})
    stage_frame(tmp_path / "fit_X.json", X)
    stage_frame(tmp_path / "fit_y.json", y)
    fitted = []
    evaluated = []
    class SpyPipeline:
        def fit(self, frame, target, context):
            fitted.extend(frame.index)
        def predict(self, frame, context):
            evaluated.extend(frame.index)
            return prediction_bundle({"answer": np.zeros(len(frame))}, frame, context, "values")
    monkeypatch.setattr("automl_agent.pipelines.build_pipeline", lambda *args: SpyPipeline())
    request = {"kind": "residuals", "task": task.model_dump(mode="json"), "pipeline": {"model": "ridge"},
        "context": {"strategy": strategy, "protocol_hash": "protocol", "workdir": str(tmp_path)},
        "source_recipe_hash": "source_recipe", "data_version": "data-v1", "protocol_hash": "protocol", "experiment_id": "fixture"}
    result = run_worker(request, tmp_path)
    assert result["status"] == "completed"
    a, b = inner_split(X, y, task, strategy, 42)
    assert fitted == list(X.iloc[a].index) and evaluated == list(X.iloc[b].index)
    assert not set(fitted) & set(evaluated)
    if "temporal" in strategy:
        assert X.loc[fitted, "when"].max() < X.loc[evaluated, "when"].min()
    if "group" in strategy:
        assert not set(X.loc[fitted, "group"]) & set(X.loc[evaluated, "group"])


def test_render_failure_keeps_numeric_report(tmp_path, monkeypatch):
    from automl_agent.executor import stage_frame
    from automl_agent.visual_diagnostics import run_worker
    stage_frame(tmp_path / "fit_X.json", pd.DataFrame({"x": np.arange(60)}))
    def unavailable(*args):
        raise RuntimeError("Synthetic renderer failure")
    monkeypatch.setattr("automl_agent.visual_diagnostics.render_report", unavailable)
    result = run_worker({"kind": "distributions", "scope": "fit", "data_version": "d", "protocol_hash": "p"}, tmp_path)
    assert result["status"] == "completed" and result["image_unavailable"] == "render_failed:RuntimeError"
    assert "diagnostic.png" not in result["hashes"]
    assert json.loads((tmp_path / "diagnostic_report.json").read_text())["columns"]["x"]["mean"] == 29.5


def test_snapshot_pins_png_bytes_and_changed_file_uses_numeric_mode(agent):
    raw = publish_via_client(agent, "json")
    snapshot = agent.context_memory.snapshot(SYSTEM)
    image = json.loads(snapshot.images_json)[0]
    path = agent.workdir / raw["artifacts"][0]["path"]
    path.write_bytes(b"changed after snapshot")
    prepared = agent.context_memory.compressor.prepare(snapshot, "json", budget=262144)
    blocks = json.loads(prepared.wire_json)["messages"][1]["content"]
    assert blocks[1]["image_url"]["url"] == image["data_url"]
    assert json.loads(agent.context_memory.snapshot(SYSTEM).images_json) == []


def test_frozen_research_cannot_launch_diagnostic_fit(agent, monkeypatch):
    agent.freeze("Offline finalization boundary check")
    monkeypatch.setattr(agent.executor, "run", lambda *args: pytest.fail("Frozen research cannot run diagnostics"))
    with pytest.raises(ValueError, match="searching"):
        diagnose(agent)
