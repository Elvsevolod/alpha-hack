import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from automl_agent.admission import admit, drain
from automl_agent.budget import Limits
from automl_agent.contracts import Hypothesis, PipelineSpec, ToolCommand, TrainArgs
from automl_agent.controller import Controller, SYSTEM
from automl_agent.recipe_patch import inherit_parent
from automl_agent.research import compact_research_options
from automl_agent.tool_results import payload
from automl_agent.util import canonical, read_json, atomic_json, file_hash
from conftest import write_task


def hypothesis(changed=()):
    return Hypothesis(observation="Fixed offline fixture", expected_effect="Measure alpha on the same rows",
        changed_component="alpha", acceptance="Compare untouched research metric", changed_fields=list(changed))


def parent_controller(pipeline):
    return SimpleNamespace(lookup=lambda _: (SimpleNamespace(pipeline=pipeline), None))


@pytest.mark.parametrize("field,value", [
    ("prediction_policy", "class_offsets"), ("transformations", [{"name": "standard_scale"}]),
    ("target_transform", "log1p"), ("shift_strategy", "importance_weighting"),
])
def test_omitted_component_inherits_and_an_explicit_reset_requires_changed_fields(field, value):
    config = {field: value}
    if field == "prediction_policy":
        config.update(postprocess_fit_policy="inner_holdout", class_offsets=[.1, -.1],
            decision_method="regularized", learned_decision={"fitted": True})
    parent = PipelineSpec(**config)
    c = parent_controller(parent)
    args = TrainArgs(parent_id="parent", pipeline={"model": "extra_trees"}, hypothesis=hypothesis())
    inherit_parent(c, args)
    assert getattr(args.pipeline, field) == getattr(parent, field)
    assert field in args._inherited_fields
    assert args.pipeline.class_offsets is None and not args.pipeline.learned_decision
    if field == "prediction_policy":
        assert args.pipeline.postprocess_fit_policy == "inner_holdout"
        assert args.pipeline.decision_method == "regularized"
    reset = getattr(PipelineSpec(), field)
    with pytest.raises(ValueError, match="changed_fields"):
        inherit_parent(c, TrainArgs(parent_id="parent", pipeline={field: reset}, hypothesis=hypothesis()))
    explicit = TrainArgs(parent_id="parent", pipeline={field: reset}, hypothesis=hypothesis([field]))
    inherit_parent(c, explicit)
    assert getattr(explicit.pipeline, field) == reset and field not in explicit._inherited_fields
    omitted_reset = TrainArgs(parent_id="parent", hypothesis=hypothesis([field]))
    inherit_parent(c, omitted_reset)
    assert getattr(omitted_reset.pipeline, field) == reset and field not in omitted_reset._inherited_fields


def test_inheritance_keeps_child_parameters_is_deep_and_preserves_log_inverse():
    parent = PipelineSpec(model="lightgbm", hyperparameters={"n_estimators": 9}, target_transform="log1p",
        log_inverse="oof_smearing", learned_inverse={"factor": 2},
        transformations=[{"name": "math", "parameters": {"columns": ["x"]}}])
    args = TrainArgs(parent_id="parent", pipeline={"model": "ridge", "hyperparameters": {"alpha": 7}}, hypothesis=hypothesis())
    inherit_parent(parent_controller(parent), args)
    assert args.pipeline.model == "ridge" and args.pipeline.hyperparameters == {"alpha": 7}
    assert args.pipeline.log_inverse == "oof_smearing" and not args.pipeline.learned_inverse
    args.pipeline.transformations[0].parameters["columns"].append("y")
    assert parent.transformations[0].parameters["columns"] == ["x"]


def test_partial_transform_chain_removal_requires_acknowledgment():
    parent = PipelineSpec(transformations=[{"name": "standard_scale"}, {"name": "drop_constant"}])
    c = parent_controller(parent)
    args = TrainArgs(parent_id="parent", pipeline={"transformations": [{"name": "standard_scale"}]}, hypothesis=hypothesis())
    with pytest.raises(ValueError, match="transformations"):
        inherit_parent(c, args)
    args.hypothesis.changed_fields = ["transformations"]
    inherit_parent(c, args)
    assert len(args.pipeline.transformations) == 1


@pytest.fixture
def prepared(tmp_path):
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1, memory_mb=1024), run_id="inheritance")
    c.no_llm = True
    c.prepare()
    yield c
    c.close()


def train_parent(c):
    result = payload(c.dispatch(ToolCommand(operation_id="parent", tool="train_model", arguments={
        "pipeline": {"model": "ridge", "hyperparameters": {"alpha": 2}, "transformations": [{"name": "standard_scale"}]},
        "budget_seconds": 45, "hypothesis": hypothesis().model_dump(mode="json")})))
    assert result["result"]["status"] == "completed", result
    return result["experiment"]["experiment_id"]


def child(parent, alpha=7):
    return {"parent_id": parent, "search_action": "improve", "pipeline": {"model": "ridge", "hyperparameters": {"alpha": alpha}},
        "budget_seconds": 45, "hypothesis": hypothesis().model_dump(mode="json")}


def assert_child(c, result, operation_id):
    spec, measured = c.lookup(result["experiment"]["experiment_id"])
    assert measured.status == "completed"
    assert [item.name for item in spec.pipeline.transformations] == ["standard_scale"]
    assert "transformations" in spec.inherited_fields
    events = [json.loads(row[0]) for row in c.journal.db.execute("SELECT payload FROM events WHERE kind='inherited_fields'")]
    assert any(row["operation_id"] == operation_id and "transformations" in row["inherited_fields"] for row in events)


def test_single_child_inherits_before_validation_and_operation_replay(prepared):
    c = prepared
    parent = train_parent(c)
    command = ToolCommand(operation_id="single", tool="train_model", arguments=child(parent))
    from context_compressor import Command
    c.context_memory.validate(Command(operation_id=command.operation_id, tool=command.tool, arguments_json=canonical(command.arguments)))
    result = payload(c.dispatch(command))
    assert_child(c, result, "single")
    assert payload(c.dispatch(command)) == result
    changed = copy.deepcopy(command.arguments)
    changed["pipeline"]["transformations"] = []
    with pytest.raises(ValueError, match="different arguments"):
        c.dispatch(ToolCommand(operation_id="single", tool="train_model", arguments=changed))


def test_admitted_queue_and_batch_preserve_omission_provenance(prepared):
    c = prepared
    parent = train_parent(c)
    queued = ToolCommand(operation_id="queued", tool="train_model", arguments=child(parent, 8))
    assert admit(c, queued)
    assert "transformations" not in c.journal.get("admitted_queue")[0]["command"]["arguments"]["pipeline"]
    assert drain(c, "Offline HTTP 402 fixture") == ["queued"]
    assert_child(c, payload(c.journal.operation("queued").result), "queued")
    result = payload(c.dispatch(ToolCommand(operation_id="batch", tool="run_experiments", arguments={
        "candidates": [child(parent, 9), child(parent, 10)], "budget_seconds": 120})))
    assert len(result["results"]) == 2 and all(row["status"] == "completed" for row in result["results"]), result
    for row in result["results"]:
        assert_child(c, payload(c.journal.operation(row["operation_id"]).result), row["operation_id"])


def finish_args(c):
    return {"reason": "Local verification is complete", "result_ids": [c.journal.get("baseline_id")],
        "skipped_directions": {"all": "Offline bounded mechanism check; no external model requests are authorized."}}


def lose_frozen_transform(c, identifier):
    spec, result = c.lookup(identifier)
    path = Path(result.artifacts["worker_manifest"])
    manifest = read_json(path)
    manifest["frozen_pipeline"]["transformations"] = []
    atomic_json(path, manifest)
    result.artifacts["worker_manifest_sha256"] = file_hash(path)
    c.journal.save_experiment(spec, result)


def test_finish_audits_selected_finalists_and_confirmation_is_bound_to_recipe(prepared):
    c = prepared
    parent = train_parent(c)
    c.journal.put("leader_id", parent)
    lose_frozen_transform(c, parent)
    args = finish_args(c)
    warned = payload(c.dispatch(ToolCommand(operation_id="finish_warn", tool="finish", arguments=args)))
    assert warned["status"] == "component_confirmation_required"
    assert c.state == "searching" and not c.journal.get("control_opened")
    assert not c.journal.get("finalization_plan") and not c.journal.get("finish_evidence")
    assert any("transformations" in row["missing_fields"] for row in warned["warnings"])
    stale = {**args, "component_audit_hash": "0" * 64}
    assert payload(c.dispatch(ToolCommand(operation_id="finish_stale", tool="finish", arguments=stale)))["status"] == "component_confirmation_required"
    spec, measured = c.lookup(parent)
    path = Path(measured.artifacts["worker_manifest"])
    manifest = read_json(path)
    manifest["frozen_pipeline"]["hyperparameters"]["alpha"] = 3
    atomic_json(path, manifest)
    measured.artifacts["worker_manifest_sha256"] = file_hash(path)
    c.journal.save_experiment(spec, measured)
    stale = {**args, "component_audit_hash": warned["component_audit_hash"]}
    changed = payload(c.dispatch(ToolCommand(operation_id="finish_changed", tool="finish", arguments=stale)))
    assert changed["status"] == "component_confirmation_required" and changed["component_audit_hash"] != warned["component_audit_hash"]
    confirmed = {**args, "component_audit_hash": changed["component_audit_hash"]}
    assert payload(c.dispatch(ToolCommand(operation_id="finish_confirm", tool="finish", arguments=confirmed)))["status"] == "frozen"
    assert c.journal.get("finalization_plan")["component_audit"]["warnings"]


def test_automatic_stop_excludes_damaged_finalist_without_authorizing_removal(prepared):
    c = prepared
    parent = train_parent(c)
    c.journal.put("leader_id", parent)
    lose_frozen_transform(c, parent)
    c.freeze("Absolute deadline fixture")
    plan = c.journal.get("finalization_plan")
    assert parent in plan["excluded_component_loss"] and parent not in plan["finalists"]
    assert c.journal.get("baseline_id") in plan["finalists"]


def test_no_intact_finalists_keeps_control_closed_and_reports_the_audit(prepared, monkeypatch):
    c = prepared
    parent = train_parent(c)
    lose_frozen_transform(c, parent)
    monkeypatch.setattr(c, "prospective_finalists", lambda: ([parent], {}))
    c.freeze("Unavailable model channel fixture")
    c.finalize()
    assert c.state == "completed" and not c.journal.get("control_opened")
    assert c.journal.get("control_status") == "control_skipped_component_confirmation"
    assert c.report()["finalist_component_warnings"]


def test_legacy_child_loss_and_missing_fitted_offsets_are_detected(prepared):
    from automl_agent.contracts import RefitPolicy
    from automl_agent.finalists import component_audit
    c = prepared
    parent = train_parent(c)
    spec, measured = c.lookup(parent)
    offset_parent = spec.model_copy(deep=True, update={"experiment_id": "old_offsets", "pipeline": PipelineSpec(
        prediction_policy="class_offsets", postprocess_fit_policy="inner_holdout")})
    c.journal.save_experiment(offset_parent)
    legacy_child = spec.model_copy(deep=True, update={"experiment_id": "legacy_child", "parent_id": "old_offsets",
        "pipeline": PipelineSpec()})
    c.journal.save_experiment(legacy_child)
    audit = component_audit(c, ["legacy_child"], {})
    assert any("prediction_policy" in row["missing_fields"] for row in audit["warnings"])
    acknowledged = legacy_child.model_copy(deep=True, update={"experiment_id": "acknowledged_child"})
    acknowledged.hypothesis.changed_fields = ["prediction_policy"]
    c.journal.save_experiment(acknowledged)
    assert not component_audit(c, ["acknowledged_child"], {})["warnings"]
    restored = legacy_child.model_copy(deep=True, update={"experiment_id": "restored_child", "parent_id": "legacy_child",
        "pipeline": offset_parent.pipeline.model_copy(deep=True)})
    c.journal.save_experiment(restored)
    assert not component_audit(c, ["restored_child"], {})["warnings"]
    acknowledged_later = acknowledged.model_copy(deep=True, update={"experiment_id": "acknowledged_later", "parent_id": "legacy_child"})
    c.journal.save_experiment(acknowledged_later)
    assert not component_audit(c, ["acknowledged_later"], {})["warnings"]
    frozen = offset_parent.pipeline
    policies = {"old_offsets": RefitPolicy(experiment_id="old_offsets", pipeline_hash=frozen.hash, frozen_pipeline=frozen).model_dump(mode="json")}
    assert "class_offsets" in component_audit(c, ["old_offsets"], policies)["warnings"][0]["missing_fields"]


def test_research_options_first_read_is_bounded_and_details_and_leader_remain_readable(prepared):
    c = prepared
    core = json.loads(c.context_memory.snapshot(SYSTEM).core_json)
    work = core["current_work"]
    assert work["leader_pipeline"] == {"tool": "read_run_records", "arguments": {"ids": [c.journal.get("leader_id")], "view": "pipeline"}}
    from test_layered_prompt_context import read_all
    content = read_all(c, work['research_options'])
    compact = json.loads(content)
    assert len(content.encode()) <= 3072 and "direction_map" not in compact
    details = compact["details"]
    assert details["tool"] == "read_run_records" and details["arguments"]["source"]["section"] == "details"
    response = payload(c.dispatch(ToolCommand(operation_id="options_details", tool=details["tool"], arguments=details["arguments"])))
    assert response["content"] and response["source"]["section"] == "details"


def test_context_summary_is_bounded_even_with_large_multibyte_history():
    rows = [{"branch": "ветка" * 250, "status": "measured", "measured_candidates": 1,
        "technical_failures": 0} for _ in range(100)]
    options = {"branches": rows, "direction_map": {"directions": [{}] * 1000,
        "affordable_unattempted": [{"id": "x" * 2000, "model": "ridge", "budget_seconds": 15, "cost_basis": "bounded"}]}}
    summary = compact_research_options(options, {"tool": "read_run_records", "arguments": {"source": {"section": "details"}}})
    assert len(canonical(summary).encode("utf-8")) <= 3072 and summary["branch_count"] == 100


def test_offsets_inherit_as_a_recipe_and_survive_finish_control_and_full_refit(tmp_path):
    import numpy as np
    import pandas as pd
    path = write_task(tmp_path, kind="classification")
    train = pd.read_csv(tmp_path / "train.csv")
    values = train.value.fillna(0)
    train["answer"] = np.where(values < -.4, "low", np.where(values > .4, "high", "middle"))
    train.to_csv(tmp_path / "train.csv", index=False)
    c = Controller(path, Limits(seconds=180, cpu=1, memory_mb=1024), run_id="offsets")
    c.no_llm = True
    try:
        c.prepare()
        parent = payload(c.dispatch(ToolCommand(operation_id="offset_parent", tool="train_model", arguments={
            "pipeline": {"model": "extra_trees", "hyperparameters": {"n_estimators": 8},
                "prediction_policy": "class_offsets", "postprocess_fit_policy": "inner_holdout"},
            "hypothesis": hypothesis().model_dump(mode="json"), "budget_seconds": 15})))
        assert parent["result"]["status"] == "completed", parent
        arguments = child(parent["experiment"]["experiment_id"])
        arguments["pipeline"] = {"model": "extra_trees", "hyperparameters": {"n_estimators": 12}}
        trained = payload(c.dispatch(ToolCommand(operation_id="offset_child", tool="train_model", arguments=arguments)))
        assert trained["result"]["status"] == "completed", trained
        identifier = trained["experiment"]["experiment_id"]
        spec, result = c.lookup(identifier)
        assert spec.pipeline.class_offsets is None and spec.pipeline.prediction_policy == "class_offsets"
        assert "prediction_policy" in spec.inherited_fields and "postprocess_fit_policy" in spec.inherited_fields
        manifest = read_json(Path(result.artifacts["worker_manifest"]))
        assert len(manifest["frozen_pipeline"]["class_offsets"]) == 3
        c.journal.put("leader_id", identifier)
        done = payload(c.dispatch(ToolCommand(operation_id="offset_finish", tool="finish", arguments=finish_args(c))))
        assert done["status"] == "frozen", done
        assert not c.journal.get("finalization_plan")["component_audit"]["warnings"]
        frozen = c.journal.get("finalization_plan")["policies"][identifier]["frozen_pipeline"]
        assert frozen["prediction_policy"] == "class_offsets" and len(frozen["class_offsets"]) == 3
        c.finalize()
        assert c.state == "completed" and c.journal.get("control_status") == "completed"
        assert c.journal.get("refit_status") == "completed"
        assert len(pd.read_csv(tmp_path / "submission.csv")) == 25
        _, refitted = c._prepare_finalist(identifier, list(range(len(c.dataset.train))), "refit", 30)
        assert refitted["fit_rows"] == len(c.dataset.train)
        assert refitted["frozen_pipeline"]["prediction_policy"] == "class_offsets"
        assert refitted["frozen_pipeline"]["class_offsets"] == frozen["class_offsets"]
    finally:
        c.close()
