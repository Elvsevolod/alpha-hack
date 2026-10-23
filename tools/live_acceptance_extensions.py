"""Declared temporal-role and context-restart cases for the frozen live suite."""
import json

import pandas as pd

from automl_agent.budget import Limits
from automl_agent.contracts import Hypothesis, PipelineSpec, ToolCommand, Transformation
from automl_agent.controller import Controller
from automl_agent.util import atomic_json, file_hash
from tools.acceptance_fixtures import write_task


def temporal_fixture(directory, repeat):
    path = write_task(directory, temporal=True)
    (directory / "task_spec.json").unlink()
    for name in ("train.csv", "test.csv"):
        frame = pd.read_csv(directory / name, dtype=str, keep_default_na=False).rename(columns={"when": "p"})
        frame["p"] = frame["p"].astype(int) + 2017
        frame.to_csv(directory / name, index=False)
    role = ["p: calendar year of observation.", "p: год наблюдения.",
        "Next period transfer. p is a repeated ordered measurement code."][repeat % 3]
    path.write_text("Regression task. Primary metric: RMSE. Read train.csv, test.csv and sample_submission.csv. "
        "Predict answer and preserve row identifiers. Empty cells are missing; NA and - are literal categories.\n" + role + "\n")
    return path


class InjectedRestart(BaseException):
    """Emulate abrupt controller loss without invoking its graceful finish path."""


def run_context_recovery(repeat, output, seconds, transport, fingerprint, protocol_version):
    directory = output / f"context_recovery_{repeat}"
    path = write_task(directory)
    focus = "Controlled context-recovery scenario. Preserve the task, protocol, prior leader, negative results and resource limits after a request overflow and restart. Choose a compatible, genuinely new experiment or explain a justified finish."
    c = Controller(path, Limits(seconds=seconds, cpu=1, memory_mb=2048, max_candidates=6, max_calls=7),
        run_id="live", transport=transport, research_focus=focus)
    initial_code = fingerprint()
    checks, requests = [], [0]
    restarted = False
    try:
        c.prepare()
        c.dispatch(ToolCommand(operation_id="setup_reference", tool="get_reference", arguments={
            "mode": "read", "reference_id": "candidate_interface"}))
        c.dispatch(ToolCommand(operation_id="setup_model", tool="train_model", arguments={
            "pipeline": PipelineSpec(model="ridge", hyperparameters={"alpha": 2}).model_dump(mode="json"),
            "hypothesis": Hypothesis(observation="Controlled recovery setup", expected_effect="Preserve a measured alternative",
                changed_component="model_family", acceptance="Actual fit before the declared context fault").model_dump(), "budget_seconds": 15}))
        base, result = c.lookup("baseline")
        bad = base.pipeline.model_copy(update={"transformations": [Transformation(name="select_kbest")]})
        for i in range(300):
            identifier = f"injected_history_{i}"
            spec = base.model_copy(update={"experiment_id": identifier, "pipeline": bad})
            failed = result.model_copy(update={"experiment_id": identifier, "status": "unsupported_mode", "score": None,
                "artifacts": {}, "error": "native_transformations_unsupported: select_kbest"})
            c.journal.save_experiment(spec, failed)
            c.capabilities.records[identifier] = {"model": "lightgbm", "engine": "direct", "task": "regression",
                "strategy": "random", "status": "unsupported_mode", "versions": c.capabilities.versions, "probe_id": identifier}
        atomic_json(c.capabilities.path, c.capabilities.records)
        anchors = {"task": c.task.hash, "protocol": c.plan.hash, "baseline": c.journal.get("baseline_id"),
            "leader": c.journal.get("leader_id"), "deadline": c.budget.deadline,
            "candidates": c.budget.counter("candidate_count"), "calls": c.budget.counter("llm_calls")}
        atomic_json(directory / "RECOVERY_SETUP.json", {"anchors": anchors, "synthetic_history": 300,
            "faults": ["declared provider rejection of a bounded request", "abrupt restart after compaction"],
            "history_contains_no_fabricated_scores": True})

        def attach(controller, resumed):
            original_request = controller.qwen._request
            def request(messages, tools=None, *, prepared=None):
                if prepared is None:
                    return original_request(messages, tools)
                payload = json.loads(prepared.wire_json)
                value = json.loads(payload["messages"][-1]["content"])
                core, work = value["core"], value["core"]["current_work"]
                if not resumed:
                    if prepared.stage == 0:
                        from automl_agent.qwen import QwenContextError
                        controller.journal.event("acceptance_fault", {"kind": "provider_context_rejection", "synthetic_history": 300})
                        raise QwenContextError("Declared provider context rejection")
                    controller.journal.put("context_recovery_stage", prepared.stage)
                    controller.journal.event("acceptance_fault", {"kind": "abrupt_restart_after_compaction"})
                    raise InjectedRestart()
                checks.append({"task": core["identity"]["task_hash"] == anchors["task"],
                    "protocol": core["identity"]["protocol_hash"] == anchors["protocol"],
                    "baseline": work["baseline_id"] == anchors["baseline"],
                    "leader": work["leader_id"] == (anchors["leader"] if not checks else controller.journal.get("leader_id")),
                    "negative_result": any(entry["status"] == "unsupported_mode" for entry in value["known_failures"]),
                    "reference_index": any(item["operation_id"] == "setup_reference" for item in value["observations"]),
                    "request_fits": prepared.conservative_size <= controller.qwen.context_tokens,
                    "request_sha256": prepared.request_hash})
                response = original_request(messages, tools, prepared=prepared)
                requests[0] += 1
                return response
            controller.qwen._request = request

        attach(c, False)
        try:
            report = c.run()
        except InjectedRestart:
            c.close()
            c = Controller(path, Limits(seconds=seconds + 999, cpu=1, memory_mb=2048, max_candidates=6, max_calls=7),
                run_id="live", transport=transport, research_focus=focus)
            restarted = True
            before = {"deadline": c.budget.deadline, "candidates": c.budget.counter("candidate_count"),
                "calls": c.budget.counter("llm_calls"), "compaction": c.journal.get("context_recovery_stage", 0)}
            attach(c, True)
            report = c.run()
        decisions = [json.loads(row[0])["command"] for row in c.journal.db.execute("SELECT payload FROM events WHERE kind='qwen_decision' ORDER BY seq")]
        violation = any(command["tool"] == "inspect_dataset" and command["arguments"].get("scope", "fit") != "fit"
            and (command["arguments"].get("include_target") or command["arguments"].get("predicate")) for command in decisions)
        meaningful = [command for command in decisions if command["tool"] in {
            "train_model", "tune_model", "run_automl", "run_custom_experiment", "build_ensemble", "finish"}]
        next_choice = False
        if meaningful:
            operation = c.journal.operation(meaningful[0]["operation_id"])
            next_choice = operation is not None and operation.status == "succeeded" and (
                meaningful[0]["tool"] == "finish" or (operation.result or {}).get("result", {}).get("status") == "completed")
        verification = {"real_request_succeeded": requests[0] > 0,
            "overflow_compacted_and_restarted": restarted and before["compaction"] >= 1,
            "original_deadline_and_counters": restarted and all(before[key] == anchors[key] for key in ("deadline", "candidates", "calls")),
            "memory_in_actual_requests": bool(checks) and all(all(value for key, value in check.items() if key != "request_sha256") for check in checks),
            "next_meaningful_choice_succeeded": next_choice,
            "no_forbidden_slice_attempt": not violation,
            "control_and_refit": report["control_status"] == report["refit_status"] == "completed",
            "verified_published_csv": bool(report["publication"]) and c.publisher.output.is_file()
                and report["publication"]["csv_sha256"] == file_hash(c.publisher.output),
            "code_unchanged_during_case": initial_code == fingerprint()}
        passed = all(verification.values()) and report["state"] == "completed"
        outcome = {"scenario": "context_recovery", "repeat": repeat, "passed": passed, "state": report["state"],
            "valid_csv": verification["verified_published_csv"], "trained_result": report["first_trained_csv_seconds"] is not None,
            "fault_injection": {"count": 2 if restarted else 1, "kind": "context_overflow_and_restart", "synthetic_history": 300},
            "successful_live_requests": requests[0], "calls": c.budget.counter("llm_calls"), "candidates": c.budget.counter("candidate_count"),
            "control": report["control_status"], "refit": report["refit_status"], "token_usage": report["token_usage"],
            "checks": verification, "context_checks": checks, "protocol_version": protocol_version, "code_fingerprint": initial_code,
            "forbidden_external_slice_attempt": violation,
            "decisions": decisions, "run_report": str(c.workdir / "run_report.json")}
        outcome["data_version"] = c.dataset.source_hash
        outcome["model_configuration"] = report.get("model_configuration")
        outcome["research_stop_reason"] = report.get("research_stop_reason")
        atomic_json(directory / "acceptance.json", outcome)
        return outcome
    finally:
        c.close()
