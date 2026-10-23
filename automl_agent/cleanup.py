"""Recoverable removal of reproducible inputs and unreachable model artifacts."""
from pathlib import Path
import shutil
import time

from .util import digest, file_hash
from .diagnostics import snapshot


def protected_ids(controller):
    c = controller
    roots = set(filter(None, [c.journal.get("baseline_id"), c.journal.get("leader_id"), c.journal.get("control_winner")]))
    roots.update(c.journal.get("finalization_plan", {}).get("finalists", []))
    roots.update(c.journal.get("finalist_pool_ids", []))
    for name in ("publication", "publication_pending"):
        entry = c.journal.get(name) or {}
        if entry.get("experiment_id") != "constant":
            roots.update(filter(None, [entry.get("experiment_id"), *entry.get("dependencies", [])]))
    for operation in c.journal.incomplete_operations():
        if operation.experiment_id:
            roots.add(operation.experiment_id)
    revision = c.journal.get("protocol_revision") or {}
    if revision.get("status") not in {None, "completed", "aborted"}:
        roots.update(filter(None, [revision.get("baseline_id"), revision.get("leader_id")]))
        for item in revision.get("candidates", []):
            roots.add(item["old_id"])
            spec_id = item["spec"]["experiment_id"]
            if any(spec.experiment_id == spec_id and result and result.status == "completed" for spec, result in c.records()):
                roots.add(spec_id)
    protected, pending = set(), list(roots)
    while pending:
        identifier = pending.pop()
        if identifier in protected:
            continue
        protected.add(identifier)
        try:
            spec, _ = c.lookup(identifier)
            pending.extend(spec.pipeline.members)
        except ValueError:
            pass
    return protected


def recover_cleanup(controller):
    c = controller
    operation = c.journal.get("cleanup_operation")
    if not operation or operation["status"] == "completed":
        return
    protected = protected_ids(c)
    for item in operation["items"]:
        if item["status"] != "pending":
            continue
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(c.workdir.resolve()):
            raise ValueError("Cleanup path escaped the run directory")
        if item["kind"] == "candidate" and item["experiment_id"] in protected:
            item["status"] = "newly_protected"
        elif item["kind"] == "staging":
            if path.exists() and file_hash(path) != item["sha256"]:
                item["status"] = "changed_since_plan"
            else:
                path.unlink(missing_ok=True)
                item["status"] = "removed"
        else:
            if path.is_dir():
                snapshot(path)
                for child in path.iterdir():
                    if child.is_file() and (child.suffix in {".py", ".log"} or child.name in {
                            "request.json", "worker_result.json", "evaluated_result.json", "process.json", "output_schema.json"}
                            or child.name.startswith("flaml_")):
                        continue
                    shutil.rmtree(child) if child.is_dir() else child.unlink()
            removed = c.journal.get("removed_artifacts", {})
            removed[item["experiment_id"]] = {"reason": operation["reason"], "operation_id": operation["id"]}
            c.journal.put("removed_artifacts", removed)
            item["status"] = "removed"
        c.journal.put("cleanup_operation", operation)
    operation["status"] = "completed"
    c.journal.put("cleanup_operation", operation)
    c.journal.event("cleanup_completed", operation)


def staging(controller, directory):
    """Call only after the worker exits and its outputs have been verified."""
    recover_cleanup(controller)
    files = sorted([*directory.glob("*_X.json"), *directory.glob("fit_y.json")])
    if not files:
        return
    items = [{"path": str(path), "sha256": file_hash(path), "kind": "staging", "status": "pending"} for path in files]
    operation = {"id": digest(items), "status": "pending", "reason": "verified_worker_reproducible_staging",
        "items": items, "time": time.time()}
    controller.journal.put("cleanup_operation", operation)
    controller.journal.event("cleanup_planned", operation)
    recover_cleanup(controller)


def reclaim(controller, required_free_bytes):
    c = controller
    recover_cleanup(c)
    if shutil.disk_usage(c.workdir).free >= required_free_bytes:
        return
    protected = protected_ids(c)
    items = []
    removed = c.journal.get("removed_artifacts", {})
    for spec, result in c.records():
        if spec.experiment_id in protected or spec.experiment_id in removed or not result:
            continue
        directory = (c.workdir / "attempts" / spec.experiment_id).resolve()
        if not directory.is_dir():
            continue
        # Metadata, source, scores and operation history remain in the journal.
        items.append({"path": str(directory), "experiment_id": spec.experiment_id, "kind": "candidate", "status": "pending"})
    operation = {"id": digest(items), "status": "pending", "reason": "disk_pressure_unreachable_candidates",
        "required_free_bytes": required_free_bytes, "items": items, "time": time.time()}
    c.journal.put("cleanup_operation", operation)
    c.journal.event("cleanup_planned", operation)
    recover_cleanup(c)
