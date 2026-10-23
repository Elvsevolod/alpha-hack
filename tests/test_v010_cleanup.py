import pytest

from automl_agent.budget import Limits
from automl_agent.cleanup import protected_ids, reclaim, recover_cleanup
from automl_agent.contracts import PipelineSpec
from automl_agent.controller import Controller
from automl_agent.util import file_hash
from conftest import write_task


def test_pending_nested_operation_protects_all_descendants(tmp_path):
    c = Controller(write_task(tmp_path), Limits(seconds=120, cpu=1))
    try:
        c.prepare()
        base, result = c.lookup("baseline")
        for name in ("leaf_b", "leaf_c"):
            spec = base.model_copy(update={"experiment_id": name})
            c.journal.save_experiment(spec, result.model_copy(update={"experiment_id": name}))
        nested = base.model_copy(update={"experiment_id": "nested", "pipeline": PipelineSpec(engine="ensemble", members=["baseline", "leaf_b"])})
        c.journal.save_experiment(nested, result.model_copy(update={"experiment_id": "nested"}))
        pending = base.model_copy(update={"experiment_id": "pending", "pipeline": PipelineSpec(engine="ensemble", members=["nested", "leaf_c"])})
        c.journal.save_experiment(pending)
        operation, _ = c.journal.begin_operation(c.run_id, "pending_operation", "build_ensemble", {}, experiment_id="pending")
        assert {"baseline", "leaf_b", "leaf_c", "nested", "pending"} <= protected_ids(c)
    finally:
        c.close()


def test_interrupted_cleanup_rechecks_newly_protected_nodes_and_keeps_metadata(tmp_path, monkeypatch):
    c = Controller(write_task(tmp_path), Limits(seconds=120, cpu=1))
    try:
        c.prepare()
        published = file_hash(c.publisher.output)
        base, result = c.lookup("baseline")
        for name in ("unused_a", "unused_b"):
            directory = c.workdir / "attempts" / name
            directory.mkdir()
            (directory / "model.pkl").write_bytes(b"reproducible model fixture")
            (directory / "worker_result.json").write_text('{"status":"completed"}')
            spec = base.model_copy(update={"experiment_id": name})
            c.journal.save_experiment(spec, result.model_copy(update={"experiment_id": name, "artifacts": {"directory": str(directory)}}))
        original = c.journal.put
        def stop_after_first(key, value):
            original(key, value)
            if key == "cleanup_operation" and any(i["status"] == "removed" for i in value["items"]):
                raise KeyboardInterrupt()
        monkeypatch.setattr(c.journal, "put", stop_after_first)
        with pytest.raises(KeyboardInterrupt):
            reclaim(c, 10**18)
        monkeypatch.undo()
        c.journal.put("leader_id", "unused_b")
        recover_cleanup(c)
        assert (c.workdir / "attempts/unused_b/model.pkl").exists()
        assert not (c.workdir / "attempts/unused_a/model.pkl").exists()
        assert (c.workdir / "attempts/unused_a/worker_result.json").exists()
        assert file_hash(c.publisher.output) == published
        recover_cleanup(c)
        assert (c.workdir / "attempts/unused_b/model.pkl").exists()
        assert c.journal.get("cleanup_operation")["status"] == "completed"
    finally:
        c.close()
