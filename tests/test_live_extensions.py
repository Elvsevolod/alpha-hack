import json

import httpx
import pytest

from automl_agent.data import Dataset
from automl_agent.qwen import QwenClient
from automl_agent.util import atomic_json
from tools.live_acceptance import PROTOCOL_VERSION, code_fingerprint, run_case
from tools.live_acceptance_extensions import run_context_recovery, temporal_fixture


@pytest.mark.parametrize("repeat", [0, 1, 2])
def test_temporal_live_fixture_requires_the_intended_role_resolution(tmp_path, repeat):
    path = temporal_fixture(tmp_path, repeat)
    data = Dataset(path)
    task, questions = data.draft()
    if repeat < 2:
        assert task and task.time_column == "p" and not questions
    else:
        assert task is None and "time_column" in questions


@pytest.mark.parametrize("scenario", ["context_recovery", "temporal_role"])
def test_live_harness_exercises_recovery_and_roles_without_network(tmp_path, monkeypatch, scenario):
    monkeypatch.setenv("OPENAI_API_KEY", "offline_fixture")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fixture/v1")
    original = QwenClient.__init__
    def init(client, *args, **kwargs):
        original(client, *args, **kwargs)
        client.client.close()
        def respond(request):
            body = json.loads(request.content)
            content = json.loads(body['messages'][-1]['content'])
            if 'request_id' in content and 'sources' in content:
                dataset = Dataset(client.journal.directory.parent.parent/'task.md')
                task, questions = dataset.draft()
                assert task and not questions
                spec = task.model_dump(mode='json')
                spec.update(contract_version=2,file_roles={name:name for name in dataset.paths if dataset.paths[name]},source_hash=content['source_hash'])
                proposal = {'request_id':content['request_id'],'source_hash':content['source_hash'],'status':'proposed',
                    'task_spec':spec,'evidence':{field:['task','inventory'] for field in spec}}
                return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(proposal)}}]})
            assert "declared_acceptance_padding" not in body["messages"][-1]["content"]
            arguments = {"operation_id": "after_recovery_finish", "reason": "Finish after the declared context fault and restored research memory",
                "result_ids":["baseline"], "imputation_skip_reason": "Offline harness test; preserve the verified models and exercise complete finalization",
                "skipped_directions": {"all": "This fixed offline recovery fixture verifies finalization without testing model quality."}}
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"tool_calls": [
                {"id": "mock_finish", "type": "function", "function": {"name": "finish", "arguments": json.dumps(arguments)}}]}}]})
        client.client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(QwenClient, "__init__", init)
    outcome = run_context_recovery(0, tmp_path, 180, "auto", code_fingerprint, PROTOCOL_VERSION) if scenario == "context_recovery" else run_case("temporal_role", 0, tmp_path, 180)
    assert outcome["passed"], outcome
    assert outcome["successful_live_requests"] == (2 if scenario == 'temporal_role' else 1)
    assert outcome["fault_injection"]["count"] == (2 if scenario == "context_recovery" else 0)
    atomic_json(tmp_path / "MOCK_MODEL_SCOPE.json", {"scope": "offline harness test with mocked HTTP; no organizer API calls",
        "checks": outcome["checks"], "not_a_live_acceptance_result": True})
