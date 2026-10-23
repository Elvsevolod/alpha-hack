"""Bound the complete routed discovery response, including both continuations."""
from test_tool_api_v2 import agent, call
from automl_agent.controller import SYSTEM
from automl_agent.util import canonical
from automl_agent.tool_results import envelope
from types import SimpleNamespace


def test_search_space_full_envelope_respects_intermediate_budgets(agent):
    agent.context_memory.snapshot(SYSTEM)
    for maximum in (3000, 3500, 4000, 6000):
        result = call(agent, 'search_knowledge', {
            'collection': 'search_spaces',
            'query': 'LightGBM optuna search space n_estimators learning_rate num_leaves min_child_samples subsample colsample_bytree',
            'limit': 5, 'max_chars': maximum, 'strategy': 'auto'}, 'space_budget_' + str(maximum))
        assert len(canonical(result)) <= maximum, (maximum, len(canonical(result)), result['status'])
        worst = envelope(agent, SimpleNamespace(tool='search_knowledge', operation_id='s' * 100), result['data'])
        assert len(canonical(worst)) <= maximum
        if maximum >= 4000:
            assert result['status'] == 'success' and result['data']['entries']
            assert result['continuation']['arguments']['cursor'] == result['data']['next_cursor']
