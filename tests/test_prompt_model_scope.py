import pytest

from automl_agent.action_cards import cards
from automl_agent.compatibility import require_compatible
from automl_agent.contracts import PipelineSpec
from automl_agent.custom import validate_code
from automl_agent.journal import Journal
from automl_agent.prompting import SECTIONS, bind_prompt, build_system_prompt
from automl_agent.research import RESEARCH_POLICY
from automl_agent.util import digest


def test_authority_acknowledges_existing_optional_neural_action(tabular):
    _, _, task = tabular
    card = cards(task, 'random')['action:torch_tabular']
    assert card['applicable']
    require_compatible(PipelineSpec.model_validate(card['pipeline']), task, 'random')
    authority = SECTIONS['authority_and_scope']
    assert 'torch_tabular' in authority, 'An executable optional card must not contradict higher-priority scope'
    assert 'experimental' in authority.lower() and 'optional' in authority.lower()
    assert 'not a default' in authority.lower()
    assert 'torch_tabular' in RESEARCH_POLICY and 'torch_tabular' in build_system_prompt()


@pytest.mark.parametrize('code', [
    'import torch\ndef build_transformer(context): return None',
    'from sklearn.neural_network import MLPRegressor\ndef build_transformer(context): return None',
])
def test_scope_description_does_not_enable_arbitrary_custom_neural_imports(code):
    with pytest.raises(ValueError):
        validate_code(code, 'transformer')


def test_sixth_prompt_remains_pinned_and_new_scope_has_a_distinct_version(tmp_path):
    with_journal = Journal(tmp_path)
    saved = dict(version=6, system='Frozen classical-only predecessor',
        system_hash=digest('Frozen classical-only predecessor'), guides={}, guides_hash=digest({}), sections={})
    try:
        with_journal.put('llm_prompt_contract', saved)
        assert bind_prompt(with_journal, 'legacy', existing_run=True) == saved
    finally:
        with_journal.close()
    fresh = Journal(tmp_path/'fresh')
    try:
        bound = bind_prompt(fresh, 'legacy', existing_run=False)
        assert bound['version'] == 7
        assert 'torch_tabular' in bound['system']
        assert bind_prompt(fresh, 'different', existing_run=True) == bound
    finally:
        fresh.close()
