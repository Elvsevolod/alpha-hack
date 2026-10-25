"""Parameter inspection is read-only; the normal controller fixture prepares a baseline."""
from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from automl_agent.contracts import KnowledgeReadArgs,KnowledgeSearchArgs,ToolCommand
from automl_agent.controller import Controller
from automl_agent.budget import Limits
from automl_agent.knowledge_tools import CursorError
from automl_agent.parameter_docs import parameter_section
from automl_agent.tool_results import envelope
from automl_agent.validation import choose_validation
from conftest import write_task


@pytest.fixture(scope='module')
def agent(tmp_path_factory):
    root=tmp_path_factory.mktemp('parameters')
    a=Controller(write_task(root/'task'),Limits(seconds=180,cpu=1),run_id='parameters')
    a.no_llm=True
    a.prepare()
    yield a
    assert a.budget.counter('llm_calls')==0
    a.close()


def args(agent,**kwargs):
    return KnowledgeReadArgs(collection='documents',id='api_sklearn_ensemble_HistGradientBoostingClassifier',
        corpus_version=agent.references.version,view='parameters',**kwargs)


def test_existing_exact_symbol_search_and_parameter_read(agent):
    found=agent.knowledge_tools.discover(KnowledgeSearchArgs(collection='documents',query='sklearn.ensemble.HistGradientBoostingClassifier',strategy='exact_symbol'))
    assert found['entries']
    read=agent.knowledge_tools.read(args(agent,parameter='max_iter'))
    assert read['status']=='success',read
    assert 'max_iter : int, default=100' in read['entry']['content']
    assert 'number of iterations' in read['entry']['content']
    assert read['entry']['version']=='1.5.2'
    assert read['entry']['parameter_view']['adapter']['model']=='hist_gradient_boosting'
    assert 'max_iter' in read['entry']['parameter_view']['adapter']['allowed_hyperparameters']
    assert 'n_estimators' not in read['entry']['parameter_view']['adapter']['allowed_hyperparameters']
    assert read['entry']['doc_sha256']==agent.references.document(args(agent).id)['doc_sha256']
    assert 'n_jobs' not in read['entry']['parameter_view']['adapter']['allowed_hyperparameters']


def test_parameter_pagination_is_bounded_stable_and_original_source_offsets(agent):
    request=args(agent,max_chars=5000)
    pages=[];cursor=None
    for _ in range(50):
        response=agent.knowledge_tools.read(request.model_copy(update={'cursor':cursor}))
        assert response['status']=='success',response
        public=envelope(agent,SimpleNamespace(tool='read_knowledge',operation_id='0'*100),response)
        assert len(json.dumps(public,ensure_ascii=False,sort_keys=True,separators=(',',':')))<=5000
        record=agent.references.document(request.id)
        start,end=response['range']['start'],response['range']['end']
        assert response['entry']['content']==record['text'][start:end]
        pages.append(response)
        if not response['next_cursor']:break
        cursor=response['next_cursor']
    else:pytest.fail('Parameter pagination did not terminate')
    assert len(pages)>1
    assert all(a['range']['end']==b['range']['start'] for a,b in zip(pages,pages[1:]))
    assert 'max_iter' in ''.join(p['entry']['content'] for p in pages)
    first=agent.knowledge_tools.read(request)
    assert first==pages[0]
    with pytest.raises(CursorError):
        agent.knowledge_tools.read(request.model_copy(update={'parameter':'max_iter','cursor':first['next_cursor']}))
    with pytest.raises(CursorError):
        agent.knowledge_tools.read(request.model_copy(update={'view':'document','cursor':first['next_cursor']}))


def test_library_defaults_are_not_adapter_permission(agent):
    response=agent.knowledge_tools.read(args(agent,parameter='class_weight'))
    assert response['status']=='success'
    assert 'class_weight' in response['entry']['content']
    assert 'class_weight' in response['entry']['parameter_view']['adapter']['library_only_parameters']
    unknown=agent.knowledge_tools.read(args(agent,parameter='n_estimators'))
    assert unknown['status']=='no_matches' and unknown['error_code']=='unknown_parameter'
    assert agent.knowledge_tools.read(args(agent).model_copy(update={'corpus_version':'wrong'}))['status']=='missing_version'


def test_parameter_view_is_available_through_normal_dispatch(agent):
    result=agent.dispatch(ToolCommand(tool='read_knowledge',operation_id='inspect-parameters',
        arguments=args(agent,parameter='max_iter').model_dump(mode='json')))
    assert result['status']=='success'
    assert 'max_iter' in result['data']['entry']['content']
    assert agent.journal.operation('inspect-parameters').status=='succeeded'


def test_parameter_projection_handles_unknown_formats_without_inventing_values():
    assert parameter_section({'text':'an API fragment without a parameter section'}) is None
    with pytest.raises(ValueError,match='requires view'):
        KnowledgeReadArgs(collection='documents',id='x',corpus_version='v',parameter='x')


@pytest.mark.parametrize('identifier,parameter,model',[
    ('api_sklearn_ensemble_RandomForestClassifier','n_estimators','random_forest'),
    ('api_lightgbm_LGBMClassifier','n_estimators','lightgbm'),
    ('api_sklearn_preprocessing_OneHotEncoder','handle_unknown',None),
])
def test_parameter_views_cover_other_models_and_preprocessing(agent,identifier,parameter,model):
    request=args(agent,parameter=parameter).model_copy(update={'id':identifier})
    result=agent.knowledge_tools.read(request)
    assert result['status']=='success',result
    assert parameter in result['entry']['content']
    adapter=result['entry']['parameter_view']['adapter']
    assert (adapter['model'] if adapter else None)==model

    if identifier=='api_lightgbm_LGBMClassifier':
        view=result['entry']['parameter_view']
        assert view['detail_status']=='signature_only'
        assert view['declarations']==[{'name':'n_estimators','has_default':True,'default_text':'100','annotation_text':'int'}]
        assert view['further_documentation']['arguments']['filters']=={'library':'lightgbm','version':'4.6.0'}
        assert not result['entry']['answer_usable']


def test_document_view_keeps_legacy_cursor_and_default_argument_identity(agent):
    from automl_agent.knowledge_tools import cursor
    from automl_agent.contracts import request_arguments
    from automl_agent.util import digest
    request = KnowledgeReadArgs(collection='documents', id='api_sklearn_ensemble_HistGradientBoostingClassifier',
        corpus_version=agent.references.version, max_chars=4000)
    record = agent.references.document(request.id)
    expected = {'tool': 'read_knowledge', 'corpus': agent.references.version,
        'request': digest({'collection': request.collection, 'id': request.id, 'corpus_version': request.corpus_version}),
        'doc_sha256': record['doc_sha256']}
    response = agent.knowledge_tools.read(request)
    assert response['next_cursor'] == cursor(expected, response['range']['end'])
    assert 'view' not in response['continuation']['arguments'] and 'parameter' not in response['continuation']['arguments']
    assert 'view' not in request_arguments(request)
    assert response == agent.knowledge_tools.read(request.model_copy(update={'view': 'document'}))
