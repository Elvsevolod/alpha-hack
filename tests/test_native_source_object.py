import copy
import json

import pytest

from automl_agent.contracts import TOOL_SCHEMAS
from context_compressor.transport import Transport


SOURCE={'source_id':'contract_v1','version':'actual_sha','scope':'contract','offset':0,'limit':768}


def native(source):
    return {'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[
        {'type':'function','function':{'name':'read_run_records',
            'arguments':json.dumps({'operation_id':'read_source','source':source})}}]}}]}


def transport():
    return Transport(system='One operation.',tool_schemas={k:v.model_json_schema() for k,v in TOOL_SCHEMAS.items()})


def test_only_declared_native_source_object_is_decoded_losslessly_once():
    raw=native(json.dumps(SOURCE));original=copy.deepcopy(raw)
    command=transport().parse_response(raw,'native')
    args=TOOL_SCHEMAS[command.tool].model_validate(command.as_dict()['arguments'])
    assert args.source.model_dump()=={**SOURCE,'section':''} and raw==original
    ordinary=transport().parse_response(native(SOURCE),'native')
    assert ordinary.as_dict()==command.as_dict()


@pytest.mark.parametrize('encoded',['[]','null','"nested"','{} trailing',
    '{"scope":"contract","scope":"fit"}','{"offset":NaN}','{"offset":1e999}'])
def test_invalid_source_encoding_never_becomes_an_operation(encoded):
    with pytest.raises(ValueError):
        command=transport().parse_response(native(encoded),'native')
        TOOL_SCHEMAS[command.tool].model_validate(command.as_dict()['arguments'])


@pytest.mark.parametrize('change',[{'scope':'control'},{'limit':100000},{'offset':-1},{'hidden_labels':True}])
def test_decoding_does_not_bypass_source_permissions_or_bounds(change):
    raw=native(json.dumps({**SOURCE,**change}))
    raw['choices'][0]['message']['tool_calls'].append({'type':'function','function':{
        'name':'get_dataset_report','arguments':json.dumps({'operation_id':'valid_second','scope':'fit'})}})
    command=transport().parse_response(raw,'native')
    assert command.tool=='read_run_records' and command.operation_id=='read_source'
    with pytest.raises(ValueError):TOOL_SCHEMAS[command.tool].model_validate(command.as_dict()['arguments'])


def test_string_source_schema_and_json_transport_are_not_coerced():
    schemas={k:v.model_json_schema() for k,v in TOOL_SCHEMAS.items()}
    schemas['read_run_records']['properties']['source']={'type':'string'}
    value=json.dumps(SOURCE)
    command=Transport(system='Literal string source.',tool_schemas=schemas).parse_response(native(value),'native')
    assert command.as_dict()['arguments']['source']==value
    raw={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({
        'operation_id':'read_source','tool':'read_run_records','arguments':{'source':value}})}}]}
    command=transport().parse_response(raw,'json')
    with pytest.raises(ValueError):TOOL_SCHEMAS[command.tool].model_validate(command.as_dict()['arguments'])
