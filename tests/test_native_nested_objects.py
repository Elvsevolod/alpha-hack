import copy
import json

import pytest

from automl_agent.contracts import TOOL_SCHEMAS
from context_compressor.transport import Transport


def native(pipeline=None, **extra):
    args = {'operation_id': 'nested_objects', 'pipeline': pipeline or json.dumps({'model': 'ridge'}),
        'hypothesis': json.dumps({'observation': 'Fixed split', 'expected_effect': 'Measure error',
            'changed_component': 'model', 'acceptance': 'Validation score'}), **extra}
    return {'choices': [{'finish_reason': 'tool_calls', 'message': {'tool_calls': [
        {'type': 'function', 'function': {'name': 'train_model', 'arguments': json.dumps(args)}}]}}]}


def transport():
    return Transport(system='One allowed operation.',
        tool_schemas={n: s.model_json_schema() for n, s in TOOL_SCHEMAS.items()})


def test_native_json_encoded_objects_reach_the_same_typed_command_without_changing_raw_response():
    response = native(); original = copy.deepcopy(response)
    command = transport().parse_response(response, 'native')
    args = TOOL_SCHEMAS[command.tool].model_validate(command.as_dict()['arguments'])
    assert args.pipeline.model == 'ridge' and args.hypothesis.changed_component == 'model'
    assert response == original


@pytest.mark.parametrize('encoded', ['[]', '"nested string"', '{"model":"ridge"} trailing',
    '{"model":"ridge","model":"lightgbm"}', '{"hyperparameters":{"alpha":NaN}}',
    '{"hyperparameters":{"alpha":1e999}}'])
def test_bad_native_object_encoding_cannot_be_admitted(encoded):
    with pytest.raises(ValueError):
        command = transport().parse_response(native(encoded), 'native')
        TOOL_SCHEMAS[command.tool].model_validate(command.as_dict()['arguments'])


def test_decoding_does_not_bypass_pipeline_schema_or_decode_hyperparameters():
    command = transport().parse_response(native(json.dumps({'model': 'ridge', 'download_model': True})), 'native')
    with pytest.raises(ValueError):
        TOOL_SCHEMAS[command.tool].model_validate(command.as_dict()['arguments'])
    encoded = '{"a":1}'
    command = transport().parse_response(native(json.dumps({'model': 'ridge', 'hyperparameters': {'literal': encoded}})), 'native')
    args = TOOL_SCHEMAS[command.tool].model_validate(command.as_dict()['arguments'])
    assert args.pipeline.hyperparameters['literal'] == encoded


def test_json_channel_still_requires_objects_and_does_not_repair_strings():
    response = native()
    args = json.loads(response['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
    operation = args.pop('operation_id')
    response = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(
        {'operation_id': operation, 'tool': 'train_model', 'arguments': args})}}]}
    command = transport().parse_response(response, 'json')
    with pytest.raises(ValueError):
        TOOL_SCHEMAS[command.tool].model_validate(command.as_dict()['arguments'])
