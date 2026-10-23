"""Assemble a complete Chat Completions SSE response before parsing actions."""
import json


class IncompleteModelStream(ValueError):
    pass


class ProviderStreamError(ValueError):
    """An explicitly terminated provider failure; never contains an action."""
    def __init__(self, code):
        self.code = code
        super().__init__(f"Provider stream failed with code {code}")


def validate_completion(body):
    if not isinstance(body, dict):
        raise IncompleteModelStream('Expected a complete response object')
    choices, usage = body.get('choices'), body.get('usage')
    if (not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict)
            or not isinstance(choices[0].get('message'), dict)
            or not isinstance(choices[0].get('finish_reason'), str)
            or choices[0].get('finish_reason') not in {'stop','tool_calls','length','content_filter','function_call'}):
        raise IncompleteModelStream('Missing complete response choice')
    if usage is not None and (not isinstance(usage, dict)
            or usage.get('completion_tokens_details') is not None
            and not isinstance(usage['completion_tokens_details'], dict)):
        raise IncompleteModelStream('Invalid provider usage object')
    if usage is None:
        body['usage'] = {}
    return body


def completion_from_sse(lines):
    def events():
        data = []
        for line in lines:
            if not line:
                if data:
                    yield '\n'.join(data), True
                    data.clear()
            elif line.startswith('data:'):
                data.append(line[5:].removeprefix(' '))
        if data:
            yield '\n'.join(data), False

    body, content, calls = {}, [], {}
    finish, done, reasoning_chars, refusal = None, False, 0, None
    try:
        for data, framed in events():
            if data == '[DONE]':
                done = True
                break
            chunk = json.loads(data)
            if not isinstance(chunk, dict):
                raise ValueError('Invalid provider stream event')
            if 'error' in chunk:
                error = chunk['error']
                code = error.get('code') if isinstance(error, dict) else None
                if (not framed or not isinstance(error, dict)
                        or not isinstance(error.get('message'), str) or not error['message'].strip()
                        or type(code) not in (int, str) or str(code) not in {str(n) for n in range(400, 600)}):
                    raise ValueError('Malformed provider error event')
                raise ProviderStreamError(int(code))
            if chunk.get('usage') is not None and not isinstance(chunk['usage'], dict):
                raise ValueError('Invalid provider usage object')
            details = (chunk.get('usage') or {}).get('completion_tokens_details')
            if details is not None and not isinstance(details, dict):
                raise ValueError('Invalid provider usage object')
            for key in ('id', 'model', 'usage'):
                if chunk.get(key) is not None:
                    body[key] = chunk[key]
            for choice in chunk.get('choices', []):
                if choice.get('index', 0) != 0:
                    raise ValueError('Expected one response choice')
                reason = choice.get('finish_reason')
                if reason is not None and (not isinstance(reason, str)
                        or reason not in {'stop','tool_calls','length','content_filter','function_call'}):
                    raise ValueError('Invalid final choice')
                delta = choice.get('delta') or {}
                if finish is not None and any(delta.get(key) for key in
                        ('content', 'tool_calls', 'reasoning_content', 'reasoning', 'refusal')):
                    raise ValueError('Received output after final choice')
                if finish is not None and choice.get('finish_reason') not in (None, finish):
                    raise ValueError('Conflicting final choices')
                if delta.get('content') is not None:
                    if not isinstance(delta['content'], str):
                        raise ValueError('Expected text content')
                    content.append(delta['content'])
                reasoning_chars += len(delta.get('reasoning_content') or delta.get('reasoning') or '')
                refusal = delta.get('refusal') or refusal
                for call in delta.get('tool_calls') or []:
                    index = call['index']
                    if type(index) is not int or index < 0:
                        raise ValueError('Invalid tool call index')
                    current = calls.setdefault(index, {'id':'', 'type':'function',
                        'function':{'name':'', 'arguments':''}})
                    if call.get('type') not in (None, 'function'):
                        raise ValueError('Expected a function call')
                    current['id'] += call.get('id') or ''
                    function = call.get('function') or {}
                    for key in ('name', 'arguments'):
                        current['function'][key] += function.get(key) or ''
                finish = choice.get('finish_reason') or finish
        if not done or finish is None:
            raise ValueError('Missing final choice or stream terminator')
    except ProviderStreamError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise IncompleteModelStream('Incomplete or malformed model stream') from error
    message = {'role':'assistant', 'content':''.join(content), 'refusal':refusal}
    if calls:
        message['tool_calls'] = [calls[index] for index in sorted(calls)]
    body.update(choices=[{'index':0, 'finish_reason':finish, 'message':message}],
        stream_reasoning_chars=reasoning_chars)
    return validate_completion(body)
