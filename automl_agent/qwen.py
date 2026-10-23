from __future__ import annotations

import json
import os
import signal
import threading
import time
from typing import Any

import httpx

from .contracts import TOOL_SCHEMAS, TOOL_DESCRIPTIONS, ToolCommand
from .util import atomic_json, canonical, digest
from .diagnostics import emit
from .qwen_stream import IncompleteModelStream, ProviderStreamError, completion_from_sse, validate_completion


GENERATION = {"temperature": 1.0, "top_p": .95, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
OUTPUT_TOKENS = {"medium": 16384, "xhigh": 32768}


class QwenError(RuntimeError):
    pass


class QwenUnavailable(QwenError):
    """The supplied credentials or account budget cannot be retried usefully."""


class QwenOutcomeUnknown(QwenUnavailable):
    """A sent generation may have completed; do not send another blindly."""


class QwenRetryableError(QwenError):
    """A transient failure does not establish that tool calling is unsupported."""


class QwenContextError(QwenError):
    """The mandatory request cannot fit; requires stronger controller compaction."""


def request_size(payload, response_reserve):
    return len(canonical(payload).encode("utf-8")) + response_reserve


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    elif text.startswith("```\n") and text.endswith("```"):
        text = text[4:-3].strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("Expected a single JSON object")
    result, end = json.JSONDecoder().raw_decode(text, start)
    if text[end:].strip():
        raise ValueError("Unexpected content after the single JSON object")
    if not isinstance(result, dict):
        raise ValueError("Expected a single JSON object")
    return result


def native_tools():
    result = []
    descriptions = TOOL_DESCRIPTIONS
    for name, schema in TOOL_SCHEMAS.items():
        parameters = schema.model_json_schema()
        parameters.setdefault("properties", {})["operation_id"] = {"type": "string", "description": "Stable unique ID; reuse for retries only"}
        parameters.setdefault("required", []).append("operation_id")
        result.append({"type": "function", "function": {"name": name, "description": descriptions[name], "parameters": parameters}})
    return result


def json_tool_schemas():
    definitions, arguments = {}, {}
    for name, model in TOOL_SCHEMAS.items():
        schema = model.model_json_schema()
        for identifier, value in schema.pop("$defs", {}).items():
            if identifier in definitions and definitions[identifier] != value:
                raise ValueError("Conflicting tool schema definitions")
            definitions[identifier] = value
        arguments[name] = schema
    return arguments, definitions


def command_messages(system, context, native):
    if context.get("research_focus"):
        system += "\nDevelopment objective, subject to all fixed contracts and permissions: " + canonical(context["research_focus"])
    request = context
    if not native:
        arguments, definitions = json_tool_schemas()
        request = {**context, "command_schema": ToolCommand.model_json_schema(),
            "tool_argument_schemas": arguments, "$defs": definitions}
        system += "\nReturn exactly one JSON ToolCommand object. No prose or code fences."
    return [{"role": "system", "content": system}, {"role": "user", "content": canonical(request)}]


class QwenClient:
    def __init__(self, journal, budget, transport="auto"):
        self.journal, self.budget = journal, budget
        self.key = os.environ.get("OPENAI_API_KEY", "")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
        self.model = os.environ.get("OPENAI_MODEL", "").strip() or "qwen/qwen3.8-27b"
        self.context_tokens = int(os.environ.get("AGENT_CONTEXT_TOKENS", "262144"))
        self.output_floor = max(0, int(os.environ.get("AGENT_OUTPUT_TOKENS", "0")))
        self.output_by_effort = journal.get("qwen_configuration", {}).get("output_by_effort", {})
        self.extra = json.loads(os.environ.get("OPENAI_EXTRA_BODY", "{}"))
        if {"model", "messages", "tools", "stream", "stream_options", "max_tokens", "max_completion_tokens", "tool_choice", "parallel_tool_calls", "response_format"} & set(self.extra):
            raise ValueError("Request extras cannot override the transport contract")
        managed = set(GENERATION) | {"reasoning", "reasoning_effort", "enable_thinking", "chat_template_kwargs", "include_reasoning"}
        ignored = sorted(set(self.extra) & managed)
        if ignored:
            journal.event("managed_generation_parameters", {"ignored_environment_fields": ignored,
                "reason": "Use the requested temperature and automatic medium/xhigh policy"})
        self.extra = {k: v for k, v in self.extra.items() if k not in managed}
        self.extra.update(GENERATION)
        self.transport = journal.get("transport", transport)
        self.reserve_seconds = 0
        self.client = httpx.Client(follow_redirects=False)
        self.select_reasoning("medium", "ordinary_step")

    def select_reasoning(self, effort, reason):
        self.effort = effort
        self.output_tokens = max(OUTPUT_TOKENS[effort], self.output_floor, self.output_by_effort.get(effort, 0))
        self.extra["reasoning"] = {"enabled": True, "effort": effort}
        self.reasoning_reason = reason
        self.journal.put("qwen_configuration", {"model": self.model, "model_fingerprint": digest(self.model),
            "endpoint_fingerprint": digest(self.base_url), "context_tokens": self.context_tokens,
            "output_tokens": self.output_tokens, "output_by_effort": self.output_by_effort,
            "generation": self.extra, "reasoning_policy": {"ordinary": "medium", "complex": "xhigh"},
            "reasoning_reason": reason,
            "token_accounting": "server_usage; context_guard_uses_conservative_utf8_bytes"})

    def select_for_step(self):
        last = self.journal.get("last_tool_result") or {}
        tool = self.journal.get("last_research_tool")
        memory = getattr(self, "context_memory", None)
        note = memory.store.active_note() if memory is not None else None
        if last.get("error") or last.get("status") in {"failed", "error", "invalid_command_or_api", "search_direction_review"}:
            self.select_reasoning("xhigh", "error_or_stalled_search")
        elif self.journal.get("stagnation_count", 0) >= 3:
            self.select_reasoning("xhigh", "reconsider_search_direction")
        elif note and note.get("status") == "ready_to_compare":
            self.select_reasoning("xhigh", "evidence_ready_for_complex_step")
        elif tool in {"train_model", "run_automl", "tune_model", "run_custom_experiment", "analyze_experiments", "build_ensemble", "revise_validation"}:
            self.select_reasoning("xhigh", "interpret_experiment_and_plan_next_change")
        elif tool is None:
            self.select_reasoning("xhigh", "initial_research_plan")
        else:
            self.select_reasoning("medium", "ordinary_evidence_step")

    @property
    def configured(self):
        return bool(self.key and self.base_url and self.model)

    def _request(self, messages, tools=None, *, prepared=None):
        if threading.current_thread() is not threading.main_thread():
            raise QwenError("Model requests require the controller main thread for deadline enforcement")
        unknown = self.journal.get('qwen_unknown_outcome')
        inflight = self.journal.get('llm_inflight')
        if inflight and not unknown:
            unknown = {'call_id':inflight.get('call_id',self.journal.get('llm_calls')),'epoch':time.time(),
                'error_type':'InterruptedRequest','reason':'inflight_request_from_previous_process'}
            self.journal.put('qwen_unknown_outcome',unknown)
        if unknown:
            raise QwenOutcomeUnknown('Previous model generation has unknown outcome; call_id=' + str(unknown['call_id']))
        if self.journal.get('qwen_provider_failure_streak', 0) >= 3:
            raise QwenUnavailable('Three consecutive explicit provider stream failures; retain saved results')
        if not self.configured:
            raise QwenError("OPENAI_API_KEY and OPENAI_BASE_URL are required")
        url = httpx.URL(self.base_url)
        if url.scheme not in {"http", "https"} or not url.host:
            raise QwenUnavailable("The model endpoint must be an HTTP or HTTPS URL")
        payload = {"model": self.model, "messages": messages, "max_tokens": self.output_tokens,
            "stream": True, "stream_options": {"include_usage": True}, **self.extra}
        if tools:
            payload.update(tools=tools, parallel_tool_calls=False, tool_choice="required")
        else:
            # Every non-tool channel returns a JSON contract, including the
            # semantic task proposal before compressed research begins.
            payload["response_format"] = {"type": "json_object"}
        if prepared is not None:
            payload = json.loads(prepared.wire_json)
            tools = payload.get("tools")
        if request_size(payload, self.output_tokens) > self.context_tokens:
            raise QwenContextError("Conservative context guard exceeded; compact optional context first")
        if prepared is not None:
            self.context_memory.c.knowledge.audit(prepared, payload)
        reserve = max(self.reserve_seconds, self.budget.final_reserve())
        call_id = self.budget.claim_call(reserve)
        self.last_call_id = call_id
        request_path = self.journal.directory / "requests" / f"{call_id:08d}.json"
        atomic_json(request_path, payload)
        self.journal.event("qwen_request", {"call_id": call_id, "request_sha256": digest(payload),
            "artifact": str(request_path), "transport": "native" if tools else "json",
            "reasoning_effort": payload.get("reasoning", {}).get("effort"),
            "reasoning_reason": self.reasoning_reason, "temperature": payload.get("temperature"),
            "conservative_size": request_size(payload, self.output_tokens)})
        started = time.monotonic()
        self.journal.put("llm_inflight", {"call_id":call_id,"started": time.time(), "transport": "native" if tools else "json"})
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        available = self.budget.remaining - reserve
        if self.budget.limits.max_llm_seconds is not None:
            available = min(available, self.budget.limits.max_llm_seconds - self.journal.get("llm_seconds", 0))
        timeout = max(.01, available)
        wire = prepared.wire_json.encode('utf-8') if prepared is not None else json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
        import hashlib
        emit('qwen_exchange_start', call_id=call_id, request_id=prepared.request_id if prepared is not None else None,
             request_id_missing_reason=None if prepared is not None else 'task_resolution_or_legacy_transport',
             wire_sha256=hashlib.sha256(wire).hexdigest(), payload=payload, reserve_seconds=reserve,
             response_deadline_epoch=time.time()+timeout)
        def alarm(signum, frame):
            raise QwenOutcomeUnknown("Model request exceeded its wall-clock allocation; outcome unknown")
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        unknown_reason = 'wall_clock_outcome_unknown'
        try:
            with self.client.stream("POST", self.base_url + "/chat/completions",
                **({"content": prepared.wire_json.encode("utf-8")} if prepared is not None else {"json": payload}),
                headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"},
                timeout=httpx.Timeout(timeout, connect=min(15, timeout), pool=min(15, timeout))) as response:
                if response.status_code == 200 and 'text/event-stream' in response.headers.get('content-type', '').lower():
                    try:
                        body = completion_from_sse(response.iter_lines())
                    except IncompleteModelStream as error:
                        unknown_reason = 'stream_incomplete'
                        raise QwenOutcomeUnknown("Model stream ended without a complete response; outcome unknown") from error
                else:
                    # Some compatible providers return a complete JSON response
                    # even when streaming was requested. This uses the same call.
                    response.read()
                    try:
                        body = validate_completion(response.json()) if response.status_code == 200 else None
                    except ValueError as error:
                        unknown_reason = 'invalid_response_body'
                        raise QwenOutcomeUnknown("Provider response is not complete JSON; outcome unknown") from error
            emit('qwen_http_received', call_id=call_id, request_id=prepared.request_id if prepared is not None else None,
                 status=response.status_code, wire_sha256=hashlib.sha256(wire).hexdigest())
            if response.status_code != 200:
                if response.status_code in {401, 402, 403}:
                    raise QwenUnavailable(f"Model API returned HTTP {response.status_code}; check credentials or account budget")
                if response.status_code in {408, 409, 429} or response.status_code >= 500:
                    raise QwenRetryableError(f"Model API returned temporary HTTP {response.status_code}")
                if response.status_code == 413 or response.status_code in {400, 422} and any(code in response.text.lower()
                        for code in ("context_length_exceeded", "maximum context length", "context window", "too many tokens")):
                    raise QwenContextError("Provider rejected the request context size")
                raise QwenError(f"Model API returned HTTP {response.status_code}")
            self.journal.put('qwen_provider_failure_streak', 0)
            if prepared is not None:
                self.context_memory.c.knowledge.provided(prepared)
            usage = body.get("usage", {})
            total = self.journal.get("token_usage", {})
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if isinstance(usage.get(key), int):
                    total[key] = total.get(key, 0) + usage[key]
            reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            if isinstance(reasoning_tokens, int):
                total["reasoning_tokens"] = total.get("reasoning_tokens", 0) + reasoning_tokens
            self.journal.put("token_usage", total)
            if prepared is not None and all(type(usage.get(key)) is int for key in ("prompt_tokens", "completion_tokens")):
                self.context_memory.store.record_usage(prepared.request_id, prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"], elapsed_seconds=time.monotonic() - started)
            choices = body.get("choices", [])
            if len(choices) != 1:
                raise QwenRetryableError("Missing model response")
            self.journal.event("qwen_response", {"call_id": call_id, "model": body.get("model"),
                "finish_reason": choices[0].get("finish_reason"), "reasoning_tokens": reasoning_tokens,
                "requested_reasoning_effort": self.effort, "elapsed_seconds": time.monotonic() - started})
            if choices[0].get("finish_reason") == "length":
                previous = self.output_tokens
                input_size = request_size(payload, 0)
                self.output_tokens = max(previous, min(previous * 2, self.context_tokens - input_size))
                if self.output_tokens == previous:
                    raise QwenContextError("Truncated response cannot grow within the context window; compact before retrying")
                self.output_by_effort[self.effort] = self.output_tokens
                configuration = self.journal.get("qwen_configuration", {})
                configuration["output_tokens"] = self.output_tokens
                configuration["output_by_effort"] = self.output_by_effort
                self.journal.put("qwen_configuration", configuration)
                self.journal.event("qwen_response_truncated", {"previous_output_tokens": previous,
                    "next_output_tokens": self.output_tokens})
                raise QwenRetryableError("Truncated model response; retry with the recorded output allocation")
            return body if prepared is not None else choices[0]["message"]
        except ProviderStreamError as error:
            streak = self.journal.get('qwen_provider_failure_streak', 0) + 1
            self.journal.put('qwen_provider_failure_streak', streak)
            self.journal.event('qwen_provider_failure', {'call_id': call_id, 'code': error.code,
                'consecutive_failures': streak, 'partial_response_discarded': True})
            if error.code in {408, 409, 429} or error.code >= 500:
                if streak < 3:
                    raise QwenRetryableError(str(error)) from error
            raise QwenUnavailable(str(error) + '; stop after nonretryable or three consecutive provider failures') from error
        except QwenOutcomeUnknown as error:
            self.journal.put('qwen_unknown_outcome', {'call_id':call_id,'epoch':time.time(),
                'error_type':type(error).__name__,'reason':unknown_reason})
            raise
        except (KeyboardInterrupt,SystemExit) as error:
            self.journal.put('qwen_unknown_outcome', {'call_id':call_id,'epoch':time.time(),
                'error_type':type(error).__name__,'reason':'request_interrupted'})
            raise
        except httpx.HTTPError as error:
            if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
                raise QwenRetryableError(f"Model request not sent: {type(error).__name__}") from error
            self.journal.put('qwen_unknown_outcome', {'call_id':call_id,'epoch':time.time(),
                'error_type':type(error).__name__,'reason':'sent_request_outcome_unknown'})
            raise QwenOutcomeUnknown(f"Model generation outcome unknown: {type(error).__name__}") from error
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, max(.001, previous_timer[0] - (time.monotonic() - started)), previous_timer[1])
            self.journal.put("llm_seconds", self.journal.get("llm_seconds", 0) + time.monotonic() - started)
            self.journal.put("llm_inflight", None)

    def resolve(self, system, request):
        self.select_reasoning("xhigh", "resolve_task_contract")
        message = self._request([{"role": "system", "content": system}, {"role": "user", "content": canonical(request)}])
        return parse_json_object(message.get("content") or "")

    def next_command(self, system, context):
        from context_compressor.contracts import Snapshot
        if isinstance(context, Snapshot):
            return self.next_compressed(context)
        self.select_for_step()
        state = context.get("state", {})
        self.reserve_seconds = state.get("remaining", {}).get("reserve_seconds", 0) if isinstance(state, dict) else 0
        messages = command_messages(system, context, True)
        if self.transport in {"auto", "native"}:
            try:
                message = self._request(messages, native_tools())
                calls = message.get("tool_calls", [])
                if not calls:
                    raise ValueError("Expected a tool call, received none")
                if len(calls) > 1:
                    self.journal.event("extra_native_calls_deferred", {"received": len(calls),
                        "executed": 1, "reason": "One operation per updated research state"})
                call = calls[0]
                arguments = parse_json_object(call["function"]["arguments"])
                operation_id = arguments.pop("operation_id", call["id"])
                command = ToolCommand(operation_id=operation_id, tool=call["function"]["name"], arguments=arguments)
                TOOL_SCHEMAS[command.tool].model_validate(command.arguments)
                self.transport = "native"
                self.journal.put("transport", self.transport)
                return command, message.get("content") or ""
            except (QwenUnavailable, QwenRetryableError, QwenContextError):
                raise
            except (ValueError, KeyError, TypeError, QwenError) as error:
                self.journal.event("native_fallback", {"reason": str(error)[:500]})
                self.transport = "json"
                self.journal.put("transport", self.transport)
        messages = command_messages(system, context, False)
        message = self._request(messages)
        command = ToolCommand.model_validate(parse_json_object(message.get("content") or ""))
        TOOL_SCHEMAS[command.tool].model_validate(command.arguments)
        return command, ""

    def next_compressed(self, snapshot):
        from context_compressor.contracts import ContextError, BudgetExceeded
        memory = self.context_memory
        compressor = memory.compressor
        self.reserve_seconds = json.loads(snapshot.core_json)["policies_resources"]["reserve_seconds"]
        modes = ["json"] if self.transport == "json" else ["native", "json"]
        for mode in modes:
            minimum_stage = 0
            while True:
                prepared = compressor.prepare(snapshot, mode, budget=self.context_tokens, minimum_stage=minimum_stage)
                try:
                    response = self._request(None, prepared=prepared)
                    command = compressor.accept_response(prepared.request_id, response, memory.validate)
                    calls = response['choices'][0]['message'].get('tool_calls') or []
                    if mode == 'native' and len(calls) > 1:
                        self.journal.event('extra_native_calls_deferred', {'received': len(calls),
                            'accepted': 1, 'discarded': len(calls) - 1, 'queued': 0,
                            'reason': 'One operation per updated research state; only the first call was validated'})
                    emit('qwen_parse_outcome', call_id=getattr(self,'last_call_id',None), request_id=prepared.request_id,
                         accepted=True, command=command.as_dict())
                    self.transport = mode
                    self.journal.put("transport", mode)
                    memory.persist_note()
                    return ToolCommand.model_validate({k: v for k, v in command.as_dict().items() if k != "research_note"}), ""
                except QwenContextError as error:
                    emit('qwen_parse_outcome', call_id=getattr(self,'last_call_id',None), request_id=prepared.request_id,
                         accepted=False, error_type=type(error).__name__, error=str(error))
                    memory.store.channel_error(prepared.request_id, type(error).__name__, str(error))
                    if prepared.stage == 3:
                        raise BudgetExceeded("Provider rejected the protected minimum context; finalize the saved result") from error
                    minimum_stage = prepared.stage + 1
                except (QwenUnavailable, QwenRetryableError, TimeoutError) as error:
                    emit('qwen_parse_outcome', call_id=getattr(self,'last_call_id',None), request_id=prepared.request_id,
                         accepted=False, error_type=type(error).__name__, error=str(error))
                    memory.store.channel_error(prepared.request_id, type(error).__name__, str(error))
                    raise
                except (ContextError, ValueError, KeyError, TypeError, QwenError) as error:
                    emit('qwen_parse_outcome', call_id=getattr(self,'last_call_id',None), request_id=prepared.request_id,
                         accepted=False, error_type=type(error).__name__, error=str(error))
                    from .util import redact
                    memory.store.channel_error(prepared.request_id, type(error).__name__, redact(str(error)))
                    if mode == "json":
                        raise
                    self.journal.event("native_fallback", {"reason": redact(str(error))[:500]})
                    self.transport = "json"
                    self.journal.put("transport", "json")
                    break
        raise QwenError("No valid compressed-context command")

    def command_size(self, system, context):
        sizes = []
        for native in ([False] if self.transport == "json" else [True, False]):
            payload = {"model": self.model, "messages": command_messages(system, context, native),
                "max_tokens": self.output_tokens, "stream": True, "stream_options": {"include_usage": True}, **self.extra}
            if native:
                payload.update(tools=native_tools(), parallel_tool_calls=False, tool_choice="required")
            sizes.append(request_size(payload, self.output_tokens))
        return max(sizes)

    def close(self):
        self.client.close()
