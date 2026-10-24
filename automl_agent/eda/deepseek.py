"""Development-only provider and durable, fail-closed USD reservations."""
import fcntl
from functools import wraps
import json
import math
import os
from pathlib import Path

import httpx
from dotenv import dotenv_values

from ..qwen import QwenClient, QwenError, QwenUnavailable, QwenOutcomeUnknown, parse_json_object
from .trace import event, sanitized
from .planning import generation_inputs
from ..util import atomic_json, canonical, digest

# Peak, cache-miss USD per million tokens; official pricing snapshot 2026-10-24.
PRICING = {"input": .30, "output": 1.20, "model": "deepseek-flash",
    "source": "https://api-docs.deepseek.com/quick_start/pricing/", "date": "2026-10-24"}


def locked(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.path.exists():
                self.state = json.loads(self.path.read_text())
                if self.state["pricing"] != PRICING:
                    raise ValueError("Frozen pricing changed")
            return method(self, *args, **kwargs)
    return invoke


class CostLedger:
    def __init__(self, path, run_id, cap=.20, series_cap=.85):
        self.path, self.run_id, self.cap, self.series_cap = Path(path), run_id, cap, series_cap
        if not 0 < cap <= .20 or not 0 < series_cap <= .85:
            raise ValueError("Development cost caps exceed approved allocation")
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"pricing": PRICING, "requests": []}
        if self.state["pricing"] != PRICING:
            raise ValueError("Cannot change frozen prices in an existing ledger")

    @locked
    def reserve(self, payload):
        if any(r["state"] == "reserved" for r in self.state["requests"]):
            raise QwenOutcomeUnknown("Unreconciled request: new API requests disabled")
        caps = self.state.setdefault("run_caps", {})
        if self.run_id in caps and caps[self.run_id] != self.cap:
            raise ValueError("Cannot change a saved run cost allocation")
        caps[self.run_id] = self.cap
        rows = self.state["requests"]
        # UTF-8 bytes conservatively bound text tokenization, plus framing overhead.
        upper = ((len(canonical(payload).encode()) + 4096) * PRICING["input"] + 2048 * PRICING["output"]) / 1e6
        used = sum(r["usd"] for r in rows)
        local = sum(r["usd"] for r in rows if r["run_id"] == self.run_id)
        if used + upper > self.series_cap or local + upper > self.cap:
            raise QwenUnavailable("USD reservation would exceed development allocation")
        row = {"run_id": self.run_id, "state": "reserved", "usd": upper, "upper_usd": upper}
        rows.append(row)
        atomic_json(self.path, self.state)
        return len(rows)-1

    @locked
    def settle(self, index, usage):
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if any(type(n) is not int or n < 0 for n in (prompt, completion)):
            raise QwenOutcomeUnknown("Provider usage missing or invalid; reservation retained")
        cost = (prompt * PRICING["input"] + completion * PRICING["output"]) / 1e6
        row = self.state["requests"][index]
        if not math.isfinite(cost) or cost > row["upper_usd"]:
            raise QwenOutcomeUnknown("Usage exceeds reservation; reconciliation required")
        row.update(state="settled", usd=cost, usage=usage)
        atomic_json(self.path, self.state)

    @locked
    def summary(self):
        rows = [r for r in self.state["requests"] if r["run_id"] == self.run_id]
        return {"usd": sum(r["usd"] for r in rows), "requests": len(rows),
            "outcome_unknown": any(r["state"] == "reserved" for r in rows), "cap_usd": self.cap}


class MalformedDecision(QwenError):
    """A charged, received response did not decode to an executable command."""


class DeepSeekClient(QwenClient):
    def __init__(self, journal, budget, ledger, transport="native", client=None):
        config = dotenv_values(Path.cwd() / ".env")
        self.key = os.getenv("DEEPSEEK_API_KEY") or config.get("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL") or config.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        self.model = os.getenv("DEEPSEEK_MODEL") or config.get("DEEPSEEK_MODEL") or "deepseek-flash"
        if self.model != PRICING["model"]:
            raise ValueError("Unpriced model: freeze reviewed tariffs before changing model")
        self.journal, self.budget, self.ledger, self.transport = journal, budget, ledger, transport
        self.client = client or httpx.Client(timeout=60, follow_redirects=False)
        self.offline_fixture = client is not None and isinstance(getattr(client,"_transport",None),httpx.MockTransport)
        self.reserve_seconds = 0
        self.journal.put("development_provider", {"provider": "deepseek", "model": self.model,
            "thinking": "disabled", "max_output_tokens": 2048, "pricing": PRICING})

    def next_command(self, system, context):
        before = self.budget.counter("llm_calls")
        try:
            return super().next_command(system, context)
        except (ValueError, KeyError, TypeError) as error:
            current = self.budget.counter("llm_calls")
            if current > before and self.journal.get("eda_received_response_call") == current:
                raise MalformedDecision(str(error)) from error
            raise

    def select_reasoning(self, *args):
        pass

    def select_for_step(self):
        pass

    def resolve(self, system, request):
        import copy
        request = copy.deepcopy(request)
        # The proposal carries citations. TaskSpec.evidence is a controller-owned
        # projection; advertising both led live models to put citations in the wrong place.
        spec_schema = request["schema"]["$defs"]["TaskSpec"]
        spec_schema["properties"].pop("evidence", None)
        known_sources = list(request["sources"])
        request["allowed_evidence_source_ids"] = known_sources
        request["schema"]["required"] = list(dict.fromkeys([*request["schema"].get("required", []), "evidence"]))
        request["schema"]["properties"]["evidence"] = {"type":"object", "additionalProperties":{
            "type":"array", "items":{"type":"string", "enum":known_sources}}}
        system += "\nPut evidence at the TOP LEVEL of TaskResolutionProposal, alongside task_spec. TaskSpec.evidence is filled by the verifier after confirmation. Each citation must be an EXACT allowed_evidence_source_ids value, never a sources.* path. Include citations for kind and every nonempty declared field, including feature roles, metric, output and file_roles."
        request["schema"]["$defs"]["TaskSpec"]["properties"]["feature_columns"] = {
            "anyOf": [{"type": "array", "items": {"type": "string"}},
                {"const": "all_nonservice_inputs"}]}
        system += "\nFor wide tables, feature_columns may be the exact string all_nonservice_inputs: the verifier expands train headers minus explicitly declared targets, id_columns and service_columns. Explicitly choose these roles from evidence. This is an encoding shorthand, not role inference."
        result = self._request([{"role": "system", "content": system}, {"role": "user", "content": canonical(request)}])
        proposal = parse_json_object(result.get("content") or "")
        spec = proposal.get("task_spec") or {}
        if spec.get("feature_columns") == "all_nonservice_inputs":
            inventory = json.loads(request["sources"]["inventory"]["text"])
            role = spec.get("file_roles", {}).get("train.csv")
            omitted = set(spec.get("targets", []) + spec.get("id_columns", []) + spec.get("service_columns", []))
            spec["feature_columns"] = [c for c in inventory[role]["columns"] if c not in omitted]
        return proposal

    def _request(self, messages, tools=None, *, prepared=None):
        if not self.offline_fixture:
            raise QwenUnavailable("Historical DeepSeek live generations are disabled; use the organizer Qwen EDA CLI")
        if not self.configured:
            raise QwenUnavailable("DEEPSEEK_API_KEY is missing")
        allowed = getattr(self, "allowed_tools", None)
        messages, tools, allowed = generation_inputs(messages, tools, allowed, self.budget)
        if allowed:
            self.allowed_tools = allowed
        self.budget.claim_call()
        if allowed and not tools:
            content = json.loads(messages[-1]["content"])
            if "tool_argument_schemas" in content:
                names = [item["function"]["name"] for item in allowed]
                content["tool_argument_schemas"] = {k:v for k,v in content["tool_argument_schemas"].items() if k in names}
                content["command_schema"]["properties"]["tool"]["enum"] = names
                messages = [*messages[:-1], {**messages[-1], "content": canonical(content)}]
        payload = {"model": self.model, "messages": messages, "max_tokens": 2048,
            "thinking": {"type": "disabled"}, "stream": False}
        if tools:
            tools = getattr(self, "allowed_tools", tools)
            payload.update(tools=tools, tool_choice="required", parallel_tool_calls=False)
        ticket = self.ledger.reserve(payload)
        # Headers are never persisted. Redact the configured credential even if echoed in source text.
        safe_payload=sanitized(payload,self.key)
        atomic_json(self.journal.directory / "provider" / f"request_{ticket}.json",safe_payload)
        event(self.journal.directory,"provider_request",ticket=ticket,model=self.model,
            mode=getattr(self,"provider_mode","live"),
            request_hash=digest(payload), request_bytes=len(canonical(payload).encode()),
            maximum_output_tokens=2048,reserved_usd=self.ledger.summary()["usd"])
        try:
            response = self.client.post(self.base_url.rstrip("/") + "/chat/completions", json=payload,
                headers={"Authorization": "Bearer " + self.key}, timeout=min(60, max(.1, self.budget.remaining)))
            response.raise_for_status()
            body = response.json()
            self.ledger.settle(ticket, body.get("usage") or {})
            message = body["choices"][0]["message"]
            atomic_json(self.journal.directory / "provider" / f"response_{ticket}.json",
                sanitized({"message":message,"usage":body.get("usage"),"http_status":response.status_code},self.key))
            event(self.journal.directory,"provider_response",ticket=ticket,http_status=response.status_code,
                usage=body.get("usage"),cost=self.ledger.summary())
            self.journal.event("development_usage", self.ledger.summary())
            self.journal.put("eda_received_response_call", self.budget.counter("llm_calls"))
            return message
        except QwenOutcomeUnknown:
            event(self.journal.directory,"provider_unknown",ticket=ticket,reservation_retained=True)
            raise
        except Exception as error:
            status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
            self.journal.event("development_provider_failure", {"http_status": status,
                "error_type": type(error).__name__, "reservation_retained": True})
            event(self.journal.directory,"provider_failure",ticket=ticket,http_status=status,
                error_type=type(error).__name__,reservation_retained=True)
            # Never log httpx request/response: those can carry authorization headers.
            raise QwenOutcomeUnknown("DeepSeek outcome unknown: " + type(error).__name__ +
                (" HTTP " + str(status) if status else "")) from None

    def close(self):
        self.client.close()
