"""At most four pure fitting attempts against one durable, immutable snapshot."""
from __future__ import annotations

import json
import time
import uuid

from .contracts import (B_MIN, OUTPUT_RESERVE, SAFETY_MARGIN, BudgetExceeded,
                        ContextError, PreparedRequest, Snapshot, canonical,
                        identifier, size, text_hash)
from .core import make_snapshot, validate_core
from .store import ContextStore
from .transport import Transport


def _context(snapshot: Snapshot, stage: int, now: float):
    core = json.loads(snapshot.core_json)
    resources = core["policies_resources"]
    resources["remaining_seconds"] = max(0, resources["deadline_epoch"] - now)
    validate_core(core)
    cards = [json.loads(value) for value in snapshot.cards]
    if stage == 2:
        cards = cards[:8]
    elif stage == 3:
        cards = []
    failures = json.loads(snapshot.failures_json) if stage < 2 else []
    profile = json.loads(snapshot.profile_json) if stage < 3 else None
    if stage == 2 and profile:
        profile = {"summary": profile["summary"], "read": profile["read"]}
    selected = [card["record_id"] for card in cards]
    return {"core": core, "observations": json.loads(snapshot.observations_json),
            "history": cards, "history_details": [json.loads(value) for value in snapshot.details] if stage == 0 else [],
            "known_failures": failures, "profile": profile,
            "compaction": {"snapshot_id": snapshot.snapshot_id, "stage": stage,
                           "history_count": snapshot.history_count, "selected_ids": selected,
                           "pending_observations": snapshot.pending_count,
                           "retained_observation_count": snapshot.pending_count - sum(item["complete"] for item in json.loads(snapshot.observations_json)),
                           "included_observation_ids": [item["observation_id"] for item in json.loads(snapshot.observations_json)],
                           "exclusions": {"max_record_seq": snapshot.max_record_seq,
                                          "count": snapshot.history_count - len(selected),
                                          "rule": "permitted records at/below snapshot seq, excluding selected_ids; enumerate with excluded_ids(request_id)"}}}


class ContextCompressor:
    def __init__(self, store: ContextStore, transport: Transport):
        self.store, self.transport = store, transport

    def snapshot(self):
        return make_snapshot(self.store)

    def _budget(self, budget):
        required = B_MIN + max(0, self.transport.response_reserve - OUTPUT_RESERVE)
        if type(budget) is not int or budget < required:
            raise BudgetExceeded(f"Unsupported budget: require >= {required}; response reserve is never reduced")

    def preflight(self, snapshot: Snapshot, *, budget=B_MIN, now=None):
        """Check the minimum request in BOTH transports before starting research."""
        self._budget(budget)
        now = time.time() if now is None else now
        context = _context(snapshot, 3, now)
        if size(context["observations"]) > 8192:
            raise BudgetExceeded("Observation batch exceeds its protected 8192-byte budget")
        result = {}
        for mode in ("native", "json"):
            overhead = self.transport.overhead_bytes(mode)
            if overhead > 49152:
                raise BudgetExceeded(f"{mode} instructions/schemas/framing exceed 49152 bytes: {overhead}")
            wire = self.transport.render(context, mode)
            estimate = len(wire.encode("utf-8")) + self.transport.response_reserve
            if estimate + SAFETY_MARGIN > budget:
                raise BudgetExceeded(f"{mode} minimum request cannot fit: {estimate} + {SAFETY_MARGIN} > {budget}")
            result[mode] = {"overhead_bytes": overhead, "wire_bytes": len(wire.encode("utf-8")),
                            "core_bytes": size(context["core"]), "observations_bytes": size(context["observations"]),
                            "response_reserve": self.transport.response_reserve, "conservative_size": estimate,
                            "safety_margin": SAFETY_MARGIN, "budget": budget}
        return result

    def fit(self, snapshot: Snapshot, mode: str, *, budget=B_MIN, request_id=None,
            now=None, minimum_stage=0) -> PreparedRequest:
        """Pure fitting: no database/profile reads, no receipts, no source mutation.

        minimum_stage is useful after an actual provider context rejection. Such a
        retry stays attached to the same snapshot and does not touch research counters.
        """
        started = time.perf_counter()
        self._budget(budget)
        if type(minimum_stage) is not int or minimum_stage not in range(4):
            raise ContextError("minimum_stage must be 0..3")
        now = time.time() if now is None else now
        core = json.loads(snapshot.core_json)
        remaining = core["policies_resources"]["deadline_epoch"] - now
        if remaining <= core["policies_resources"]["reserve_seconds"]:
            raise BudgetExceeded("Original deadline/finalization reserve reached; finalize the existing result")
        if self.transport.overhead_bytes(mode) > 49152:
            raise BudgetExceeded("Instructions/schemas/framing exceed 49152 UTF-8 bytes")
        request_id = request_id or uuid.uuid4().hex
        identifier(request_id, "request_id")
        attempts = 0
        trace = []
        for stage in range(minimum_stage, 4):
            attempts += 1
            context = _context(snapshot, stage, now)
            if size(context["observations"]) > 8192:
                raise BudgetExceeded("Protected observation batch exceeds 8192 bytes")
            wire = self.transport.render(context, mode)
            wire_bytes = len(wire.encode("utf-8"))
            estimate = wire_bytes + self.transport.response_reserve
            trace.append(canonical({"stage": stage, "wire_bytes": wire_bytes, "conservative_size": estimate,
                                    "selected_ids": context["compaction"]["selected_ids"],
                                    "block_bytes": {key: size(value) for key, value in context.items()}}))
            if estimate + SAFETY_MARGIN <= budget:
                return PreparedRequest(request_id, snapshot.snapshot_id, mode, wire, text_hash(wire), wire_bytes,
                                       self.transport.response_reserve, estimate, budget, attempts, stage,
                                       snapshot.build_seconds + time.perf_counter() - started,
                                       tuple(context["compaction"]["selected_ids"]),
                                       {key: size(value) for key, value in context.items()}, tuple(trace))
        raise BudgetExceeded(f"Minimal {mode} request does not fit after {attempts} attempts: {estimate} + {SAFETY_MARGIN} > {budget}")

    def prepare(self, snapshot: Snapshot, mode: str, **options) -> PreparedRequest:
        """Refresh remaining time, fit, and persist the exact body immediately before send."""
        prepared = self.fit(snapshot, mode, **options)
        self.store.register_request(prepared, json.loads(snapshot.observations_json))
        return prepared

    def accept_response(self, request_id, response, validate):
        request = self.store.request(request_id)
        command = self.transport.parse_response(response, request["transport"])
        return self.store.accept_command(request_id, command, validate)
