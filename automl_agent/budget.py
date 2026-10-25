from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Limits:
    seconds: float = 180 * 60
    memory_mb: float = 28000
    cpu: int = 7
    max_candidates: int | None = None
    max_calls: int | None = None
    max_llm_seconds: float | None = None
    max_debug: int | None = None
    stagnation: int = 3


class Budget:
    def __init__(self, journal, limits: Limits, started_at: float):
        self.journal, self.limits = journal, limits
        previous = journal.get("deadline")
        self.deadline = previous if previous is not None else started_at + limits.seconds
        if previous is None:
            journal.put("deadline", self.deadline)
            journal.put("limits", vars(limits))
        else:
            saved = journal.get("limits")
            if saved:
                self.limits = Limits(**saved)

    @property
    def remaining(self):
        return max(0.0, self.deadline - time.time())

    def final_reserve(self, costs=(), train_ratio=1.5):
        minimum = max(min(300.0, self.limits.seconds * (5 / 180)), min(900.0, self.limits.seconds * 0.06))
        forecast = getattr(costs, "forecast", None)
        if forecast is not None:
            return max(minimum, forecast["seconds"])
        costs = list(costs)
        if not costs:
            return minimum
        preparation = sum(costs) * train_ratio * 1.5
        refit = sum(costs) * train_ratio * 1.5
        predict_control_write = max(2.0, sum(costs) * 0.1)
        return max(minimum, preparation + refit + predict_control_write + min(60, self.limits.seconds * 0.03))

    def grant_training(self, requested, costs=(), full_comparison=False):
        available = self.remaining - self.final_reserve(costs)
        if available <= 1:
            raise TimeoutError("Finalization reserve exhausted; freeze search")
        return min(float(requested), available)

    def counter(self, key, increment=0):
        value = self.journal.get(key, 0)
        if increment:
            value += increment
            self.journal.put(key, value)
        return value

    def claim_candidate(self):
        if self.limits.max_candidates is not None and self.counter("candidate_count") >= self.limits.max_candidates:
            raise TimeoutError("Research candidate limit reached")
        return self.counter("candidate_count", 1)

    def claim_call(self, reserve_seconds=0):
        if self.limits.max_calls is not None and self.counter("llm_calls") >= self.limits.max_calls:
            raise TimeoutError("Qwen call limit reached")
        if self.limits.max_llm_seconds is not None and self.journal.get("llm_seconds", 0) >= self.limits.max_llm_seconds:
            raise TimeoutError("Qwen time budget exhausted")
        if self.remaining <= max(reserve_seconds, self.final_reserve()):
            raise TimeoutError("Insufficient time for another Qwen request")
        return self.counter("llm_calls", 1)

    def snapshot(self, costs=()):
        return {"seconds": self.remaining, "reserve_seconds": self.final_reserve(costs),
            "candidates": None if self.limits.max_candidates is None else self.limits.max_candidates - self.counter("candidate_count"),
            "qwen_calls": None if self.limits.max_calls is None else self.limits.max_calls - self.counter("llm_calls"),
            "llm_seconds_used": self.journal.get("llm_seconds", 0)}
