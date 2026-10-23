"""Persistent evidence-based dead-end detection, independent of model prose."""
import time

from .util import digest


def record(controller, source, evidence):
    return record_many(controller, source, [evidence])


def record_many(controller, source, evidence):
    c = controller
    keys = list(dict.fromkeys(digest({"source": source, "evidence": item}) for item in evidence))
    state = c.journal.get("progress_state", {"last_useful_at": time.time(), "failures": 0, "seen": []})
    new = [key for key in keys if key not in state["seen"]]
    useful = bool(new)
    if useful:
        state.update(last_useful_at=time.time(), failures=0)
        state["seen"].extend(new)
        c.journal.event("useful_progress", {"source": source, "evidence_hashes": new})
    else:
        state["failures"] += 1
    c.journal.put("progress_state", state)
    return useful


def failure(controller, kind):
    c = controller
    state = c.journal.get("progress_state", {"last_useful_at": time.time(), "failures": 0, "seen": []})
    state["failures"] += 1
    state["last_failure_kind"] = kind
    c.journal.put("progress_state", state)


def exhausted(controller):
    c = controller
    state = c.journal.get("progress_state", {})
    return state.get("failures", 0) >= 3 and time.time() - state.get("last_useful_at", time.time()) >= 600
