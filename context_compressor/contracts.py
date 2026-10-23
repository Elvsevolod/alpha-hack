"""Wire contracts for the independent, rule-based context compressor.

Only the controller may create confirmed state and publish permitted sources.
All source ranges and byte limits use UTF-8, never Python character indices.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Mapping


BUILDER_VERSION = "context-compressor/2"
B_MIN = 131072
OUTPUT_RESERVE = 32768
SAFETY_MARGIN = 8192
CORE_LIMITS = {
    "identity": 2048, "task": 3072, "policies_resources": 3072,
    "current_work": 4096, "research_note": 3072, "reading_routes": 1024,
}
SCOPES = frozenset({"contract", "policy", "fit", "validation_features", "test_features",
                    "registered_feedback", "reference", "code", "environment", "controller"})
TRUST = frozenset({"controller_fact", "external_data", "agent_assumption"})
NOTE_STATUSES = ("preparing", "awaiting_evidence", "ready_to_compare", "closed")
TOOLS = ("inspect_dataset", "get_reference", "train_model", "run_automl", "tune_model",
         "analyze_experiments", "build_ensemble", "run_custom_experiment", "finish",
         "get_dataset_report", "revise_validation", "search_knowledge", "list_knowledge", "read_knowledge",
         "search_run_records", "list_run_records", "read_run_records")


class ContextError(ValueError):
    """Invalid input, immutable-ID conflict, or unsupported context projection."""


class BudgetExceeded(ContextError):
    """The controller must finalize using its existing result, without retrying research."""


class AccessDenied(ContextError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def size(value: Any) -> int:
    return len(canonical(value).encode("utf-8"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def bounded(value: str, limit: int, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not empty) or len(value.encode("utf-8")) > limit:
        raise ContextError(f"{name} must be {'0' if empty else '1'}..{limit} UTF-8 bytes")
    return value


def identifier(value: str, name: str = "ID") -> str:
    return bounded(value, 96, name)


def finite(value: float, name: str, minimum: float = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ContextError(f"{name} must be finite and >= {minimum}")


@dataclass(frozen=True)
class ResearchNote:
    question: str
    reading_goal: str
    expected_evidence: str
    related_ids: tuple[str, ...] = ()
    status: str = "preparing"

    def __post_init__(self):
        for name in ("question", "reading_goal", "expected_evidence"):
            bounded(getattr(self, name), 512, name)
        if not isinstance(self.related_ids, tuple) or len(self.related_ids) > 8:
            raise ContextError("research_note.related_ids must be a tuple with at most 8 IDs")
        for item in self.related_ids:
            identifier(item, "research_note.related_id")
        if self.status not in NOTE_STATUSES:
            raise ContextError("Unknown research_note status")

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> ResearchNote:
        if not isinstance(value, dict) or set(value) - {"question", "reading_goal", "expected_evidence", "related_ids", "status"}:
            raise ContextError("Invalid research_note fields")
        raw = dict(value)
        if "related_ids" in raw:
            if not isinstance(raw["related_ids"], list):
                raise ContextError("research_note.related_ids must be an array")
            raw["related_ids"] = tuple(raw["related_ids"])
        try:
            return cls(**raw)
        except TypeError as error:
            raise ContextError("Incomplete research_note") from error

    def persisted(self, command_id: str) -> dict[str, Any]:
        identifier(command_id, "command_id")
        value = {**asdict(self), "source": "agent_assumption", "command_id": command_id,
                 "note_id": digest({"command_id": command_id, "note": asdict(self)})}
        if size(value) > 3072:
            raise ContextError("research_note including provenance exceeds 3072 UTF-8 bytes")
        return value


@dataclass(frozen=True)
class Source:
    source_id: str
    version: str
    content: str
    scope: str
    trust: str = "external_data"
    section: str = ""
    origin: str = "tool_result"

    def __post_init__(self):
        identifier(self.source_id, "source_id")
        bounded(self.version, 128, "source version")
        bounded(self.section, 128, "source section", empty=True)
        bounded(self.origin, 128, "source origin")
        if not isinstance(self.content, str):
            raise ContextError("Source content must be text; serialize structured results explicitly")
        # Reject invalid surrogate strings before they reach durable storage.
        self.content.encode("utf-8")
        if self.scope not in SCOPES or self.trust not in TRUST:
            raise AccessDenied("Unsupported source scope or trust designation")

    @property
    def reference(self) -> dict[str, Any]:
        return read_ref(self.source_id, self.version, self.scope, self.section)


def read_ref(source_id: str, version: str, scope: str, section: str = "", offset: int = 0) -> dict[str, Any]:
    return {"tool": "analyze_experiments", "arguments": {"context_read": {
        "source_id": source_id, "version": version, "scope": scope, "section": section,
        "offset": offset, "limit": 768}}}


@dataclass(frozen=True)
class Emission:
    """A new source and requested UTF-8 byte range (the whole source by default)."""
    source: Source
    start: int = 0
    end: int | None = None
    kind: str = "reference"

    def __post_init__(self):
        raw = self.source.content.encode("utf-8")
        end = len(raw) if self.end is None else self.end
        if type(self.start) is not int or type(end) is not int or not 0 <= self.start <= end <= len(raw):
            raise ContextError("Invalid emitted source range")
        if any(index < len(raw) and raw[index] & 0xC0 == 0x80 for index in (self.start, end)):
            raise ContextError("Emitted byte range splits a UTF-8 character")
        try:
            raw[self.start:end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ContextError("Emitted byte range splits a UTF-8 character") from error
        if self.kind not in {"training_card", "reference", "diagnostic", "code", "analysis"}:
            raise ContextError("Unknown observation kind")

    def observation_id(self, operation_id: str, result_version: str) -> str:
        start = self.start
        end = len(self.source.content.encode("utf-8")) if self.end is None else self.end
        return digest({"operation_id": operation_id, "result_version": result_version,
                       "source_id": self.source.source_id, "version": self.source.version,
                       "section": self.source.section, "start": start, "end": end,
                       "kind": self.kind})


@dataclass(frozen=True)
class SourceSlice:
    """A permitted read page of an ALREADY persisted source; no whole-source load."""
    source_id: str
    version: str
    scope: str
    section: str
    start: int
    end: int
    source_sha256: str
    content_sha256: str
    kind: str = "reference"

    def __post_init__(self):
        identifier(self.source_id, "source_id")
        bounded(self.version, 128, "source version")
        bounded(self.section, 128, "source section", empty=True)
        if self.scope not in SCOPES or self.kind not in {"training_card", "reference", "diagnostic", "code", "analysis"}:
            raise ContextError("Unsupported read page scope/kind")
        if type(self.start) is not int or type(self.end) is not int or not 0 <= self.start <= self.end <= self.start + 768:
            raise ContextError("A persisted source read page must be at most 768 UTF-8 bytes")
        for value in (self.source_sha256, self.content_sha256):
            if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
                raise ContextError("Read pages require SHA-256 source and content hashes")

    @classmethod
    def from_fragment(cls, fragment: dict, *, kind="reference"):
        try:
            source, interval = fragment["source"], fragment["range"]
            if interval["unit"] != "utf8_bytes" or not isinstance(fragment["content"], str):
                raise ContextError("Expected a source read page with UTF-8 byte ranges")
            result = cls(source["source_id"], source["version"], source["scope"], source["section"],
                         interval["start"], interval["end"], source["sha256"], fragment["content_sha256"], kind)
            if len(fragment["content"].encode()) != result.end - result.start or text_hash(fragment["content"]) != result.content_sha256:
                raise ContextError("Read page content does not match its range and hash")
            return result
        except (KeyError, TypeError) as error:
            raise ContextError("Incomplete source read page") from error

    def observation_id(self, operation_id: str, result_version: str) -> str:
        return digest({"operation_id": operation_id, "result_version": result_version,
                       "source_id": self.source_id, "version": self.version, "section": self.section,
                       "start": self.start, "end": self.end, "kind": self.kind})


@dataclass(frozen=True)
class Record:
    """Bounded metadata only. Full code, evidence, and diagnostics live in Source."""
    record_id: str
    source_id: str
    source_version: str
    data_scope: str = "registered_feedback"
    kind: str = "experiment"
    component: str = "unknown"
    engine: str = "unknown"
    status: str = "outcome_unknown"
    error_code: str | None = None
    protocol: str = "unspecified"
    environment: str = "unspecified"
    parent_id: str | None = None
    tool: str = "train_model"
    evaluation_scope: str = "full"
    metric: str | None = None
    maximize: bool | None = None
    score: float | None = None
    seconds: float = 0
    peak_memory_mib: float = 0
    checked_change: str = "unspecified"
    dependencies: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("record_id", "source_id"):
            identifier(getattr(self, name), name)
        for name in ("source_version", "kind", "component", "engine", "status", "protocol", "environment", "tool", "evaluation_scope"):
            bounded(getattr(self, name), 128, name)
        for name in ("parent_id", "error_code", "metric"):
            value = getattr(self, name)
            if value is not None:
                bounded(value, 96 if name == "parent_id" else 128, name)
        bounded(self.checked_change, 512, "checked_change")
        if self.data_scope not in SCOPES:
            raise AccessDenied("Unsupported record scope")
        if self.maximize is not None and type(self.maximize) is not bool:
            raise ContextError("maximize must be boolean")
        if self.score is not None:
            finite(self.score, "score", -float("inf"))
        finite(self.seconds, "seconds")
        finite(self.peak_memory_mib, "peak_memory_mib")
        if not isinstance(self.dependencies, tuple) or len(self.dependencies) > 8:
            raise ContextError("Record metadata supports up to eight dependencies; keep larger graphs in the source")
        for item in self.dependencies:
            identifier(item, "dependency")
        if size(asdict(self)) > 3072:
            raise ContextError("Record metadata exceeds 3072 bytes")


@dataclass(frozen=True)
class Command:
    operation_id: str
    tool: str
    arguments_json: str
    note: ResearchNote | None = None

    @classmethod
    def parse(cls, raw: Any, tools: tuple[str, ...] = TOOLS) -> Command:
        if not isinstance(raw, dict) or set(raw) - {"operation_id", "tool", "arguments", "research_note"}:
            raise ContextError("Expected one ToolCommand with known fields")
        identifier(raw.get("operation_id"), "operation_id")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", raw["operation_id"]):
            raise ContextError("operation_id must contain ASCII letters, digits, underscore, or hyphen")
        if raw.get("tool") not in tools or not isinstance(raw.get("arguments"), dict):
            raise ContextError("Unknown tool or invalid argument object")
        note = ResearchNote.parse(raw["research_note"]) if raw.get("research_note") is not None else None
        if note:
            note.persisted(raw["operation_id"])
        return cls(raw["operation_id"], raw["tool"], canonical(raw["arguments"]), note)

    def as_dict(self) -> dict[str, Any]:
        value = {"operation_id": self.operation_id, "tool": self.tool, "arguments": json.loads(self.arguments_json)}
        if self.note:
            value["research_note"] = asdict(self.note)
        return json.loads(canonical(value))


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    core_json: str
    observations_json: str
    cards: tuple[str, ...]
    details: tuple[str, ...]
    failures_json: str
    profile_json: str
    max_record_seq: int
    history_count: int
    pending_count: int
    created_at: float
    build_seconds: float
    source_reads: int


@dataclass(frozen=True)
class PreparedRequest:
    request_id: str
    snapshot_id: str
    transport: str
    wire_json: str
    request_hash: str
    wire_bytes: int
    response_reserve: int
    conservative_size: int
    budget: int
    attempts: int
    stage: int
    build_seconds: float
    selected_ids: tuple[str, ...]
    block_bytes: Mapping[str, int] = field(default_factory=dict)
    fit_trace: tuple[str, ...] = ()
