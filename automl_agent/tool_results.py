"""Typed public result envelopes; private executor payloads retain their shapes."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .contracts import Contract
from .util import digest


class ToolResult(Contract):
    tool: str
    tool_api_version: Literal[2] = 2
    operation_id: str
    observation_id: str
    evidence_ref: dict[str, str]
    status: Literal["success", "no_matches", "ambiguous", "missing_version", "unknown_id", "incompatible",
        "technical_failure", "invalid_cursor", "budget_too_small", "interrupted", "outcome_unknown",
        "hypothesis_revision_required", "hypothesis_review_unavailable"]
    error_code: str | None = None
    provenance: dict[str, Any]
    complete: bool
    continuation: dict[str, Any] | None = None
    data: dict[str, Any]


class ReportData(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str = "success"
    report: dict[str, Any] | None = None
    content: str | None = None


class RevisionData(BaseModel):
    model_config = ConfigDict(extra="allow")
    decision: Literal["admitted", "rejected"]
    revision_status: Literal["rejected", "reevaluating", "committed", "completed", "interrupted_before_commit", "outcome_unknown"]
    previous_protocol_hash: str
    proposed_protocol_hash: str | None
    active_protocol_hash: str
    reevaluated_ids: list[str]
    pending_ids: list[str]
    outcomes: list[dict[str, str]]
    reason: str | None = None


class SearchData(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    entries: list[dict[str, Any]] = []
    total: int = 0
    method: str | None = None
    next_cursor: str | None = None


class KnowledgeData(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    entry: dict[str, Any] | None = None
    content: str | None = None


class RunData(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    records: list[dict[str, Any]] = []
    content: str | None = None
    next_cursor: str | None = None


class TrainingData(BaseModel):
    model_config = ConfigDict(extra="allow")
    experiment: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class FinishData(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    reason: str | None = None
    error: str | None = None


class DatasetReportResult(ToolResult):
    tool: Literal["get_dataset_report"]
    data: ReportData


class DiagnosticReportResult(ToolResult):
    tool: Literal["get_diagnostic_report"]
    data: ReportData


class ValidationRevisionResult(ToolResult):
    tool: Literal["revise_validation"]
    data: RevisionData


class KnowledgeSearchResult(ToolResult):
    tool: Literal["search_knowledge"]
    data: SearchData


class KnowledgeListResult(ToolResult):
    tool: Literal["list_knowledge"]
    data: SearchData


class KnowledgeReadResult(ToolResult):
    tool: Literal["read_knowledge"]
    data: KnowledgeData


class RunSearchResult(ToolResult):
    tool: Literal["search_run_records"]
    data: RunData


class RunListResult(ToolResult):
    tool: Literal["list_run_records"]
    data: RunData


class RunReadResult(ToolResult):
    tool: Literal["read_run_records"]
    data: RunData


class TrainResult(ToolResult):
    tool: Literal["train_model"]
    data: TrainingData


class AutoMLResult(ToolResult):
    tool: Literal["run_automl"]
    data: TrainingData


class TuneResult(ToolResult):
    tool: Literal["tune_model"]
    data: TrainingData


class CustomResult(ToolResult):
    tool: Literal["run_custom_experiment"]
    data: TrainingData


class EnsembleResult(ToolResult):
    tool: Literal["build_ensemble"]
    data: TrainingData


class FinishResult(ToolResult):
    tool: Literal["finish"]
    data: FinishData


RESULT_SCHEMAS = dict(zip(("get_dataset_report", "revise_validation", "search_knowledge", "list_knowledge", "read_knowledge",
    "search_run_records", "list_run_records", "read_run_records", "train_model", "run_automl", "tune_model",
    "run_custom_experiment", "build_ensemble", "finish"), (DatasetReportResult, ValidationRevisionResult,
    KnowledgeSearchResult, KnowledgeListResult, KnowledgeReadResult, RunSearchResult, RunListResult, RunReadResult,
    TrainResult, AutoMLResult, TuneResult, CustomResult, EnsembleResult, FinishResult)))
RESULT_SCHEMAS["get_diagnostic_report"] = DiagnosticReportResult


def envelope(c, operation, payload):
    if payload.get("tool_api_version") == 2:
        return payload
    status = payload.get("status", "success")
    if status in {"completed", "frozen", "available", "complete", "partial", "revised"}:
        status = "success"
    if status == "rejected":
        status = "incompatible"
    measured = (payload.get("result") or {}).get("status")
    if measured in {"timeout", "memory_limit", "error", "failed"}:
        status = "technical_failure"
    elif measured in {"interrupted", "outcome_unknown"}:
        status = measured
    if status not in ToolResult.model_fields["status"].annotation.__args__:
        status = "technical_failure"
    version = digest(payload)
    observation = digest({"run": c.run_id, "operation": operation.operation_id, "result": version})
    return RESULT_SCHEMAS[operation.tool](tool=operation.tool, tool_api_version=2, operation_id=operation.operation_id,
        observation_id=observation, evidence_ref={"kind": "observation", "id": observation, "version": version},
        status=status, error_code=payload.get("error_code") or (measured if status == "technical_failure" else None), data=payload,
        provenance={"run_id": c.run_id, "result_hash": version, "data_version": c.dataset.source_hash,
            "protocol_hash": c.plan.hash, "corpus_version": c.references.version},
        complete=payload.get("complete", True), continuation=payload.get("continuation")).model_dump(mode="json", exclude_unset=True)


def payload(result):
    return result.get("data", result) if isinstance(result, dict) else result
