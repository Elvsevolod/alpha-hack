"""Coordinate permitted EDA, immutable reports and the shared worker budget."""
import time

import numpy as np

from context_compressor import ProfileKey

from ..contracts import InspectArgs
from ..executor import stage_frame
from ..knowledge_tools import CursorError
from ..profiling import (MAX_CORRELATION_COLUMNS, MAX_CORRELATION_ROWS, PROFILE_VERSION,
    profile_key, semantic_type)
from ..util import digest


class EDAService:
    def __init__(self, dataset, repository, executor, budget, workdir, finalist_costs):
        self.dataset = dataset
        self.repository = repository
        self.executor = executor
        self.budget = budget
        self.workdir = workdir
        self.finalist_costs = finalist_costs

    def profile(self, plan, scope="fit"):
        indices = list(range(len(self.dataset.test))) if scope == "test_features" else getattr(plan,
            "fit" if scope == "fit" else "validation")
        key = ProfileKey(self.dataset.source_hash, profile_key(self.dataset, indices, scope),
            scope=scope, format_version=PROFILE_VERSION)
        def calculate():
            try:
                return self._profile(plan, indices, scope)
            except (ValueError, OSError, TimeoutError) as error:
                from ..contracts import DataProfile
                return DataProfile(data_version=self.dataset.source_hash, scope=scope, source="immutable_csv",
                    version=int(PROFILE_VERSION), rows={}, columns={}, status="budget_too_small" if isinstance(error,TimeoutError) else "technical_failure",
                    error="Optional profile failed: " + type(error).__name__).model_dump(mode="json")
        return self.repository.profile(key, calculate)

    def _profile(self, plan, indices, scope, selected=None):
        from ..contracts import DataProfile
        failure = dict(data_version=self.dataset.source_hash, scope=scope, source="immutable_csv",
            version=int(PROFILE_VERSION), rows={"train": len(self.dataset.train), "test": len(self.dataset.test)},
            columns={}, observations=[], findings=[])
        end = min(self.budget.deadline - self.budget.final_reserve(self.finalist_costs()), time.time() + 20)
        if end <= time.time():
            return DataProfile(**failure, status="budget_too_small", error="Profile unknown: original deadline exhausted").model_dump(mode="json")
        directory = self.workdir / "analyses" / ("profile_" + profile_key(self.dataset, indices, scope))
        directory.mkdir(parents=True, exist_ok=True)
        frame = selected if selected is not None else (self.dataset.test if scope == "test_features" else self.dataset.train.iloc[indices])
        columns = self.dataset.task.feature_columns + (self.dataset.task.targets if scope == "fit" else [])
        try:
            stage_frame(directory / "profile_rows.json", frame[columns], deadline=end, memory_mb=self.executor.memory_mb)
            stage_frame(directory / "profile_test.json", self.dataset.test[self.dataset.task.feature_columns], deadline=end, memory_mb=self.executor.memory_mb)
        except (OSError, ValueError, TimeoutError) as error:
            return DataProfile(**failure, status="budget_too_small" if isinstance(error,TimeoutError) else "technical_failure",
                error="Profile staging incomplete: " + type(error).__name__).model_dump(mode="json")
        if end <= time.time():
            return DataProfile(**failure, status="budget_too_small", error="Profile staging exhausted deadline").model_dump(mode="json")
        result = self.executor.run(directory, {"mode": "profile", "task": self.dataset.task.model_dump(mode="json"),
            "scope": scope, "train_rows": len(self.dataset.train), "data_version": self.dataset.source_hash}, end-time.time())
        if result["status"] != "completed":
            return DataProfile(**failure, status="technical_failure", error="Profile worker: " + result["status"]).model_dump(mode="json")
        return result["report"]

    def report(self, args, plan):
        self.repository.check_scope(args.scope)
        raw = args.model_dump(exclude={"cursor", "max_chars", "knowledge_gap", "evaluation"})
        identity = {"data": self.dataset.source_hash, "protocol": plan.hash, "args": raw,
            "task": self.dataset.task.hash, "analyzer": PROFILE_VERSION,
            "evaluation":args.evaluation.model_dump(mode="json") if args.evaluation else None,
            "active_evaluation":self.evaluation_identity() if args.report == "errors" and hasattr(self,"evaluation_identity") else None}
        value = self.repository.report(identity)
        if value is not None:
            return value, identity
        if args.cursor:
            raise CursorError("invalid_cursor: report snapshot is unavailable")
        if args.report == "errors":
            if args.scope != "fit" or not args.include_target or args.predicate or args.columns or args.group_by:
                raise ValueError("Error diagnosis requires full fit scope, include_target=true and no projection/filter")
            handler = getattr(self, "error_runner", None)
            value = handler() if handler else {"status": "incompatible", "error": "No diagnostic fit allowance in this runtime"}
            self.repository.save_report(identity, value)
            return value, identity
        document = None
        if args.report == "profile" and not args.group_by:
            selected, _ = self.dataset.select_report(args, plan)
            if args.predicate:
                document = self._profile(plan, list(selected.index), args.scope, selected)
            else:
                _, document = self.profile(plan, args.scope)
        if args.report == "correlations":
            value = self.correlation_report(args, plan)
        elif args.report in {"missing", "categories", "slice"} or (args.group_by and args.report != "dictionary"):
            value = self.descriptive_report(args, plan)
        else:
            value = self.dataset.inspect(InspectArgs(**raw), plan, profile_document=document)
        value["report_schema_version"] = 3
        self.repository.save_report(identity, value)
        return value, identity

    def descriptive_report(self, args, plan):
        frame, columns = self.dataset.select_report(args, plan)
        result = {"report": args.report, "scope": args.scope, "data_version": self.dataset.source_hash,
            "protocol_hash": plan.hash, "rows": len(frame),
            "selection": {"columns": columns, "predicate": args.predicate, "group_by": args.group_by}}
        end = min(self.budget.deadline - self.budget.final_reserve(self.finalist_costs()), time.time() + 20)
        if end <= time.time():
            return {**result, "status": "budget_too_small", "error": "Original analysis deadline exhausted; aggregates unknown."}
        directory = self.workdir / "analyses" / digest({"result": result, "task": self.dataset.task.hash})
        try:
            directory.mkdir(parents=True, exist_ok=True)
            staged = list(dict.fromkeys([*columns, *([args.group_by] if args.group_by else [])]))
            stage_frame(directory / "analysis_X.json", frame[staged], deadline=end, memory_mb=self.executor.memory_mb)
            if end <= time.time():
                return {**result, "status": "budget_too_small", "error": "Staging exhausted original analysis deadline."}
            measured = self.executor.run(directory, {"mode": "descriptive_report", "columns": columns,
                "report": args.report, "group_by": args.group_by}, end-time.time())
        except TimeoutError:
            return {**result, "status": "budget_too_small", "error": "Staging exhausted the original analysis deadline; result unknown."}
        except (OSError, ValueError) as error:
            return {**result, "status": "technical_failure", "diagnostic_status": type(error).__name__,
                "error": "Optional aggregate analysis failed; aggregates unknown."}
        if measured["status"] != "completed":
            return {**result, "status": "technical_failure", "diagnostic_status": measured["status"],
                "error": "Aggregate worker did not complete; aggregates unknown."}
        return {**result, "status": "available", **measured["report"]}

    def correlation_report(self, args, plan):
        frame, columns = self.dataset.select_report(args, plan)
        if len(columns) > MAX_CORRELATION_COLUMNS:
            return {"status": "incompatible", "scope": args.scope, "report": "correlations",
                "error": "Select at most 30 columns explicitly; the default selection is too wide."}
        excluded = {name: "unsupported_type" for name in columns if semantic_type(self.dataset.task, name) != "numeric"}
        numeric = [name for name in columns if name not in excluded]
        source_rows = len(frame)
        if source_rows > MAX_CORRELATION_ROWS:
            frame = frame.iloc[np.sort(np.random.default_rng(plan.seed).choice(source_rows,
                MAX_CORRELATION_ROWS, replace=False))]
        result = {"report": "correlations", "scope": args.scope, "data_version": self.dataset.source_hash,
            "protocol_hash": plan.hash, "source_rows": source_rows, "sample_rows": len(frame),
            "sampled": len(frame) != source_rows, "sample_seed": plan.seed,
            "selection": {"columns": columns, "predicate": args.predicate},
            "limitation": "Association does not establish causality, leakage or usefulness of removing a feature."}
        identity = digest({"result": result, "task": self.dataset.task.hash, "version": PROFILE_VERSION})
        directory = self.workdir / "analyses" / identity
        reserve = self.budget.final_reserve(self.finalist_costs())
        end = min(self.budget.deadline - reserve, time.time() + 20)
        if end - time.time() <= 1:
            return {**result, "status": "budget_too_small", "error": "Finalization reserve protected; correlation is unknown."}
        try:
            directory.mkdir(parents=True, exist_ok=True)
            stage_frame(directory / "analysis_X.json", frame[numeric], deadline=end, memory_mb=self.executor.memory_mb)
            seconds = end - time.time()
            if seconds <= 0:
                return {**result, "status": "budget_too_small", "error": "Staging exhausted the diagnostic budget; correlation is unknown."}
            measured = self.executor.run(directory, {"mode": "correlations", "excluded_columns": excluded}, seconds)
        except TimeoutError:
            return {**result, "status": "budget_too_small", "error": "Staging exhausted the original analysis deadline; result unknown."}
        except (OSError, ValueError) as error:
            return {**result, "status": "technical_failure", "diagnostic_status": type(error).__name__,
                "error": "Optional correlation staging/execution failed; correlations are unknown."}
        if measured["status"] != "completed":
            return {**result, "status": "technical_failure", "diagnostic_status": measured["status"],
                "error": "Optional correlation analysis did not complete; correlations are unknown."}
        return {**result, "status": "available", **measured["report"]}
