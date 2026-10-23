from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .util import digest

Kind = Literal["regression", "classification", "multioutput_regression", "multioutput_classification", "multilabel", "forecasting", "ranking"]
PredictionType = Literal["values", "labels", "probabilities", "scores"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @property
    def hash(self) -> str:
        return digest(self.model_dump(mode="json"))


class MetricSpec(Contract):
    name: Literal["rmse", "mse", "mae", "r2", "f1", "accuracy", "roc_auc", "log_loss", "average_precision", "ndcg", "map"]
    average: Literal["macro", "micro", "weighted", "binary", "samples"] = "macro"
    positive_label: str | int | float | None = None
    k: int | None = Field(default=None, ge=1)
    aggregation: Literal["uniform", "weighted"] = "uniform"
    target_weights: list[float] | None = None
    relevance_threshold: float = 0.0

    @property
    def maximize(self) -> bool:
        return self.name not in {"rmse", "mse", "mae", "log_loss"}


class OutputSpec(Contract):
    columns: list[str] = Field(min_length=1)
    prediction_type: PredictionType
    # Each probability column maps to one target and one explicit class label.
    probability_columns: dict[str, tuple[str, str | int | float]] = Field(default_factory=dict)


class TaskSpec(Contract):
    kind: Kind
    targets: list[str] = Field(min_length=1)
    id_columns: list[str] = Field(default_factory=list)
    feature_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    text_columns: list[str] = Field(default_factory=list)
    time_column: str | None = None
    group_column: str | None = None
    series_columns: list[str] = Field(default_factory=list)
    horizon: int | None = Field(default=None, ge=1)
    metric: MetricSpec
    output: OutputSpec
    classes: dict[str, list[str | int | float]] = Field(default_factory=dict)
    source_hash: str
    evidence: dict[str, list[str]] = Field(default_factory=dict)
    explicit_validation: Literal["random", "group", "temporal"] | None = None
    feature_constraints: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def roles(self):
        if not self.feature_columns and self.kind != "forecasting":
            raise ValueError("A non-forecasting task requires declared input features")
        for name, columns in (("targets", self.targets), ("identifiers", self.id_columns), ("features", self.feature_columns)):
            if len(columns) != len(set(columns)):
                raise ValueError(f"Duplicate {name}")
        if set(self.targets) & set(self.feature_columns + self.id_columns):
            raise ValueError("Targets overlap identifiers or features")
        if set(self.id_columns) & set(self.feature_columns):
            raise ValueError("Identifiers cannot be training features")
        if not set(self.categorical_columns + self.text_columns) <= set(self.feature_columns):
            raise ValueError("Unknown categorical/text feature")
        if set(self.categorical_columns) & set(self.text_columns):
            raise ValueError("Text and categorical roles overlap")
        if self.kind == "ranking" and (not self.group_column or self.metric.name not in {"ndcg", "map"}):
            raise ValueError("Ranking requires groups and a ranking metric")
        if self.kind == "forecasting" and (not self.time_column or not self.horizon):
            raise ValueError("Forecasting requires time and horizon")
        if len(self.targets) > 1 and self.kind in {"regression", "classification", "ranking"}:
            raise ValueError("Multiple targets require an explicit multioutput family")
        if self.metric.aggregation == "weighted":
            weights = self.metric.target_weights
            if not weights or len(weights) != len(self.targets) or min(weights) < 0 or sum(weights) <= 0:
                raise ValueError("Invalid target weights")
        if self.metric.name in {"ndcg", "map"} and self.metric.k is None:
            raise ValueError("Ranking requires an explicit k")
        classification = self.kind in {"classification", "multioutput_classification", "multilabel"}
        if classification and self.metric.name in {"rmse", "mse", "mae", "r2", "ndcg", "map"}:
            raise ValueError("Metric is incompatible with classification")
        if self.kind in {"regression", "multioutput_regression", "forecasting"} and self.metric.name not in {"rmse", "mse", "mae", "r2"}:
            raise ValueError("Numeric task requires a supported regression metric")
        if classification and self.output.prediction_type not in {"labels", "probabilities"}:
            raise ValueError("Classification requires labels or probabilities")
        if not set(self.feature_constraints) <= set(self.feature_columns):
            raise ValueError("Unknown feature constraint")
        return self


class TaskResolutionProposal(Contract):
    request_id: str
    source_hash: str
    status: Literal["proposed", "unresolved"]
    task_spec: TaskSpec | None = None
    evidence: dict[str, list[str]] = Field(default_factory=dict)
    unresolved_fields: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def complete(self):
        if self.status == "proposed" and (self.task_spec is None or self.unresolved_fields or self.contradictions):
            raise ValueError("A proposed contract must be complete and consistent")
        if self.status == "unresolved" and self.task_spec is not None:
            raise ValueError("An unresolved proposal cannot confirm a contract")
        return self


class DataProfile(Contract):
    data_version: str
    scope: str
    source: str
    version: int = 1
    rows: dict[str, int]
    columns: dict[str, dict[str, Any]]
    observations: list[dict[str, Any]] = Field(default_factory=list)


class FeedbackPolicy(Contract):
    min_slice_rows: int = Field(default=30, ge=30)
    report_ids: list[str] = Field(default_factory=list)
    version: int = 1
    arbitrary_external_slices: Literal[False] = False


class SelectionPolicy(Contract):
    maximize: bool
    min_delta_abs: float = Field(default=0.001, ge=0)
    min_delta_rel: float = Field(default=0.0, ge=0)
    tie_order: list[str] = Field(default_factory=lambda: ["finish_seconds", "peak_memory_mb", "created_index"])
    control_tie: Literal["baseline"] = "baseline"

    def tolerance(self, reference: float) -> float:
        return max(self.min_delta_abs, self.min_delta_rel * abs(reference))

    def improvement(self, score: float, reference: float) -> bool:
        gain = score - reference if self.maximize else reference - score
        threshold = self.tolerance(reference)
        return gain > threshold and not math.isclose(gain, threshold, rel_tol=1e-12, abs_tol=0.0)

    def tied(self, score: float, reference: float) -> bool:
        gap, threshold = abs(score - reference), self.tolerance(reference)
        return gap <= threshold or math.isclose(gap, threshold, rel_tol=1e-12, abs_tol=0.0)


class ValidationPlan(Contract):
    strategy: Literal["random", "group", "temporal"]
    reason: str
    fit: list[int]
    validation: list[int]
    control: list[int]
    seed: int = 42
    version: int = 1
    data_version: str
    feedback_policy: FeedbackPolicy
    selection_policy: SelectionPolicy
    class_coverage: dict[str, Any] = Field(default_factory=dict)
    observations: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def disjoint(self):
        a, b, c = map(set, (self.fit, self.validation, self.control))
        if not a or not b or not c or a & b or a & c or b & c:
            raise ValueError("Fit, validation and control must be nonempty and disjoint")
        if any(len(x) != len(set(x)) for x in (self.fit, self.validation, self.control)):
            raise ValueError("Duplicate split indices")
        return self


class ImputationPlan(Contract):
    numeric: Literal["median", "iterative_bayesian_ridge", "iterative_trees", "knn", "native"] = "median"
    categorical: Literal["missing_category", "most_frequent", "model"] = "missing_category"
    indicators: bool = True
    max_iter: int = Field(default=3, ge=1, le=10)
    n_nearest_features: int = Field(default=20, ge=1, le=100)
    empty_numeric: float = 0.0
    categorical_model_columns: list[str] = Field(default_factory=list)
    constraints: dict[str, dict[str, Any]] = Field(default_factory=dict)


class Transformation(Contract):
    name: Literal["standard_scale", "robust_scale", "yeo_johnson", "spline", "drop_constant", "drop_correlated", "math", "datetime", "log1p", "select_kbest"]
    parameters: dict[str, Any] = Field(default_factory=dict)


class PipelineSpec(Contract):
    engine: Literal["direct", "flaml", "autogluon", "optuna", "custom", "ensemble"] = "direct"
    model: str = "extra_trees"
    representation: Literal["tabular", "text", "mixed", "lags", "native"] = "tabular"
    imputation: ImputationPlan = Field(default_factory=ImputationPlan)
    transformations: list[Transformation] = Field(default_factory=list)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    training_objective: str | None = None
    prediction_policy: Literal["default", "threshold", "calibrated"] = "default"
    postprocess_fit_policy: Literal["none", "inner_holdout"] = "none"
    target_transform: Literal["none", "log1p"] = "none"
    custom_module: str | None = None
    custom_hash: str | None = None
    custom_semantic_hash: str | None = None
    custom_component: Literal["pipeline", "transformer", "analysis"] = "pipeline"
    members: list[str] = Field(default_factory=list)
    search_owner: Literal["none", "optuna", "flaml", "autogluon"] = "none"
    learners: list[str] = Field(default_factory=list)
    search_space: dict[str, Any] = Field(default_factory=dict)
    n_trials: int | None = Field(default=8, ge=1)
    text_analyzer: Literal["word", "char", "char_wb", "hashing"] = "word"
    max_text_features: int = Field(default=20000, ge=100, le=100000)
    resampling: Literal["none", "random_over", "smote"] = "none"

    @model_validator(mode="after")
    def search(self):
        expected = self.engine if self.engine in {"flaml", "autogluon", "optuna"} else "none"
        if self.search_owner not in {"none", expected}:
            raise ValueError("Nested optimizers are forbidden")
        if self.engine in {"flaml", "autogluon"} and self.search_space:
            raise ValueError("AutoML search spaces require an engine-specific verified adapter")
        if self.prediction_policy != "default" and self.postprocess_fit_policy != "inner_holdout":
            raise ValueError("Postprocessing must be learned inside fit")
        if self.engine == "ensemble" and len(self.members) < 2:
            raise ValueError("An ensemble requires at least two members")
        return self


class Hypothesis(Contract):
    observation: str = Field(min_length=1)
    expected_effect: str = Field(min_length=1)
    changed_component: str = Field(min_length=1)
    acceptance: str = Field(min_length=1)
    evidence_refs: list[dict[str, str]] = Field(default_factory=list, max_length=10,
        description="Evidence: exact kind/id/version/collection returned by tools.")


class ExperimentSpec(Contract):
    experiment_id: str
    parent_id: str | None = None
    hypothesis: Hypothesis
    pipeline: PipelineSpec
    task_hash: str
    data_version: str
    validation_hash: str
    budget_seconds: float = Field(gt=0)
    budget_mode: Literal["research", "full_comparison"] = "research"
    search_action: Literal["draft", "improve", "debug"] = "draft"
    debug_root: str | None = None
    method_refs: list[str] = Field(default_factory=list)
    recipe_ref: str | None = None
    reference_ids: list[str] = Field(default_factory=list)
    search_space_ref: str | None = None
    knowledge_bundle_version: str | None = None
    status: Literal["pilot", "full", "baseline"] = "full"
    purpose: str = "research"
    library_versions: dict[str, str] = Field(default_factory=dict)

    @property
    def computation_hash(self) -> str:
        pipeline = self.pipeline.model_dump(mode="json")
        if self.pipeline.engine == "custom" and self.pipeline.custom_semantic_hash:
            pipeline["custom_hash"] = pipeline["custom_module"] = self.pipeline.custom_semantic_hash
        return digest({
            "pipeline": pipeline, "task_hash": self.task_hash,
            "data": self.data_version, "validation": self.validation_hash,
            "libraries": self.library_versions, "status": self.status,
        })


class PredictionBundle(Contract):
    data_version: str
    split_id: str
    protocol_hash: str
    row_keys: list[str]
    row_order_hash: str
    target_names: list[str]
    prediction_type: PredictionType
    values: dict[str, Any]
    classes: dict[str, list[str | int | float]] = Field(default_factory=dict)


class TrainingSummary(Contract):
    rows: int = Field(ge=0)
    seconds: float = Field(ge=0)
    selected_parameters: dict[str, Any] = Field(default_factory=dict)
    imputation: dict[str, Any] = Field(default_factory=dict)
    inner_scores: dict[str, Any] = Field(default_factory=dict)
    auxiliary_fits: int = 0
    refit_supported: bool = True
    preprocessing: dict[str, Any] | None = None
    budget: dict[str, Any] | None = None
    execution_phases: dict[str, float] | None = None


class FitStatistic(Contract):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    aggregation: Literal["mean", "std", "min", "max", "median", "count", "correlation", "quantile", "custom_statistic"]
    columns: list[str] = Field(default_factory=list, max_length=6)
    value: float | int | bool | None
    support_rows: int = Field(ge=0)


class CustomAnalysisReport(Contract):
    statistics: list[FitStatistic] = Field(max_length=30)


class ExperimentResult(Contract):
    experiment_id: str
    status: Literal["completed", "import_error", "unsupported_mode", "invalid_prediction", "timeout", "memory_limit", "disk_limit", "error", "interrupted", "outcome_unknown"]
    validation_hash: str
    score: float | None = None
    seconds: float = 0
    peak_memory_mb: float = 0
    finish_seconds: float = 0
    created_index: int = 0
    training_summary: dict[str, Any] = Field(default_factory=dict)
    feedback: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    scope: Literal["full", "pilot", "baseline"] = "full"


class RefitPolicy(Contract):
    experiment_id: str
    pipeline_hash: str
    frozen_pipeline: PipelineSpec
    dependencies: list[str] = Field(default_factory=list)
    rows_policy: Literal["fit_plus_validation_then_all_train"] = "fit_plus_validation_then_all_train"
    new_search_allowed: Literal[False] = False
    iteration_policy: Literal["saved_configuration"] = "saved_configuration"


class OperationRecord(Contract):
    run_id: str
    operation_id: str
    tool: str
    arguments_hash: str
    request_hash: str | None = None
    experiment_id: str | None = None
    status: Literal["pending", "running", "succeeded", "failed"]
    outcome: Literal["confirmed", "outcome_unknown"] = "outcome_unknown"
    started_at: float
    ended_at: float | None = None
    budget_seconds: float = 0
    result: dict[str, Any] | None = None
    component: str | None = None
    engine: str | None = None
    parent_id: str | None = None
    protocol_hash: str | None = None


class ExecutionPolicy(Contract):
    mode: Literal["isolated", "limited_trust"] = "limited_trust"
    writable_directories: list[str]
    tests: dict[str, Any]
    limitations: list[str]
    worker_network_allowed: Literal[False] = False
    credential_environment_allowed: Literal[False] = False


class ResearchState(Contract):
    version: int = 1
    task_hash: str
    validation_hash: str
    facts: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    raw_best: float | None = None
    leader: str | None = None
    directions: list[dict[str, Any]] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    current_hypothesis: dict[str, Any] | None = None
    counters: dict[str, int] = Field(default_factory=dict)
    remaining: dict[str, Any] = Field(default_factory=dict)


class InspectArgs(Contract):
    scope: Literal["fit", "validation_features", "test_features"] = "fit"
    report: Literal["profile", "dictionary", "missing", "categories", "correlations", "slice", "validation_proposal"] = "profile"
    columns: list[str] = Field(default_factory=list, max_length=30)
    group_by: str | None = None
    predicate: dict[str, Any] | None = None
    include_target: bool = False
    rationale: str | None = None
    proposed_strategy: Literal["random", "group", "temporal"] | None = None


class ReferenceArgs(Contract):
    mode: Literal["methods", "recipes", "api", "search", "read", "search_space"] = "search"
    query: str = Field(default="", description="For api use a qualified symbol such as sklearn.linear_model.Ridge, or an unambiguous short name such as Ridge. Use search for broad natural-language queries.")
    reference_id: str | None = Field(default=None, description="Exact ID returned by an earlier result, including IDs in a method card's references. Use read with this ID to obtain the document directly.")
    library: str | None = None
    version: str | None = None
    cursor: str | Literal[0] | None = Field(default=0, description="Opaque continuation returned by get_reference; begin with null or 0. Keep the same query, mode and filters.")
    limit: int = Field(default=5, ge=1, le=5)
    max_chars: int = Field(default=10000, ge=100, le=20000)
    knowledge_gap: str | None = None


class TrainArgs(Contract):
    pipeline: PipelineSpec
    hypothesis: Hypothesis
    parent_id: str | None = None
    search_action: Literal["draft", "improve", "debug"] = "draft"
    budget_seconds: float = Field(default=120, gt=0)
    pilot: bool = False
    method_refs: list[str] = Field(default_factory=list)
    recipe_ref: str | None = None
    reference_ids: list[str] = Field(default_factory=list)
    search_space_ref: str | None = None


class CustomArgs(TrainArgs):
    code: str = Field(min_length=1, max_length=60000)


class AnalyzeArgs(Contract):
    experiment_ids: list[str] = Field(default_factory=list)
    report_id: str | None = None
    knowledge_gap: str | None = None
    include_code: bool = False
    operation_ids: list[str] = Field(default_factory=list, max_length=20)
    cursor: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=20)


class EnsembleArgs(Contract):
    members: list[str] = Field(min_length=2, max_length=5)
    hypothesis: Hypothesis
    budget_seconds: float = Field(default=120, gt=0)


class FinishArgs(Contract):
    reason: str = Field(min_length=1)
    imputation_skip_reason: str | None = None


LEGACY_TOOL_SCHEMAS = {
    "inspect_dataset": InspectArgs, "get_reference": ReferenceArgs,
    "train_model": TrainArgs, "run_automl": TrainArgs, "tune_model": TrainArgs,
    "analyze_experiments": AnalyzeArgs, "build_ensemble": EnsembleArgs,
    "run_custom_experiment": CustomArgs, "finish": FinishArgs,
}


class DatasetReportArgs(Contract):
    scope: Literal["fit", "validation_features", "test_features"] = "fit"
    report: Literal["profile", "dictionary", "missing", "categories", "correlations", "slice"] = "profile"
    columns: list[str] = Field(default_factory=list, max_length=30)
    group_by: str | None = None
    predicate: dict[str, Any] | None = None
    include_target: bool = False
    knowledge_gap: str | None = None
    cursor: str | None = None
    max_chars: int = Field(default=6000, ge=2000, le=12000)


class ReviseValidationArgs(Contract):
    proposed_strategy: Literal["random", "group", "temporal"]
    rationale: str = Field(min_length=1)


class KnowledgeFilters(Contract):
    library: str | None = None
    version: str | None = None


class KnowledgeListArgs(Contract):
    collection: Literal["documents", "methods", "recipes", "search_spaces"] = "documents"
    filters: KnowledgeFilters = Field(default_factory=KnowledgeFilters)
    cursor: str | None = None
    limit: int = Field(default=5, ge=1, le=10)
    max_chars: int = Field(default=6000, ge=2000, le=12000)
    knowledge_gap: str | None = None


class KnowledgeSearchArgs(KnowledgeListArgs):
    query: str = Field(min_length=1, max_length=2000)
    strategy: Literal["auto", "exact_symbol", "fts", "lexical"] = "auto"


class KnowledgeReadArgs(Contract):
    collection: Literal["documents", "methods", "recipes", "search_spaces"]
    id: str
    corpus_version: str
    cursor: str | None = None
    max_chars: int = Field(default=6000, ge=2000, le=12000)
    knowledge_gap: str | None = None


class RunFilters(Contract):
    kind: Literal["experiment", "operation", "observation", "note", "source"] | None = None
    component: str | None = None
    engine: str | None = None
    status: str | None = None
    error_code: str | None = None
    protocol: str | None = None
    parent_id: str | None = None
    environment: str | None = None
    tool: str | None = None
    evaluation_scope: str | None = None
    record_id: str | None = None


class RunListArgs(Contract):
    filters: RunFilters = Field(default_factory=RunFilters)
    cursor: str | None = None
    limit: int = Field(default=5, ge=1, le=10)
    max_chars: int = Field(default=6000, ge=2000, le=12000)
    knowledge_gap: str | None = None


class RunSearchArgs(RunListArgs):
    query: str = Field(default="", max_length=2000)


class RunSourceRef(Contract):
    source_id: str
    version: str
    scope: Literal["contract", "policy", "fit", "validation_features", "test_features", "registered_feedback", "reference", "code", "environment", "controller"]
    section: str = ""
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=768, ge=4, le=768)


class RunReadArgs(Contract):
    ids: list[str] = Field(default_factory=list, max_length=10)
    source: RunSourceRef | None = None
    include_code: bool = False
    cursor: str | None = None
    max_chars: int = Field(default=6000, ge=2000, le=12000)
    knowledge_gap: str | None = None

    @model_validator(mode="after")
    def selection(self):
        if bool(self.ids) == bool(self.source):
            raise ValueError("Choose ids (up to ten) OR one versioned source")
        if self.include_code and (self.source or len(self.ids) != 1):
            raise ValueError("Code requires exactly one record ID")
        if self.source and self.cursor:
            raise ValueError("Versioned source reads use their returned byte offset")
        return self


TOOL_API_VERSION = 2
TOOL_SCHEMAS = {
    "get_dataset_report": DatasetReportArgs, "revise_validation": ReviseValidationArgs,
    "search_knowledge": KnowledgeSearchArgs, "list_knowledge": KnowledgeListArgs, "read_knowledge": KnowledgeReadArgs,
    "search_run_records": RunSearchArgs, "list_run_records": RunListArgs, "read_run_records": RunReadArgs,
    "train_model": TrainArgs, "run_automl": TrainArgs, "tune_model": TrainArgs,
    "run_custom_experiment": CustomArgs, "build_ensemble": EnsembleArgs, "finish": FinishArgs,
}

TOOL_DESCRIPTIONS = {
    "get_dataset_report": "Compute one permitted data report; target analysis and arbitrary slices use fit only.",
    "revise_validation": "State-changing action: may re-evaluate candidates and apply a new active validation protocol in this call, without further confirmation. Metric, roles and closed control cannot change.",
    "search_knowledge": "Search one collection. Ranked candidates are not confirmed answers; read evidence next. Exact symbol, FTS5 BM25 or lexical matching.",
    "list_knowledge": "Browse a filtered catalog, without claiming a search match.",
    "read_knowledge": "Read a specific versioned knowledge source, including conditions, warnings and continuation.",
    "search_run_records": "Find permitted past results, errors, notes or observations by filters and literal normalized words.",
    "list_run_records": "Browse the permitted run index by type and filters.",
    "read_run_records": "Read up to ten saved records, one saved code object, or a versioned contract/source page.",
    "train_model": "Train and evaluate one direct pipeline.",
    "run_automl": "Run bounded FLAML or AutoGluon search using fit only.",
    "tune_model": "Run bounded sequential Optuna search using fit only.",
    "run_custom_experiment": "Execute a Python pipeline, transformer or fit-only analysis under controller limits.",
    "build_ensemble": "Combine compatible saved predictions as a separate evaluated candidate.",
    "finish": "Irreversibly freeze research; controller selects finalists and performs control/refit.",
}


class ToolCommand(Contract):
    operation_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    # The three legacy names are private executor compatibility only. Neither
    # public transport advertises or accepts them.
    tool: Literal["get_dataset_report", "revise_validation", "search_knowledge", "list_knowledge", "read_knowledge", "search_run_records", "list_run_records", "read_run_records", "train_model", "run_automl", "tune_model", "build_ensemble", "run_custom_experiment", "finish", "inspect_dataset", "get_reference", "analyze_experiments"] = Field(json_schema_extra={"enum": list(TOOL_SCHEMAS)})
    arguments: dict[str, Any]
