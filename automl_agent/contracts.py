from __future__ import annotations

import math
from typing import Any, Literal, TypedDict, NotRequired, Annotated

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from .util import digest

Kind = Literal["regression", "classification", "multioutput_regression", "multioutput_classification", "multilabel", "forecasting", "ranking"]
PredictionType = Literal["values", "labels", "probabilities", "scores"]
LEARNED_PIPELINE_FIELDS = {"custom_module", "custom_hash", "custom_semantic_hash", "class_offsets", "learned_decision",
    "frozen_iterations", "iteration_schedule", "iteration_scale_rows", "ensemble_weights", "ensemble_meta", "learned_inverse"}


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @property
    def hash(self) -> str:
        return digest(self.model_dump(mode="json"))


class UnsupportedMetricError(ValueError):
    code = "unsupported_metric"


class MetricSpec(Contract):
    name: Literal["rmse", "mse", "mae", "r2", "f1", "accuracy", "roc_auc", "log_loss", "average_precision", "ndcg", "map", "mape", "balanced_accuracy"] = Field(description="MAPE is a fraction using max(abs(y), float64 epsilon), including negative/zero targets. Balanced accuracy is unadjusted mean recall of classes observed in truth; absent declared classes are excluded. Each produces one value per target, then uses uniform or declared target weights.")
    average: Literal["macro", "micro", "weighted", "binary", "samples"] = "macro"
    positive_label: str | int | float | None = None
    k: int | None = Field(default=None, ge=1)
    aggregation: Literal["uniform", "weighted"] = "uniform"
    target_weights: list[float] | None = None
    relevance_threshold: float = 0.0

    @field_validator("name", mode="before")
    @classmethod
    def supported_metric(cls, value):
        if not isinstance(value, str) or value not in {"rmse", "mse", "mae", "r2", "f1", "accuracy", "roc_auc", "log_loss", "average_precision", "ndcg", "map", "mape", "balanced_accuracy"}:
            raise UnsupportedMetricError(f"unsupported_metric: {value}")
        return value

    @property
    def maximize(self) -> bool:
        return self.name not in {"rmse", "mse", "mae", "mape", "log_loss"}

    @property
    def requires_probabilities(self) -> bool:
        return self.name in {"roc_auc", "log_loss", "average_precision"}


class OutputSpec(Contract):
    columns: list[str] = Field(min_length=1)
    prediction_type: PredictionType
    # Each probability column maps to one target and one explicit class label.
    probability_columns: dict[str, tuple[str, str | int | float]] = Field(default_factory=dict)


class RelatedTable(Contract):
    file: str
    keys: list[str] = Field(min_length=1)
    base_keys: list[str] = Field(min_length=1)
    prefix: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,30}$")
    aggregations: dict[str, list[Literal["mean", "sum", "count", "min", "max", "last"]]]
    time_column: str | None = None
    cutoff_column: str | None = None


class TaskSpec(Contract):
    contract_version: Literal[1, 2] = 1
    file_roles: dict[str, str] = Field(default_factory=dict)
    service_columns: list[str] = Field(default_factory=list)
    related_tables: list[RelatedTable] = Field(default_factory=list, max_length=5)
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
    explicit_validation: Literal["random", "group", "temporal", "temporal_group"] | None = Field(
        default=None,
        description="Set only when a cited task or organizer text explicitly prescribes this validation protocol; otherwise leave null for automatic inference.",
    )
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
        if set(self.service_columns) & set(self.targets + self.id_columns + self.feature_columns):
            raise ValueError("Service columns overlap model roles")
        if len(self.service_columns) != len(set(self.service_columns)):
            raise ValueError("Duplicate service columns")
        if set(self.targets) & set(self.series_columns + [self.time_column, self.group_column]):
            raise ValueError("Targets cannot be structural input columns")
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
        if set(self.classes) - set(self.targets):
            raise ValueError("Class labels must refer to declared targets")
        if self.classes and not classification:
            raise ValueError("Class labels require a classification task")
        if any(not labels or len(labels) != len(set(labels)) for labels in self.classes.values()):
            raise ValueError("Class labels must be nonempty and unique when declared")
        if any(isinstance(value, float) and not math.isfinite(value) for labels in self.classes.values() for value in labels):
            raise ValueError("Class labels must be finite")
        if classification and self.metric.name in {"rmse", "mse", "mae", "mape", "r2", "ndcg", "map"}:
            raise ValueError("Metric is incompatible with classification")
        if self.kind in {"regression", "multioutput_regression", "forecasting"} and self.metric.name not in {"rmse", "mse", "mae", "mape", "r2"}:
            raise ValueError("Numeric task requires a supported regression metric")
        if classification and self.output.prediction_type not in {"labels", "probabilities"}:
            raise ValueError("Classification requires labels or probabilities")
        if self.metric.requires_probabilities and self.output.prediction_type != "probabilities":
            raise ValueError("Probability metrics require probability output")
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
    status: Literal["available", "partial", "unavailable"] = "available"
    error: str | None = None
    targets: dict[str, dict[str, Any]] = Field(default_factory=dict)
    findings: list[dict[str, Any]] = Field(default_factory=list)


class FeedbackPolicy(Contract):
    min_slice_rows: int = Field(default=30, ge=1)
    report_ids: list[str] = Field(default_factory=list)
    version: int = 2
    class_order: list[str] = Field(default_factory=list)
    joint_disclosure: Literal["conservative_v2", "research_v3"] = "conservative_v2"
    history_guard: Literal["initial_protocol", "restricted_after_revision"] = "initial_protocol"
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
    strategy: Literal["random", "group", "temporal", "temporal_group"]
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
    evaluation_mode: Literal["holdout", "exact_group", "low_overlap", "rolling"] = "holdout"
    development_folds: list[dict[str, list[int]]] = Field(default_factory=list)

    @model_validator(mode="after")
    def disjoint(self):
        a, b, c = map(set, (self.fit, self.validation, self.control))
        if not a or not b or not c or a & b or a & c or b & c:
            raise ValueError("Fit, validation and control must be nonempty and disjoint")
        if any(len(x) != len(set(x)) for x in (self.fit, self.validation, self.control)):
            raise ValueError("Duplicate split indices")
        pool = a | b
        held = set()
        for fold in self.development_folds:
            if set(fold) != {"fit", "validation"}:
                raise ValueError("Development folds require fit and validation rows")
            f, v = set(fold["fit"]), set(fold["validation"])
            if not f or not v or f & v or not (f | v) <= pool or v & held:
                raise ValueError("Invalid development fold boundaries")
            held |= v
        if self.evaluation_mode == "rolling" and not self.development_folds:
            raise ValueError("Rolling protocol requires explicit development folds")
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
    name: Literal["standard_scale", "robust_scale", "yeo_johnson", "spline", "drop_constant", "drop_correlated", "math", "datetime", "log1p", "aggregate", "select_kbest", "history", "target_encode"]
    parameters: dict[str, Any] = Field(default_factory=dict,
        description="target_encode uses the JOINT key of columns, smoothing>=1 and split-aware fit labels; it does not implement an external encoder's target_type option.")


class PipelineSpec(Contract):
    engine: Literal["direct", "flaml", "autogluon", "lightautoml", "optuna", "custom", "ensemble"] = "direct"
    model: str = "extra_trees"
    representation: Literal["tabular", "text", "mixed", "lags", "native"] = "tabular"
    imputation: ImputationPlan = Field(default_factory=ImputationPlan)
    transformations: list[Transformation] = Field(default_factory=list)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    training_objective: str | None = None
    prediction_policy: Literal["default", "threshold", "calibrated", "class_offsets"] = "default"
    fit_diagnostics: Literal["none", "inner_holdout"] = Field(default="none",
        description="inner_holdout requires single-target classification with engine direct/optuna. For regression use none and get_diagnostic_report(kind=residuals).")
    class_offsets: list[float] | None = Field(default=None, max_length=1000)
    postprocess_fit_policy: Literal["none", "inner_holdout", "pooled_oof"] = "none"
    target_transform: Literal["none", "log1p"] = "none"
    custom_module: str | None = None
    custom_hash: str | None = None
    custom_semantic_hash: str | None = None
    custom_component: Literal["pipeline", "transformer", "analysis"] = "pipeline"
    members: list[str] = Field(default_factory=list)
    search_owner: Literal["none", "optuna", "flaml", "autogluon", "lightautoml"] = "none"
    learners: list[str] = Field(default_factory=list)
    search_space: dict[str, Any] = Field(default_factory=dict)
    n_trials: int | None = Field(default=12, ge=1)
    text_analyzer: Literal["word", "char", "char_wb", "hashing"] = "word"
    max_text_features: int = Field(default=20000, ge=100, le=100000)
    resampling: Literal["none", "random_over", "smote"] = "none"
    early_stopping_rounds: int | None = Field(default=None, ge=2, le=200)
    evaluation_interval: int = Field(default=25, ge=1, le=1000)
    frozen_iterations: int | None = Field(default=None, ge=1)
    iteration_schedule: dict[str, int] = Field(default_factory=dict)
    iteration_scale_rows: int | None = Field(default=None, ge=1)
    refit_iteration_policy: Literal["fixed", "sqrt_rows"] = "fixed"
    learned_decision: dict[str, Any] = Field(default_factory=dict)
    supervised_transform: bool = True
    autogluon_profile: Literal["selection", "bagging", "stacking"] = "selection"
    bag_folds: int = Field(default=3, ge=2, le=5)
    ensemble_strategy: Literal["uniform", "weighted_oof", "stacked_oof"] = "uniform"
    ensemble_weights: list[float] | None = Field(default=None, min_length=2, max_length=5)
    ensemble_meta: dict[str, Any] = Field(default_factory=dict)
    shift_strategy: Literal["none", "importance_weighting", "pseudo_label"] = "none"
    include_features: list[str] | None = None
    exclude_features: list[str] = Field(default_factory=list)
    decision_method: Literal["legacy", "regularized"] = "legacy"
    log_inverse: Literal["naive", "oof_smearing"] = "naive"
    learned_inverse: dict[str, Any] = Field(default_factory=dict)
    hpo_sampler: Literal["auto", "random", "tpe"] = "auto"
    continue_study: bool = True
    autogluon_configurations: dict[str, list[dict[str, Any]]] = Field(default_factory=dict,
        description="Explicit native CPU tree configurations. GBM class_weight and CAT class_weights dictionaries are unsupported here; use direct/Optuna for external-label weights. Native CAT auto_class_weights remains supported.")
    custom_oof: bool = False
    neighbor_scale: Literal["standard", "robust", "none"] = "standard"
    checkpoint_counts: list[int] = Field(default_factory=list, max_length=3)

    def research_recipe(self):
        """Complete choices for a fresh fit, without reusing learned controller state."""
        return self.model_dump(mode="json", exclude=LEARNED_PIPELINE_FIELDS | {"search_owner", "members", "ensemble_strategy"})

    @model_validator(mode="after")
    def search(self):
        if any(type(value) is not int or not 1 <= value <= 10000 for value in self.checkpoint_counts):
            raise ValueError("Checkpoint counts require at most three positive bounded iterations")
        if self.class_offsets is not None:
            if self.prediction_policy != "class_offsets" or not all(math.isfinite(v) and abs(v) <= 1 for v in self.class_offsets):
                raise ValueError("Frozen class offsets require bounded finite multiclass decision values")
        expected = self.engine if self.engine in {"flaml", "autogluon", "lightautoml", "optuna"} else "none"
        if self.search_owner not in {"none", expected}:
            raise ValueError("Nested optimizers are forbidden")
        if self.engine in {"flaml", "autogluon", "lightautoml"} and self.search_space:
            raise ValueError("AutoML search spaces require an engine-specific verified adapter")
        if self.prediction_policy != "default" and self.postprocess_fit_policy not in {"inner_holdout", "pooled_oof"}:
            raise ValueError("Postprocessing must be learned inside fit")
        if self.engine == "ensemble" and len(self.members) < 2:
            raise ValueError("An ensemble requires at least two members")
        if self.ensemble_weights is not None and (len(self.ensemble_weights) != len(self.members)
                or any(not math.isfinite(value) or value < 0 for value in self.ensemble_weights) or sum(self.ensemble_weights) <= 0):
            raise ValueError("Blend weights must be finite nonnegative values for every member")
        return self


class EvidenceRef(TypedDict):
    kind: Literal["knowledge", "observation", "record"]
    id: Annotated[str, Field(min_length=1)]
    version: Annotated[str, Field(min_length=1)]
    collection: NotRequired[str]


class Hypothesis(Contract):
    observation: str = Field(min_length=1)
    expected_effect: str = Field(min_length=1)
    changed_component: str = Field(min_length=1)
    acceptance: str = Field(min_length=1)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=10,
        description="Copy the exact returned evidence_ref or knowledge reference, including its real version; document types such as method/experiment are not evidence kinds.")
    changed_fields: list[str] = Field(default_factory=list, max_length=60,
        description="All omitted fresh-fit parent recipe fields inherit. List a field here for an intentional removal/reset; learned state never inherits.")


class ExperimentSpec(Contract):
    experiment_id: str
    parent_id: str | None = None
    inherited_fields: list[str] = Field(default_factory=list)
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
        identity = {
            "pipeline": pipeline, "task_hash": self.task_hash,
            "data": self.data_version, "validation": self.validation_hash,
            "libraries": self.library_versions, "status": self.status,
        }
        if self.pipeline.engine != "ensemble":
            identity["budget_seconds"] = self.budget_seconds
            identity["budget_mode"] = self.budget_mode
        return digest(identity)

    @property
    def fixed_recipe_hash(self) -> str | None:
        # Adaptive searches and early stopping can legitimately depend on time.
        if self.pipeline.engine != "direct" or self.pipeline.early_stopping_rounds or self.status == "pilot":
            return None
        pipeline = self.pipeline.research_recipe()
        pipeline.pop("evaluation_interval", None)
        return digest({"pipeline": pipeline, "task_hash": self.task_hash, "data": self.data_version,
            "validation": self.validation_hash, "libraries": self.library_versions})


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
    raw_probabilities: dict[str, Any] = Field(default_factory=dict)
    calibrated_probabilities: dict[str, Any] = Field(default_factory=dict)
    decision_values: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


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
    fit_diagnostics: dict[str, Any] | None = None
    cost_curve: dict[str, Any] | None = None
    learning_curve: dict[str, Any] | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


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
    proposed_strategy: Literal["random", "group", "temporal", "temporal_group"] | None = None


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


class RecipeChange(Contract):
    path: str = Field(min_length=1, max_length=160)
    operation: Literal["set", "remove"] = "set"
    value: Any = None


class RecipeApplication(Contract):
    source: Literal["experiment", "knowledge"] = "experiment"
    base_id: str = Field(min_length=1)
    base_hash: str = Field(min_length=64, max_length=64)
    changes: list[RecipeChange] = Field(default_factory=list, max_length=30)


class TrainArgs(Contract):
    _inherited_fields: list[str] = PrivateAttr(default_factory=list)
    pipeline: PipelineSpec = Field(default_factory=PipelineSpec)
    hypothesis: Hypothesis
    parent_id: str | None = None
    search_action: Literal["draft", "improve", "debug"] = "draft"
    budget_seconds: float = Field(default=120, gt=0)
    pilot: bool = False
    method_refs: list[str] = Field(default_factory=list)
    recipe_ref: str | None = Field(default=None,
        description="Reference/check only; does not choose or apply a pipeline. Encode recipe settings in pipeline.")
    reference_ids: list[str] = Field(default_factory=list)
    search_space_ref: str | None = None
    apply_recipe: RecipeApplication | None = None


class BatchArgs(Contract):
    candidates: list[dict[str, Any]] = Field(min_length=2, max_length=4,
        description="2-4 independent TrainArgs objects: pipeline, hypothesis, optional parent_id/search_action/budget_seconds. Custom code and ensembles use their separate tools.")
    budget_seconds: float = Field(default=240, gt=0)


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
    """Uniform accepts compatible saved predictions; weighted_oof/stacked_oof require built-in members, excluding custom and ensemble pipelines."""

    strategy: Literal["uniform", "weighted_oof", "stacked_oof"] = "uniform"
    members: list[str] = Field(min_length=2, max_length=5)
    hypothesis: Hypothesis
    budget_seconds: float = Field(default=120, gt=0)
    final_decision: Literal["auto", "argmax", "regularized"] = "auto"


class FinishArgs(Contract):
    reason: str = Field(min_length=1)
    result_ids: list[str] = Field(default_factory=list)
    imputation_skip_reason: str | None = None
    skipped_directions: dict[str, str] = Field(default_factory=dict)
    component_audit_hash: str | None = Field(default=None, min_length=64, max_length=64,
        description="Confirm removals with the warning hash from a prior finish, or retrain.")


LEGACY_TOOL_SCHEMAS = {
    "inspect_dataset": InspectArgs, "get_reference": ReferenceArgs,
    "train_model": TrainArgs, "run_automl": TrainArgs, "tune_model": TrainArgs, "run_experiments": BatchArgs,
    "analyze_experiments": AnalyzeArgs, "build_ensemble": EnsembleArgs,
    "run_custom_experiment": CustomArgs, "finish": FinishArgs,
}


class DatasetReportArgs(Contract):
    """For transfer/domain use all confirmed features: columns must be empty, predicate/group_by absent. Domain never receives task labels."""

    scope: Literal["fit", "validation_features", "test_features"] = "fit"
    report: Literal["profile", "dictionary", "missing", "categories", "correlations", "slice", "transfer", "domain"] = "profile"
    columns: list[str] = Field(default_factory=list, max_length=30)
    group_by: str | None = None
    predicate: dict[str, Any] | None = None
    include_target: bool = False
    knowledge_gap: str | None = None
    cursor: str | None = None
    max_chars: int = Field(default=6000, ge=2000, le=12000)
    budget_seconds: float = Field(default=60, ge=5, le=300)
    correlation_method: Literal["legacy", "supported"] = "legacy"


class DiagnosticReportArgs(Contract):
    kind: Literal["distributions", "confusion_matrix", "residuals"]
    columns: list[str] = Field(default_factory=list, max_length=3)
    scope: Literal["fit", "validation_features", "test_features"] = "fit"
    include_target: bool = False
    experiment_id: str | None = Field(default=None,
        description="Required for residuals/confusion_matrix: a completed direct/Optuna experiment ID, scope=fit, columns=[], include_target=false. Distributions require no ID.")
    budget_seconds: float = Field(default=60, ge=5, le=300)
    knowledge_gap: str | None = None

    @model_validator(mode="after")
    def diagnostic_arguments(self):
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Diagnostic columns must be distinct")
        if self.kind != "distributions" and not self.experiment_id:
            raise ValueError("Model diagnostics require experiment_id")
        if self.kind != "distributions" and (self.scope != "fit" or self.include_target or self.columns):
            raise ValueError("Model diagnostics require scope=fit, include_target=false, columns=[], and experiment_id")
        if self.kind == "distributions" and self.experiment_id is not None:
            raise ValueError("Distributions do not use experiment_id")
        return self


class ReviseValidationArgs(Contract):
    proposed_strategy: Literal["random", "group", "temporal", "temporal_group"]
    rationale: str = Field(min_length=1)
    evaluation_mode: Literal["holdout", "exact_group", "low_overlap", "rolling"] = "holdout"


class KnowledgeFilters(Contract):
    library: str | None = None
    version: str | None = None


class KnowledgeListArgs(Contract):
    collection: Literal["documents", "methods", "recipes", "search_spaces", "actions"] = "documents"
    filters: KnowledgeFilters = Field(default_factory=KnowledgeFilters)
    cursor: str | None = None
    limit: int = Field(default=5, ge=1, le=10)
    max_chars: int = Field(default=6000, ge=2000, le=12000)
    knowledge_gap: str | None = None


class KnowledgeSearchArgs(KnowledgeListArgs):
    query: str = Field(min_length=1, max_length=2000)
    strategy: Literal["auto", "exact_symbol", "fts", "lexical"] = "auto"


class KnowledgeReadArgs(Contract):
    collection: Literal["documents", "methods", "recipes", "search_spaces", "actions"]
    id: str
    corpus_version: str
    cursor: str | None = None
    max_chars: int = Field(default=6000, ge=2000, le=12000)
    knowledge_gap: str | None = None
    view: Literal["document", "parameters"] = "document"
    parameter: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def parameter_view(self):
        if self.parameter is not None and self.view != "parameters":
            raise ValueError("parameter requires view=parameters")
        return self


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
    source_id: str = Field(min_length=1, max_length=96,
        description="Exact source_id from a returned source/continuation. A record ID belongs in ids, not here; copy the returned read arguments unchanged.")
    version: str = Field(min_length=1, max_length=128,
        description="Exact nonempty source version from the returned source/continuation; never guess or send an empty string.")
    scope: Literal["contract", "policy", "fit", "validation_features", "test_features", "registered_feedback", "reference", "code", "environment", "controller"]
    section: str = Field(default="", max_length=128)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=768, ge=4, le=768)

    @field_validator('source_id', 'version', 'section')
    @classmethod
    def source_byte_bounds(cls, value, info):
        maximum = 96 if info.field_name == 'source_id' else 128
        if len(value.encode('utf-8')) > maximum:
            raise ValueError(info.field_name + ' exceeds its UTF-8 byte limit')
        return value


class RunReadArgs(Contract):
    view: Literal["full", "pipeline"] = Field(default="full",
        description="For a parent/leader recipe use ids=[experiment_id], view=pipeline, include_code=false. Copy returned recipe_hash into preview/apply base_hash; diagnostic source_recipe_hash is different provenance.")
    ids: list[str] = Field(default_factory=list, max_length=10,
        description="Exact record IDs from a returned read.arguments.ids route, including guides, options and reviews as well as experiments. Copy the route unchanged.")
    source: RunSourceRef | None = Field(default=None,
        description="Alternative to ids, only for an exact source descriptor returned by a byte continuation. Never turn a record ID into source_id or invent a version.")
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
        if self.view == "pipeline" and (self.source or len(self.ids) != 1 or self.include_code):
            raise ValueError("Pipeline view requires exactly one experiment ID and include_code=false")
        if self.source and self.cursor:
            raise ValueError("Versioned source reads use their returned byte offset")
        return self


TOOL_API_VERSION = 2
TOOL_SCHEMAS = {
    "preview_recipe": RecipeApplication,
    "get_dataset_report": DatasetReportArgs, "revise_validation": ReviseValidationArgs,
    "get_diagnostic_report": DiagnosticReportArgs,
    "search_knowledge": KnowledgeSearchArgs, "list_knowledge": KnowledgeListArgs, "read_knowledge": KnowledgeReadArgs,
    "search_run_records": RunSearchArgs, "list_run_records": RunListArgs, "read_run_records": RunReadArgs,
    "train_model": TrainArgs, "run_automl": TrainArgs, "tune_model": TrainArgs, "run_experiments": BatchArgs,
    "run_custom_experiment": CustomArgs, "build_ensemble": EnsembleArgs, "finish": FinishArgs,
}

TOOL_DESCRIPTIONS = {
    "preview_recipe": "Copy recipe_hash from the complete base recipe read into base_hash (never diagnostic source_recipe_hash). Apply typed set/remove changes; return the normalized recipe, hash and diff without training. To execute, copy the same application into TrainArgs.apply_recipe.",
    "get_dataset_report": "Compute one permitted data report; target analysis and arbitrary slices use fit only. For transfer/domain, columns must be empty and predicate/group_by absent; all confirmed features are used. Domain never receives task labels.",
    "get_diagnostic_report": "Numeric evidence and optional PNG. For residuals/confusion_matrix supply experiment_id of a completed direct/Optuna model, scope=fit, columns=[], include_target=false. A bounded fit-inner copy includes training/rendering cost. Distributions use up to three columns without experiment_id.",
    "revise_validation": "State-changing action: may re-evaluate candidates and apply a new active validation protocol in this call, without further confirmation. Metric, roles and closed control cannot change.",
    "search_knowledge": "Search one collection. Ranked candidates are not confirmed answers; read evidence next. Exact symbol, FTS5 BM25 or lexical matching.",
    "list_knowledge": "Browse a filtered catalog, without claiming a search match.",
    "read_knowledge": "Read a specific versioned knowledge source, including conditions, warnings and continuation.",
    "search_run_records": "Find permitted past results, errors, notes or observations by filters and literal normalized words.",
    "list_run_records": "Browse the permitted run index by type and filters.",
    "read_run_records": "For one complete recipe use ids=[experiment_id], view=pipeline, include_code=false. Otherwise read up to ten saved records, one custom code object, or the exact returned versioned source route.",
    "train_model": "Train and evaluate one direct pipeline.",
    "run_automl": "Run bounded FLAML or AutoGluon search using fit only.",
    "tune_model": "Run bounded sequential Optuna search using fit only.",
    "run_custom_experiment": "Execute a Python pipeline, transformer or fit-only analysis under controller limits.",
    "run_experiments": "Run 2-4 independent hypotheses sequentially under one shared time/CPU/memory budget; return compact results and durable child operation IDs.",
    "build_ensemble": "Combine compatible saved predictions. Uniform supports custom members; weighted_oof/stacked_oof require built-in members, excluding custom and ensemble pipelines, with common fold/row IDs.",
    "finish": "Irreversibly freeze research; controller selects finalists and performs control/refit.",
}


class ToolCommand(Contract):
    operation_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,96}$")
    # The three legacy names are private executor compatibility only. Neither
    # public transport advertises or accepts them.
    tool: Literal["preview_recipe", "get_dataset_report", "get_diagnostic_report", "revise_validation", "search_knowledge", "list_knowledge", "read_knowledge", "search_run_records", "list_run_records", "read_run_records", "train_model", "run_automl", "tune_model", "run_experiments", "build_ensemble", "run_custom_experiment", "finish", "inspect_dataset", "get_reference", "analyze_experiments"] = Field(json_schema_extra={"enum": list(TOOL_SCHEMAS)})
    arguments: dict[str, Any]


def request_arguments(args):
    """Keep legacy operation identities; preserve child omissions where meaningful."""
    result = args.model_dump(mode="json", exclude_unset=isinstance(args, TrainArgs) and bool(args.parent_id))
    if isinstance(args, KnowledgeReadArgs) and args.view == "document":
        result.pop("view", None)
        result.pop("parameter", None)
    if isinstance(args, DatasetReportArgs) and args.correlation_method == "legacy":
        result.pop("correlation_method", None)
    return result


def public_tool_schema(model):
    """Expose actionable settings; frozen learned state is controller-owned."""
    schema = model.model_json_schema()
    pipeline = schema.get("$defs", {}).get("PipelineSpec", {})
    hidden = LEARNED_PIPELINE_FIELDS | {"ensemble_strategy", "members", "search_owner"}
    for name in hidden:
        pipeline.get("properties", {}).pop(name, None)
    return schema
