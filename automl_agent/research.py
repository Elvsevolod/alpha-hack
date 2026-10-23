"""Task-independent research guidance and controller-measured experiment reviews."""
from __future__ import annotations

from context_compressor import Record, Source

from .contracts import ExperimentResult, ExperimentSpec
from .util import atomic_json, canonical, digest


RESEARCH_POLICY = """Read the research_guide in current_work.related_ids once via read_run_records.
Hypotheses state observation, expected_effect, changed_component, acceptance and evidence_refs.
"""

RESEARCH_GUIDE = """
Visual diagnostics: get_diagnostic_report returns numeric evidence plus an optional
image. Compare quantiles, skew, missingness and histogram shape; a long tail suggests
a transform/loss hypothesis, never automatic deletion. For confusion_matrix use the
full fit-inner matrix, declared class order and per-class precision/recall/F1;
rare-class suppression cannot be reconstructed. Residuals are y_true-minus-y_pred
in the original scale: inspect bias and aggregated prediction intervals. Curvature
or changing spread is a hypothesis, not proof of a cause. Cite the numeric report,
consider an alternative explanation and compare one bounded experiment. Reports
are exploratory fit-only evidence; group/time boundaries and final reserves bind.
Research cycle: observed problem -> targeted source reading -> falsifiable hypothesis ->
bounded experiment -> comparison -> conditional lesson -> next question. Use the permitted
local corpus as your library; the runtime has no web access. Search by the observed failure,
confirmed metric and installed version. Read relevant conditions and limitations, then cite
the returned versioned evidence_refs. Search hits alone are not read evidence. If a needed
source is absent, record the gap and use a justified supported alternative; do not invent it.
Before training, state a comparator, expected metric direction, acceptance criterion and
one main change in the existing hypothesis fields. Copy exact settings into the pipeline;
the controller freezes this specification before execution. Include auxiliary fits and full
refit in the measured cost. Never tune a criterion, split, seed or stopping rule to a score.
After a completed experiment, read its research_review and compare on the SAME protocol.
An improvement is conditional evidence on this task, not a reusable universal winner.
Review acceptance yourself: a numeric delta does not prove the textual hypothesis or causality.
For a negative or inconclusive outcome, preserve the incumbent, identify what the experiment
ruled out, and reopen research_note with the next source question when useful and affordable.
A timeout/OOM/unknown outcome requires technical repair, not a negative ML conclusion.
Do not rerun unchanged failures, demand a positive result, or widen a grid without new evidence.
Keep negative results and costs. Finish for reasoned exhaustion or resource reserve, not merely
because one idea failed. Research reviews and source reading never authorize extra label access.

Transferable candidate ideas, always conditional and optional:
* Classification: distinguish probability quality from the required decision metric/output.
  Compare weighted/unweighted training and the default decision rule with a supported
  fit-only threshold or multiclass class_offsets rule when appropriate. Better log-loss
  does not imply better Macro F1. Binary thresholds do not generalize independently to
  mutually exclusive multiclass labels. Do not apply class offsets to a probability output.
* For any learned preprocessing, target aggregate, calibration, threshold, early stopping,
  or ensemble weight, fit it inside each training fold. To evaluate a tuned procedure,
  use untouched outer rows; cross-fitting a calibrator is insufficient if its base models
  have seen the outer labels. Use the controller's fit-only inner paths or a valid custom
  implementation. A small inner holdout is acceptable under budget; disclose uncertainty.
* For genuine temporal tasks, investigate causal history and residual correction against
  a simple available baseline. Require confirmed time/group semantics and strictly earlier
  training targets for history; metadata coincidences are not verified entity identities.
  Diagnose rows with/without usable history separately and test a feature-only fallback.
  Evaluate errors on the original target scale. No history feature is mandatory.
* Compare model families/representations with the same metric, split and learning scope;
  report fit size, cost and auxiliary fits. For an ablation keep other settings fixed.
  A changed training size, reused holdout, multiple changed components or repeated selection
  limits causal claims. Preserve frozen control; validation revisions are not score hunting.
No dataset names, feature keys, class labels, historical scores or winning hyperparameters
from development tasks define these rules. Revalidate every candidate on the current task.
"""


def experiment_review(spec, result, previous, task, policy):
    """Summarize permitted scores without promoting a local outcome into global knowledge."""
    candidates = [(s, r) for s, r in previous
        if r.status == "completed" and r.score is not None and r.scope != "pilot"
        and (s.task_hash, s.data_version, s.validation_hash, s.library_versions)
        == (spec.task_hash, spec.data_version, spec.validation_hash, spec.library_versions)
        and r.validation_hash == result.validation_hash == spec.validation_hash]
    parent = next(((s, r) for s, r in candidates if s.experiment_id == spec.parent_id), None)
    incumbent = (max if task.metric.maximize else min)(candidates, key=lambda pair: pair[1].score) if candidates else None
    comparator = parent or incumbent
    gain = None
    if result.status in {"outcome_unknown", "interrupted"}:
        outcome, next_step = "unresolved", "Recover the existing operation before interpreting or repeating it."
    elif result.status != "completed":
        outcome, next_step = "technical_failure", "Read the diagnostic; repair the evidenced technical cause within budget."
    elif result.scope == "pilot":
        outcome, next_step = "pilot_only", "Use this only for feasibility/cost; obtain a comparable full result if affordable."
    elif result.score is None or not comparator:
        outcome, next_step = "not_comparable", "Establish a scored comparator on the same protocol before a quality claim."
    else:
        reference = comparator[1].score
        gain = result.score - reference if task.metric.maximize else reference - result.score
        improved = policy.improvement(result.score, reference)
        outcome = "improved_locally" if improved else "no_material_gain"
        next_step = ("Check acceptance, costs and failure slices; retain this as conditional evidence only." if improved else
            "Preserve the incumbent; record the negative result and read sources for a distinct error-based hypothesis if affordable.")
        if improved and not policy.improvement(result.score, incumbent[1].score):
            outcome, next_step = "improved_parent_only", "Parent improved but the incumbent remains better; preserve it and review the hypothesis."
    return {"experiment_id": spec.experiment_id, "outcome": outcome,
        "hypothesis": {"source": "agent_assumption", **spec.hypothesis.model_dump(mode="json")},
        "measurement": {"status": result.status, "score": result.score, "scope": result.scope,
            "metric": task.metric.model_dump(mode="json"), "maximize": task.metric.maximize,
            "seconds": result.seconds, "peak_memory_mb": result.peak_memory_mb,
            "fit_rows": result.training_summary.get("rows") if type(result.training_summary.get("rows")) is int else None,
            "auxiliary_fits": result.training_summary.get("auxiliary_fits") if type(result.training_summary.get("auxiliary_fits")) is int else None},
        "comparison": {"experiment_id": comparator[0].experiment_id if comparator else None,
            "basis": "parent" if parent else "best_prior_comparable" if comparator else "none",
            "score": comparator[1].score if comparator else None, "gain": gain,
            "incumbent_id": incumbent[0].experiment_id if incumbent else None,
            "incumbent_score": incumbent[1].score if incumbent else None,
            "acceptance": "requires_hypothesis_review", "causality_established": False},
        "reproduction": {"task_hash": spec.task_hash, "data_version": spec.data_version,
            "validation_hash": spec.validation_hash, "pipeline_hash": spec.pipeline.hash,
            "pipeline": spec.pipeline.model_dump(mode="json"), "library_versions": spec.library_versions,
            "budget_seconds": spec.budget_seconds, "method_refs": spec.method_refs,
            "reference_ids": spec.reference_ids},
        "source_status": "citations_are_not_proof_of_reading_or_application",
        "next_step": next_step, "transfer_status": "task_local_requires_revalidation"}


def index_reviews(context_memory):
    """Idempotent projection of durable experiment records; never read worker/data files."""
    c, store = context_memory.c, context_memory.store
    guide_id = "research_guide:" + digest(RESEARCH_GUIDE)[:16]
    if not store.record(guide_id):
        guide = Source(guide_id, digest(RESEARCH_GUIDE), RESEARCH_GUIDE, "reference", "external_data",
            origin="project_research_guidance")
        store.put_source(guide)
        store.publish_records([Record(guide_id, guide.source_id, guide.version, guide.scope,
            kind="reference", component="research_guide", engine="controller", status="available",
            checked_change="Read once: source-led research cycle; conditional methods, honest validation and negative results.")])
    context_memory.research_guide_id = guide_id
    previous, latest = [], None
    for row in c.journal.experiments():
        spec = ExperimentSpec.model_validate(row["spec"])
        if row["result"] is None:
            continue
        result = ExperimentResult.model_validate(row["result"])
        # Reviews contain an explicit whitelist; private feedback/predictions are
        # never included. The digest also changes if a recovered comparator changes.
        public = c.public_result(result)
        measured = result.model_copy(update={"score": public.get("score"), "seconds": public.get("seconds", 0),
            "training_summary": public.get("training_summary", {})})
        review = experiment_review(spec, measured, previous, c.task, c.plan.selection_policy)
        identifier = "research_review:" + digest(review)[:24]
        if not store.record(identifier):
            source = Source(identifier, digest(review), canonical(review), "registered_feedback",
                "external_data", origin="research_review")
            store.put_source(source)
            atomic_json(c.workdir / "research_reviews" / (identifier.split(":")[1] + ".json"), review)
            store.publish_records([Record(identifier, source.source_id, source.version, source.scope,
                kind="observation", component="research_review", engine="controller", status=result.status,
                protocol=spec.validation_hash, environment=digest(spec.library_versions),
                parent_id=spec.experiment_id, metric=c.task.metric.name, maximize=c.task.metric.maximize,
                evaluation_scope=result.scope, checked_change=review["outcome"] + ": " + review["next_step"])])
        previous.append((spec, measured))
        latest = identifier
    return latest
