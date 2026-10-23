---
name: qwen-ml-best-practices
description: Guide evidence-led ML experiments for the Alpha Qwen agent when diagnosing baseline errors, choosing a representation or decision rule, or reviewing an experiment. Use within the confirmed task and available tools.
---

# Qwen ML best practices

Use human ML cases to choose a useful next experiment. They suggest conditional ideas; their published scores are not evidence about this task. Skip questions already resolved. This guide does not change the controller's task, permissions, resources or tool contracts.

## Choose the next action

- Start from an observed problem or a specific unresolved question. Distinguish task meaning, validation, representation, decision metric and execution cost. Name plausible alternatives; use the cheapest informative check to distinguish them.
- Ground the question in available data reports or completed experiments. Use permitted fit analysis and fit-only inner holdouts for exploratory error analysis. External validation reports stay fixed; suppression binds. Workers keep individual errors and labels local.
- Match the check to intended generalization: later dates, unseen entities, rare categories or ordinary random observations. A convenient identifier does not establish entity or time semantics.
- Search the local corpus by the observed problem and current metric. Read the relevant conditions before using a source. Existing source IDs and search hits alone do not prove support or applicability. Record a knowledge gap when material is unavailable.

## Form and check an experiment

Use the existing `Hypothesis` fields: `observation`, `expected_effect`, `changed_component`, `acceptance`, `evidence_refs`. State a comparable parent or incumbent and what result would weaken the hypothesis. Separate measured facts from your interpretation.

Prefer one main change for a diagnostic comparison. Keep the protocol, metric and learning scope comparable; disclose other differences. Fit preprocessing, target aggregates and learned decision rules inside training folds. Temporal history must precede the predicted event. Check the concrete adapter before proposing unsupported operations.

Count preparation, auxiliary fits, inference and frozen refit in the cost. Use a pilot for feasibility and cost, not as comparable quality evidence. The existing critic reviews costly proposals; it does not need another automatic review call.

## Interpret the result

- Improvement: review the acceptance criterion, affected error groups and cost. Preserve the result as task-local evidence; a positive score delta does not establish a causal explanation.
- No material gain: preserve the incumbent, record what was tested and choose a different supported question when useful.
- Inconclusive: keep the uncertainty explicit; do not force acceptance or rejection.
- Timeout, OOM or unknown outcome: resolve the technical outcome before drawing an ML conclusion or repeating the operation.

Use the existing research note, experiment review and memory. Keep negative results and reasons for revisiting them. Reopening an idea needs changed evidence, conditions or implementation. Continue while an informative admissible action fits the budget; protect the controller's finalization reserve.

## Conditional patterns

Read only the relevant section of [diagnostic patterns](references/diagnostic-patterns.md): validation and time; lost feature information; rare entities and missingness; decision metrics; cost and ablation. A pattern supplies candidate checks, not a required model or a fixed experiment order.

These links describe source files for the host. Qwen uses `search_knowledge` and `read_knowledge` after the host indexes the material; it must not invent filesystem tools or reference IDs. If a section is not available, record the gap and use available evidence.
