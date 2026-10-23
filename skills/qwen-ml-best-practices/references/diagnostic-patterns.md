# Conditional diagnostic patterns

This is project guidance synthesized from published human case studies. Checks below are proposals for a new task, not reproduced human action traces. Source reports, project suggestions and current-task measurements remain distinct. Read one relevant section, then return to the actual evidence.

## Validation and time

**Signal:** plausible future or unseen-entity use; an unexplained gap between comparable evaluations; performance varying across permitted periods.

**Check:** establish time and entity meaning from the task and dictionary. Inspect permitted temporal or entity-level error aggregates. A small fit-only stress test can distinguish dependence on familiar entities from transferable signal. Propose an external protocol revision only through the controller and with semantic evidence.

**Candidate experiment:** remove or replace one suspected unstable feature block and compare under the same valid protocol. An isolated weak feature is not enough to justify removal when interactions may matter.

**Limits:** do not choose splits by their scores, infer identities from convenient coincidences or read future labels. A recovered time order needs independent justification. Batch contest features that use later intervals do not become causal forecasting features.

Sources: [Rossmann author interview](https://medium.com/kaggle-blog/rossmann-store-sales-winners-interview-1st-place-gert-jacobusse-a14b271659b); [Kaggle's adaptation of the IEEE-CIS authors' report](https://www.kaggle.com/solution-write-up-documentation); [Optiver author report](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970).

## Lost feature information

**Signal:** errors concentrate in a meaningful group; text processing loses quantities or units; event aggregation hides differences between histories.

**Check:** use worker-local fit analysis to test a concrete explanation. Report allowed aggregates with adequate support rather than individual examples. Establish that the proposed information exists at prediction time.

**Candidate experiment:** preserve one discarded piece of information with a transformer compatible with the current pipeline. For repeated events, consider richer time-aware aggregates; a model-derived event summary is a later option if honest cross-fitting and full refit fit the budget.

**Limits:** high feature importance does not establish improvement. Auxiliary predictions for training rows require appropriate out-of-fold generation; repeated entities and time constraints survive every fold. If related tables are unavailable, record that limitation.

Sources: [Home Credit author report](https://github.com/NoxMoon/home-credit-default-risk#feature-engineering); [Mercari authors' report, Feature preprocessing](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/writeups/pawe-and-konstantin-1st-place-solution).

## Rare entities, missingness and zeros

**Signal:** rare entities have little history; missingness differs between permitted populations; a filter predicts zero or omits a subgroup.

**Check:** distinguish missing observations, unavailable products and observed zero outcomes when the data supports that distinction. Quantify how much data a filter excludes and evaluate its group inside a suitable inner holdout. Suppressed or small groups remain uncertain.

**Candidate experiment:** compare an unchanged baseline with one supported fallback, broader-group historical summary, missingness indicator or removal of an unjustified filter. Choose the alternative using the observed mechanism.

**Limits:** zero does not identify its cause. Historical target aggregates must precede prediction; other target aggregates require proper cross-fitting. A published case does not supply an unreported smoothing recipe. Public feedback alone cannot validate an imputation mechanism.

Sources: [Instacart engineering report](https://tech.instacart.com/predicting-real-time-availability-of-200-million-grocery-items-in-us-canada-stores-61f43a16eafe); [Favorita author's post-mortem](https://medium.com/data-science/review-kaggle-corporaci%C3%B3n-favorita-grocery-sales-forecasting-part-i-9330b7350713).

## Decision metric and output

**Signal:** probability quality and the required label or set metric move differently; minority-class errors dominate an allowed aggregate report.

**Check:** confirm whether output is a value, mutually exclusive label, multilabel set, probability or ranking score. Verify the exact primary metric. Separate errors of probability estimation from errors introduced by the decision rule.

**Candidate experiment:** compare the supported default rule with a rule learned inside fit. For mutually exclusive multiclass labels, use an appropriate joint decision rule; binary thresholds cannot simply be applied independently. Set-level expected-F1 procedures require a genuine set-prediction task and an implementation supported by the host.

**Limits:** tuning and evaluating a decision rule on the same labels inflates evidence. Outer evaluation must also be unseen by all fitted dependencies. Threshold changes do not improve the ordering measured by AUC. Keep probability outputs probabilistic.

Source: [Instacart author's report, F1 maximization](https://github.com/KazukiOnodera/Instacart#f1-maximization). Inner/outer evaluation requirements here are project guidance, not a claim that the README documents that protocol.

## Cost and ablation

**Signal:** one experiment consumes most of the remaining budget; an ensemble has unmeasured extra members; a plausible explanation of improvement has not been tested.

**Check:** measure the whole candidate path and dependencies. Ask which uncertainty a cheaper comparison would resolve. Preserve fit size, protocol and settings when attributing a gain to a component.

**Candidate experiment:** add or remove one feature block or ensemble member, compare a simpler representation, or compare forecast strategies on a representative temporal subset. Confirm promising findings under the intended learning scope and on fresh evaluation data when available.

**Limits:** changing training size or several components weakens attribution. CPU-only does not mean short runtime. A recipe successful elsewhere can fail here; a failed configuration does not rule out its entire model family. Inconclusive or unaffordable tests remain open questions.

Sources: [Optiver author's ablations](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/302626); [M5 author's implementation notes](https://github.com/Mcompetitions/M5-methods/blob/master/Code%20of%20Winning%20Methods/A1/0.%20README.md); [Rohlik author's report](https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2/writeups/golden-1st-place-solution); [Mercari authors' report](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/writeups/pawe-and-konstantin-1st-place-solution).
