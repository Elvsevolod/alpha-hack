# Autonomous EDA development mode

**Current interface:** `--agent` uses organizer Qwen and `--stop-units`. The DeepSeek section below describes historical development experiments, not the active competition or acceptance path. Adaptive evaluation v3 at the end supersedes v2 methodology.

`python -m automl_agent.eda task.md --agent --output outputs/new-run --time-limit 600`

This is a separate development session. The default `solution.py` Qwen connection and competition finalization remain the default competition path. Offline `python -m automl_agent.eda` still requires an explicit contract and makes no API calls or fits.

Architecture: `EDAAgent` reuses `Controller._task`, API v2 command decoding, Pydantic schemas, journal operations, knowledge/history handlers, `RunTools`, `EDAService`, and `Executor`. It overrides direct training to use the existing `error_analysis`/`build_pipeline` path with one fit-only inner split. It does not construct a Publisher or run `Controller.prepare`/control/refit. After upstream c35e09b, API v2 has 15 actions, including get_diagnostic_report; nine remain available in this EDA mode. The new production diagnostic action is retained, but is not exposed here because auxiliary fits must share the EDA allowance. Disabled actions are rejected before dispatch, independent of advertised schemas.

A confirmed semantic contract precedes the base profile. The provider then selects one action based on the current profile and saved results. There is no mandatory analysis or training sequence. Every request includes the compact profile and recent results, and earlier operations remain readable through history tools. `finish.reason` must record conclusions, unknowns and a proposed next experiment. It requires no completed model. Finishing never publishes a CSV.

## Computation and recovery

Profile analyzer version 5 runs in the monitored worker (20 seconds maximum within the original research deadline). CSV parsing also uses the same executor/deadline. Metadata inventory and frame decoding remain controller responsibilities. Frame staging streams bounded row/cell chunks into an atomic temporary file, checking the original deadline and controller RSS between chunks. An interrupted write retains the previous artifact. Deadline checks after inventory/decoding prevent starting downstream work after exhaustion. They are not an OS-level hard timeout for each individual controller instruction.

Profile facts are immutable per data hash, contract/roles, row selection, scope and analyzer. Numeric categories receive frequencies/cardinality, never numeric moments or quantiles. Risk-flag definitions are retained. Filtered profiles also use the worker. Correlations retain Pearson and add Spearman, pair support/status; semantic categories are excluded and the deterministic 20,000-row/30-column/minimum-30-pair limits apply. Failed optional reports retain their unknown status in storage and delivery.

Dataset report schema version 3 adds `errors` under `get_dataset_report`. In this development mode it requires full `fit`, `include_target=true`, no filter/projection/grouping; it performs one fixed LightGBM inner experiment. Direct `train_model` can compare a permitted pipeline on the identical inner split. Every attempt reserves one of two fits before staging, even if it crashes or times out. Nested search, calibration, fit diagnostics, learned imputation, custom code, ensembles and validation revision are rejected. Each worker gets at most 60 seconds including staging, 2 CPU and 4096 MiB. A failed fit is not retried automatically. Diagnostic pagination reuses the saved result and does not allocate a fit.

Tool responses distinguish `analysis_status` from `delivery_status` (complete/partial pages); technical/budget/incompatibility failures propagate to the operation status. Old profile runtimes must resume with their frozen runtime or a new run ID. Autonomous sessions persist costs, history and fit reservations; recovery terminates a verified owned worker before any new loader starts, restores the saved split, and honors durable completion intents. Other incomplete operations become unknown failures rather than rerunning a fit. Decisions are durably saved before dispatch. Recovery reconnects committed operation receipts to history when a crash interrupted delivery; unavailable historical arguments are explicitly left unknown, never invented.

## DeepSeek and cost

Only the development client reads `DEEPSEEK_API_KEY`, optional `DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL` from the environment or `.env`. Workers receive an allowlisted environment without credentials. The historical development default was `deepseek-flash`, thinking disabled, max 2048 output tokens. Unpriced model overrides fail closed. No Qwen-specific generation parameters are sent. For wide-table interpretation, the model may explicitly encode `feature_columns` as `all_nonservice_inputs`; the verifier expands exact train headers minus the explicitly selected target/ID/service roles. This saves output space without inferring roles.

The durable locked ledger reserves UTF-8 request bytes plus framing allowance as a conservative input-token bound and maximum output tokens at frozen peak/cache-miss prices. On 2026-10-24 these are $0.30/M input and $1.20/M output for flash. [Official pricing](https://api-docs.deepseek.com/quick_start/pricing/). Settlement uses reported input/output usage at these conservative rates; it does not claim an account-balance measurement or a cached/off-peak discount. Missing usage, HTTP/transport failure, or an unexpectedly larger cost retains the reservation and blocks every subsequent series request until reviewed. There are no automatic provider retries. A process lock prevents concurrent ledger writers. Model generation count also includes semantic resolution and malformed responses.

One series: connection <=$0.05; four tasks <=$0.20 each, seven generations, 600 seconds, two fits per task. The ledger has an absolute spendable ceiling of $0.85; the $0.15 reserve cannot be used automatically. The connection probe uses one generation. `tools/run_eda_series.py` runs sequentially and refuses a second invocation against an already started series directory.

## Data and verification

Prepare with `python tools/prepare_eda_benchmark.py --output outputs/eda_autonomy_20261024`. Source originals, UCI pages, licenses and SHA-256 are preserved. Only `tasks/<dataset>/` is visible to dataset inventory; sealed labels/reference contracts and expected criteria are siblings outside it. Adult/Bank use stratified 80/20 seed 42, then fixed stratified 20,000-row train cap. Bike uses the last ceil(20% distinct days) as external test and excludes rental components; HAR keeps the official subject split. All tasks stay below 20,000 training rows without splitting days/groups. Raw archives never enter the competition package.

Offline acceptance covers semantic categories, nonlinear associations with weak Pearson, budget expiry, durable cost reservations, unknown HTTP outcome, allowed-tool enforcement, fit limits, result feedback, pagination cache, restart without new requests, no CSV, and both native/JSON command formats. Existing profile, memory, API and data-boundary suites remain necessary. A live series establishes only behavior on these four development tasks; the live organizer Qwen development results are recorded in `outputs/qwen_policy_comparison_20261024/REPORT.md`; improved final competition predictions need separate validation.

## Debug evidence and provider identity

A team key for `changellenge.up.railway.app` belongs in `OPENAI_API_KEY` with the organizer Qwen settings. It is not a direct DeepSeek credential. `DEEPSEEK_API_KEY` must be issued by DeepSeek for `api.deepseek.com`; using the team key there returns 401. The historical autonomous CLI provider was DeepSeek; configuring the standard organizer connection does not by itself establish a live autonomous Qwen run or authorize replacing USD accounting with unknown organizer units.

`tools/check_deepseek_connection.py` performs only read-only model/balance requests and records status codes without credentials. Organizer checks use the project's alpha-api-limits skill, with before/after snapshots.

`tools/evaluate_eda_reports.py` independently checks stored facts, pair support and correlations against permitted fit values; protocol checks use only structural features. `protocol_assessment.json` exposes support, chronological/group integrity and horizon mismatches without changing the split. Agent requests also carry a compact fit-target overview so an inactive historical heuristic cannot be mistaken for balanced data.

`tools/debug_eda_algorithm.py` is a controlled offline scenario using HTTP mock decisions with real bounded fits. Its role proposals come from the reviewer's contract. Its model usage/cost ledger is explicitly simulated and is isolated from real provider accounting. It cannot establish autonomous decision quality. Each task writes `events.jsonl`, exact redacted provider bodies, `trajectory.json` and worker artifacts. API keys and Authorization headers are not persisted.

The HAR control run exposed LightGBM's rejection of JSON punctuation in native feature names. The shared pipeline now uses collision-free positional model aliases only when needed, retaining the mapping and original source/feature roles. Regression coverage checks categories, prediction, serialization and unchanged input values.


## Decision-policy comparison

The tool/execution architecture is unchanged. Three provider-independent decision policies can be selected with `--agent --policy free|evidence|alternatives`:

| Policy | Mechanism | Expected benefit | Main limitation |
|---|---|---|---|
| `free` | Base profile, recent results, existing history tools | Least added context; comparison baseline | Earlier grounds can disappear from the recent-results window |
| `evidence` (new-run default) | Same free choice plus compact receipt/hypothesis ledger | Keeps questions, expected effects, acceptance conditions, hashes, statuses and exact references available | Receipt existence does not prove a claim; additional context costs tokens |
| `alternatives` | Compare two next questions inside the existing action rationale | Makes opportunity cost explicit without another call | The comparison is free text and may be superficial |

No policy requires missingness, correlations or training. All use the same schemas, nine permitted actions out of the current 15-action API, two-fit and seven-generation limits. Policy is persisted and cannot change during resume. Historical sessions with history and no saved policy retain `free`.

`decision_ledger.json` is derived from original history receipts. It preserves analysis success separately from page delivery, reference linkage separately from protocol compatibility, and marks claim support `unassessed`. Entries expose exact evidence references for reuse; knowledge/history references not found among previous outer receipts remain unresolved here, rather than being declared invalid. Existing executor-side evidence validation remains authoritative. The ledger prescribes no features, datasets, transformation or order of tools.

`summary.json` retains the existing execution state for compatibility and adds `execution_status`, `decision_policy` and `decision_assessment`. A successful `finish` means execution ended; scientific interpretation and predictive improvement remain unassessed. This also applies to a convincing-looking mock trajectory.

Class diagnostics use separate support for precision and recall. A class with at least 30 actual examples and no predictions exposes recall=0 and `no_predictions=true`; precision remains unavailable. Small-support metrics are omitted, with per-metric reasons. This avoids hiding a supported failure while preserving aggregate limits.

The counterfactual HTTP tests exercise all three policies on changed facts under arbitrary schemas. They establish feedback availability and enforcement, not superiority of one policy's live reasoning. A real comparison requires a separately bounded provider series. Existing source/limit/packaging skills are not converted into dataset recipes; runtime decision rules live in the EDA policy module.


## Organizer Qwen development runner after c35e09b

`python tools/qwen_eda_live.py outputs/eda_autonomy_20261024/tasks --output outputs/new-qwen-run`

This runner injects the organizer client into the same EDA controller and executors. It uses OPENAI_* settings, native tools/JSON decoding, Qwen generation parameters and medium reasoning with an 8192-token development output ceiling. These parameters differ from the competition client's automatic medium/xhigh and output ceilings; this test is not an exact competition-runtime quality comparison. No new executors or model-specific data recipes are added.

Account checks stay in the development tool, outside solution.py and production runtime. Request artifacts omit credentials. The gateway's reported response charges are tracked in gateway units and separately compared with account-counter changes, which can include other users/processes. A stop threshold is checked between sequential requests. Without a disclosed tariff or server-enforced allocation it cannot be asserted to be a hard cap on an in-flight generation. Unknown outcomes retain a pending entry and stop further calls. Snapshots are saved before and after the actual series.

Live tests exposed confusion between nested TaskSpec.evidence and top-level TaskResolutionProposal.evidence. The development interpretation channel now advertises citations only in the proposal, requires that field, and restricts source identifiers to the supplied exact keys. It does not fill citations, infer roles or change the verifier. Original failed trajectories are kept.

## Current stage and live review

The live organizer development runner is `tools/qwen_eda_live.py`; it injects the provider into the same EDAAgent and retains the shared fit limits. This is still reconnaissance, including short hypothesis checks, before full research/finalization. The chronological review and failures are recorded in `outputs/qwen_policy_comparison_20261024/REPORT.md`. It is not a controlled policy comparison or competition-score reproduction.

`read_knowledge(view="parameters", parameter="...")` reads parameter documentation from the existing pinned corpus. Signature-only entries explicitly have `answer_usable=false`; adapter restrictions are separate from library API capabilities. No runtime introspection, network retrieval or new action is introduced.

## Budget-aware reconnaissance revision — 24 October 2026

The seven-generation allowance is unchanged and includes contract resolution. The final available generation advertises only finish in both transports, and dispatch refuses further analysis in that generation. This reserves a model-written conclusion; it does not manufacture one after malformed replies, API failures or an exhausted deadline. Earlier turns retain all permitted actions and can finish without training.

Each turn carries action_window and experiment_context. The latter preserves measured pipelines, split hashes, support limitations and component differences even when older receipts leave recent_results. Score differences are computed only for matching metric and fit/evaluation rows; they never establish causality or transfer. The decision instruction asks for one scientifically meaningful change per hypothesis test, permits explicitly described algorithm comparisons, discourages repeated successful reads, and asks for falsifiable next experiments without invented numerical gains. Weak time/group support can justify stopping and proposing an improved future evaluation; this run's split is unchanged.

Offline and live revision evidence is stored under outputs/eda_completion_revision_20261024. These changes are general decision support, not dataset-specific prescriptions.

A live Bike regression exposed an inherited native-to-JSON fallback consuming another generation inside one decision. The development providers now recompute action_window at every actual HTTP request, refresh remaining generations, and restrict the last JSON/native request to finish. The dispatch guard checks the actual post-request call counter. A known malformed response is not treated as an executed experiment; unknown outcomes still stop without retry. Competition Qwen transport is unchanged.

The completed Adult/Bike reruns and the preserved failed Bike transport regression are reviewed in `outputs/eda_completion_revision_20261024/REPORT.md`. Completion improved; low-support causal/transfer claims still require review.

## General temporal evaluation v2

EDA freezes a new versioned evaluation contract before fitting. Non-forecast temporal outer partitions reserve the last 15% of observed periods each for validation/control, keeping complete periods and any declared groups intact. Competition defaults remain unchanged; the shared split functions accept an explicit window override. The EDA inner window targets max(20% of available fit periods, observed external-test periods), capped at half the fit history. Declared forecasting horizons are preserved. No dataset or column names select this policy. If the cap prevents covering the external horizon, the report states the mismatch. Period count is not elapsed calendar duration on irregular data.

Temporal regression errors include per-period aggregate losses inside the bounded worker. Matching candidates receive a deterministic paired moving-block percentile interval (1,000 resamples; block length ceil(sqrt(periods)); minimum 20 periods). For RMSE/MSE/MAE, support on this historical origin requires a 95% delta interval wholly beyond a 1% baseline-score improvement floor and a covered horizon. These are engineering defaults, with stationarity/block assumptions; they do not prove causal effects, generalization or reliable future transfer. Unsupported metrics and inadequate support stay inconclusive. Training attempts remain limited to two. Period aggregates remain in the local diagnosis artifact; compact tool results carry the comparison.

Aggregate bias does not identify high-target errors, skewness does not prove a target transform helps squared-error loss, and duplicate observations do not prove their cause or harmlessness. These limits are included in the model's evaluation contract. Old EDA runs must use their frozen runtime; they cannot silently resume under v2.

Every EDA run emits human-readable REPORT.md and machine-readable evaluation_report.json, separating measured facts/limitations from unverified model interpretation. Committed review evidence lives in docs/reports; datasets, credentials and complete network logs stay outside Git.

## Adaptive evaluation v3 — supersedes v2 defaults

The v2 fixed fractions, effect floor and mandatory block-bootstrap computation were a development hypothesis, not organizer requirements. They are retained only in historical reports. The active `EvaluationDesign` instead carries an agent's rationale, assumptions, optional temporal window/gap, `descriptive` or optional `moving_block`, explicitly selected block size/confidence/resamples, and optional practical effect floor. Before a temporal fit the window must be selected; the controller enforces chronology, history, groups, and declared horizons, not a universal sample fraction. The design freezes at the first fit attempt. No floor means no claim of practical improvement; descriptive differences have no confidence interval.

Outer competition split defaults are preserved. EDA context describes outer validation/control as closed, never as the support of an inner model score. Each receipt gives actual inner evaluation rows/periods and inner unseen-category rates. `revise_validation` in the full controller can now change temporal window/gap on the same strategy, retaining the closed control and durably re-evaluating candidates. It cannot reuse scores across protocols.

See [requirements traceability](requirements_traceability.md), [methods](universal_research_methods.md), and [the general approach](eda_research_approach.md). Development provider clients are excluded from competition source packaging; only the normal Qwen path is used in a competition run. Historical DeepSeek experiments are not organizer-compliant acceptance evidence. New paid checks use organizer Qwen.
