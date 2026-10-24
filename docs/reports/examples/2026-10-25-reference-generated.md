# Participant reference experiments

Offline development reference; this is not a test of autonomous Qwen decisions.
Test labels are unavailable. All scores below are internal validation scores.

## regression

Split: fit years 1-3; tune year 4; control year 5; external year 6.
Counts: {'fit': 22054, 'tune': 12946, 'control': 18308}.

| Candidate | Class correction exponent | Tuning score | Iterations |
|---|---:|---:|---:|
| cb6 | 0 | 44.368473 | 723 |
| cb8 | 0 | 43.425701 | 1199 |
| lgb31 | 0 | 43.353857 | 460 |
| lgb63 | 0 | 43.398431 | 596 |
| cb8_building | 0 | 41.672161 | 1199 |
| cb6_log | 0 | 46.610433 | 467 |

Selected: [{'name': 'cb8_building', 'iterations': 1199, 'alpha': 0}, {'name': 'lgb31', 'iterations': 460, 'alpha': 0}].
Control score: 46.315304; first baseline: 50.043123.
Output: 22449 rows; SHA-256 `535ad099cbb80b424b468831b30790c034c321a6103451d47e8a8400f25fc98f`.
Limitation: Only one untouched forward control origin; yearly transfer uncertainty remains.

## classification

Split: stratified identical-feature groups.
Counts: {'fit': 52903, 'tune': 17626, 'control': 17141}.

| Candidate | Class correction exponent | Tuning score | Iterations |
|---|---:|---:|---:|
| lgb31 | 0 | 0.569301 | 173 |
| lgb31 | 0.25 | 0.613314 | 173 |
| lgb31 | 0.5 | 0.623093 | 173 |
| lgb31_balanced | 0 | 0.607917 | 551 |
| lgb31_balanced | 0.25 | 0.599017 | 551 |
| lgb31_balanced | 0.5 | 0.582354 | 551 |
| lgb63 | 0 | 0.568039 | 112 |
| lgb63 | 0.25 | 0.611611 | 112 |
| lgb63 | 0.5 | 0.626916 | 112 |
| cb6 | 0 | 0.490656 | 200 |
| cb6 | 0.25 | 0.541611 | 200 |
| cb6 | 0.5 | 0.585867 | 200 |
| cb6_sqrt | 0 | 0.612943 | 200 |
| cb6_sqrt | 0.25 | 0.555837 | 200 |
| cb6_sqrt | 0.5 | 0.506246 | 200 |

Selected: [{'name': 'lgb63', 'iterations': 112, 'alpha': 0.5}, {'name': 'lgb31', 'iterations': 173, 'alpha': 0.5}].
Control score: 0.644310; first baseline: 0.586697.
Output: 41166 rows; SHA-256 `c22ad4daa2ae3ab4d8f6bf6d8de7bf5c6b0fe9c20e76d4ad878ce73c32a81931`.
Limitation: No external labels; internal group split does not reproduce official capture-domain shift.

## Verification boundaries

- Input and split hashes, random seeds, frozen recipes and fit logs are saved.
- Identical feature groups are disjoint in classification; temporal ordering is preserved in regression.
- Control labels are not used for early stopping or recipe selection in this initial study.
- CSV row count, identifier order, finite values and allowed labels are checked.
- No API requests, downloaded data, weights or packages are used by the experiment.
- This does not establish contest container readiness, autonomous decision quality, or external-test quality.
