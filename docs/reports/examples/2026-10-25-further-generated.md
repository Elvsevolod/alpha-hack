# Source-informed reference study

Offline experiments; development research is separate from contest runtime.
All audit splits were inspected previously; these are reused audits, not fresh blind tests.

## regression

| Candidate | Best selection score | Fold scores | Tau | Offsets |
|---|---:|---|---:|---|
| tweedie15 | 46.366302 | [54.10256711075954, 38.63003593944148] | 0 | False |
| gamma | 46.554732 | [53.80382315844834, 39.30564110420912] | 0 | False |
| poisson | 46.722389 | [54.651594088108624, 38.79318319226245] | 0 | False |
| current_cb | 49.305086 | [55.6169208694683, 42.9932505869862] | 0 | False |
| current_lgb | 50.199382 | [58.979968194501495, 41.41879498905743] | 0 | False |

Frozen choice: {'names': ['tweedie15', 'gamma'], 'weights': [0.75, 0.25], 'tau': 0, 'offsets': False, 'fold_scores': [53.88905878257271, 38.66261881778173], 'score': 46.27583880017722, 'learned_offsets': [], 'sealed_before_reused_audit': True, 'at_unix': 1792900979.279098}.
Reused audit: 39.748070; previous recipe: 39.036147.
Iterations: {'tweedie15': 905, 'gamma': 1119}.
Submission: 22449 rows; SHA-256 `4d7496b3f853343ccf5a635d4f99717cc5f9d2db2ad073f3dfdd6900103736ec`.
External-test quality remains unknown. This does not evaluate autonomous Qwen decisions.

Completed paired recipes: 5. All attempted fits and optional auxiliary fits are recorded in job artifacts.
## classification

| Candidate | Best selection score | Fold scores | Tau | Offsets |
|---|---:|---|---:|---|
| current_weighted | 0.633445 | [0.6408419298543888, 0.6260482353921162] | 0 | False |
| hierarchy_balanced | 0.631497 | [0.6352923988451984, 0.6277006440695156] | 0 | True |
| current_plain | 0.631107 | [0.638098802576956, 0.6241152218374995] | 0.5 | False |
| hierarchy | 0.626569 | [0.6212670087839461, 0.6318705842899875] | 0.5 | True |

Frozen choice: {'names': ['current_weighted', 'hierarchy_balanced'], 'weights': [0.5, 0.5], 'tau': 0.5, 'offsets': True, 'fold_scores': [0.6476582933693851, 0.6395338752111228], 'score': 0.643596084290254, 'learned_offsets': [[-0.635036496350365, -0.647887323943662, 0.9215070643642073, 0.9705535924617197, 0.47517378351539225, 0.488339552238806, 0.4911378943637008, 0.0, -0.5, 0.0], [-0.33766233766233766, -0.6527777777777778, 0.9347258485639687, 0.9726626571897211, 0.47384937238493724, 0.9728997289972899, 0.49130737134909597, 0.0, -0.576271186440678, 0.0]], 'sealed_before_reused_audit': True, 'at_unix': 1792901148.035606}.
Reused audit: 0.656526; previous recipe: 0.647494.
Iterations: {'current_weighted': 589, 'hierarchy_balanced': 757}.
Submission: 41166 rows; SHA-256 `31a52b15c3490d53506b1804b8d1d923b894323726435cef9c659cb5a5b9fda2`.
External-test quality remains unknown. This does not evaluate autonomous Qwen decisions.

Completed paired recipes: 4. All attempted fits and optional auxiliary fits are recorded in job artifacts.


# Source-informed reference study

Offline experiments; development research is separate from contest runtime.
All audit splits were inspected previously; these are reused audits, not fresh blind tests.

## regression

| Candidate | Best selection score | Fold scores | Tau | Offsets |
|---|---:|---|---:|---|
| tweedie15 | 46.366302 | [54.10256711075954, 38.63003593944148] | 0 | False |
| gamma | 46.554732 | [53.80382315844834, 39.30564110420912] | 0 | False |
| poisson | 46.722389 | [54.651594088108624, 38.79318319226245] | 0 | False |
| tabm_mini | 47.642234 | [53.393153910298466, 41.891314128793645] | 0 | False |
| current_cb | 49.305086 | [55.6169208694683, 42.9932505869862] | 0 | False |
| current_lgb | 50.199382 | [58.979968194501495, 41.41879498905743] | 0 | False |

Frozen choice: {'names': ['tweedie15', 'gamma'], 'weights': [0.75, 0.25], 'tau': 0, 'offsets': False, 'fold_scores': [53.88905878257271, 38.66261881778173], 'score': 46.27583880017722, 'learned_offsets': [], 'sealed_before_reused_audit': True, 'at_unix': 1792901412.192551}.
Reused audit: 39.748070; previous recipe: 39.036147.
Iterations: {'tweedie15': 905, 'gamma': 1119}.
Submission: 22449 rows; SHA-256 `4d7496b3f853343ccf5a635d4f99717cc5f9d2db2ad073f3dfdd6900103736ec`.
External-test quality remains unknown. This does not evaluate autonomous Qwen decisions.

Completed paired recipes: 6. All attempted fits and optional auxiliary fits are recorded in job artifacts.


# Source-informed reference study

Offline experiments; development research is separate from contest runtime.
All audit splits were inspected previously; these are reused audits, not fresh blind tests.

## regression

| Candidate | Best selection score | Fold scores | Tau | Offsets |
|---|---:|---|---:|---|
| tweedie15 | 46.366302 | [54.10256711075954, 38.63003593944148] | 0 | False |
| gamma | 46.554732 | [53.80382315844834, 39.30564110420912] | 0 | False |
| poisson | 46.722389 | [54.651594088108624, 38.79318319226245] | 0 | False |
| tabm_mini | 47.642234 | [53.393153910298466, 41.891314128793645] | 0 | False |
| current_cb | 49.305086 | [55.6169208694683, 42.9932505869862] | 0 | False |
| current_lgb | 50.199382 | [58.979968194501495, 41.41879498905743] | 0 | False |

Frozen choice: {'names': ['tweedie15', 'tabm_mini'], 'weights': [0.75, 0.25], 'tau': 0, 'offsets': False, 'fold_scores': [53.63565753994459, 38.635699946819166], 'score': 46.135678743381874, 'learned_offsets': [], 'sealed_before_reused_audit': True, 'at_unix': 1792901578.7377112}.
Reused audit: 40.359258; previous recipe: 39.036147.
Iterations: {'tweedie15': 905, 'tabm_mini': 37}.
Submission: 22449 rows; SHA-256 `3abd4be793cb565480df3a401b9f933507e0d08504d15c069a22484ca0aeb201`.
External-test quality remains unknown. This does not evaluate autonomous Qwen decisions.

Completed paired recipes: 6. All attempted fits and optional auxiliary fits are recorded in job artifacts.
