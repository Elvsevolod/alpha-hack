# Source-informed reference study

Offline experiments; development research is separate from contest runtime.
All audit splits were inspected previously; these are reused audits, not fresh blind tests.

## regression

| Candidate | Best selection score | Fold scores | Tau | Offsets |
|---|---:|---|---:|---|
| regularized_lgb | 47.818356 | [54.93762802048622, 40.69908391164846] | 0 | False |
| engineered_cb | 48.226741 | [54.243033324265156, 42.210449441105524] | 0 | False |
| imputed_cb | 48.524719 | [54.45996497675471, 42.58947331886216] | 0 | False |
| anchor_cb | 49.098930 | [56.52569980857672, 41.67216106577835] | 0 | False |
| extra_trees | 49.101738 | [55.52762390217408, 42.675852422428754] | 0 | False |
| xgb_building | 49.138585 | [56.43985851728954, 41.83731218207129] | 0 | False |
| xgb6 | 50.390974 | [57.80110297743688, 42.980845999119715] | 0 | False |
| mlp | 51.635447 | [58.1024003532651, 45.16849282551723] | 0 | False |
| anchor_lgb | 52.218275 | [61.082692932546024, 43.35385703231087] | 0 | False |

Frozen choice: {'names': ['regularized_lgb', 'engineered_cb'], 'weights': [0.5, 0.5], 'tau': 0, 'offsets': False, 'fold_scores': [54.256330999609354, 40.08116341908786], 'score': 47.16874720934861, 'learned_offsets': [], 'sealed_before_reused_audit': True, 'at_unix': 1792863682.266471}.
Reused audit: 39.036147; previous recipe: 46.315304.
Iterations: {'regularized_lgb': 1035, 'engineered_cb': 979}.
Submission: 22449 rows; SHA-256 `33443b9d9b93d709a69f472b393470b1b71b93fdb0a4082924f2150d732564fd`.
External-test quality remains unknown. This does not evaluate autonomous Qwen decisions.

Completed paired recipes: 9. All attempted fits and optional auxiliary fits are recorded in job artifacts.
## classification

| Candidate | Best selection score | Fold scores | Tau | Offsets |
|---|---:|---|---:|---|
| xgb5_weighted | 0.634833 | [0.6430576928677968, 0.6266088905306739] | 0 | False |
| anchor63 | 0.629633 | [0.6325860189521355, 0.6266799312164819] | 0.5 | True |
| anchor31 | 0.629284 | [0.6297831536934432, 0.6287845717180975] | 0.5 | True |
| macro63 | 0.626885 | [0.6290205575169148, 0.6247501739904456] | 0.5 | False |
| xgb5 | 0.622927 | [0.6296078423979112, 0.6162454658935059] | 0.5 | False |
| forest | 0.613380 | [0.630581784244559, 0.5961790887288325] | 0.5 | True |
| macro31 | 0.612865 | [0.5991177415856862, 0.6266125695715641] | 0.5 | False |
| mlp | 0.566699 | [0.5612832745113969, 0.572114163582712] | 0 | True |

Frozen choice: {'names': ['xgb5_weighted'], 'weights': [1.0], 'tau': 0, 'offsets': False, 'fold_scores': [0.6430576928677968, 0.6266088905306739], 'score': 0.6348332916992354, 'learned_offsets': [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], 'sealed_before_reused_audit': True, 'at_unix': 1792863979.876153}.
Reused audit: 0.643890; previous recipe: 0.644310.
Iterations: {'xgb5_weighted': 589}.
Submission: 41166 rows; SHA-256 `1f70a7c7d0764057b33fca1d28553579dc9f3a97cdcd20947f4c2d53e21ea94f`.
External-test quality remains unknown. This does not evaluate autonomous Qwen decisions.

Completed paired recipes: 8. All attempted fits and optional auxiliary fits are recorded in job artifacts.


# Source-informed reference study

Offline experiments; development research is separate from contest runtime.
All audit splits were inspected previously; these are reused audits, not fresh blind tests.

## classification

| Candidate | Best selection score | Fold scores | Tau | Offsets |
|---|---:|---|---:|---|
| xgb5_weighted | 0.634833 | [0.6430576928677968, 0.6266088905306739] | 0 | False |
| xgb_engineered | 0.633461 | [0.6305665596683356, 0.6363552607415894] | 0.5 | True |
| xgb_raw_weighted | 0.633280 | [0.6419581169264532, 0.6246028033965494] | 0 | False |
| anchor63 | 0.629633 | [0.6325860189521355, 0.6266799312164819] | 0.5 | True |
| anchor31 | 0.629284 | [0.6297831536934432, 0.6287845717180975] | 0.5 | True |
| xgb5 | 0.622927 | [0.6296078423979112, 0.6162454658935059] | 0.5 | False |

Frozen choice: {'names': ['xgb5_weighted', 'xgb_engineered'], 'weights': [0.5, 0.5], 'tau': 0.5, 'offsets': True, 'fold_scores': [0.6518654800948425, 0.6229150242641635], 'score': 0.637390252179503, 'learned_offsets': [[0.0, -0.647887323943662, 0.46075353218210363, 0.48527679623085984, 0.47517378351539225, 0.976679104477612, 0.4911378943637008, -0.4562937062937063, -0.25, 0.0], [-0.6753246753246753, -0.6527777777777778, 0.0, 0.0, 0.0, 0.0, 0.0, -0.45495495495495497, 0.288135593220339, 0.0]], 'sealed_before_reused_audit': True, 'at_unix': 1792864281.317835}.
Reused audit: 0.647494; previous recipe: 0.644310.
Iterations: {'xgb5_weighted': 589, 'xgb_engineered': 558}.
Submission: 41166 rows; SHA-256 `ea625c3e3f16cb9d6c6b710e90840ef7abcb362ed10ec45387c6d861425e1df3`.
External-test quality remains unknown. This does not evaluate autonomous Qwen decisions.

Completed paired recipes: 6. All attempted fits and optional auxiliary fits are recorded in job artifacts.
