"""One declared offline intervention: v116 group identities in v120 execution."""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/archive_recipe_parity_20261026_r2'
OUT = ROOT / 'outputs/archive_identity_diagnostic_20261026'


def main():
    import numpy as np
    import pandas as pd
    OUT.mkdir(exist_ok=False)
    freeze = json.loads((BASE / 'freeze.json').read_text())
    (OUT / 'freeze.json').write_text(json.dumps(freeze, indent=2)+'\n')
    sys.path.insert(0, freeze['runtimes']['v120']['path'])
    import automl_agent.identity as identity
    source = Path(freeze['runtimes']['v116']['path']) / 'automl_agent/identity.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == freeze['runtimes']['v116']['sources']['automl_agent/identity.py']
    definition = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'feature_groups')
    namespace = dict(identity.__dict__)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), 'exec'), namespace)
    old = namespace['feature_groups']
    data = freeze['tasks']['classification']
    task = json.loads((ROOT/data['task']).read_text())
    plan = json.loads((ROOT/data['plan']).read_text())
    frame = pd.read_csv(ROOT/data['train'], usecols=task['feature_columns']).iloc[plan['fit']]
    before, after = old(frame), identity.feature_groups(frame)
    # Renaming an equivalence class is not a change of the duplicate relation.
    equivalent = bool(np.array_equal(pd.factorize(before, sort=False)[0], pd.factorize(after, sort=False)[0]))
    from sklearn.model_selection import GroupKFold
    folds = [list(GroupKFold(3).split(frame, groups=groups)) for groups in [before, after]]
    changed = [int(len(set(a[1]) ^ set(b[1]))) for a, b in zip(*folds)]
    declaration = dict(intervention='v120 exact runtime with only feature_groups function replaced by exact v116 function',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        expected_falsification='If only group-label ordering caused decision difference, complete labels and raw probabilities reproduce v116',
        full_fit_duplicate_relation_identical=equivalent, changed_oof_heldout_rows_per_fold=changed,
        fit_rows=len(frame), control_scored=False, validation_previously_viewed=True, api_calls=0,
        not_production_patch=True, reason_not_revert='v120 exact integer identity avoids float precision collisions; this is a diagnostic intervention')
    (OUT/'intervention.json').write_text(json.dumps(declaration,indent=2)+'\n')
    identity.feature_groups = old
    sys.path.insert(0, str(ROOT))
    from tools.archive_recipe_parity import worker
    worker(OUT, 'v120', 'classification_offsets')
    for name in ['values', 'probabilities']:
        a=np.load(BASE/'v116_classification_offsets'/f'{name}.npy')
        b=np.load(OUT/'v120_classification_offsets'/f'{name}.npy')
        declaration[name+'_matches_v116_exactly']=bool(np.array_equal(a,b))
    declaration['score']=json.loads((OUT/'v120_classification_offsets/result.json').read_text())['score']
    (OUT/'intervention.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(declaration),flush=True)


if __name__ == '__main__':
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        os.environ[name]='1'
    main()
