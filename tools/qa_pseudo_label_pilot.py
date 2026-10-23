"""One-iteration teacher/student experiment with explicit source and phase authority."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from automl_agent.util import digest
from automl_agent.validation import stratified_parts


PROTOCOL = {'dataset_seeds': [17, 42, 73], 'training_seed': 42,
    'thresholds': [.8, .95], 'pseudo_weight': .25, 'iterations': 1,
    'tuning': 'Only inner holdout of research fit; ties prefer higher confidence threshold',
    'phases': ['research_fit', 'control_preparation', 'full_train_refit'],
    'source': 'Separate explicit approved unlabeled CSV; train/test aliases and ID overlap rejected',
    'promotion': 'Report synthetic effects; no automatic activation or default-policy change'}


def frame_hash(frame):
    body = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(body + json.dumps(list(frame)).encode()).hexdigest()


@dataclass(frozen=True)
class UnlabeledSource:
    source_id: str
    features: pd.DataFrame
    features_hash: str
    labeled_ids: frozenset[str]
    test_ids: frozenset[str]


def approved_source(root, manifest):
    if manifest.get('allow_pseudo_labels') is not True:
        raise ValueError('Unlabeled source is not explicitly approved')
    root = root.resolve()
    paths = {}
    for role in ('train', 'test', 'unlabeled'):
        relative = Path(manifest['file_roles'][role])
        resolved = (root / relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not resolved.is_relative_to(root):
            raise ValueError('Source path escapes task directory')
        paths[role] = resolved
    if len(set(paths.values())) != 3:
        raise ValueError('Unlabeled data cannot alias train or test')
    identifier, columns = manifest['id_column'], manifest['features']
    if not columns or len(set(columns)) != len(columns) or set(columns) & {identifier, manifest['target']}:
        raise ValueError('Invalid unlabeled feature roles')
    frame = pd.read_csv(paths['unlabeled'], dtype={identifier: 'string'})
    if set(frame) != set(columns + [identifier]) or frame.empty:
        raise ValueError('Unlabeled source has targets, undeclared columns or no rows')
    ids = frame[identifier]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError('Unlabeled IDs are missing or duplicated')
    known = {}
    for role in ('train', 'test'):
        values = pd.read_csv(paths[role], usecols=[identifier], dtype={identifier: 'string'})[identifier]
        if values.isna().any() or values.duplicated().any():
            raise ValueError('Known source IDs are missing or duplicated')
        known[role] = frozenset(values.astype(str))
    if set(ids) & (known['train'] | known['test']):
        raise ValueError('Unlabeled source overlaps labeled or test IDs')
    features = frame[columns].apply(pd.to_numeric, errors='raise')
    features.index = ids.astype(str)
    if np.isinf(features.to_numpy(dtype=float)).any():
        raise ValueError('Infinite unlabeled feature value')
    source_id = digest({'role': 'unlabeled', 'path': manifest['file_roles']['unlabeled'],
                        'sha256': hashlib.sha256(paths['unlabeled'].read_bytes()).hexdigest()})
    return UnlabeledSource(source_id, features, frame_hash(features), known['train'], known['test'])


@dataclass(frozen=True)
class PseudoRecipe:
    threshold: float
    pseudo_weight: float = .25
    seed: int = 42
    iterations: int = 1

    def __post_init__(self):
        if not np.isfinite(self.threshold) or not .5 <= self.threshold <= 1:
            raise ValueError('Invalid confidence threshold')
        if not np.isfinite(self.pseudo_weight) or not 0 < self.pseudo_weight <= 1:
            raise ValueError('Invalid pseudo-label weight')
        if type(self.iterations) is not int or self.iterations != 1:
            raise ValueError('Only one teacher-to-student iteration is supported')


@dataclass(frozen=True)
class PhaseAuthority:
    phase: str
    allowed_ids: frozenset[str]


def estimator(seed):
    return Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('scale', StandardScaler()),
                     ('model', LogisticRegression(C=1., max_iter=300, random_state=seed))])


@dataclass
class PseudoModel:
    teacher: Pipeline
    student: Pipeline
    recipe: PseudoRecipe
    phase: str
    columns: tuple[str, ...]
    provenance: list[dict]
    fit_ids: tuple[str, ...]
    teacher_hash: str

    def predict(self, features):
        if tuple(features) != self.columns:
            raise ValueError('Prediction feature order differs from fitted recipe')
        return self.student.predict(features)


def fit_one_iteration(features, labels, source, recipe, authority):
    ids = set(features.index)
    if authority.phase not in PROTOCOL['phases']:
        raise ValueError('Unknown training phase')
    if not features.index.is_unique or not features.index.equals(labels.index) or not len(features):
        raise ValueError('Labels and unique training rows must align')
    if not ids <= authority.allowed_ids or not ids <= source.labeled_ids or ids & source.test_ids:
        raise ValueError('Labeled rows are outside this phase authority')
    if tuple(features) != tuple(source.features) or frame_hash(source.features) != source.features_hash:
        raise ValueError('Approved source changed or feature schema differs')
    teacher = estimator(recipe.seed)
    teacher.fit(features, labels)
    probability = teacher.predict_proba(source.features)
    confidence = probability.max(axis=1)
    selected = np.flatnonzero(confidence >= recipe.threshold)
    classes = teacher[-1].classes_
    pseudo = classes[np.argmax(probability[selected], axis=1)]
    combined = pd.concat([features, source.features.iloc[selected]])
    target = np.concatenate([labels.to_numpy(), pseudo])
    weights = np.concatenate([np.ones(len(features)), np.full(len(selected), recipe.pseudo_weight)])
    student = estimator(recipe.seed)  # A fresh preprocessor and estimator, never the teacher.
    student.fit(combined, target, model__sample_weight=weights)
    teacher_hash = hashlib.sha256(pickle.dumps(teacher, protocol=5)).hexdigest()
    provenance = [{'row_id': str(source.features.index[position]), 'source_id': source.source_id,
        'teacher_phase': authority.phase, 'teacher_sha256': teacher_hash,
        'label': pseudo[index].item() if isinstance(pseudo[index], np.generic) else pseudo[index],
        'confidence': float(confidence[position]), 'weight': recipe.pseudo_weight}
        for index, position in enumerate(selected)]
    return PseudoModel(teacher, student, recipe, authority.phase, tuple(features), provenance,
                       tuple(features.index), teacher_hash)


def macro(labels, prediction):
    return float(f1_score(labels, prediction, average='macro', zero_division=0))


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'PREREGISTRATION.json').write_text(json.dumps(PROTOCOL, indent=2) + '\n')
    cases = []
    for seed in PROTOCOL['dataset_seeds']:
        case = output / str(seed)
        case.mkdir()
        matrix, values = make_classification(n_samples=3000, n_features=10, n_informative=6,
            n_redundant=2, n_classes=3, n_clusters_per_class=2, weights=[.70, .22, .08],
            class_sep=.8, flip_y=.025, random_state=seed)
        columns = [f'x{i}' for i in range(10)]
        frame = pd.DataFrame(matrix[:1500], columns=columns, index=[f'train:{i}' for i in range(1500)])
        labels = pd.Series(values[:1500], index=frame.index, name='y')
        frame.assign(y=labels).rename_axis('row_id').to_csv(case / 'labeled.csv')
        pd.DataFrame(matrix[1500:2500], columns=columns, index=[f'unlabeled:{i}' for i in range(1000)]).rename_axis('row_id').to_csv(case / 'approved_unlabeled.csv')
        pd.DataFrame(matrix[2500:], columns=columns, index=[f'test:{i}' for i in range(500)]).rename_axis('row_id').to_csv(case / 'future.csv')
        manifest = {'allow_pseudo_labels': True, 'file_roles': {'train': 'labeled.csv', 'test': 'future.csv',
            'unlabeled': 'approved_unlabeled.csv'}, 'features': columns, 'id_column': 'row_id', 'target': 'y'}
        (case / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
        source = approved_source(case, manifest)
        fit, validation, control = stratified_parts(labels.reset_index(drop=True).to_frame(), 42)
        inner_a, inner_b, inner_c = stratified_parts(labels.iloc[fit].reset_index(drop=True).to_frame(), 43)
        training = [fit[position] for position in inner_a]
        tuning = [fit[position] for position in inner_b + inner_c]
        authority = PhaseAuthority('research_fit', frozenset(frame.iloc[fit].index))
        trials = []
        for threshold in PROTOCOL['thresholds']:
            recipe = PseudoRecipe(threshold)
            model = fit_one_iteration(frame.iloc[training], labels.iloc[training], source, recipe, authority)
            trials.append({'threshold': threshold, 'inner_macro_f1': macro(labels.iloc[tuning], model.predict(frame.iloc[tuning]))})
        chosen = min(trials, key=lambda row: (-row['inner_macro_f1'], -row['threshold']))
        recipe = PseudoRecipe(chosen['threshold'])
        frozen_hash = digest(asdict(recipe))
        phases, models = [], []
        for phase, rows in [('research_fit', fit), ('control_preparation', sorted(fit + validation)),
                            ('full_train_refit', list(range(len(frame))))]:
            if phase == 'full_train_refit':
                # Finish control before allowing any model to fit control labels.
                candidate = models[1]
                baseline = estimator(42).fit(frame.iloc[sorted(fit + validation)], labels.iloc[sorted(fit + validation)])
                score_base = macro(labels.iloc[control], baseline.predict(frame.iloc[control]))
                score_pseudo = macro(labels.iloc[control], candidate.predict(frame.iloc[control]))
                control_result = {'baseline_macro_f1': score_base, 'pseudo_macro_f1': score_pseudo,
                                  'frozen_recipe_hash': frozen_hash}
                (case / 'CONTROL_FROZEN.json').write_text(json.dumps(control_result, indent=2) + '\n')
            model = fit_one_iteration(frame.iloc[rows], labels.iloc[rows], source, recipe,
                PhaseAuthority(phase, frozenset(frame.iloc[rows].index)))
            assert all(model.teacher is not old.teacher and model.student is not old.student for old in models)
            models.append(model)
            assert digest(asdict(model.recipe)) == frozen_hash
            ledger = gzip.compress(json.dumps(model.provenance, sort_keys=True).encode(), mtime=0)
            path = case / (phase + '_provenance.json.gz')
            path.write_bytes(ledger)
            phases.append({'phase': phase, 'labeled_rows': len(rows), 'labeled_ids_hash': digest(list(model.fit_ids)),
                'pseudo_rows': len(model.provenance), 'teacher_sha256': model.teacher_hash,
                'pseudo_weight': recipe.pseudo_weight, 'recipe_hash': frozen_hash,
                'provenance_file': path.name, 'provenance_sha256': hashlib.sha256(ledger).hexdigest()})
        assert json.loads((case / 'CONTROL_FROZEN.json').read_text()) == control_result
        def forbidden(*args, **kwargs):
            raise AssertionError('predict attempted to fit')
        candidate.teacher.fit = forbidden
        candidate.student.fit = forbidden
        candidate.student[-1].fit = forbidden
        assert np.array_equal(candidate.predict(frame.iloc[control]), candidate.predict(frame.iloc[control]))
        cases.append({'seed': seed, 'inner_trials': trials, 'selected_recipe': asdict(recipe),
            'rule_tuning_ids_hash': digest(list(frame.iloc[tuning].index)), 'phases': phases,
            'control_baseline_macro_f1': score_base, 'control_pseudo_macro_f1': score_pseudo,
            'pseudo_minus_baseline': score_pseudo - score_base, 'predict_never_fits': True})
    report = {'passed': True, 'organizer_requests': 0, 'cases': cases,
        'unlabeled_ground_truth_written_or_used': False, 'iterations': 1,
        'decision': 'staged_only; retain current runtime until quality/phase integration acceptance',
        'scope': 'Three synthetic datasets; phase mechanism refit exercised after frozen control, regardless of experimental win; no final CSV publication or full-runtime acceptance claim',
        'runtime_archives_modified': False}
    (output / 'REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output.resolve())
    print(json.dumps({'passed': report['passed'], 'cases': len(report['cases'])}))
