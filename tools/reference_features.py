"""Development-only, target-free transformations suggested by source research."""
from __future__ import annotations


def engineer(frame, plan=None):
    import numpy as np
    from scipy.special import ndtri

    result = frame.copy()
    if plan in {'energy_building', 'energy_all'}:
        result['year_built'] = result.year_built.mask(result.year_built <= 0)
        result['ref_log_area'] = np.log1p(result.floor_area.clip(lower=0))
        rating = result.energy_star_rating
        result['ref_rating_missing'] = rating.isna().astype(float)
        # A smooth alternative coordinate for a percentile score, not an
        # asserted reconstruction of the ENERGY STAR calculation.
        result['ref_rating_normal'] = ndtri(((rating + .5) / 101).clip(.005, .995))
        result['ref_state_facility'] = result.State_Factor.astype(str) + '|' + result.facility_type.astype(str)
        result['ref_facility_area'] = result.facility_type.astype(str) + '|' + np.floor(result.ref_log_area).astype('Int64').astype(str)
        if plan == 'energy_all':
            monthly = [c for c in result if c.endswith('_avg_temp_c')]
            if monthly:
                result['ref_monthly_temp_std'] = result[monthly].std(axis=1)
                result['ref_monthly_temp_range'] = result[monthly].max(axis=1) - result[monthly].min(axis=1)
    elif plan in {'network', 'network_semantic'}:
        for left, right, name in [('sbytes', 'spkts', 'source_packet_bytes'),
                                  ('dbytes', 'dpkts', 'destination_packet_bytes'),
                                  ('sbytes', 'dbytes', 'byte_ratio'), ('spkts', 'dpkts', 'packet_ratio'),
                                  ('synack', 'tcprtt', 'syn_fraction')]:
            denominator = result[right].where(result[right] != 0)
            result['ref_' + name] = (result[left] / denominator).replace([np.inf, -np.inf], np.nan)
        result['ref_ttl_difference'] = result.sttl - result.dttl
        result['ref_log_byte_ratio'] = np.log1p(result.sbytes.clip(lower=0)) - np.log1p(result.dbytes.clip(lower=0))
        if plan == 'network_semantic':
            for a, b in [('proto', 'service'), ('proto', 'state'), ('service', 'state')]:
                result[f'ref_{a}_{b}'] = result[a].fillna('__MISSING__').astype(str) + '|' + result[b].fillna('__MISSING__').astype(str)
            result['ref_service_unspecified'] = result.service.eq('-').astype(float)
            result['ref_no_reply'] = ((result.dpkts == 0) & (result.dbytes == 0)).astype(float)
            result['ref_total_packets'] = result.spkts + result.dpkts
            result['ref_total_bytes'] = result.sbytes + result.dbytes
            result['ref_packet_asymmetry'] = (result.spkts - result.dpkts) / (result.ref_total_packets + 1)
            result['ref_byte_asymmetry'] = (result.sbytes - result.dbytes) / (result.ref_total_bytes + 1)
            result['ref_tcp_time_residual'] = result.tcprtt - result.synack - result.ackdat
            result['ref_no_windows'] = ((result.swin == 0) & (result.dwin == 0)).astype(float)
            result['ref_log_duration'] = np.log1p(result.dur.clip(lower=0))
            result['ref_log_rate'] = np.log1p(result.rate.clip(lower=0))
    elif plan is not None:
        raise ValueError(f'Unknown feature plan: {plan}')
    return result


def macro_f1(y, prediction, count):
    import numpy as np

    y, prediction = np.asarray(y, dtype=int), np.asarray(prediction, dtype=int)
    matrix = np.bincount(y * count + prediction, minlength=count * count).reshape(count, count)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    return float(np.divide(2 * matrix.diagonal(), denominator, out=np.zeros(count), where=denominator != 0).mean())


def fit_offsets(probabilities, labels, initial=None):
    """Bounded coordinate search; support shrinkage limits rare-class swings.

    This is an engineering adaptation, not a theorem for multiclass macro F1.
    Its usefulness must be assessed on labels excluded from this search.
    """
    import numpy as np

    p = np.asarray(probabilities)
    k = p.shape[1]
    z = np.log(np.maximum(p, 1e-15))
    support = np.bincount(labels, minlength=k)
    shrink = support / (support + 50.)
    offsets = np.zeros(k) if initial is None else np.array(initial, dtype=float).copy()
    for _ in range(2):
        for column in range(k):
            choices = []
            for raw in [-1., -.5, 0., .5, 1.]:
                candidate = offsets.copy()
                candidate[column] = raw * shrink[column]
                score = macro_f1(labels, (z + candidate).argmax(axis=1), k)
                choices.append((score, -float(np.dot(candidate, candidate)), candidate))
            offsets = max(choices, key=lambda item: item[:2])[2]
    return offsets


class RatingImputer:
    """Auxiliary fit on observed rating values in permitted training rows only."""

    columns = ['State_Factor', 'building_class', 'facility_type', 'floor_area', 'year_built', 'ELEVATION']

    def fit(self, frame):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.impute import SimpleImputer
        from automl_agent.contracts import PipelineSpec
        from automl_agent.pipelines import make_estimator

        observed = frame.loc[frame.energy_star_rating.notna()]
        if len(observed) < 30:
            raise ValueError('Insufficient observed training ratings for auxiliary fit')
        self.preprocessor = ColumnTransformer([
            ('numeric', SimpleImputer(strategy='median', keep_empty_features=True), ['floor_area', 'year_built', 'ELEVATION']),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['State_Factor', 'building_class', 'facility_type'])])
        x = self.preprocessor.fit_transform(observed[self.columns])
        self.model = make_estimator(PipelineSpec(model='extra_trees', hyperparameters={
            'n_estimators': 100, 'max_depth': 12, 'min_samples_leaf': 5}), False, {'cpu': 4})
        self.model.fit(x, observed.energy_star_rating)
        return self

    def transform(self, frame):
        result = frame.copy()
        mask = result.energy_star_rating.isna()
        result['ref_rating_was_missing'] = mask.astype(float)
        if mask.any():
            values = self.model.predict(self.preprocessor.transform(result.loc[mask, self.columns]))
            result.loc[mask, 'energy_star_rating'] = values.clip(1, 100)
        return result
