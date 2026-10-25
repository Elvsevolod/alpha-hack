"""Small, testable probability and chronological-refit research decisions."""
from __future__ import annotations

import numpy as np


def normalize(p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(axis=1) <= 0).any():
        raise ValueError('Invalid probability matrix')
    return p / p.sum(axis=1, keepdims=True)


def unweight(p, counts, strength):
    counts = np.asarray(counts, dtype=float)
    if (counts <= 0).any():
        raise ValueError('Cannot correct unsupported classes')
    return normalize(np.asarray(p) * (counts / counts.max()) ** strength)


def temper(p, temperature):
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    z = np.log(np.maximum(p, 1e-15)) / temperature
    return normalize(np.exp(z - z.max(axis=1, keepdims=True)))


def nll(p, labels):
    return float(-np.log(np.maximum(np.asarray(p)[np.arange(len(labels)), np.asarray(labels, dtype=int)], 1e-15)).mean())


def fit_temperature(p, labels):
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(lambda log_t: nll(temper(p, np.exp(log_t)), labels),
                             bounds=(np.log(.5), np.log(2.)), method='bounded')
    return float(np.exp(result.x))


def join_hierarchy(gate, conditional, normal_index):
    gate, conditional = normalize(gate), normalize(conditional)
    result = np.zeros((len(gate), conditional.shape[1] + 1))
    attack = [i for i in range(result.shape[1]) if i != normal_index]
    result[:, normal_index] = gate[:, 0]
    result[:, attack] = gate[:, 1, None] * conditional
    return normalize(result)


def refit_iterations(rule, inner_iterations, earlier_rows, full_rows, limit=2500):
    if not 0 < earlier_rows < full_rows or inner_iterations < 1:
        raise ValueError('Refit requires strictly earlier, nonempty inner training rows')
    if rule == 'latest_inner':
        scale = 1.
    elif rule == 'latest_inner_sqrt_rows':
        scale = np.sqrt(full_rows / earlier_rows)
    else:
        raise ValueError('Unknown chronological refit rule')
    return max(1, min(limit, int(round(inner_iterations * scale))))
