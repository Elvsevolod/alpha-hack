"""CPU-only development alternatives. No downloads, pretrained weights or APIs.

MiniEnsembleEstimator is a budgeted, independently implemented TabM-mini-style
variant of arXiv:2410.24210, not the authors' full benchmark configuration.
"""
from __future__ import annotations

import time
import numpy as np


def mini_network(inputs, outputs, heads, width):
    import torch
    from torch import nn

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.scales = nn.Parameter(torch.randn(heads, inputs))
            self.backbone = nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Dropout(.1),
                                          nn.Linear(width, width), nn.ReLU(), nn.Dropout(.1))
            self.output = nn.Parameter(torch.empty(heads, width, outputs))
            self.bias = nn.Parameter(torch.zeros(heads, outputs))
            nn.init.uniform_(self.output, -width ** -.5, width ** -.5)

        def forward(self, x):
            hidden = self.backbone(x[:, None, :] * self.scales[None, :, :])
            return torch.einsum('bkh,kho->bko', hidden, self.output) + self.bias
    return Network()


class MiniEnsembleEstimator:
    def __init__(self, classification=False, epochs=40, width=128, heads=8, batch_size=512,
                 learning_rate=.001, decay=.0001, seed=42, seconds=150, threads=1):
        self.classification, self.epochs, self.width, self.heads = classification, epochs, width, heads
        self.batch_size, self.learning_rate, self.decay = batch_size, learning_rate, decay
        self.seed, self.seconds = seed, seconds
        self.threads = threads

    def fit(self, x, y, eval_set=None, sample_weight=None):
        import torch
        from torch.nn import functional as F
        from tools.reference_features import macro_f1

        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y) if self.classification else None
        self.inputs_, self.outputs_ = x.shape[1], len(self.classes_) if self.classification else 1
        net = mini_network(self.inputs_, self.outputs_, self.heads, self.width).to('cpu')
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.learning_rate, weight_decay=self.decay)
        tx = torch.from_numpy(x)
        ty = torch.as_tensor(y, dtype=torch.long if self.classification else torch.float32)
        weights = torch.ones(len(y)) if sample_weight is None else torch.as_tensor(sample_weight, dtype=torch.float32)
        weights = weights / weights.mean()
        best, stale, began = -np.inf, 0, time.monotonic()
        self.stop_reason_ = 'epoch_cap'
        self.training_curve_ = []
        for epoch in range(self.epochs):
            net.train()
            for rows in np.array_split(rng.permutation(len(y)), max(1, int(np.ceil(len(y) / self.batch_size)))):
                prediction = net(tx[rows])
                if self.classification:
                    loss = F.cross_entropy(prediction.reshape(-1, self.outputs_), ty[rows, None].expand(-1, self.heads).reshape(-1), reduction='none').reshape(len(rows), self.heads).mean(axis=1)
                else:
                    loss = ((prediction[:, :, 0] - ty[rows, None]) ** 2).mean(axis=1)
                loss = (loss * weights[rows]).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite neural training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
                optimizer.step()
            net.eval()
            if eval_set is None:
                score = float(epoch)
            else:
                values = self._predict_network(net, eval_set[0])
                score = macro_f1(np.asarray(eval_set[1]), values.argmax(axis=1), self.outputs_) if self.classification else -float(np.sqrt(np.mean((np.asarray(eval_set[1]) - values) ** 2)))
            self.training_curve_.append({'epoch': epoch + 1, 'validation_score': score if eval_set is not None else None})
            if score > best:
                best, stale = score, 0
                self.state_ = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
                self.n_iter_ = epoch + 1
            else:
                stale += 1
            self.epochs_run_ = epoch + 1
            if eval_set is not None and stale >= 8:
                self.stop_reason_ = 'early_stopping'
                break
            if time.monotonic() - began >= self.seconds:
                self.stop_reason_ = 'internal_time_cap'
                break
        return self

    def _predict_network(self, net, x):
        import torch
        values = []
        with torch.no_grad():
            for start in range(0, len(x), 2048):
                tensor = torch.as_tensor(np.asarray(x[start:start + 2048], dtype=np.float32))
                p = net(tensor)
                values.append((p.softmax(axis=2).mean(axis=1) if self.classification else p[:, :, 0].mean(axis=1)).numpy())
        return np.concatenate(values)

    def predict_proba(self, x):
        import torch
        torch.set_num_threads(self.threads)
        net = mini_network(self.inputs_, self.outputs_, self.heads, self.width)
        net.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in self.state_.items()})
        net.eval()
        return self._predict_network(net, x)

    def predict(self, x):
        p = self.predict_proba(x)
        return self.classes_[p.argmax(axis=1)] if self.classification else p


class HierarchicalLGBM:
    """Soft gate: P(type)=P(attack)*P(type|attack); no true binary input label."""

    def __init__(self, spec, normal_index, component_iterations=None):
        self.spec, self.normal_index = spec, normal_index
        self.component_iterations = component_iterations

    def fit(self, x, y, eval_set=None, **kwargs):
        import lightgbm as lgb
        from automl_agent.contracts import PipelineSpec
        from automl_agent.pipelines import make_estimator

        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        self.attack_indices_ = self.classes_[self.classes_ != self.normal_index]
        binary = (y != self.normal_index).astype(int)
        if len(np.unique(binary)) < 2:
            raise ValueError('Hierarchy requires both normal and attack training labels')
        self.models_, self.component_iterations_ = {}, {}
        for stage in ['gate', 'attack']:
            params = dict(self.spec['hyperparameters'])
            if self.component_iterations:
                params['n_estimators'] = self.component_iterations[stage]
            model = make_estimator(PipelineSpec.model_validate(dict(self.spec, hyperparameters=params)), True, {'cpu': 4})
            rows = np.ones(len(y), dtype=bool) if stage == 'gate' else binary.astype(bool)
            labels = binary if stage == 'gate' else np.searchsorted(self.attack_indices_, y[rows])
            fit = {}
            if eval_set is not None:
                vx, vy = eval_set
                vy = np.asarray(vy, dtype=int)
                keep = np.ones(len(vy), dtype=bool) if stage == 'gate' else vy != self.normal_index
                target = (vy != self.normal_index).astype(int) if stage == 'gate' else np.searchsorted(self.attack_indices_, vy[keep])
                fit.update(eval_set=(vx.iloc[keep] if hasattr(vx, 'iloc') else vx[keep], target),
                           callbacks=[lgb.early_stopping(60, verbose=False)])
            model.fit(x.iloc[rows] if hasattr(x, 'iloc') else x[rows], labels, **fit)
            self.models_[stage] = model
            self.component_iterations_[stage] = int(model.best_iteration_ or params['n_estimators'])
        self.best_iteration_ = max(self.component_iterations_.values())
        return self

    def predict_proba(self, x):
        attack = self.models_['gate'].predict_proba(x)[:, 1]
        conditional = self.models_['attack'].predict_proba(x)
        result = np.zeros((len(x), len(self.classes_)))
        result[:, self.normal_index] = 1 - attack
        result[:, self.attack_indices_] = attack[:, None] * conditional
        return result

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]
