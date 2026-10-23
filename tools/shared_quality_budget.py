"""Shared, conservative development-only accounting; never shipped in a ZIP.

The 100-unit in-flight reserve is a safety margin, not a published price bound.
Stop admitting at 300 of 400, serialize through one gateway, and stop permanently
on unknown billing. Provider-reported usage costs are counted before counters
refresh. Add the account counter high-water mark conservatively, even though
this can count our own settled charges twice; never attribute an unexplained
counter increment to our requests and thereby hide another spender's charges.
"""
from __future__ import annotations

import fcntl
import json
import math
from pathlib import Path
import threading

from automl_agent.util import atomic_json
from tools import diagnostic_3h as d


class SharedQualityBudget:
    cap = 400.
    reserve = 100.
    max_output_tokens = 65536
    max_request_bytes = 262144

    def __init__(self, path, before):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.released = False
        self.process_lock = self.path.with_suffix('.lock').open('a')
        try:
            fcntl.flock(self.process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                self.state = json.loads(self.path.read_text())
                if self.state['base']['key_fingerprint'] != before['key_fingerprint'] or self.state['base']['limits']['budget_reset_at'] != before['limits']['budget_reset_at']:
                    raise ValueError('Shared budget key or period mismatch')
            else:
                spend = before['limits']['spend']
                if type(spend) not in (int, float) or not math.isfinite(spend) or spend < 0:
                    raise ValueError('Known finite initial period spend required')
                self.state = {'base': before, 'initial_spend': spend, 'counter_spend': spend,
                    'reported_cost': 0., 'requests': {}, 'inflight': None, 'closed': None}
                self.save()
            if self.state['inflight'] is not None:
                self.stop('previous_request_billing_unknown')
            self.observe(before)
        except BaseException:
            self.process_lock.close()
            raise

    def save(self):
        atomic_json(self.path, self.state)

    def stop(self, reason):
        with self.lock:
            self.state['closed'] = self.state['closed'] or reason
            self.save()
            return False

    @property
    def spent(self):
        return self.state['counter_spend'] + self.state['reported_cost']

    def observe(self, snapshot):
        with self.lock:
            if self.released:
                return False
            base = self.state['base']
            if snapshot['limits']['budget_reset_at'] != base['limits']['budget_reset_at']:
                return self.stop('period_changed')
            comparison = d.limits.compare(base, snapshot)
            if comparison.get('increase') is None:
                return self.stop('incomparable_budget_counters')
            measured = self.state['initial_spend'] + comparison['increase']
            if measured > self.state['counter_spend']:
                self.state['counter_spend'] = measured
                self.save()
            if snapshot.get('budget_exhausted') or snapshot['limits'].get('blocked'):
                return self.stop('account_unavailable')
            if self.spent >= self.cap - self.reserve:
                return self.stop('shared_300_admission_threshold')
            return self.state['closed'] is None

    def begin(self, identifier, body, raw):
        with self.lock:
            if self.released:
                return False
            tokens = body.get('max_tokens')
            if type(tokens) is not int or not 0 < tokens <= self.max_output_tokens or len(raw) > self.max_request_bytes:
                return self.stop('development_request_size_limit')
            if self.state['closed'] or self.spent >= self.cap - self.reserve:
                return self.stop('shared_300_admission_threshold')
            if self.state['inflight'] is not None or identifier in self.state['requests']:
                return self.stop('concurrent_or_duplicate_paid_admission')
            self.state['inflight'] = identifier
            self.save()  # Durable before the HTTP send; a crash cannot erase it.
            return True

    def complete(self, identifier, status, usage):
        with self.lock:
            if self.released:
                return False
            if self.state['inflight'] != identifier:
                return self.stop('billing_request_identity_mismatch')
            cost = usage.get('cost') if isinstance(usage, dict) else None
            if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
                return self.stop('response_billing_unknown')
            self.state['requests'][identifier] = {'status': status, 'cost': cost}
            self.state['reported_cost'] += cost
            self.state['inflight'] = None
            self.save()
            if cost > self.reserve:
                return self.stop('observed_charge_exceeds_safety_margin')
            if self.spent >= self.cap - self.reserve:
                return self.stop('shared_300_admission_threshold')
            return True

    def cancel_before_send(self, identifier):
        with self.lock:
            if self.released or self.state['inflight'] != identifier:
                return False
            self.state['requests'][identifier] = {'not_sent': True, 'cost': 0.}
            self.state['inflight'] = None
            self.save()
            return True

    def release(self):
        with self.lock:
            if not self.released:
                if self.state['inflight'] is not None:
                    self.stop('request_still_inflight_at_release')
                self.released = True
                self.process_lock.close()
