from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .contracts import OperationRecord
from .util import canonical, digest, utf8_short
from .diagnostics import emit

TRANSITIONS = {
    "preparing": {"searching", "failed", "contract_unresolved"},
    "searching": {"frozen", "failed"}, "frozen": {"control", "refitting", "completed", "failed"},
    "control": {"refitting", "completed", "failed"}, "refitting": {"completed", "failed"},
    "completed": set(), "failed": set(), "contract_unresolved": set(),
}


class Journal:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.db = sqlite3.connect(directory / "journal.sqlite", timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS operations (id TEXT PRIMARY KEY, record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, spec TEXT NOT NULL, result TEXT);
            CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, time REAL, kind TEXT, payload TEXT);
        """)
        self.db.commit()

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        previous = self.get(key)
        with self.db:
            self.db.execute("INSERT INTO metadata VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, canonical(value)))
        if previous != value:
            emit('metadata_changed', run_id=self.directory.name, key=key, previous=previous, value=value)

    def put_many(self, values, event=None):
        """One durable transaction for related controller state and publication intent."""
        with self.db:
            self.db.executemany("INSERT INTO metadata VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [(key, canonical(value)) for key, value in values.items()])
            if event:
                self.db.execute("INSERT INTO events(time,kind,payload) VALUES (?,?,?)", (time.time(), event[0], canonical(event[1])))

    def event(self, kind, value):
        with self.db:
            self.db.execute("INSERT INTO events(time,kind,payload) VALUES (?,?,?)", (time.time(), kind, canonical(value)))
        emit('journal_event', run_id=self.directory.name, event=kind, value=value)

    def transition(self, state, reason):
        old = self.get("state", "preparing")
        if state not in TRANSITIONS[old]:
            raise ValueError(f"Forbidden controller transition: {old} -> {state}")
        with self.db:
            self.db.execute("INSERT INTO metadata VALUES ('state',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (canonical(state),))
            self.db.execute("INSERT INTO events(time,kind,payload) VALUES (?,?,?)", (time.time(), "transition", canonical({"from": old, "to": state, "reason": reason})))
        emit('phase_transition', run_id=self.directory.name, previous=old, state=state, reason=reason)

    def operation(self, operation_id):
        row = self.db.execute("SELECT record FROM operations WHERE id=?", (operation_id,)).fetchone()
        return OperationRecord.model_validate_json(row[0]) if row else None

    def begin_operation(self, run_id, operation_id, tool, arguments, experiment_id=None, budget_seconds=0):
        old = self.operation(operation_id)
        argument_hash = digest(arguments)
        request_hash = digest(arguments.get("arguments", arguments))
        if old:
            if (old.request_hash != request_hash if old.request_hash else old.arguments_hash != argument_hash) or old.tool != tool:
                raise ValueError(f"Operation ID {operation_id!r} was already used with different arguments; use a new operation_id for a changed command")
            return old, False
        operation = OperationRecord(run_id=run_id, operation_id=operation_id, tool=tool, arguments_hash=argument_hash, request_hash=request_hash,
            experiment_id=experiment_id, status="pending", started_at=time.time(), budget_seconds=budget_seconds)
        requested = arguments.get("arguments", arguments)
        operation.component = utf8_short((requested.get("hypothesis") or {}).get("changed_component") or tool, 128)
        pipeline = requested.get("pipeline")
        operation.engine = pipeline.get("engine", "controller") if isinstance(pipeline, dict) else "controller"
        operation.parent_id = requested.get("parent_id")
        operation.protocol_hash = arguments.get("protocol")
        # Validate searchable metadata before allocating any training resources.
        from context_compressor import Record
        Record("operation:" + operation_id, "operation:" + digest(operation_id), digest(arguments),
            kind="operation", component=operation.component, engine=operation.engine,
            parent_id=operation.parent_id, protocol=operation.protocol_hash or "unspecified", tool=tool)
        with self.db:
            self.db.execute("INSERT INTO operations VALUES (?,?)", (operation_id, canonical(operation)))
        emit('operation_started', record=operation.model_dump(mode='json'), arguments=arguments)
        return operation, True

    def save_operation(self, operation):
        old = self.operation(operation.operation_id)
        if old and canonical(old) == canonical(operation):
            return
        with self.db:
            self.db.execute("UPDATE operations SET record=? WHERE id=?", (canonical(operation), operation.operation_id))
        emit('operation_record', record=operation.model_dump(mode='json'))

    def incomplete_operations(self):
        return [op for row in self.db.execute("SELECT record FROM operations")
                if (op := OperationRecord.model_validate_json(row[0])).status in {"pending", "running"}]

    def save_experiment(self, spec, result=None):
        with self.db:
            self.db.execute("INSERT INTO experiments VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET result=COALESCE(excluded.result,experiments.result)",
                (spec.experiment_id, canonical(spec), canonical(result) if result is not None else None))
        emit('experiment_record', spec=spec.model_dump(mode='json'), result=result.model_dump(mode='json') if result is not None else None)

    def experiments(self):
        return [{"spec": json.loads(a), "result": json.loads(b) if b else None}
                for a, b in self.db.execute("SELECT spec,result FROM experiments ORDER BY rowid")]

    def close(self):
        self.db.close()
