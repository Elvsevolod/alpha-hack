"""EDA persistence in the existing run database; no separate database or lifecycle."""
import hashlib
import json

from context_compressor import ProfileCache

from ..util import canonical, digest


class EDARepository:
    def __init__(self, store):
        self.store = store
        self.store.db.execute("CREATE TABLE IF NOT EXISTS eda_reports(id TEXT PRIMARY KEY, content TEXT NOT NULL)")

    def check_scope(self, scope):
        self.store._check_scope(scope)

    def profile(self, key, calculate):
        reference = ProfileCache(self.store).get_or_create(key, calculate)
        source_id = "profile:" + key.hash
        row = self.store._source_row(source_id, key.format_version, "", key.scope)
        blob = self.store.db.execute("SELECT content FROM cc_sources WHERE source_id=? AND version=? AND section=''",
            (source_id, key.format_version)).fetchone()[0]
        if hashlib.sha256(blob).hexdigest() != row["sha256"]:
            raise ValueError("Cached profile source checksum mismatch")
        return reference, json.loads(blob)

    def report(self, identity):
        self.check_scope(identity["args"]["scope"])
        row = self.store.db.execute("SELECT content FROM eda_reports WHERE id=?", (digest(identity),)).fetchone()
        return json.loads(row[0]) if row else None

    def save_report(self, identity, value):
        self.check_scope(identity["args"]["scope"])
        with self.store.transaction():
            self.store.db.execute("INSERT OR IGNORE INTO eda_reports VALUES (?,?)", (digest(identity), canonical(value)))
