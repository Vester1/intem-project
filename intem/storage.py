"""Индекс прототипа. Профили и история диалога отсутствуют."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, source TEXT NOT NULL, page INTEGER NOT NULL,
                text TEXT NOT NULL, vector TEXT NOT NULL, metadata TEXT NOT NULL)''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def chunks(self):
        with self.connect() as db:
            return [{**dict(r), "vector": json.loads(r["vector"]), "metadata": json.loads(r["metadata"])}
                    for r in db.execute("SELECT * FROM chunks ORDER BY id")]
