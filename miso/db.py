"""SQLite connection management.

Tries to load sqlite-vec for fast vector search; falls back to in-Python
cosine if the extension isn't installed. The fallback is adequate at Miso's
scale (hundreds to low-thousands of summaries).
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
EMBEDDING_DIM = 768  # bge-base-en-v1.5


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_PATH.read_text())
    conn.commit()
    _try_load_sqlite_vec(conn)
    return conn


def _try_load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    try:
        conn.enable_load_extension(True)
        import sqlite_vec  # type: ignore
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as e:
        log.info(
            "sqlite-vec not loaded (%s); dense retrieval will use the Python "
            "fallback. Install with `pip install sqlite-vec`.", e,
        )
        return False
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS summary_vectors USING vec0("
        f"note_id TEXT PRIMARY KEY, embedding FLOAT[{EMBEDDING_DIM}])"
    )
    conn.commit()
    return True


def has_sqlite_vec(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='summary_vectors'"
    ).fetchone()
    return row is not None


def reset(path: Path) -> None:
    if path.exists():
        path.unlink()
