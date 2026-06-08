"""Per-note summary storage."""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime

from miso.types import ExtractedNote, Note, Summary

log = logging.getLogger(__name__)

EMBEDDING_DIM = 768  # bge-base-en-v1.5


class SummaryStore:
    def __init__(self, conn: sqlite3.Connection, *, embedder=None):
        """When `embedder` is None, the dense side stores zero vectors."""
        self.conn = conn
        self.embedder = embedder

    def add(self, extracted: ExtractedNote, note: Note) -> Summary:
        ts = note.timestamp.isoformat()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO notes(note_id, course_id, image_path,
                                         raw_ocr_json, extracted_json,
                                         processing_order, pipeline_version, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                note.note_id, note.course_id, str(note.image_path),
                None,
                json.dumps(extracted.structured_json),
                note.processing_order,
                "v1.0.0",
                ts,
            ),
        )
        bm25_text = f"{extracted.summary_topic_line} {extracted.summary_gist}"
        self.conn.execute(
            """
            INSERT OR REPLACE INTO summaries(note_id, course_id, topic_line, gist,
                                             processing_order, timestamp, bm25_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                note.note_id, note.course_id,
                extracted.summary_topic_line, extracted.summary_gist,
                note.processing_order, ts, bm25_text,
            ),
        )
        self._maybe_write_embedding(note.note_id, bm25_text)
        self.conn.commit()
        return Summary(
            note_id=note.note_id,
            course_id=note.course_id,
            topic_line=extracted.summary_topic_line,
            gist=extracted.summary_gist,
            processing_order=note.processing_order,
            pipeline_version="v1.0.0",
            timestamp=note.timestamp,
        )

    def _maybe_write_embedding(self, note_id: str, text: str) -> None:
        from miso.db import has_sqlite_vec
        if not has_sqlite_vec(self.conn):
            return
        # Zero vector when no embedder is wired.
        vec = [0.0] * EMBEDDING_DIM if self.embedder is None else list(self.embedder.encode([text])[0])
        self.conn.execute(
            "INSERT OR REPLACE INTO summary_vectors(note_id, embedding) VALUES (?, ?)",
            (note_id, json.dumps(vec)),
        )

    def all_for_course(self, course_id: str) -> list[Summary]:
        rows = self.conn.execute(
            """
            SELECT note_id, course_id, topic_line, gist, processing_order, timestamp
            FROM summaries
            WHERE course_id = ?
            ORDER BY processing_order
            """,
            (course_id,),
        ).fetchall()
        return [
            Summary(
                note_id=r["note_id"],
                course_id=r["course_id"],
                topic_line=r["topic_line"],
                gist=r["gist"],
                processing_order=r["processing_order"],
                pipeline_version="v1.0.0",
                timestamp=datetime.fromisoformat(r["timestamp"]),
            )
            for r in rows
        ]

    def bm25_corpus(self, course_id: str) -> list[tuple[str, str]]:
        rows = self.conn.execute(
            "SELECT note_id, bm25_text FROM summaries WHERE course_id = ? "
            "ORDER BY processing_order",
            (course_id,),
        ).fetchall()
        return [(r["note_id"], r["bm25_text"]) for r in rows]

    def count_for_course(self, course_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM summaries WHERE course_id = ?",
            (course_id,),
        ).fetchone()
        return int(row["n"]) if row else 0
