"""Per-course token lexicon — the cache's OCR read-path.

Two sides:
- WRITE: `harvest()` records sightings of distinctive tokens from each
  LLM-extracted note. `promote_pending()` admits a sighting to the lexicon
  once it has recurred enough times and is not a common-English word.
- READ:  `correct()` walks an OCR result, finds low-confidence tokens, and
  fuzzy-matches them against the per-course lexicon using a shape-aware edit
  distance. Confidence is soft-reweighted toward strong matches — bounded so
  the lexicon can tip a close call but cannot override a confident OCR reading.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from typing import Iterable

from miso.config import LexiconConfig
from miso.types import (
    CorrectedOCR, ExtractedNote, LexiconCorrection, OCRResult, OCRWord,
)

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Bigram and single-char substitution costs for OCR-typical confusions.
# Costs below 1.0 mean "we expect this swap in handwriting/print" — Levenshtein
# treats them as cheaper than arbitrary substitutions.
_SHAPE_CONFUSIONS: dict[tuple[str, str], float] = {
    ("rn", "m"): 0.3, ("m", "rn"): 0.3,
    ("cl", "d"): 0.3, ("d", "cl"): 0.3,
    ("l", "1"): 0.2, ("1", "l"): 0.2,
    ("l", "I"): 0.2, ("I", "l"): 0.2,
    ("0", "O"): 0.2, ("O", "0"): 0.2,
    ("o", "0"): 0.3, ("0", "o"): 0.3,
    ("e", "c"): 0.4, ("c", "e"): 0.4,
    ("u", "v"): 0.4, ("v", "u"): 0.4,
    ("nn", "m"): 0.4, ("m", "nn"): 0.4,
}


def _shape_aware_distance(a: str, b: str) -> float:
    """Edit distance with shape-aware substitution costs, including bigrams."""
    n, m = len(a), len(b)
    if n == 0:
        return float(m)
    if m == 0:
        return float(n)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = float(i)
    for j in range(m + 1):
        dp[0][j] = float(j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                sub_cost = 0.0
            else:
                sub_cost = _SHAPE_CONFUSIONS.get((a[i - 1], b[j - 1]), 1.0)
            best = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + sub_cost,
            )
            # Allow 2:1 swaps like "rn"→"m" and 1:2 like "m"→"rn".
            if i >= 2 and j >= 1:
                bg = _SHAPE_CONFUSIONS.get((a[i - 2:i], b[j - 1]))
                if bg is not None:
                    best = min(best, dp[i - 2][j - 1] + bg)
            if i >= 1 and j >= 2:
                bg = _SHAPE_CONFUSIONS.get((a[i - 1], b[j - 2:j]))
                if bg is not None:
                    best = min(best, dp[i - 1][j - 2] + bg)
            dp[i][j] = best
    return dp[n][m]


class LexiconLayer:
    def __init__(self, conn: sqlite3.Connection, *, common_words: set[str] | None = None):
        """`common_words` are filtered from admission. Defaults to wordfreq's top-N
        via `miso.wordlists.load_common_words()`; pass a SCOWL-derived set instead
        for production-grade filtering.
        """
        self.conn = conn
        if common_words is None:
            from miso.wordlists import load_common_words
            common_words = load_common_words()
        self.common_words = common_words

    def correct(
        self,
        ocr: OCRResult,
        course_id: str,
        cfg: LexiconConfig,
    ) -> CorrectedOCR:
        terms = self._load_course_terms(course_id)
        corrections: list[LexiconCorrection] = []
        new_words: list[OCRWord] = list(ocr.words)
        touched: list[str] = []

        for i, w in enumerate(ocr.words):
            if w.confidence >= cfg.confidence_threshold:
                continue
            best_term, best_dist = self._best_match(w.text, terms, cfg.max_edit_distance)
            if best_term is None:
                continue
            strength = max(0.0, 1.0 - best_dist / max(cfg.max_edit_distance, 1e-6))
            new_conf = min(1.0, w.confidence + cfg.boost_magnitude * strength)
            new_words[i] = replace(w, text=best_term, confidence=new_conf)
            corrections.append(LexiconCorrection(
                token_index=i,
                original=w.text,
                suggested=best_term,
                match_strength=strength,
                ocr_confidence=w.confidence,
                accepted=True,
            ))
            touched.append(best_term)

        return CorrectedOCR(
            words=new_words,
            corrected_text=" ".join(w.text for w in new_words),
            corrections=corrections,
            touched_terms=touched,
        )

    @staticmethod
    def _best_match(
        token: str,
        terms: Iterable[str],
        d_max: float,
    ) -> tuple[str | None, float]:
        best, best_d = None, d_max + 1.0
        token_lower = token.lower()
        for t in terms:
            d = _shape_aware_distance(token_lower, t.lower())
            if d < best_d:
                best_d = d
                best = t
        if best is None or best_d > d_max:
            return None, best_d
        return best, best_d

    def harvest(self, extracted: ExtractedNote, course_id: str) -> None:
        """Record term sightings from the extraction. Promotion happens separately
        in `promote_pending()` so the N-recurrence knob can be swept at read time.
        """
        for term in self._candidates_from_extraction(extracted):
            if term.lower() in self.common_words:
                continue
            self._record_sighting(course_id, term, _now_iso())

    def _record_sighting(self, course_id: str, term: str, now: str) -> None:
        self.conn.execute(
            """
            INSERT INTO lexicon_sightings(course_id, term, first_seen, last_seen,
                                          sighting_count, context_snippet)
            VALUES (?, ?, ?, ?, 1, NULL)
            ON CONFLICT(course_id, term) DO UPDATE SET
                sighting_count = sighting_count + 1,
                last_seen = excluded.last_seen
            """,
            (course_id, term, now, now),
        )
        self.conn.commit()

    def _candidates_from_extraction(self, extracted: ExtractedNote) -> list[str]:
        strings: list[str] = []
        self._walk_strings(extracted.structured_json, strings)
        tokens: list[str] = []
        for s in strings:
            tokens.extend(re.findall(r"[A-Za-z][A-Za-z0-9_\-]*", s))
        return tokens

    def _walk_strings(self, value, out: list[str]) -> None:
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                self._walk_strings(v, out)
        elif isinstance(value, (list, tuple)):
            for v in value:
                self._walk_strings(v, out)

    def size(self, course_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM lexicon_terms WHERE course_id = ?",
            (course_id,),
        ).fetchone()
        return int(row["n"]) if row else 0

    def _load_course_terms(self, course_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT term FROM lexicon_terms WHERE course_id = ?",
            (course_id,),
        ).fetchall()
        return [r["term"] for r in rows]

    def promote_pending(self, course_id: str, n_recurrence: int) -> int:
        """Move sightings past the recurrence threshold into the lexicon. Returns
        the number of newly-admitted terms.
        """
        now = _now_iso()
        cur = self.conn.execute(
            """
            SELECT course_id, term, first_seen, last_seen, sighting_count, context_snippet
            FROM lexicon_sightings
            WHERE course_id = ? AND sighting_count >= ?
            """,
            (course_id, n_recurrence),
        )
        promoted = 0
        for row in cur.fetchall():
            try:
                self.conn.execute(
                    """
                    INSERT INTO lexicon_terms(course_id, term, frequency,
                                              context_snippet, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (row["course_id"], row["term"], row["sighting_count"],
                     row["context_snippet"], row["first_seen"], row["last_seen"]),
                )
                promoted += 1
            except sqlite3.IntegrityError:
                self.conn.execute(
                    """
                    UPDATE lexicon_terms
                    SET frequency = ?, last_seen = ?
                    WHERE course_id = ? AND term = ?
                    """,
                    (row["sighting_count"], now, row["course_id"], row["term"]),
                )
        self.conn.commit()
        return promoted
