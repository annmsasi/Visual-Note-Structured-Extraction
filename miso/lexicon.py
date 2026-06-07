"""Per-course token lexicon for OCR correction."""
from __future__ import annotations

import logging
import math
import re
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone

from miso.config import LexiconConfig
from miso.types import (
    Candidate, CorrectedOCR, ExtractedNote, LexiconCorrection,
    OCRResult, OCRWord, WordFlag,
)

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Bigram and single-char substitution costs for OCR-typical confusions.
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
            # 2:1 and 1:2 bigram swaps
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
        """Words in `common_words` are filtered from admission."""
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
        if cfg.mode == "off":
            return self._passthrough(ocr)
        vocab = self._load_course_vocab(course_id)
        if not vocab:
            # Cold/empty course lexicon: nothing to match against.
            return self._passthrough(ocr)
        if cfg.mode == "replace":
            return self._correct_replace(ocr, vocab, cfg)
        return self._flag(ocr, vocab, cfg)

    @staticmethod
    def _passthrough(ocr: OCRResult) -> CorrectedOCR:
        return CorrectedOCR(
            words=list(ocr.words),
            corrected_text=ocr.raw_text,
            corrections=[],
            touched_terms=[],
            layout_text=ocr.layout_text,
            flags=[],
        )

    def _flag(
        self, ocr: OCRResult, vocab: list[tuple[str, int]], cfg: LexiconConfig,
    ) -> CorrectedOCR:
        """Annotate uncertain words with candidate terms; never mutate the OCR.

        The LLM, which also has the page image, makes the final call.
        """
        flags: list[WordFlag] = []
        touched: list[str] = []
        for i, w in enumerate(ocr.words):
            if w.confidence >= cfg.search_ceiling:
                continue
            cands = self._candidates(w.text, w.confidence, vocab, cfg)
            if not cands:
                continue
            flags.append(WordFlag(
                token_index=i,
                original=w.text,
                confidence=w.confidence,
                candidates=cands,
            ))
            touched.extend(c.term for c in cands)
        return CorrectedOCR(
            words=list(ocr.words),
            corrected_text=ocr.raw_text,
            corrections=[],
            touched_terms=touched,
            layout_text=ocr.layout_text,
            flags=flags,
        )

    def _correct_replace(
        self, ocr: OCRResult, vocab: list[tuple[str, int]], cfg: LexiconConfig,
    ) -> CorrectedOCR:
        """Ablation arm: hard-swap the top candidate before the LLM sees it."""
        new_words: list[OCRWord] = list(ocr.words)
        corrections: list[LexiconCorrection] = []
        touched: list[str] = []
        for i, w in enumerate(ocr.words):
            if w.confidence >= cfg.confidence_threshold:
                continue
            cands = self._candidates(w.text, w.confidence, vocab, cfg)
            if not cands:
                continue
            best = cands[0]
            strength = max(0.0, 1.0 - best.distance)
            new_conf = min(1.0, w.confidence + cfg.boost_magnitude * strength)
            new_words[i] = replace(w, text=best.term, confidence=new_conf)
            corrections.append(LexiconCorrection(
                token_index=i,
                original=w.text,
                suggested=best.term,
                match_strength=strength,
                ocr_confidence=w.confidence,
                accepted=True,
            ))
            touched.append(best.term)

        from miso.layout import render_layout_text
        return CorrectedOCR(
            words=new_words,
            corrected_text=" ".join(w.text for w in new_words),
            corrections=corrections,
            touched_terms=touched,
            layout_text=render_layout_text(new_words),
            flags=[],
        )

    def _candidates(
        self, word: str, confidence: float,
        vocab: list[tuple[str, int]], cfg: LexiconConfig,
    ) -> list[Candidate]:
        """Top course terms for an OCR word, ranked by a bounded relevance score.

            relevance = (1 - confidence)                  # suspicion        (<= 1)
                        * exp(-distance_decay * d_norm)    # OCR likelihood   (<= 1)
                        * count / (count + freq_prior_k)   # frequency prior  (<= 1)

        Every factor is <= 1, so frequency can only down-weight a candidate and
        the likelihood term makes distance dominate the tail — no hard edit band
        is needed. The saturating prior is robust to Zipfian skew, so a rare but
        shape-close term is never crushed below the floor. Selection is a single
        absolute floor (relevance_floor) plus an optional relative gate (keep
        candidates within relative_gate of the best).
        """
        wl = word.lower()
        suspicion = 1.0 - confidence
        out: list[Candidate] = []
        for term, freq in vocab:
            tl = term.lower()
            if tl == wl:
                continue
            d_norm = _shape_aware_distance(wl, tl) / max(len(wl), len(tl), 1)
            prior = freq / (freq + cfg.freq_prior_k)
            relevance = suspicion * math.exp(-cfg.distance_decay * d_norm) * prior
            if relevance < cfg.relevance_floor:
                continue
            out.append(Candidate(
                term=term, distance=d_norm, frequency=freq, relevance=relevance,
            ))
        out.sort(key=lambda c: -c.relevance)
        if out and cfg.relative_gate > 0.0:
            cut = cfg.relative_gate * out[0].relevance
            out = [c for c in out if c.relevance >= cut]
        return out[: cfg.max_candidates]

    def harvest(self, extracted: ExtractedNote, course_id: str) -> None:
        """Record term sightings from the extraction."""
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

    def _load_course_vocab(self, course_id: str) -> list[tuple[str, int]]:
        rows = self.conn.execute(
            "SELECT term, frequency FROM lexicon_terms WHERE course_id = ?",
            (course_id,),
        ).fetchall()
        return [(r["term"], int(r["frequency"])) for r in rows]

    def promote_pending(self, course_id: str, n_recurrence: int) -> int:
        """Move sightings past the recurrence threshold into the lexicon, returning the count admitted."""
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
