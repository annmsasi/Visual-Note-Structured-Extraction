"""Course-distinctive recurring vocabulary — the terms the cache targets.

A term is distinctive if it is (a) not general-English (per the lexicon's own
common-words filter) and (b) recurs across ≥ N notes of the course. This mirrors
the lexicon's admission rule (cache_design_v1.md §4.2) so the term metrics score
exactly the vocabulary the lexicon is built to capture.
"""
from __future__ import annotations

import re
from collections import Counter

from miso.wordlists import load_common_words

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]{3,}")


def course_distinctive_terms(
    note_bodies: list[str],
    min_recurrence: int = 2,
    common: set[str] | None = None,
) -> tuple[set[str], list[list[str]]]:
    """Returns (course term set, per-note present-terms). `min_recurrence` counts
    the number of notes a term appears in (presence), matching the lexicon's N."""
    common = common if common is not None else {w.lower() for w in load_common_words()}
    per_note_sets: list[set[str]] = []
    counts: Counter[str] = Counter()
    for body in note_bodies:
        toks = {m.group(0).lower() for m in _WORD.finditer(body)}
        toks = {w for w in toks if w not in common}
        per_note_sets.append(toks)
        counts.update(toks)
    distinctive = {w for w, c in counts.items() if c >= min_recurrence}
    per_note = [sorted(s & distinctive) for s in per_note_sets]
    return distinctive, per_note
