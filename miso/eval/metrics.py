"""Pure metric functions — no I/O, no module-level state.

Character / word error rate, structural F1 over JSON paths, lexicon
precision/recall/over-correction, and a percentile-bootstrap CI helper.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict, deque
from typing import Iterable, Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance over any sequence (characters or token lists)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def align_tokens(a: Sequence, b: Sequence) -> list[tuple]:
    """Needleman-Wunsch token alignment (unit costs). Returns ordered
    (a_tok | None, b_tok | None) pairs — None marks an insertion/deletion gap.
    Shared by the term and correction metrics so they judge by *position*.
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            out.append((a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            out.append((a[i - 1], None)); i -= 1
        else:
            out.append((None, b[j - 1])); j -= 1
    out.reverse()
    return out


_PUNCT = ".,;:!?()[]{}\"'`"


def _norm_token(t: str) -> str:
    """Lowercase + strip surrounding punctuation — for token comparison/alignment."""
    return t.lower().strip(_PUNCT)


def _norm_phrase(s: str) -> str:
    """Lowercase, non-alphanumerics → single spaces, collapsed. For term matching."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", s.lower()).split())


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate. May exceed 1.0 when the hypothesis is longer."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def normalized_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate on normalized tokens (lowercased, surrounding punctuation
    stripped, empties dropped). Spacing, case, line breaks, and punctuation are
    collapsed away, so this scores WORD CHOICE only — the right lens for judging a
    recognizer (or its note) independently of how it laid the text out. Stays strict
    on spelling, like term_recall, so genuine OCR misreads still count."""
    ref = [t for t in (_norm_token(w) for w in reference.split()) if t]
    hyp = [t for t in (_norm_token(w) for w in hypothesis.split()) if t]
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def _stem(tok: str) -> str:
    """Tiny plural stemmer: regular noun plurals collapse to the singular
    (eigenvector(s), theorem(s), box/boxes, theory/theories), but misspellings do
    NOT — `eigenvecter` stays distinct from `eigenvector`. That distinction is the
    whole point: term-recall must keep SEEING the OCR errors the cache fixes, so
    edit-distance fuzzing (which would forgive them and flatten the cache's benefit)
    is deliberately avoided. Crude by design; irregular plurals (matrix/matrices)
    still need a curated variant in the gold term list.
    """
    if len(tok) <= 3 or tok.endswith("ss"):
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith(("ses", "xes", "zes", "ches", "shes")):
        return tok[:-2]
    if tok.endswith("s"):
        return tok[:-1]
    return tok


def _stem_phrase(s: str) -> str:
    return " ".join(_stem(w) for w in _norm_phrase(s).split())


def term_recall(terms: Iterable[str], text: str) -> float | None:
    """Fraction of distinct distinctive terms recovered in `text`. Returns None when
    there are no terms, so callers can exclude the note from the average.

    Matching is case-insensitive, phrase-boundary aware, and plural-normalised (via a
    light stemmer) so the LLM's legitimate morphology counts — but it stays STRICT on
    spelling, so the OCR errors the cache is meant to fix still register as misses
    (edit-distance fuzzing would forgive them and hide the cache's benefit). This is
    the END-TO-END headline metric: the number that should move when the cache helps.
    """
    norm_terms = {_stem_phrase(t) for t in terms}
    norm_terms.discard("")
    if not norm_terms:
        return None
    hay = f" {_stem_phrase(text)} "
    hits = sum(1 for t in norm_terms if f" {t} " in hay)
    return hits / len(norm_terms)


def term_restricted_cer(reference: str, hypothesis: str, terms: Iterable[str]) -> float | None:
    """CER restricted to occurrences of `terms` in the reference, token-aligned to
    the hypothesis. Returns None if no term occurs in the reference.

    This is the INTRINSIC lexicon metric: score the OCR-stage text against a
    verbatim reference, over only the distinctive-term spans. Global CER hides the
    ~5% of tokens the lexicon actually moves; this exposes them.
    """
    ref = [_norm_token(t) for t in reference.split()]
    hyp = [_norm_token(t) for t in hypothesis.split()]
    seqs = [[_norm_token(w) for w in t.split()] for t in terms]
    seqs = [s for s in seqs if s and all(s)]
    covered: set[int] = set()
    for s in seqs:
        L = len(s)
        for i in range(len(ref) - L + 1):
            if ref[i:i + L] == s:
                covered.update(range(i, i + L))
    if not covered:
        return None
    ri, total, dist = -1, 0, 0
    for r, h in align_tokens(ref, hyp):
        if r is None:
            continue
        ri += 1
        if ri in covered:
            total += len(r)
            dist += levenshtein(r, h or "")
    return dist / total if total else None


def structural_f1(reference_json, hypothesis_json) -> float:
    """F1 over dotted-path keys present in each JSON tree. Captures whether the
    extraction recovered the right shape (headings, sections, code blocks).
    """
    ref_keys = _collect_paths(reference_json)
    hyp_keys = _collect_paths(hypothesis_json)
    if not ref_keys and not hyp_keys:
        return 1.0
    tp = len(ref_keys & hyp_keys)
    fp = len(hyp_keys - ref_keys)
    fn = len(ref_keys - hyp_keys)
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)


def _collect_paths(value, prefix: str = "") -> set[str]:
    out: set[str] = set()
    if isinstance(value, dict):
        for k, v in value.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            out.add(path)
            out |= _collect_paths(v, path)
    elif isinstance(value, list):
        path = f"{prefix}[]" if prefix else "[]"
        out.add(path)
        for v in value:
            out |= _collect_paths(v, path)
    return out


def correction_precision_recall(
    corrections: list[dict],
    gold_text: str,
    raw_text: str,
    max_diff: int = 2,
) -> tuple[float, float, float]:
    """Returns (precision, recall, over_correction_rate), alignment-based.

    Raw OCR tokens are token-aligned to the gold, so each correction is judged
    against the gold token at *its own position* — not mere set membership, which
    the earlier version used and which over-credited a suggestion that happened to
    appear somewhere else in the gold. A correction "helped" if it changed the
    token to its aligned gold token; "hurt" if it changed an already-correct token
    away from gold. Recall's denominator is raw tokens aligned to a different but
    close (≤max_diff) gold token — an approximation of "fixable" errors.
    """
    if not corrections:
        return (0.0, 0.0, 0.0)

    gold_tokens = [_norm_token(t) for t in gold_text.split()]
    raw_tokens = [_norm_token(t) for t in raw_text.split()]
    gold_set = set(gold_tokens)
    pairs = align_tokens(raw_tokens, gold_tokens)

    aligned: dict[str, deque] = defaultdict(deque)
    for r, g in pairs:
        if r is not None:
            aligned[r].append(g)  # g may be None (raw token with no gold counterpart)

    _MISSING = object()
    helped = hurt = 0
    for c in corrections:
        if not c.get("accepted"):
            continue
        orig = _norm_token(c.get("original", ""))
        sug = _norm_token(c.get("suggested", ""))
        g = aligned[orig].popleft() if aligned[orig] else _MISSING
        if g is _MISSING or g is None:
            # No aligned gold token (inserted/garbage OCR) — fall back to membership.
            in_o, in_s = orig in gold_set, sug in gold_set
            if in_s and not in_o:
                helped += 1
            elif in_o and not in_s:
                hurt += 1
            continue
        if sug == g and orig != g:
            helped += 1
        elif orig == g and sug != g:
            hurt += 1

    decisive = helped + hurt
    precision = helped / decisive if decisive else 0.0
    over_correction = hurt / decisive if decisive else 0.0
    fixable = sum(
        1 for r, g in pairs
        if r is not None and g is not None and r != g and levenshtein(r, g) <= max_diff
    )
    recall = helped / fixable if fixable else 0.0
    return (precision, recall, over_correction)


def bootstrap_ci(
    values: Iterable[float],
    confidence: float = 0.95,
    n_samples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap of the mean. Returns (mean, lower, upper).

    Designed for per-note delta vectors. With ~60 notes the CI will be wide;
    that's the honest treatment of the under-power risk.
    """
    vals = list(values)
    if not vals:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(vals)
    means: list[float] = []
    for _ in range(n_samples):
        s = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = means[int(alpha * n_samples)]
    hi = means[int((1.0 - alpha) * n_samples)]
    mean = sum(vals) / n
    return (mean, lo, hi)
