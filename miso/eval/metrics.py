"""Metric functions: error rates, structural F1, correction stats, bootstrap CI."""
from __future__ import annotations

import random
from typing import Iterable, Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance over any sequence."""
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


def structural_f1(reference_json, hypothesis_json) -> float:
    """F1 over dotted-path keys present in each JSON tree."""
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
) -> tuple[float, float, float]:
    """Return (precision, recall, over_correction_rate) for accepted corrections."""
    if not corrections:
        return (0.0, 0.0, 0.0)

    gold_tokens = set(gold_text.lower().split())
    helped = 0
    hurt = 0
    for c in corrections:
        if not c.get("accepted"):
            continue
        orig = c.get("original", "").lower()
        sug = c.get("suggested", "").lower()
        in_gold_orig = orig in gold_tokens
        in_gold_sug = sug in gold_tokens
        if in_gold_sug and not in_gold_orig:
            helped += 1
        elif in_gold_orig and not in_gold_sug:
            hurt += 1

    decisive = helped + hurt
    precision = helped / decisive if decisive else 0.0
    over_correction = hurt / decisive if decisive else 0.0

    raw_tokens = raw_text.lower().split()
    gold_list = list(gold_tokens)
    fixable = sum(
        1
        for r in raw_tokens
        if r not in gold_tokens and any(_close(r, w) for w in gold_list)
    )
    recall = helped / fixable if fixable else 0.0
    return (precision, recall, over_correction)


def _close(a: str, b: str, max_diff: int = 2) -> bool:
    if abs(len(a) - len(b)) > max_diff:
        return False
    return levenshtein(a, b) <= max_diff


def bootstrap_ci(
    values: Iterable[float],
    confidence: float = 0.95,
    n_samples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap of the mean. Return (mean, lower, upper)."""
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
