"""Common-words filter used by the lexicon to skip generic English vocabulary.

Primary source is `wordfreq.top_n_list("en", N)`. If wordfreq isn't installed,
falls back to a small embedded stop-word set. `/usr/share/dict/words` is *not*
included by default — it's an "is this an English word at all" list and would
filter out distinctive technical terms (e.g. "eigenvector") that the lexicon
exists to capture.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)


BUILTIN_FALLBACK: frozenset[str] = frozenset({
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but",
    "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will",
    "my", "one", "all", "would", "there", "their", "what", "so", "up", "out",
    "if", "about", "who", "get", "which", "go", "me", "when", "make", "can",
    "like", "time", "no", "just", "him", "know", "take", "people", "into",
    "year", "your", "good", "some", "could", "them", "see", "other", "than",
    "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well",
    "way", "even", "new", "want", "any", "these", "give", "day", "most", "us",
    "is", "are", "was", "were", "been", "being", "has", "had", "did",
})


DEFAULT_TOP_N = 5000


def load_common_words(
    *,
    top_n: int = DEFAULT_TOP_N,
    extra_paths: list[Path] | None = None,
    include_system_words: bool = False,
) -> set[str]:
    out: set[str] = set()
    wf = _wordfreq_top_n(top_n)
    if wf is not None:
        out |= wf
        log.info("loaded %d common words from wordfreq (top %d)", len(wf), top_n)
    else:
        out |= {w.lower() for w in BUILTIN_FALLBACK}
        log.warning(
            "wordfreq not available; using BUILTIN_FALLBACK (%d words). "
            "Install with `pip install wordfreq` for a real filter.",
            len(BUILTIN_FALLBACK),
        )

    if include_system_words:
        sys_path = Path("/usr/share/dict/words")
        if sys_path.exists():
            try:
                out |= _read_wordlist(sys_path)
            except Exception as e:
                log.warning("could not read %s: %s", sys_path, e)

    for p in extra_paths or []:
        if p.exists():
            try:
                out |= _read_wordlist(p)
            except Exception as e:
                log.warning("could not read %s: %s", p, e)

    return out


@lru_cache(maxsize=4)
def _wordfreq_top_n(n: int) -> frozenset[str] | None:
    try:
        from wordfreq import top_n_list
    except ImportError:
        return None
    return frozenset(w.lower() for w in top_n_list("en", n))


def _read_wordlist(path: Path) -> set[str]:
    out: set[str] = set()
    for line in path.read_text(errors="ignore").splitlines():
        w = line.strip().lower()
        if w and not w.startswith("#"):
            out.add(w)
    return out
