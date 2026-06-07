"""Evaluate OCR engines on their own, decoupled from the LLM grid.

Runs each engine over the corpus (cached via run_ocr_dir, so we never re-bill),
then scores the recognizer's WORD CHOICE against gold — normalized so spacing,
case, line breaks, and punctuation don't count. The note's layout/formatting is
irrelevant here; only which words it recovered matters. Reports a per-engine ladder.

    python -m miso.eval.ocr_eval corpora/tim172a --course tim172a \
        --gold corpora/tim172a_gold --engines azure tesseract paddle

Two metrics, both formatting-insensitive and spelling-strict (so real misreads
still count):
  * word-choice WER : normalized word error rate vs the verbatim transcription.
  * term-recall     : fraction of the gold distinctive terms the OCR recovered.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from miso.eval.gold import GoldNote, load_gold
from miso.eval.metrics import normalized_wer, term_recall
from miso.eval.ocr_runner import run_ocr_dir

_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _images(corpus_dir: Path) -> list[Path]:
    """Sorted corpus pages, matching build_notes / run_ocr_dir selection exactly."""
    return sorted(p for p in corpus_dir.glob("*")
                  if p.suffix.lower() in _EXTS and ".prepared" not in p.name)


def score_pages(
    imgs: list[Path], course: str,
    ocr_by_stem: dict[str, dict], gold: dict[str, GoldNote],
) -> dict:
    """Average word-choice scores over the pages, aligned to gold by the same
    sequential note_id the grid uses (sorted images -> ``{course}-{i:03d}``)."""
    wers: list[float] = []
    recalls: list[float] = []
    n = 0
    for i, img in enumerate(imgs):
        g = gold.get(f"{course}-{i:03d}")
        if g is None:
            continue
        text = (ocr_by_stem.get(img.stem) or {}).get("text", "")
        wers.append(normalized_wer(g.transcription, text))
        tr = term_recall(g.distinctive_terms, text)
        if tr is not None:
            recalls.append(tr)
        n += 1
    return {
        "n": n,
        "word_wer": (sum(wers) / len(wers)) if wers else None,
        "term_recall": (sum(recalls) / len(recalls)) if recalls else None,
    }


def score_engine(corpus_dir: Path, course: str, engine: str,
                 gold: dict[str, GoldNote]) -> dict:
    imgs = _images(corpus_dir)
    ocr = run_ocr_dir(corpus_dir, engine=engine)  # {stem: {"text", "words"}}, cached
    return {"engine": engine, **score_pages(imgs, course, ocr, gold)}


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate OCR engines on their own — word choice vs gold, "
                    "normalized so spacing/formatting doesn't count.")
    ap.add_argument("corpus_dir")
    ap.add_argument("--course", required=True)
    ap.add_argument("--gold", required=True, help="gold dir (GoldNote JSON per note)")
    ap.add_argument("--engines", nargs="+", default=["azure"],
                    choices=["stub", "azure", "paddle", "tesseract"],
                    help="OCR ladder to compare, strong -> weak")
    args = ap.parse_args(argv)

    corpus = Path(args.corpus_dir)
    gold = load_gold(Path(args.gold))
    if not gold:
        print(f"No gold found in {args.gold}", file=sys.stderr)
        return 1

    rows = [score_engine(corpus, args.course, e, gold) for e in args.engines]
    print(f"\nOCR-only eval — {len(gold)} gold notes; word choice only "
          f"(case / spacing / line breaks / punctuation ignored)\n")
    print("| engine | n | word-choice WER | term-recall |")
    print("|---|---:|---:|---:|")
    for r in rows:
        print(f"| {r['engine']} | {r['n']} | {_fmt(r['word_wer'])} | {_fmt(r['term_recall'])} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
