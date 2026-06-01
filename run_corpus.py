"""Run the full pipeline over a corpus directory as one course and report cache warm-up.

    python run_corpus.py corpora/biology --course biology
    python run_corpus.py corpora/math --course math --limit 15
    python run_corpus.py corpora/biology --course biology --config nocache
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

from miso.config import RunConfig
from miso.replay import _configure_logging, _load_env, _prepare_image, run
from miso.types import Note

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def build_notes(corpus_dir: str, course: str, limit: int | None) -> list[Note]:
    imgs = sorted(p for p in Path(corpus_dir).glob("*")
                  if p.suffix.lower() in _IMG_EXTS and ".prepared" not in p.name)
    if limit:
        imgs = imgs[:limit]
    base = datetime(2026, 1, 1)
    return [
        Note(note_id=f"{course}-{i:03d}", course_id=course,
             image_path=_prepare_image(img), processing_order=i,
             timestamp=base + timedelta(days=i))
        for i, img in enumerate(imgs)
    ]


def main(argv: list[str] | None = None) -> int:
    _load_env()
    _configure_logging()
    ap = argparse.ArgumentParser(description="Run the full pipeline over a corpus as one course.")
    ap.add_argument("corpus_dir")
    ap.add_argument("--course", required=True)
    ap.add_argument("--config", choices=["full", "lexicon", "nocache"], default="full")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cold-start", type=int, default=None,
                    help="override retrieval cold_start_note_count")
    args = ap.parse_args(argv)

    notes = build_notes(args.corpus_dir, args.course, args.limit)
    if not notes:
        print("No images found.")
        return 1

    if args.config == "full":
        cfg = RunConfig.config_6_full(tag=f"{args.course}_full")
    elif args.config == "lexicon":
        cfg = RunConfig.config_4_lexicon_only(tag=f"{args.course}_lexicon")
    else:
        cfg = RunConfig.config_3_llm_ocr_only(tag=f"{args.course}_nocache")
    cfg.ocr.engine = "azure"
    cfg.extraction.model_id = args.model
    if args.cold_start is not None:
        cfg.retrieval.cold_start_note_count = args.cold_start
    cfg.cache_path = Path(f"./cache_{args.course}_{args.config}.db")
    cfg.traces_dir = Path("./runs")

    print(f"Running {len(notes)} notes | course={args.course} | config={args.config} | model={args.model}")
    run_dir = run(cfg, notes)

    records = [json.loads(l) for l in (run_dir / "trace.jsonl").read_text().splitlines()]
    print("\n===== per-note cache activation =====")
    print(f'{"note":14}{"lex_size":>9}{"corrections":>12}{"glossary":>9}{"retr_inj":>9}{"cold":>6}')
    for r in records:
        corr = r.get("corrected_ocr") or {}
        retr = r.get("retrieval") or {}
        gate = r.get("gate") or {}
        print(f'{r["note_id"]:14}{r.get("lexicon_size_at_time", 0):>9}'
              f'{len(corr.get("corrections") or []):>12}{len(r.get("glossary_to_llm") or []):>9}'
              f'{len(retr.get("injected") or []):>9}{str(gate.get("cold_start_skip", "")):>6}')

    if args.config == "full":
        conn = sqlite3.connect(str(cfg.cache_path))
        conn.row_factory = sqlite3.Row
        terms = [r["term"] for r in conn.execute(
            "SELECT term FROM lexicon_terms WHERE course_id=? ORDER BY frequency DESC", (args.course,))]
        conn.close()
        print(f"\nPromoted lexicon terms ({len(terms)}): {', '.join(terms[:40])}")
    print(f"\nTraces: {run_dir / 'trace.jsonl'}   Cache: {cfg.cache_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
