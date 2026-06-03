"""Run the C3-C6 ablation grid over a real corpus (Azure OCR + Claude extraction),
then print the eval report (term-recall, term CER, 2x2 attribution, ramp curves).

    python -m miso.eval.run_grid corpora/biology --course biology \
        --gold corpora/biology_gold --model claude-sonnet-4-6 --limit 8

Each config is a separate run (own cache, from empty), processed in chronological
filename order, so the cache warms up across the note sequence. OCR + extraction
run once per config (the ablation requires it).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from miso.config import RunConfig
from miso.eval import cli as eval_cli
from miso.eval.ocr_runner import _load_env  # reads miso/.env explicitly
from miso.replay import _configure_logging, _prepare_image, run
from miso.types import Note

_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

_CONFIGS = [
    RunConfig.config_3_llm_ocr_only,
    RunConfig.config_4_lexicon_only,
    RunConfig.config_5_retrieval_only,
    RunConfig.config_6_full,
]


def build_notes(corpus_dir: str, course: str, limit: int | None) -> list[Note]:
    imgs = sorted(p for p in Path(corpus_dir).glob("*")
                  if p.suffix.lower() in _EXTS and ".prepared" not in p.name)
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
    _configure_logging()
    _load_env()
    ap = argparse.ArgumentParser(description="Run the C3-C6 ablation grid over a real corpus.")
    ap.add_argument("corpus_dir")
    ap.add_argument("--course", required=True)
    ap.add_argument("--gold", required=True, help="gold dir (GoldNote JSON per note)")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cold-start", type=int, default=2)
    args = ap.parse_args(argv)

    notes = build_notes(args.corpus_dir, args.course, args.limit)
    if not notes:
        print("No images found.", file=sys.stderr)
        return 1

    run_dirs: list[str] = []
    for factory in _CONFIGS:
        cfg = factory()  # canonical tags C3_*/C4_*/C5_*/C6_* (the analyze 2x2 keys)
        cfg.ocr.engine = "azure"
        cfg.extraction.model_id = args.model
        cfg.retrieval.cold_start_note_count = args.cold_start
        cfg.cache_path = Path(f"./cache_{args.course}_{cfg.config_tag}.db")
        cfg.traces_dir = Path("runs")
        print(f"\n=== running {cfg.config_tag} ({len(notes)} notes, model={args.model}) ===")
        run_dir = run(cfg, notes)
        run_dirs.append(str(run_dir))
        print(f"{cfg.config_tag}: {run_dir}")

    print("\n" + "=" * 60)
    return eval_cli.main(["analyze", "--runs", *run_dirs, "--gold", args.gold])


if __name__ == "__main__":
    sys.exit(main())
