"""Compose Teklia/IAM-line samples into page-like real-handwriting calibration
images + transcription gold (Arm C). Streamed — no full-dataset download.

    python -m miso.synth.iam --pages 8 --lines 12
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

_W, _MARGIN, _GAP, _BG = 1700, 90, 26, (252, 250, 246)


def _text_of(row: dict) -> str:
    for k in ("text", "transcription", "label", "sentence", "gt"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def build_iam(out: str = "corpora", name: str = "iam", n_pages: int = 8,
              lines_per_page: int = 12, split: str = "test") -> int:
    """Stack IAM lines into page images; gold transcription = the lines joined."""
    from datasets import load_dataset

    ds = iter(load_dataset("Teklia/IAM-line", split=split, streaming=True))
    img_dir = Path(out) / name
    gold_dir = Path(out) / f"{name}_gold"
    img_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    budget = _W - 2 * _MARGIN
    for _ in range(n_pages):
        scaled: list[tuple[Image.Image, str]] = []
        while len(scaled) < lines_per_page:
            row = next(ds, None)
            if row is None:
                break
            im = row.get("image")
            txt = _text_of(row)
            if im is None or not txt:
                continue
            if not isinstance(im, Image.Image):
                im = Image.fromarray(im)
            im = im.convert("RGB")
            s = min(1.0, budget / im.width)
            scaled.append((im.resize((int(im.width * s), int(im.height * s))), txt))
        if not scaled:
            break
        h = _MARGIN * 2 + sum(im.height for im, _ in scaled) + _GAP * (len(scaled) - 1)
        canvas = Image.new("RGB", (_W, h), _BG)
        y = _MARGIN
        for im, _ in scaled:
            canvas.paste(im, (_MARGIN, y))
            y += im.height + _GAP
        nid = f"{name}-{made:03d}"
        canvas.save(img_dir / f"{nid}.jpg", quality=92)
        (gold_dir / f"{nid}.json").write_text(json.dumps({
            "note_id": nid, "transcription": "\n".join(t for _, t in scaled),
            "extracted_json": {}, "distinctive_terms": [],
        }, ensure_ascii=False, indent=2))
        made += 1
        log.info("IAM page %s: %d lines", nid, len(scaled))
    print(f"wrote {made} IAM calibration pages to {img_dir}/ (+ gold in {gold_dir}/)")
    return made


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build IAM real-handwriting calibration pages.")
    ap.add_argument("--out", default="corpora")
    ap.add_argument("--name", default="iam")
    ap.add_argument("--pages", type=int, default=8)
    ap.add_argument("--lines", type=int, default=12)
    ap.add_argument("--split", default="test")
    args = ap.parse_args(argv)
    build_iam(args.out, args.name, args.pages, args.lines, args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
