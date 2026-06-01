"""Run the full miso pipeline on a note image and export the result.

Usage:
    python run_full_pipeline.py [data/inbox/notes.jpg] [--drive]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from miso import export
from miso.config import RunConfig
from miso.replay import _configure_logging, _load_env, _prepare_image, run
from miso.types import Note

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _default_image() -> Path:
    imgs = sorted(p for p in Path("data/inbox").glob("*") if p.suffix.lower() in _IMAGE_EXTS)
    if not imgs:
        raise SystemExit("No image in data/inbox; pass one explicitly.")
    return imgs[0]


def main(argv: list[str] | None = None) -> int:
    _load_env()
    _configure_logging()

    ap = argparse.ArgumentParser(description="Run the full miso pipeline on a note image.")
    ap.add_argument("image", nargs="?", help="note image (default: first in data/inbox)")
    ap.add_argument("--course", default="adhoc")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--drive", action="store_true", help="also upload to Google Docs")
    args = ap.parse_args(argv)

    src = Path(args.image) if args.image else _default_image()
    image_path = _prepare_image(src)

    cfg = RunConfig.config_6_full(tag="full_pipeline")
    cfg.ocr.engine = "azure"
    cfg.extraction.model_id = args.model

    note = Note(
        note_id=image_path.stem[:60],
        course_id=args.course,
        image_path=image_path,
        processing_order=0,
        timestamp=datetime(2026, 1, 1),
    )
    run(cfg, [note])

    docs = export.load_notes(cfg.cache_path, note_id=note.note_id)
    if not docs:
        print("Pipeline produced no extracted note (check API keys / logs above).")
        return 1
    _, course_id, doc = docs[0]

    html = export.render_note_html(doc)
    out = args.out or Path(f"{note.note_id}.html")
    out.write_text(html)
    print(f"Structured note: {doc.get('title')!r}")
    print(f"Wrote {out}")

    if args.drive:
        url = export.upload_html_to_drive(html, name=doc.get("title") or note.note_id, folder=course_id)
        print(f"Google Doc: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
