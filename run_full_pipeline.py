"""Run the base (no-cache) miso pipeline on a note image and export the result.

    python run_full_pipeline.py data/inbox/notes.jpg
    python run_full_pipeline.py notes.jpg --ocr tesseract --model qwen/qwen2.5-vl-72b-instruct
    python run_full_pipeline.py --drive          # also create a Google Doc
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

    ap = argparse.ArgumentParser(description="Run the base (no-cache) miso pipeline on a note image.")
    ap.add_argument("image", nargs="?", help="note image (default: first in data/inbox)")
    ap.add_argument("--course", default="adhoc", help="Drive folder name for --drive")
    ap.add_argument("--ocr", default="azure", choices=["stub", "azure", "paddle", "tesseract"],
                    help="OCR engine; paddle/tesseract are free + local")
    ap.add_argument("--model", default="claude-sonnet-4-6",
                    help="extraction model: a claude-* id, or an open VLM id like "
                         "qwen/qwen2.5-vl-72b-instruct (served via OPENROUTER_API_KEY)")
    ap.add_argument("--out", type=Path, help="HTML output path (default: <note_id>.html)")
    ap.add_argument("--drive", action="store_true", help="also upload to Google Docs")
    args = ap.parse_args(argv)

    src = Path(args.image) if args.image else _default_image()
    image_path = _prepare_image(src)

    cfg = RunConfig.base(tag="full_pipeline")
    cfg.ocr.engine = args.ocr
    cfg.extraction.model_id = args.model

    note = Note(
        note_id=image_path.stem[:60],
        course_id=args.course,
        image_path=image_path,
        processing_order=0,
        timestamp=datetime(2026, 1, 1),
    )
    extracted = run(cfg, [note])
    if not extracted:
        print("Pipeline produced no note (check API keys / logs above).")
        return 1
    doc = extracted[0].structured_json

    html = export.render_note_html(doc)
    out = args.out or Path(f"{note.note_id}.html")
    out.write_text(html)
    print(f"Structured note: {doc.get('title')!r}")
    print(f"Wrote {out}")

    if args.drive:
        url = export.upload_html_to_drive(html, name=doc.get("title") or note.note_id, folder=args.course)
        print(f"Google Doc: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
