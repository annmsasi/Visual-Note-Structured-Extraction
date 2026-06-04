"""Run the base (no-cache) miso pipeline on a note image or PDF, and export the result.

A multi-page PDF is processed as ONE document via map-reduce: each page is extracted
independently (one image per call — works with any model), then the per-page results
are merged into a single combined note.

    python run_full_pipeline.py data/inbox/notes.jpg
    python run_full_pipeline.py lecture.pdf --ocr tesseract --model qwen/qwen2.5-vl-72b-instruct
    python run_full_pipeline.py lecture.pdf --drive          # one combined Google Doc
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from miso import export
from miso.config import RunConfig
from miso.replay import _configure_logging, _load_env, _prepare_image, run_document

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _default_image() -> Path:
    imgs = sorted(p for p in Path("data/inbox").glob("*") if p.suffix.lower() in _IMAGE_EXTS)
    if not imgs:
        raise SystemExit("No image in data/inbox; pass one explicitly.")
    return imgs[0]


def _pages_from_input(src: Path) -> list[Path]:
    """Prepared page image(s): render every page of a PDF, else just the one image."""
    if src.suffix.lower() == ".pdf":
        import os
        import shutil
        from pdf2image import convert_from_path
        poppler = shutil.which("pdfinfo")
        rendered = convert_from_path(
            str(src), dpi=200,
            poppler_path=os.path.dirname(poppler) if poppler else None,
        )
        out = []
        for i, pg in enumerate(rendered):
            raw = src.with_name(f"{src.stem}.p{i:03d}.jpg")
            pg.convert("RGB").save(raw, "JPEG", quality=90)
            out.append(_prepare_image(raw))
        return out
    return [_prepare_image(src)]


def main(argv: list[str] | None = None) -> int:
    _load_env()
    _configure_logging()

    ap = argparse.ArgumentParser(description="Run the base (no-cache) miso pipeline on a note image or PDF.")
    ap.add_argument("image", nargs="?", help="note image or multi-page PDF (default: first in data/inbox)")
    ap.add_argument("--course", default="adhoc", help="Drive folder name for --drive")
    ap.add_argument("--ocr", default="azure", choices=["stub", "azure", "paddle", "tesseract"],
                    help="OCR engine; paddle/tesseract are free + local")
    ap.add_argument("--model", default="claude-sonnet-4-6",
                    help="extraction model: a claude-* id, or an open VLM id like "
                         "qwen/qwen2.5-vl-72b-instruct (served via OPENROUTER_API_KEY)")
    ap.add_argument("--out", type=Path, help="HTML output path (default: <name>.html)")
    ap.add_argument("--drive", action="store_true", help="also upload to Google Docs")
    args = ap.parse_args(argv)

    src = Path(args.image) if args.image else _default_image()
    pages = _pages_from_input(src)

    cfg = RunConfig.base(tag="full_pipeline")
    cfg.ocr.engine = args.ocr
    cfg.extraction.model_id = args.model

    doc = run_document(
        cfg, note_id=src.stem[:60], course=args.course,
        pages=pages, timestamp=datetime(2026, 1, 1),
    )
    if not doc.get("title") and not doc.get("blocks"):
        print("Pipeline produced no note (check API keys / logs above).")
        return 1

    html = export.render_note_html(doc)
    out = args.out or Path(f"{src.stem[:60]}.html")
    out.write_text(html)
    print(f"Structured note: {doc.get('title')!r}  ({len(pages)} page(s) merged)")
    print(f"Wrote {out}")

    if args.drive:
        url = export.upload_html_to_drive(html, name=doc.get("title") or src.stem, folder=args.course)
        print(f"Google Doc: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
