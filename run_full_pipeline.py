"""Full miso pipeline runner  (branch: full-pipeline).

Runs the complete vendored miso pipeline on a note image:

    preprocess -> Azure OCR -> lexicon correction -> retrieval -> Claude
    (schema-forced document IR) -> write-back to miso_cache.db

then renders the structured note to HTML (and to a Google Doc with --drive).

This branch keeps the original team scripts (preprocess_test.py, ocr_test.py,
extract_test.py) untouched — it adds the full pipeline alongside them via the
vendored `miso/` package.

Setup:
    pip install -r requirements.txt

.env (repo root) needs:
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, AZURE_DOCUMENT_INTELLIGENCE_KEY
    ANTHROPIC_API_KEY
    (optional) MISO_EXTRACTOR=claude-sonnet-4-6
    (optional, for --drive) credentials.json from a GCP OAuth client

Usage:
    python run_full_pipeline.py                         # first image in data/inbox
    python run_full_pipeline.py data/inbox/notes.jpg
    python run_full_pipeline.py --drive                 # also create a Google Doc
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
    ap.add_argument("--course", default="adhoc", help="course_id grouping (default: adhoc)")
    ap.add_argument("--model", default="claude-sonnet-4-6", help="extraction model id")
    ap.add_argument("--out", type=Path, help="HTML output path (default: <note_id>.html)")
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
    _, doc = docs[0]

    html = export.render_note_html(doc)
    out = args.out or Path(f"{note.note_id}.html")
    out.write_text(html)
    print(f"\nStructured note: {doc.get('title')!r}")
    print(f"Wrote {out}")

    if args.drive:
        url = export.upload_html_to_drive(html, name=doc.get("title") or note.note_id)
        print(f"Google Doc: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
