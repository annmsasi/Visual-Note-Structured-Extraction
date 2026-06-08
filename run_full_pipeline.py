"""Run the full miso pipeline (with cache) on a note image or PDF, and export the result.

A multi-page PDF is processed as ONE document via map-reduce: each page is extracted
independently (one image per call — works with any model), with the cache (lexicon +
retrieval) warming across the pages, then the per-page results are merged into one note.

    python run_full_pipeline.py data/inbox/notes.jpg
    python run_full_pipeline.py lecture.pdf --ocr tesseract --model qwen/qwen2.5-vl-72b-instruct
    python run_full_pipeline.py lecture.pdf --drive          # one combined Google Doc
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from miso import export
from miso.config import RunConfig
from miso.replay import _configure_logging, _load_env, _prepare_image, run_document

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

import os

# Finds the folder where run_full_pipeline.py lives, on ANY computer
base_folder = os.path.dirname(os.path.abspath(__file__))

# Builds paths relative to that folder (DOUBLE CHECK THIS)
image_folder = os.path.join(base_folder, "images")
output_folder = os.path.join(base_folder, "output")


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

    ap = argparse.ArgumentParser(description="Run the full miso pipeline (with cache) on a note image or PDF.")
    ap.add_argument("image", nargs="?", help="note image or multi-page PDF (default: first in data/inbox)")
    ap.add_argument("--course", default="adhoc", help="course id (cache namespace) + Drive folder for --drive")
    ap.add_argument("--ocr", default="azure", choices=["stub", "azure", "paddle", "tesseract"],
                    help="OCR engine; paddle/tesseract are free + local")
    ap.add_argument("--model", default="claude-sonnet-4-6",
                    help="extraction model: a claude-* id, or an open VLM id like "
                         "qwen/qwen2.5-vl-72b-instruct (served via OPENROUTER_API_KEY)")
    ap.add_argument("--format", choices=["md", "html", "pdf"], default="md",
                    help="output format (default: md); pdf embeds rendered figure diagrams")
    ap.add_argument("--out", type=Path, help="output path (default: <name>.<format>)")
    ap.add_argument("--drive", action="store_true", help="also upload to Google Docs")
    args = ap.parse_args(argv)

    src = Path(args.image) if args.image else _default_image()
    pages = _pages_from_input(src)

    cfg = RunConfig.config_6_full(tag="full_pipeline")
    cfg.ocr.engine = args.ocr
    cfg.extraction.model_id = args.model
    cfg.from_empty_cache = False  # persist the cache across CLI runs so course knowledge accrues
    cfg.extraction.figures_dir = Path("figures")  # render figure diagrams here (namespaced by note id)

    doc = run_document(
        cfg, note_id=src.stem[:60], course=args.course,
        pages=pages, timestamp=datetime(2026, 1, 1),
    )
    if not doc.get("title") and not doc.get("blocks"):
        print("Pipeline produced no note (check API keys / logs above).")
        return 1

    stem = src.stem[:60]
    if args.format == "pdf":
        out = args.out or Path(f"{stem}.pdf")
        export.render_note_pdf(doc, out)
    else:
        if args.format == "md":
            body, default_out = export.render_note_markdown(doc), Path(f"{stem}.md")
        else:
            body, default_out = export.render_note_html(doc), Path(f"{stem}.html")
        out = args.out or default_out
        out.write_text(body)
    # save the IR alongside, so you can re-render to any format later without re-running
    out.with_suffix(".json").write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    print(f"Structured note: {doc.get('title')!r}  ({len(pages)} page(s) merged)")
    print(f"Wrote {out}")

    if args.drive:
        # HTML import gives the better-looking Doc; markdown is the local format.
        # upload_note_to_drive also embeds any figure images inline via the Docs API.
        url = export.upload_note_to_drive(
            doc, name=doc.get("title") or stem, folder=args.course, fmt="html")
        print(f"Google Doc: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
