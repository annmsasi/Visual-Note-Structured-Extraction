"""Inbox watcher: turn dropped note PDFs/images into Markdown.

Drop a PDF (or image) into a COURSE subfolder of the inbox. Each run of this
script processes every file waiting there: it runs the full miso pipeline
(OCR + cache + LLM), writes a Markdown note locally, then moves the source into
the 'processed' folder so it is never handled twice. Pass --drive to also upload
each note to Google Drive as a Google Doc.

A subfolder of the inbox names the course: dropping a note into
data/inbox/cse138/ processes it under course "cse138" (its own cache namespace
and output/processed subfolder). Loose files dropped straight into data/inbox/
use --course (default "notes").

This is the only script the scheduled job needs to call. To run on a schedule,
point cron (or launchd) at it — nothing else required (./setup.sh does this).
Example crontab line, every 15 minutes:

    */15 * * * * cd /ABSOLUTE/PATH/TO/repo && .venv/bin/python process_inbox.py >> data/process.log 2>&1

Run it by hand to process whatever is waiting right now:

    python process_inbox.py
    python process_inbox.py --drive                     # also upload a Google Doc
    python process_inbox.py --course cse138 --model claude-opus-4-8

Folders (override with flags):
    data/inbox/      drop PDFs/images here; a subfolder is a course (see above)
    data/output/     per course: data/output/<course>/<name>.md + .json
    data/processed/  sources moved to data/processed/<course>/ after success
    data/failed/     sources moved to data/failed/<course>/ on error (see data/process.log)

One-time setup: ./install.sh and a .env with ANTHROPIC_API_KEY and
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT/KEY. Drive upload (--drive) additionally
needs credentials.json plus one interactive run to create token.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from miso import export
from miso.config import RunConfig
from miso.replay import _configure_logging, _load_env, _prepare_image, run_document

log = logging.getLogger("process_inbox")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _is_note(p: Path) -> bool:
    """A file we should process: a PDF or image, not an intermediate .prepared."""
    return (p.is_file()
            and (p.suffix.lower() in IMAGE_EXTS or p.suffix.lower() == ".pdf")
            and ".prepared" not in p.name)


def collect_sources(inbox: Path, default_course: str) -> list[tuple[Path, str]]:
    """Notes waiting in the inbox, each paired with its course.

    A subfolder of the inbox is a course — inbox/cse138/lec1.pdf -> "cse138".
    Loose files dropped straight into the inbox use `default_course`."""
    if not inbox.exists():
        return []
    items: list[tuple[Path, str]] = []
    for p in sorted(inbox.glob("*")):          # loose files in the inbox root
        if _is_note(p):
            items.append((p, default_course))
    for course_dir in sorted(p for p in inbox.iterdir() if p.is_dir()):
        for p in sorted(course_dir.glob("*")):
            if _is_note(p):
                items.append((p, course_dir.name))
    return items


def render_pages(src: Path, work: Path) -> list[Path]:
    """Prepared page image(s) in `work` (kept out of the inbox): every page of a
    PDF, or the single image. `work` is a throwaway dir the caller deletes."""
    if src.suffix.lower() != ".pdf":
        local = work / src.name
        shutil.copy(src, local)
        return [_prepare_image(local)]

    import os
    from pdf2image import convert_from_path
    poppler = shutil.which("pdfinfo")
    rendered = convert_from_path(
        str(src), dpi=200, poppler_path=os.path.dirname(poppler) if poppler else None)
    pages: list[Path] = []
    for i, page in enumerate(rendered):
        raw = work / f"{src.stem}.p{i:03d}.jpg"
        page.convert("RGB").save(raw, "JPEG", quality=90)
        pages.append(_prepare_image(raw))
    return pages


def process_one(src: Path, *, course: str, ocr: str, model: str, out_dir: Path,
                cache_path: Path, traces_dir: Path, drive: bool, folder: str) -> dict:
    """Run the pipeline on one file: write Markdown (+ the IR JSON) and optionally
    upload it to Drive. A multi-page PDF is merged into one note via map-reduce, with
    the course cache warming across its pages. Returns a small result dict."""
    work = Path(tempfile.mkdtemp(prefix="inbox_"))
    try:
        pages = render_pages(src, work)
        cfg = RunConfig.config_6_full(tag="inbox")
        cfg.ocr.engine = ocr
        cfg.extraction.model_id = model
        cfg.from_empty_cache = False        # persist the course cache across runs
        cfg.cache_path = cache_path
        cfg.traces_dir = traces_dir
        cfg.extraction.figures_dir = out_dir / "figures"   # crop figures next to the output
        doc = run_document(cfg, note_id=src.stem[:60], course=course,
                           pages=pages, timestamp=datetime(2026, 1, 1))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if not doc.get("title") and not doc.get("blocks"):
        raise RuntimeError("pipeline produced no note (check API keys / the log above)")

    out_dir.mkdir(parents=True, exist_ok=True)
    md = export.render_note_markdown(doc)
    md_path = out_dir / f"{src.stem}.md"
    md_path.write_text(md)
    (out_dir / f"{src.stem}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2))

    url = ""
    if drive:
        url = export.upload_note_to_drive(
            doc, name=doc.get("title") or src.stem, folder=folder or course, fmt="markdown")
    return {"md": md_path, "title": doc.get("title"), "pages": len(pages), "url": url}


def main(argv: list[str] | None = None) -> int:
    _load_env()
    _configure_logging()
    ap = argparse.ArgumentParser(description="Turn inbox PDFs/images into Markdown + Google Docs.")
    ap.add_argument("--inbox", type=Path, default=Path("data/inbox"))
    ap.add_argument("--output", type=Path, default=Path("data/output"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--failed", type=Path, default=Path("data/failed"))
    ap.add_argument("--cache", type=Path, default=Path("data/miso_cache.db"),
                    help="persistent course cache; vocabulary accrues across runs")
    ap.add_argument("--course", default="notes",
                    help="course for loose files in the inbox root; a subfolder "
                         "name overrides it (subfolder = course)")
    ap.add_argument("--ocr", default="azure", choices=["azure", "paddle", "tesseract"])
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--drive", action=argparse.BooleanOptionalAction, default=False,
                    help="also upload each note to Google Drive (default off; local Markdown only)")
    ap.add_argument("--folder", default="", help="Drive folder name (default: the course id)")
    args = ap.parse_args(argv)

    items = collect_sources(args.inbox, args.course)
    if not items:
        print(f"Nothing to do — no notes in {args.inbox}/.")
        return 0

    ok = 0
    for src, course in items:
        # Mirror the course into output/processed/failed so courses never collide.
        out_dir = args.output / course
        processed_dir = args.processed / course
        failed_dir = args.failed / course
        try:
            r = process_one(src, course=course, ocr=args.ocr, model=args.model,
                            out_dir=out_dir, cache_path=args.cache,
                            traces_dir=Path("data/runs"), drive=args.drive, folder=args.folder)
            processed_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(processed_dir / src.name))
            link = f"  ->  {r['url']}" if r["url"] else ""
            print(f"OK    [{course}] {src.name}  ->  {r['md']}  ({r['pages']} page(s)){link}")
            ok += 1
        except Exception as e:  # noqa: BLE001 — one bad file must not stop the batch
            log.exception("failed on %s", src.name)
            failed_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(failed_dir / src.name))
            print(f"FAIL  [{course}] {src.name}: {e}")
    print(f"\nprocessed {ok}/{len(items)} file(s)")
    return 0 if ok == len(items) else 1


if __name__ == "__main__":
    sys.exit(main())
