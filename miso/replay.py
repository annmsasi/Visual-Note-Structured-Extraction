"""Run a note sequence through the base pipeline and write a per-note JSONL trace.

    python -m miso.replay demo            # smoke test on fake notes (stub OCR/LLM)
    python -m miso.replay note <image>    # one real image
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from miso import __version__
from miso.config import RunConfig
from miso.extraction import StubExtractor
from miso.ocr import StubOCR
from miso.pipeline import process_note
from miso.trace import TraceWriter
from miso.types import ExtractedNote, Note

log = logging.getLogger(__name__)


def _make_ocr(cfg: RunConfig):
    if cfg.ocr.engine in ("paddle", "tesseract"):
        try:
            from miso.ocr import CachedOCR, make_ocr
            return CachedOCR(make_ocr(cfg.ocr.engine))  # disk-cache so re-runs skip re-OCR
        except Exception as e:
            log.warning("%s construction failed (%s); falling back to StubOCR", cfg.ocr.engine, e)
        return StubOCR()
    if cfg.ocr.engine == "azure":
        endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        if endpoint and key:
            try:
                from miso.ocr import AzureOCR, CachedOCR
                return CachedOCR(AzureOCR(endpoint, key))
            except Exception as e:
                log.warning("AzureOCR construction failed (%s); falling back to StubOCR", e)
        else:
            log.warning("Azure env vars not set; falling back to StubOCR")
    return StubOCR()


def _make_extractor(cfg: RunConfig):
    model = cfg.extraction.model_id
    if model.startswith("claude"):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            try:
                from miso.extraction import AnthropicExtractor
                return AnthropicExtractor(api_key=key, model_id=model)
            except Exception as e:
                log.warning("AnthropicExtractor construction failed (%s); falling back to stub", e)
        else:
            log.warning("ANTHROPIC_API_KEY not set; falling back to StubExtractor")
    elif model not in ("stub", "") and (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_KEY")
    ):
        try:
            from miso.extraction import OpenAIVisionExtractor
            return OpenAIVisionExtractor(model)  # open VLM via OpenRouter/vLLM/Ollama
        except Exception as e:
            log.warning("OpenAIVisionExtractor construction failed (%s); falling back to stub", e)
    return StubExtractor()


def run(cfg: RunConfig, notes: list[Note]) -> list[ExtractedNote]:
    """Stateless: OCR -> LLM per note, write a trace, return the extractions."""
    ocr = _make_ocr(cfg)
    extractor = _make_extractor(cfg)
    run_dir = cfg.traces_dir / cfg.run_id
    cfg.save(run_dir / "config.json")
    out: list[ExtractedNote] = []
    with TraceWriter(run_dir) as trace:
        for note in notes:
            out.append(process_note(note, cfg, ocr=ocr, extractor=extractor, trace=trace))
    log.info("trace written to %s", run_dir / "trace.jsonl")
    return out


def run_document(cfg: RunConfig, *, note_id: str, course: str,
                 pages: list[Path], timestamp: datetime) -> dict:
    """Map-reduce over a multi-page document: extract each page (MAP), then merge
    the per-page IRs into one combined document (REDUCE). Returns the merged IR.

    Each MAP call sees a single page image and the REDUCE call is text-only, so this
    works with any model — including small / short-context open VLMs.
    """
    ocr = _make_ocr(cfg)
    extractor = _make_extractor(cfg)
    run_dir = cfg.traces_dir / cfg.run_id
    cfg.save(run_dir / "config.json")
    page_docs: list[dict] = []
    with TraceWriter(run_dir) as trace:
        for i, p in enumerate(pages):
            note = Note(note_id=f"{note_id}-p{i:03d}", course_id=course,
                        image_path=p, processing_order=i, timestamp=timestamp)
            page_docs.append(
                process_note(note, cfg, ocr=ocr, extractor=extractor, trace=trace).structured_json
            )
    if len(page_docs) <= 1:
        return page_docs[0] if page_docs else {}
    log.info("merging %d page extractions into one document", len(page_docs))
    return extractor.combine(page_docs)


def _make_fake_notes(course_id: str, n: int) -> list[Note]:
    base = datetime(2026, 5, 1, 12, 0, 0)
    return [
        Note(
            note_id=f"{course_id}-note-{i:03d}",
            course_id=course_id,
            image_path=Path(f"fake_images/{course_id}-note-{i:03d}.jpg"),
            processing_order=i,
            timestamp=base + timedelta(days=i),
        )
        for i in range(n)
    ]


def _load_env() -> None:
    """Load the project-root .env into the environment if python-dotenv is present."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # override=True: the project .env is authoritative, even over an already-exported
    # (possibly stale) shell variable.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _default_extractor_model() -> str:
    return os.environ.get(
        "MISO_EXTRACTOR",
        "claude-sonnet-4-6" if os.environ.get("ANTHROPIC_API_KEY") else "stub",
    )


def _prepare_image(path: Path, max_edge: int = 4000) -> Path:
    """Normalise an image for OCR/LLM: RGB JPEG with bounded dimensions."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_edge:
        scale = max_edge / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    out = path.with_name(path.stem + ".prepared.jpg")
    img.save(out, "JPEG", quality=90)
    log.info("prepared %s -> %s (%dx%d)", path.name, out.name, img.width, img.height)
    return out


def cmd_demo(args) -> int:
    _configure_logging()
    cfg = RunConfig.base(tag="demo")
    cfg.extraction.model_id = _default_extractor_model()
    run(cfg, _make_fake_notes("CS101", 3))
    return 0


def cmd_note(args) -> int:
    _configure_logging()
    image_path = _prepare_image(Path(args.image))
    cfg = RunConfig.base(tag="single_note")
    cfg.ocr.engine = args.ocr
    cfg.extraction.model_id = args.model or _default_extractor_model()
    note = Note(
        note_id=image_path.stem[:40],
        course_id="adhoc",
        image_path=image_path,
        processing_order=0,
        timestamp=datetime(2026, 5, 1, 12, 0, 0),
    )
    extracted = run(cfg, [note])[0]
    print(json.dumps(extracted.structured_json, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_env()
    parser = argparse.ArgumentParser(prog="miso-replay", description=f"miso base pipeline v{__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="Smoke test on fake notes.")
    note = sub.add_parser("note", help="Run the pipeline on one real image.")
    note.add_argument("image", help="Path to a note image (webp/png/jpg/...).")
    note.add_argument("--ocr", default="azure", choices=["stub", "azure", "paddle", "tesseract"],
                      help="OCR engine (paddle/tesseract are free + local).")
    note.add_argument("--model", default=None, help="Extraction model id.")
    args = parser.parse_args(argv)
    if args.cmd == "demo":
        return cmd_demo(args)
    if args.cmd == "note":
        return cmd_note(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
