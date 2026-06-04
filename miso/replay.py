"""Run a fixed note sequence under a RunConfig and write a per-note JSONL trace.

    python -m miso.replay demo       # smoke test: 5 fake notes
    python -m miso.replay ablation   # the 4 cache-cell configs
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from miso import __version__
from miso.config import RunConfig
from miso.db import open_db, reset
from miso.extraction import StubExtractor
from miso.lexicon import LexiconLayer
from miso.ocr import StubOCR
from miso.pipeline import process_note
from miso.retrieval import RetrievalLayer
from miso.summary_store import SummaryStore
from miso.trace import TraceWriter
from miso.types import Note

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _maybe_real_embedder():
    try:
        from miso.encoders import STEmbedder
        return STEmbedder()
    except Exception as e:
        log.info("Embedder unavailable (%s); BM25-only retrieval", e)
        return None


def _maybe_real_reranker():
    try:
        from miso.encoders import BGEReranker
        return BGEReranker()
    except Exception as e:
        log.info("Cross-encoder reranker unavailable (%s); Jaccard fallback", e)
        return None


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
                # disk-cache so re-runs never re-bill a page
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


def run(cfg: RunConfig, notes: list[Note]) -> Path:
    if cfg.from_empty_cache:
        reset(cfg.cache_path)

    conn = open_db(cfg.cache_path)
    conn.execute(
        "INSERT OR REPLACE INTO runs(run_id, config_tag, config_json, started_at) "
        "VALUES (?, ?, ?, ?)",
        (cfg.run_id, cfg.config_tag, json.dumps(cfg.to_dict(), default=str), _now_iso()),
    )
    conn.commit()

    # only load the heavy models when retrieval is on
    embedder = _maybe_real_embedder() if cfg.retrieval.enabled else None
    reranker = _maybe_real_reranker() if cfg.retrieval.enabled else None
    ocr = _make_ocr(cfg)
    lexicon_layer = LexiconLayer(conn)
    summary_store = SummaryStore(conn, embedder=embedder)
    retrieval_layer = RetrievalLayer(conn, summary_store,
                                     embedder=embedder, reranker=reranker)
    extractor = _make_extractor(cfg)

    run_dir = cfg.traces_dir / cfg.run_id
    cfg.save(run_dir / "config.json")
    with TraceWriter(run_dir) as trace:
        for note in notes:
            process_note(
                note, cfg,
                ocr=ocr,
                lexicon_layer=lexicon_layer,
                summary_store=summary_store,
                retrieval_layer=retrieval_layer,
                extractor=extractor,
                trace=trace,
            )

    conn.execute(
        "UPDATE runs SET finished_at = ? WHERE run_id = ?",
        (_now_iso(), cfg.run_id),
    )
    conn.commit()
    conn.close()
    return run_dir


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
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers",
                  "transformers", "filelock", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _default_extractor_model() -> str:
    return os.environ.get(
        "MISO_EXTRACTOR",
        "claude-haiku-4-5-20251001" if os.environ.get("ANTHROPIC_API_KEY") else "stub",
    )


def _prepare_image(path: Path, max_edge: int = 4000) -> Path:
    """Normalise an image for Azure: RGB JPEG with bounded dimensions."""
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
    cfg = RunConfig.config_6_full(tag="demo_full_system")
    cfg.retrieval.cold_start_note_count = 2
    cfg.extraction.model_id = _default_extractor_model()
    run_dir = run(cfg, _make_fake_notes("CS101", 5))
    print(f"Trace written to: {run_dir / 'trace.jsonl'}")
    print(f"Config snapshot:   {run_dir / 'config.json'}")
    return 0


def cmd_ablation(args) -> int:
    _configure_logging()
    notes = _make_fake_notes("CS101", 5)
    configs = [
        RunConfig.config_3_llm_ocr_only(),
        RunConfig.config_4_lexicon_only(),
        RunConfig.config_5_retrieval_only(),
        RunConfig.config_6_full(),
    ]
    model_id = _default_extractor_model()
    for cfg in configs:
        cfg.retrieval.cold_start_note_count = 2
        cfg.extraction.model_id = model_id
        run_dir = run(cfg, notes)
        print(f"{cfg.config_tag}: {run_dir / 'trace.jsonl'}")
    return 0


def cmd_note(args) -> int:
    """Run the baseline on one real image."""
    _configure_logging()
    image_path = _prepare_image(Path(args.image))
    cfg = RunConfig.config_3_llm_ocr_only(tag="single_note")
    cfg.ocr.engine = "azure"
    cfg.extraction.model_id = args.model or "claude-sonnet-4-6"
    note = Note(
        note_id=image_path.stem[:40],
        course_id="adhoc",
        image_path=image_path,
        processing_order=0,
        timestamp=datetime(2026, 5, 1, 12, 0, 0),
    )
    run_dir = run(cfg, [note])

    record = json.loads((run_dir / "trace.jsonl").read_text().splitlines()[0])
    ocr = record.get("ocr_raw") or {}
    extraction = record.get("extraction") or {}
    print("\n===== Azure OCR (prebuilt-read) =====")
    print(ocr.get("raw_text", "(no OCR text)"))
    print(f"\n===== Extraction ({extraction.get('model_id', '?')}) =====")
    print(json.dumps(extraction.get("structured_json", {}), indent=2, ensure_ascii=False))
    print(f"\nTrace: {run_dir / 'trace.jsonl'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_env()
    parser = argparse.ArgumentParser(prog="miso-replay", description=f"miso v{__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="Run a single end-to-end smoke test.")
    sub.add_parser("ablation", help="Run the 4-cell cache-ablation grid.")
    note = sub.add_parser("note", help="Run the baseline on one real image.")
    note.add_argument("image", help="Path to a note image (webp/png/jpg/...).")
    note.add_argument("--model", default=None,
                      help="Override the extraction model id (default claude-sonnet-4-6).")
    args = parser.parse_args(argv)
    if args.cmd == "demo":
        return cmd_demo(args)
    if args.cmd == "ablation":
        return cmd_ablation(args)
    if args.cmd == "note":
        return cmd_note(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
