"""Per-note pipeline: OCR, lexicon, retrieval, extraction, write-back."""
from __future__ import annotations

import logging
import time

from miso.config import RunConfig
from miso.trace import TraceWriter
from miso.types import (
    CorrectedOCR, ExtractedNote, GateDecision, Note, OCRResult,
    RetrievalResult, StageLatencies, TraceRecord,
)

log = logging.getLogger(__name__)


def process_note(
    note: Note,
    cfg: RunConfig,
    *,
    ocr,
    lexicon_layer,
    retrieval_layer,
    extractor,
    trace: TraceWriter,
    prior_pages_md: list[str] | None = None,
) -> ExtractedNote:
    latencies = StageLatencies()
    record = TraceRecord(
        note_id=note.note_id,
        course_id=note.course_id,
        processing_order=note.processing_order,
        config_tag=cfg.config_tag,
        pipeline_version=cfg.pipeline_version,
        run_id=cfg.run_id,
    )

    t = time.perf_counter()
    ocr_result: OCRResult = ocr.run(note.image_path)
    latencies.ocr_ms = (time.perf_counter() - t) * 1000
    record.ocr_raw = ocr_result

    if cfg.lexicon.enabled:
        t = time.perf_counter()
        corrected: CorrectedOCR = lexicon_layer.correct(
            ocr_result, note.course_id, cfg.lexicon,
        )
        latencies.lexicon_correction_ms = (time.perf_counter() - t) * 1000
    else:
        corrected = CorrectedOCR(
            words=list(ocr_result.words),
            corrected_text=ocr_result.raw_text,
            corrections=[],
            touched_terms=[],
            layout_text=ocr_result.layout_text,
        )
    record.corrected_ocr = corrected
    record.glossary_to_llm = corrected.touched_terms
    record.lexicon_size_at_time = (
        lexicon_layer.size(note.course_id) if cfg.lexicon.enabled else 0
    )

    if cfg.retrieval.enabled:
        t = time.perf_counter()
        retrieval: RetrievalResult = retrieval_layer.retrieve(
            query_text=corrected.corrected_text,
            course_id=note.course_id,
            cfg=cfg.retrieval,
        )
        latencies.retrieval_ms = (time.perf_counter() - t) * 1000
    else:
        retrieval = RetrievalResult(
            query_text=corrected.corrected_text,
            candidates_top10=[],
            injected=[],
            cold_start_skip=False,
            filter_empty=False,
        )
    record.retrieval = retrieval
    record.gate = GateDecision(
        retrieved=bool(retrieval.injected),
        cold_start_skip=retrieval.cold_start_skip,
        filter_empty=retrieval.filter_empty,
    )

    t = time.perf_counter()
    extracted: ExtractedNote = extractor.extract(
        note=note,
        corrected_ocr=corrected if cfg.extraction.use_ocr_hint else None,
        retrieved=retrieval.injected if cfg.extraction.use_retrieved_summaries else [],
        glossary=corrected.touched_terms if cfg.extraction.use_glossary else [],
        cfg=cfg.extraction,
        prior_pages_md=prior_pages_md,
    )
    latencies.extraction_ms = (time.perf_counter() - t) * 1000
    record.extraction = extracted

    # Summary + harvest are per-NOTE, not per-page; finalize_note does them once
    # the whole note (all pages) is assembled.
    record.latencies = latencies
    trace.write(record)
    return extracted


def finalize_note(
    note_doc: dict,
    note: Note,
    *,
    extractor,
    summary_store,
    lexicon_layer,
    cfg: RunConfig,
) -> ExtractedNote:
    """Per-note finalize, run once after any multi-page merge: a dedicated
    whole-note summary (stored for retrieval) and the lexicon harvest over the
    complete note. Mutates `note_doc` to carry the summary for export."""
    topic, gist = extractor.summarize(note_doc)
    note_doc["summary_topic_line"] = topic
    note_doc["summary_gist"] = gist
    ext = ExtractedNote(
        note_id=note.note_id,
        structured_json=note_doc,
        summary_topic_line=topic,
        summary_gist=gist,
        model_id=cfg.extraction.model_id,
    )
    summary_store.add(ext, note)
    if cfg.lexicon.enabled:
        lexicon_layer.harvest(ext, note.course_id)
        lexicon_layer.promote_pending(note.course_id, cfg.lexicon.n_recurrence)
    return ext
