"""Per-note base pipeline: OCR, then schema-forced LLM extraction. No cache."""
from __future__ import annotations

import logging
import time

from miso.config import RunConfig
from miso.trace import TraceWriter
from miso.types import (
    CorrectedOCR, ExtractedNote, Note, OCRResult, StageLatencies, TraceRecord,
)

log = logging.getLogger(__name__)


def process_note(
    note: Note,
    cfg: RunConfig,
    *,
    ocr,
    extractor,
    trace: TraceWriter,
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

    # No lexicon: the OCR text passes through untouched as a weak hint to the LLM,
    # which reads the page image as the source of truth.
    corrected = CorrectedOCR(
        words=list(ocr_result.words),
        corrected_text=ocr_result.raw_text,
        corrections=[],
        touched_terms=[],
        layout_text=ocr_result.layout_text,
    )
    record.corrected_ocr = corrected

    t = time.perf_counter()
    extracted: ExtractedNote = extractor.extract(
        note=note,
        corrected_ocr=corrected if cfg.extraction.use_ocr_hint else None,
        retrieved=[],
        glossary=[],
        cfg=cfg.extraction,
    )
    latencies.extraction_ms = (time.perf_counter() - t) * 1000
    record.extraction = extracted

    record.latencies = latencies
    trace.write(record)
    return extracted
