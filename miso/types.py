"""Dataclasses passed between pipeline components.

Kept minimal and JSON-serialisable so per-note traces serialise cleanly.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class OCRWord:
    text: str
    confidence: float
    bbox: tuple[float, float, float, float] | None = None  # x, y, w, h


@dataclass
class OCRResult:
    words: list[OCRWord]
    raw_text: str

    @classmethod
    def from_words(cls, words: list[OCRWord]) -> "OCRResult":
        return cls(words=words, raw_text=" ".join(w.text for w in words))


@dataclass
class LexiconCorrection:
    token_index: int
    original: str
    suggested: str
    match_strength: float    # 1.0 = exact; lower = farther under the shape-aware metric
    ocr_confidence: float
    accepted: bool           # True if the OCR word's confidence was reweighted


@dataclass
class CorrectedOCR:
    """OCR after the lexicon layer. Reweighted, not hard-overwritten."""
    words: list[OCRWord]
    corrected_text: str
    corrections: list[LexiconCorrection]
    touched_terms: list[str]  # the matched lexicon entries; passed to the LLM as the glossary


@dataclass
class LexiconEntry:
    course_id: str
    term: str
    frequency: int
    context_snippet: str
    first_seen: datetime
    last_seen: datetime


@dataclass
class Summary:
    note_id: str
    course_id: str
    topic_line: str
    gist: str
    processing_order: int
    pipeline_version: str
    timestamp: datetime


@dataclass
class RetrievedSummary:
    summary: Summary
    retrieval_score: float           # RRF fusion score
    reranker_score: float | None
    layer_of_origin: str = "summary"  # placeholder; only one layer in v1


@dataclass
class GateDecision:
    retrieved: bool
    cold_start_skip: bool
    filter_empty: bool


@dataclass
class RetrievalResult:
    query_text: str
    candidates_top10: list[RetrievedSummary]   # pre-filter
    injected: list[RetrievedSummary]           # post-filter, capped at top_k_inject
    cold_start_skip: bool
    filter_empty: bool


@dataclass
class Note:
    note_id: str
    course_id: str
    image_path: Path
    processing_order: int
    timestamp: datetime


@dataclass
class ExtractedNote:
    """LLM extraction output. Summary fields are piggybacked from the same call."""
    note_id: str
    structured_json: dict[str, Any]
    summary_topic_line: str
    summary_gist: str
    model_id: str  # captured for reproducibility — model versions drift


@dataclass
class StageLatencies:
    preprocess_ms: float = 0.0
    ocr_ms: float = 0.0
    lexicon_correction_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    extraction_ms: float = 0.0
    writeback_ms: float = 0.0


@dataclass
class TraceRecord:
    """One JSON record per note per run. The eval harness reads these."""
    note_id: str
    course_id: str
    processing_order: int
    config_tag: str
    pipeline_version: str
    run_id: str

    ocr_raw: OCRResult | None = None
    corrected_ocr: CorrectedOCR | None = None
    glossary_to_llm: list[str] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    gate: GateDecision | None = None
    extraction: ExtractedNote | None = None
    latencies: StageLatencies = field(default_factory=StageLatencies)
    lexicon_size_at_time: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _to_json_safe(self)


def _to_json_safe(value: Any) -> Any:
    """Recursively serialise dataclasses, dates, and paths to JSON-safe values."""
    if dataclasses.is_dataclass(value):
        return {f.name: _to_json_safe(getattr(value, f.name))
                for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value
