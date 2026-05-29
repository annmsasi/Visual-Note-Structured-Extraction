"""Build the text portion of the extraction prompt.

The page image is attached separately by the extraction adapter. This module
assembles the text: system prompt → retrieved summaries (reverse-ordered so
the most relevant sits nearest the OCR) → glossary → OCR → task line.
"""
from __future__ import annotations

from miso.config import ExtractionConfig
from miso.types import CorrectedOCR, RetrievedSummary


SYSTEM_PROMPT = (
    "You extract structured notes from a page image.\n"
    "Image is the source of truth. OCR is a weak hint. Related prior notes "
    "and course terms are additional weak hints — useful for disambiguating "
    "abbreviations, recurring terminology, and math notation, but the image "
    "always wins.\n"
    "Emit STRICT JSON with the extracted fields PLUS a piggybacked summary:\n"
    "  - `summary_topic_line`: a single short line naming the topic/section.\n"
    "  - `summary_gist`: 2-4 sentences (~150 tokens) describing what the note covers.\n"
)


def assemble_prompt(
    *,
    corrected_ocr: CorrectedOCR | None,
    retrieved: list[RetrievedSummary],
    glossary: list[str],
    cfg: ExtractionConfig,
) -> str:
    parts: list[str] = [SYSTEM_PROMPT, ""]

    if cfg.use_retrieved_summaries and retrieved:
        parts.append("Related prior notes (weak context):")
        # Reverse-ordered: least-relevant first, most-relevant adjacent to the OCR.
        for r in reversed(retrieved):
            tag = f"[course={r.summary.course_id} order={r.summary.processing_order}]"
            parts.append(f"- {tag} {r.summary.topic_line}\n  {r.summary.gist}")
        parts.append("")

    if cfg.use_glossary and glossary:
        parts.append("Course terms appearing in this note:")
        parts.append("- " + ", ".join(sorted(set(glossary))))
        parts.append("")

    if cfg.use_ocr_hint and corrected_ocr is not None:
        parts.append("OCR (weak hint):")
        parts.append(corrected_ocr.corrected_text)
        parts.append("")

    parts.append("Extract the structured note as JSON. Include the piggybacked summary fields.")
    return "\n".join(parts)
