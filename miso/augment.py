"""Build the text portion of the extraction prompt."""
from __future__ import annotations

from miso.config import ExtractionConfig
from miso.types import CorrectedOCR, RetrievedSummary


SYSTEM_PROMPT = (
    "You extract a structured note from a page image.\n"
    "Image is the source of truth. OCR is a weak hint. Related prior notes "
    "and course terms are additional weak hints — useful for disambiguating "
    "abbreviations, recurring terminology, and math notation, but the image "
    "always wins.\n"
    "When a 'Possible OCR misreads' list is given, each entry is a low-confidence "
    "word with candidate course terms; choose a candidate only if it matches the "
    "writing, otherwise transcribe what you see.\n"
    "Call the `emit_structured_note` tool to return the note as an ordered list "
    "of document blocks that mirror the page's layout:\n"
    "  - `heading` (with `level` 1-3) for titles and section headings;\n"
    "  - `list` (with nested `items`) for bulleted/enumerated points — use the "
    "OCR's preserved line breaks and indentation to recover nesting;\n"
    "  - `paragraph` for running prose; `equation` (LaTeX) for math.\n"
    "Be FAITHFUL to the page's own structure: if the source is an outline of "
    "bulleted points, keep it as lists; only use `paragraph` where the writer "
    "actually wrote running prose. Do not rewrite an outline into prose, or "
    "merge separate bullets into a paragraph.\n"
    "Also fill the piggybacked summary fields:\n"
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
        # Most-relevant placed adjacent to the OCR.
        for r in reversed(retrieved):
            tag = f"[course={r.summary.course_id} order={r.summary.processing_order}]"
            parts.append(f"- {tag} {r.summary.topic_line}\n  {r.summary.gist}")
        parts.append("")

    flags = corrected_ocr.flags if corrected_ocr is not None else []
    if cfg.use_glossary and flags:
        # Flag mode: per-word candidate corrections the LLM arbitrates against the image.
        parts.append(
            "Possible OCR misreads (low-confidence words with candidate course "
            "terms — pick one only if it matches the image, else read the page):"
        )
        for f in flags:
            cands = ", ".join(c.term for c in f.candidates)
            parts.append(f'- "{f.original}" (confidence {f.confidence:.2f}) -> {cands}')
        parts.append("")
    elif cfg.use_glossary and glossary:
        parts.append("Course terms appearing in this note:")
        parts.append("- " + ", ".join(sorted(set(glossary))))
        parts.append("")

    if cfg.use_ocr_hint and corrected_ocr is not None:
        # Prefer layout text so structure survives; fall back to flat text.
        ocr_hint = corrected_ocr.layout_text or corrected_ocr.corrected_text
        parts.append("OCR (weak hint, line breaks and indentation preserved):")
        parts.append(ocr_hint)
        parts.append("")

    parts.append("Extract the structured note as JSON. Include the piggybacked summary fields.")
    return "\n".join(parts)
