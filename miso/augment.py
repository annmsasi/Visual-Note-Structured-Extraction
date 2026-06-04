"""Build the text portion of the extraction prompt (base pipeline: image + OCR hint)."""
from __future__ import annotations

from miso.config import ExtractionConfig
from miso.types import CorrectedOCR, RetrievedSummary


SYSTEM_PROMPT = (
    "You extract a structured note from a page image.\n"
    "The image is the source of truth; the OCR text is a weak hint — useful for "
    "disambiguating handwriting, but the image always wins.\n"
    "Call the `emit_structured_note` tool to return the note as an ordered list "
    "of document blocks that mirror the page's layout:\n"
    "  - `heading` (with `level` 1-3) for titles and section headings;\n"
    "  - `list` (with nested `items`) for bulleted/enumerated points — use the "
    "OCR's preserved line breaks and indentation to recover nesting;\n"
    "  - `paragraph` for running prose; `equation` (LaTeX) for math.\n"
    "Be FAITHFUL to the page's own structure: keep an outline as lists; only use "
    "`paragraph` where the writer actually wrote running prose. Do not rewrite an "
    "outline into prose, or merge separate bullets into a paragraph.\n"
    "Also fill the piggybacked summary fields:\n"
    "  - `summary_topic_line`: a single short line naming the topic/section.\n"
    "  - `summary_gist`: 2-4 sentences (~150 tokens) describing what the note covers.\n"
)


def assemble_prompt(
    *,
    corrected_ocr: CorrectedOCR | None,
    retrieved: list[RetrievedSummary] | None = None,
    glossary: list[str] | None = None,
    cfg: ExtractionConfig,
) -> str:
    parts: list[str] = [SYSTEM_PROMPT, ""]

    if cfg.use_ocr_hint and corrected_ocr is not None:
        # Prefer layout text so structure survives; fall back to flat text.
        ocr_hint = corrected_ocr.layout_text or corrected_ocr.corrected_text
        parts.append("OCR (weak hint, line breaks and indentation preserved):")
        parts.append(ocr_hint)
        parts.append("")

    parts.append("Extract the structured note as JSON. Include the piggybacked summary fields.")
    return "\n".join(parts)
