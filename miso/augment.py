"""Build the text portion of the extraction prompt."""
from __future__ import annotations

from dataclasses import replace

from miso.config import ExtractionConfig
from miso.layout import render_layout_text
from miso.types import CorrectedOCR, RetrievedSummary

# Marker wrapped around OCR words the reader was unsure of, spliced inline into
# the OCR hint. Guillemets + the literal "OCR?" tag make it unique: nobody writes
# «...» by hand, so the model can never mistake a hint for transcribed ink, and it
# never echoes the marker back. Strip/detect with the regex  «OCR\?[^»]*» .
_FLAG_OPEN = "«OCR? "
_FLAG_CLOSE = "»"


SYSTEM_PROMPT = (
    "You extract a structured note from a page image.\n"
    "The page image is the source of truth. The OCR text, the related notes from "
    "earlier in this course, and any inline term suggestions are all hints — use "
    "them to disambiguate messy handwriting, abbreviations, and notation, but when "
    "a hint disagrees with the image, follow the image.\n"
    "A few OCR words the reader was unsure of are marked inline as "
    "WORD «OCR? term | term» — the guillemets wrap course terms the word might "
    "actually be; they are not written on the page. Pick one only if it matches "
    "the writing; otherwise transcribe what you see. Never copy the «OCR? ...» "
    "marker into your output.\n"
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


def _ocr_hint(corrected_ocr: CorrectedOCR) -> str:
    """Render the OCR hint, splicing flagged candidates inline at their word.

    The markers live only in this LLM-facing string; `corrected_ocr.layout_text`
    stays clean so the structured/stub path never parses a marker.
    """
    flags = {f.token_index: f for f in corrected_ocr.flags}
    if not flags or not corrected_ocr.words:
        return corrected_ocr.layout_text or corrected_ocr.corrected_text
    annotated = []
    for i, w in enumerate(corrected_ocr.words):
        f = flags.get(i)
        if f is None:
            annotated.append(w)
            continue
        cands = " | ".join(c.term for c in f.candidates)
        annotated.append(replace(w, text=f"{w.text} {_FLAG_OPEN}{cands}{_FLAG_CLOSE}"))
    return render_layout_text(annotated)


def assemble_prompt(
    *,
    corrected_ocr: CorrectedOCR | None,
    retrieved: list[RetrievedSummary],
    glossary: list[str],
    cfg: ExtractionConfig,
) -> str:
    parts: list[str] = [SYSTEM_PROMPT, ""]

    if cfg.use_retrieved_summaries and retrieved:
        parts.append(
            "Related notes from earlier in this course (most relevant last) — "
            "background for how this course uses recurring terms and notation. "
            "They are context only: do not copy from them, and the page image "
            "still wins. Each is tagged (note N) by its order in the course:"
        )
        # Most-relevant placed last, adjacent to the OCR.
        for r in reversed(retrieved):
            parts.append(
                f"- (note {r.summary.processing_order}) {r.summary.topic_line}\n"
                f"  {r.summary.gist}"
            )
        parts.append("")

    flags = corrected_ocr.flags if corrected_ocr is not None else []
    # Flag mode splices candidates inline into the OCR hint below; the flat
    # glossary list is only the fallback when there are no per-word flags.
    if cfg.use_glossary and not flags and glossary:
        parts.append("Course terms appearing in this note:")
        parts.append("- " + ", ".join(sorted(set(glossary))))
        parts.append("")

    if cfg.use_ocr_hint and corrected_ocr is not None:
        # Prefer layout text so structure survives; splice flag candidates inline
        # unless the glossary channel is disabled.
        if cfg.use_glossary:
            ocr_hint = _ocr_hint(corrected_ocr)
        else:
            ocr_hint = corrected_ocr.layout_text or corrected_ocr.corrected_text
        parts.append("OCR (a hint — line breaks and indentation preserved):")
        parts.append(ocr_hint)
        parts.append("")

    parts.append("Extract the structured note as JSON. Include the piggybacked summary fields.")
    return "\n".join(parts)
