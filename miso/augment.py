"""Build the text portion of the extraction prompt."""
from __future__ import annotations

from dataclasses import replace

from miso.config import ExtractionConfig
from miso.layout import render_layout_text
from miso.prompts import load as load_prompt
from miso.types import CorrectedOCR, RetrievedSummary

# Marker wrapped around OCR words the reader was unsure of, spliced inline into
# the OCR hint. Guillemets + the literal "OCR?" tag make it unique: nobody writes
# «...» by hand, so the model can never mistake a hint for transcribed ink, and it
# never echoes the marker back. Strip/detect with the regex  «OCR\?[^»]*» .
_FLAG_OPEN = "«OCR? "
_FLAG_CLOSE = "»"


# The extraction system prompt is an editable Markdown file — see miso/prompts/.
SYSTEM_PROMPT = load_prompt("extraction_system")


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
