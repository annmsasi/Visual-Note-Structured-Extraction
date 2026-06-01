"""Extraction adapter — a Protocol + a deterministic stub + an Anthropic-backed implementation.

The stub returns canned JSON keyed by the assembled prompt so the pipeline
runs end-to-end without an API key. The Anthropic adapter expects the model
to emit strict JSON including the piggybacked `summary_topic_line` and
`summary_gist` fields (no separate summariser call).
"""
from __future__ import annotations

import base64
import logging
import mimetypes
from typing import Protocol

from miso.augment import assemble_prompt
from miso.config import ExtractionConfig
from miso.document import DOCUMENT_SCHEMA, validate
from miso.types import CorrectedOCR, ExtractedNote, Note, RetrievedSummary

log = logging.getLogger(__name__)

_TOOL_NAME = "emit_structured_note"


class ExtractionAdapter(Protocol):
    def extract(
        self,
        *,
        note: Note,
        corrected_ocr: CorrectedOCR | None,
        retrieved: list[RetrievedSummary],
        glossary: list[str],
        cfg: ExtractionConfig,
    ) -> ExtractedNote: ...


class StubExtractor:
    """Deterministic stub. Builds the prompt (so augmentation is exercised) and
    turns the layout-structured OCR into a document IR with a simple heuristic:
    the first line is the title, short stand-alone lines become headings, and
    everything else becomes nested list items keyed off the preserved indent.
    Lets the renderer + export path run end-to-end without an API key.
    """

    def extract(
        self,
        *,
        note: Note,
        corrected_ocr: CorrectedOCR | None,
        retrieved: list[RetrievedSummary],
        glossary: list[str],
        cfg: ExtractionConfig,
    ) -> ExtractedNote:
        assemble_prompt(  # exercise augmentation for parity with the real path
            corrected_ocr=corrected_ocr, retrieved=retrieved, glossary=glossary, cfg=cfg,
        )
        layout = (corrected_ocr.layout_text or corrected_ocr.corrected_text) if corrected_ocr else ""
        payload = _layout_to_document(layout, course_id=note.course_id)
        return _to_note(note, payload, model_id="stub-extractor-v1")


def _layout_to_document(layout_text: str, *, course_id: str) -> dict:
    """Crude layout -> IR mapping for the stub (no model judgement).

    Classifies each logical line by shape: a short top-level line that isn't a
    sentence and doesn't start with a date is a heading; long top-level prose is
    a paragraph; everything else (dated entries, fragments, indented points) is
    a list item nested by its indent depth. The real AnthropicExtractor does
    this far better off the image — this only keeps the preview sensible.
    """
    raw_lines = [ln for ln in layout_text.splitlines() if ln.strip()]
    title = raw_lines[0].strip() if raw_lines else f"Notes ({course_id})"
    blocks: list[dict] = []
    items: list[dict] = []

    def flush():
        if items:
            blocks.append({"type": "list", "items": list(items)})
            items.clear()

    for ln in raw_lines[1:]:
        depth = (len(ln) - len(ln.lstrip(" "))) // 2
        text = ln.strip()
        words = text.split()
        starts_with_date = bool(words) and words[0][0].isdigit()
        if depth == 0 and len(words) <= 6 and not starts_with_date and not text.endswith("."):
            flush()
            blocks.append({"type": "heading", "level": 2, "text": text})
        elif depth == 0 and len(words) > 12 and text.endswith("."):
            flush()
            blocks.append({"type": "paragraph", "text": text})
        else:
            items.append({"text": text, "level": depth})
    flush()
    return {
        "title": title,
        "blocks": blocks,
        "summary_topic_line": title,
        "summary_gist": " ".join(raw_lines[1:4]),
    }


class AnthropicExtractor:
    def __init__(self, api_key: str, model_id: str = "claude-opus-4-7"):
        from anthropic import Anthropic
        self._client = Anthropic(api_key=api_key)
        self.model_id = model_id

    def extract(
        self,
        *,
        note: Note,
        corrected_ocr: CorrectedOCR | None,
        retrieved: list[RetrievedSummary],
        glossary: list[str],
        cfg: ExtractionConfig,
    ) -> ExtractedNote:
        text_prompt = assemble_prompt(
            corrected_ocr=corrected_ocr,
            retrieved=retrieved,
            glossary=glossary,
            cfg=cfg,
        )

        content: list[dict] = []
        if cfg.use_image:
            mime, _ = mimetypes.guess_type(str(note.image_path))
            mime = mime or "image/jpeg"
            with open(note.image_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            })
        content.append({"type": "text", "text": text_prompt})

        # Force the document-IR tool so the model must return schema-shaped blocks
        # rather than free-text JSON we have to scrape.
        msg = self._client.messages.create(
            model=self.model_id,
            max_tokens=4096,
            tools=[{
                "name": _TOOL_NAME,
                "description": "Return the page as a structured document.",
                "input_schema": DOCUMENT_SCHEMA,
            }],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
            messages=[{"role": "user", "content": content}],
        )

        payload = None
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use" and block.name == _TOOL_NAME:
                payload = block.input
                break
        if payload is None:
            log.warning("extractor returned no tool_use block; stop_reason=%s", msg.stop_reason)
            payload = {}

        return _to_note(note, payload, model_id=self.model_id)


def _to_note(note: Note, payload: dict, *, model_id: str) -> ExtractedNote:
    doc = validate(payload)
    return ExtractedNote(
        note_id=note.note_id,
        structured_json=doc,
        summary_topic_line=doc["summary_topic_line"],
        summary_gist=doc["summary_gist"],
        model_id=model_id,
    )


def make_extractor(model_id: str) -> ExtractionAdapter:
    if model_id.startswith("stub"):
        return StubExtractor()
    if model_id.startswith("claude"):
        import os
        return AnthropicExtractor(api_key=os.environ["ANTHROPIC_API_KEY"], model_id=model_id)
    raise ValueError(f"Unknown extractor for model_id={model_id!r}")
