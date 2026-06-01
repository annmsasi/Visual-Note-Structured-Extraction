"""Extraction adapters: a Protocol, a deterministic stub, and an Anthropic implementation."""
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
    """Deterministic extractor that maps layout-structured OCR to a document IR."""

    def extract(
        self,
        *,
        note: Note,
        corrected_ocr: CorrectedOCR | None,
        retrieved: list[RetrievedSummary],
        glossary: list[str],
        cfg: ExtractionConfig,
    ) -> ExtractedNote:
        assemble_prompt(
            corrected_ocr=corrected_ocr, retrieved=retrieved, glossary=glossary, cfg=cfg,
        )
        layout = (corrected_ocr.layout_text or corrected_ocr.corrected_text) if corrected_ocr else ""
        payload = _layout_to_document(layout, course_id=note.course_id)
        return _to_note(note, payload, model_id="stub-extractor-v1")


def _layout_to_document(layout_text: str, *, course_id: str) -> dict:
    """Map layout text to a document IR by classifying each line's shape."""
    lines = [line for line in layout_text.splitlines() if line.strip()]
    title = lines[0].strip() if lines else f"Notes ({course_id})"
    blocks: list[dict] = []
    items: list[dict] = []

    def flush():
        if items:
            blocks.append({"type": "list", "items": list(items)})
            items.clear()

    for line in lines[1:]:
        depth = (len(line) - len(line.lstrip(" "))) // 2
        text = line.strip()
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
        "summary_gist": " ".join(lines[1:4]),
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

        # Force the tool so the model returns schema-shaped blocks.
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
