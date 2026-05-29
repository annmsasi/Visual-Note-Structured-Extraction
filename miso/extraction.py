"""Extraction adapter — a Protocol + a deterministic stub + an Anthropic-backed implementation.

The stub returns canned JSON keyed by the assembled prompt so the pipeline
runs end-to-end without an API key. The Anthropic adapter expects the model
to emit strict JSON including the piggybacked `summary_topic_line` and
`summary_gist` fields (no separate summariser call).
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
from typing import Protocol

from miso.augment import assemble_prompt
from miso.config import ExtractionConfig
from miso.types import CorrectedOCR, ExtractedNote, Note, RetrievedSummary

log = logging.getLogger(__name__)


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
    returns canned JSON derived from the OCR text.
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
        prompt = assemble_prompt(
            corrected_ocr=corrected_ocr,
            retrieved=retrieved,
            glossary=glossary,
            cfg=cfg,
        )
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        ocr_text = corrected_ocr.corrected_text if corrected_ocr else "(no OCR)"
        topic = ocr_text.split()[0] if ocr_text.split() else "unknown"
        return ExtractedNote(
            note_id=note.note_id,
            structured_json={
                "stub": True,
                "course_id": note.course_id,
                "ocr_text": ocr_text,
                "glossary_used": glossary,
                "retrieved_used": [r.summary.note_id for r in retrieved],
                "prompt_digest": digest,
            },
            summary_topic_line=f"Notes on {topic}",
            summary_gist=ocr_text,
            model_id="stub-extractor-v1",
        )


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

        msg = self._client.messages.create(
            model=self.model_id,
            max_tokens=4096,
            messages=[{"role": "user", "content": content}],
        )
        raw_text = "".join(
            block.text for block in msg.content if hasattr(block, "text")
        )

        try:
            payload = json.loads(_extract_json(raw_text))
        except Exception as e:
            log.warning("could not parse extractor JSON (%s); raw head=%r", e, raw_text[:200])
            payload = {"raw": raw_text, "parse_error": str(e)}

        return ExtractedNote(
            note_id=note.note_id,
            structured_json=payload,
            summary_topic_line=str(payload.get("summary_topic_line", "")),
            summary_gist=str(payload.get("summary_gist", "")),
            model_id=self.model_id,
        )


def _extract_json(s: str) -> str:
    """Locate the outermost JSON object in `s`, tolerant of code fences."""
    s = s.strip()
    if s.startswith("```"):
        s = s.lstrip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        return s[start:end + 1]
    return s


def make_extractor(model_id: str) -> ExtractionAdapter:
    if model_id.startswith("stub"):
        return StubExtractor()
    if model_id.startswith("claude"):
        import os
        return AnthropicExtractor(api_key=os.environ["ANTHROPIC_API_KEY"], model_id=model_id)
    raise ValueError(f"Unknown extractor for model_id={model_id!r}")
