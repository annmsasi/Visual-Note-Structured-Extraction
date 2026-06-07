"""Extraction adapters: a Protocol, a deterministic stub, and an Anthropic implementation."""
from __future__ import annotations

import base64
import logging
import mimetypes
from typing import Protocol

from miso.augment import assemble_prompt
from miso.config import ExtractionConfig
from miso.document import DOCUMENT_SCHEMA, validate
from miso.prompts import load as load_prompt
from miso.types import CorrectedOCR, ExtractedNote, Note, RetrievedSummary

log = logging.getLogger(__name__)

_TOOL_NAME = "emit_structured_note"
_SUMMARY_TOOL = "emit_summary"

# These two system prompts are editable Markdown files — see miso/prompts/.
COMBINE_SYSTEM = load_prompt("combine_system")
SUMMARY_SYSTEM = load_prompt("summary_system")

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "topic_line": {"type": "string"},
        "gist": {"type": "string"},
    },
    "required": ["topic_line", "gist"],
    "additionalProperties": False,
}


def _summary_input(note_doc: dict) -> str:
    from miso.export import render_note_markdown
    return ("Summarize this complete note:\n\n" + render_note_markdown(note_doc))


def _merge_prompt(page_docs: list[dict]) -> str:
    import json
    pages = "\n\n".join(
        f"=== Page {i + 1} ===\n{json.dumps(d, ensure_ascii=False)}"
        for i, d in enumerate(page_docs)
    )
    return ("Per-page structured notes of one document, in reading order:\n\n"
            + pages + "\n\nMerge them into a single document.")


class ExtractionAdapter(Protocol):
    def extract(
        self,
        *,
        note: Note,
        corrected_ocr: CorrectedOCR | None,
        retrieved: list[RetrievedSummary],
        glossary: list[str],
        cfg: ExtractionConfig,
        prior_pages_md: list[str] | None = None,
    ) -> ExtractedNote: ...

    def summarize(self, note_doc: dict) -> tuple[str, str]:
        """Whole-note summary (topic_line, gist) over a complete (merged) note IR."""
        ...


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
        prior_pages_md: list[str] | None = None,
    ) -> ExtractedNote:
        assemble_prompt(
            corrected_ocr=corrected_ocr, retrieved=retrieved, glossary=glossary, cfg=cfg,
            prior_pages_md=prior_pages_md,
        )
        layout = (corrected_ocr.layout_text or corrected_ocr.corrected_text) if corrected_ocr else ""
        payload = _layout_to_document(layout, course_id=note.course_id)
        return _to_note(note, payload, model_id="stub-extractor-v1",
                        figures_dir=cfg.figures_dir)

    def combine(self, page_docs: list[dict]) -> dict:
        """Deterministic merge: concatenate blocks, keep the first title."""
        if not page_docs:
            return {"title": "(empty)", "blocks": [], "summary_topic_line": "", "summary_gist": ""}
        return {
            "title": page_docs[0].get("title", "(untitled)"),
            "blocks": [b for d in page_docs for b in d.get("blocks", [])],
            "summary_topic_line": "",
            "summary_gist": "",
        }

    def summarize(self, note_doc: dict) -> tuple[str, str]:
        """Deterministic whole-note summary: title + the note's leading text."""
        topic = note_doc.get("title") or "(untitled)"
        texts: list[str] = []
        for b in note_doc.get("blocks", []):
            if b.get("text"):
                texts.append(b["text"])
            for it in b.get("items", []):
                if isinstance(it, dict) and it.get("text"):
                    texts.append(it["text"])
            if len(texts) >= 3:
                break
        return topic, " ".join(texts[:3])


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
        prior_pages_md: list[str] | None = None,
    ) -> ExtractedNote:
        text_prompt = assemble_prompt(
            corrected_ocr=corrected_ocr,
            retrieved=retrieved,
            glossary=glossary,
            cfg=cfg,
            prior_pages_md=prior_pages_md,
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

        return _to_note(note, payload, model_id=self.model_id, figures_dir=cfg.figures_dir)

    def summarize(self, note_doc: dict) -> tuple[str, str]:
        """Dedicated whole-note summary via a focused, text-only call."""
        msg = self._client.messages.create(
            model=self.model_id,
            max_tokens=512,
            system=SUMMARY_SYSTEM,
            tools=[{
                "name": _SUMMARY_TOOL,
                "description": "Return the note's topic line and gist.",
                "input_schema": SUMMARY_SCHEMA,
            }],
            tool_choice={"type": "tool", "name": _SUMMARY_TOOL},
            messages=[{"role": "user", "content": _summary_input(note_doc)}],
        )
        payload = next((b.input for b in msg.content
                        if getattr(b, "type", None) == "tool_use" and b.name == _SUMMARY_TOOL), None) or {}
        return (payload.get("topic_line") or note_doc.get("title", ""),
                payload.get("gist") or "")

    def combine(self, page_docs: list[dict]) -> dict:
        """REDUCE: merge per-page IRs into one document via a text-only call."""
        msg = self._client.messages.create(
            model=self.model_id,
            max_tokens=8192,
            system=COMBINE_SYSTEM,
            tools=[{
                "name": _TOOL_NAME,
                "description": "Return the merged document.",
                "input_schema": DOCUMENT_SCHEMA,
            }],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
            messages=[{"role": "user", "content": _merge_prompt(page_docs)}],
        )
        payload = next((b.input for b in msg.content
                        if getattr(b, "type", None) == "tool_use" and b.name == _TOOL_NAME), None)
        if payload is None:
            log.warning("combine returned no tool_use block; stop_reason=%s", msg.stop_reason)
            payload = {}
        return validate(payload)


class OpenAIVisionExtractor:
    """Extraction via any OpenAI-compatible vision endpoint — a local server
    (vLLM / Ollama) or a hosted gateway (OpenRouter, Together) serving an
    open-weight VLM. Drop-in for `AnthropicExtractor`: same image + OCR-hint
    prompt, same schema-forced `emit_structured_note` tool, same document IR.

    Recommended model: Qwen2.5-VL — the strongest open VLM for handwriting +
    structured extraction. Point it at a server via OPENAI_BASE_URL / OPENAI_API_KEY.
    """

    _OPENROUTER_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model_id: str, *, base_url: str | None = None, api_key: str | None = None):
        import os
        from openai import OpenAI
        # Zero-setup default: set OPENROUTER_API_KEY and pass a model id like
        # "qwen/qwen2.5-vl-72b-instruct". Falls back to a generic OpenAI-compatible
        # server (vLLM/Ollama) via OPENAI_BASE_URL / OPENAI_API_KEY.
        router_key = os.environ.get("OPENROUTER_API_KEY")
        if base_url is None and api_key is None and router_key:
            base_url, api_key = self._OPENROUTER_URL, router_key
        self._client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
        )
        self.model_id = model_id

    def extract(
        self,
        *,
        note: Note,
        corrected_ocr: CorrectedOCR | None,
        retrieved: list[RetrievedSummary],
        glossary: list[str],
        cfg: ExtractionConfig,
        prior_pages_md: list[str] | None = None,
    ) -> ExtractedNote:
        text_prompt = assemble_prompt(
            corrected_ocr=corrected_ocr, retrieved=retrieved, glossary=glossary, cfg=cfg,
            prior_pages_md=prior_pages_md,
        )
        content: list[dict] = []
        if cfg.use_image:
            mime, _ = mimetypes.guess_type(str(note.image_path))
            mime = mime or "image/jpeg"
            with open(note.image_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}})
        content.append({"type": "text", "text": text_prompt})

        tools = [{"type": "function", "function": {
            "name": _TOOL_NAME,
            "description": "Return the page as a structured document.",
            "parameters": DOCUMENT_SCHEMA,
        }}]
        msg = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=4096,
            messages=[{"role": "user", "content": content}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": _TOOL_NAME}},
        )
        return _to_note(note, _payload_from_openai(msg), model_id=self.model_id,
                        figures_dir=cfg.figures_dir)

    def summarize(self, note_doc: dict) -> tuple[str, str]:
        """Dedicated whole-note summary via a focused, text-only call."""
        msg = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=512,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": _summary_input(note_doc)},
            ],
            tools=[{"type": "function", "function": {
                "name": _SUMMARY_TOOL,
                "description": "Return the note's topic line and gist.",
                "parameters": SUMMARY_SCHEMA,
            }}],
            tool_choice={"type": "function", "function": {"name": _SUMMARY_TOOL}},
        )
        payload = _payload_from_openai(msg)
        return (payload.get("topic_line") or note_doc.get("title", ""),
                payload.get("gist") or "")

    def combine(self, page_docs: list[dict]) -> dict:
        """REDUCE: merge per-page IRs into one document (text-only — works on any model)."""
        msg = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=8192,
            messages=[
                {"role": "system", "content": COMBINE_SYSTEM},
                {"role": "user", "content": _merge_prompt(page_docs)},
            ],
            tools=[{"type": "function", "function": {
                "name": _TOOL_NAME,
                "description": "Return the merged document.",
                "parameters": DOCUMENT_SCHEMA,
            }}],
            tool_choice={"type": "function", "function": {"name": _TOOL_NAME}},
        )
        return validate(_payload_from_openai(msg))


def _payload_from_openai(msg) -> dict:
    """Pull the tool-call arguments, falling back to a JSON object in the content."""
    import json
    choice = msg.choices[0].message
    calls = getattr(choice, "tool_calls", None)
    if calls:
        try:
            return json.loads(calls[0].function.arguments)
        except (ValueError, AttributeError):
            pass
    text = getattr(choice, "content", None) or ""
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        log.warning("OpenAI-compatible extractor returned no parseable JSON")
        return {}


def _to_note(note: Note, payload: dict, *, model_id: str,
             figures_dir=None) -> ExtractedNote:
    doc = validate(payload)
    # Per-page seam for figure extraction: `note.image_path` is this single page, so
    # each figure's normalized bbox maps cleanly onto it. Done here (not after the
    # multi-page combine) because the merge loses page-to-block association.
    if figures_dir is not None:
        from miso.figures import crop_figures
        crop_figures(doc, note.image_path, figures_dir, note_id=note.note_id)
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
    # any other id → an OpenAI-compatible vision endpoint (open-weight VLM)
    return OpenAIVisionExtractor(model_id)
