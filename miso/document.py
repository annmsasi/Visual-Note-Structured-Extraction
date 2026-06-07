"""Document IR: the block structure shared between extraction and rendering."""
from __future__ import annotations

from typing import Any

# JSON Schema handed to the model as a tool's `input_schema`.
DOCUMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Page/section title, e.g. 'Chapter 3 (1763–1783)'."},
        "blocks": {
            "type": "array",
            "description": "Ordered document body, top to bottom.",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string",
                             "enum": ["heading", "paragraph", "list", "equation", "figure"]},
                    "level": {"type": "integer", "minimum": 1, "maximum": 3,
                              "description": "Heading depth (heading blocks only)."},
                    "text": {"type": "string", "description": "Text for heading/paragraph blocks."},
                    "latex": {"type": "string", "description": "LaTeX source for equation blocks."},
                    "description": {"type": "string",
                                    "description": "What a figure depicts (figure blocks only)."},
                    "bbox": {"type": "array", "items": {"type": "number"},
                             "description": "Optional figure location, normalized 0–1 as "
                                            "[x, y, width, height] (figure blocks only)."},
                    "image": {"type": "string",
                              "description": "Reserved for the extracted figure image, filled by a "
                                             "later step — leave unset (figure blocks only)."},
                    "items": {
                        "type": "array",
                        "description": "List entries (list blocks only).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "level": {"type": "integer", "minimum": 0,
                                          "description": "Nesting depth, 0 = top level."},
                            },
                            "required": ["text"],
                        },
                    },
                },
                "required": ["type"],
            },
        },
        "summary_topic_line": {"type": "string", "description": "One short line naming the topic."},
        "summary_gist": {"type": "string", "description": "2–4 sentences on what the note covers."},
    },
    "required": ["title", "blocks", "summary_topic_line", "summary_gist"],
}

_BLOCK_TYPES = {"heading", "paragraph", "list", "equation", "figure"}


def validate(payload: Any) -> dict[str, Any]:
    """Coerce a model payload into a well-formed document, dropping malformed blocks."""
    if not isinstance(payload, dict):
        return _empty("(extraction returned no object)")

    title = _as_str(payload.get("title")) or "(untitled)"
    blocks_out: list[dict[str, Any]] = []
    for raw in payload.get("blocks") or []:
        block = _clean_block(raw)
        if block is not None:
            blocks_out.append(block)

    return {
        "title": title,
        "blocks": blocks_out,
        "summary_topic_line": _as_str(payload.get("summary_topic_line")),
        "summary_gist": _as_str(payload.get("summary_gist")),
    }


def _clean_block(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    t = raw.get("type")
    if t not in _BLOCK_TYPES:
        return None
    if t == "heading":
        text = _as_str(raw.get("text"))
        if not text:
            return None
        level = raw.get("level")
        level = level if isinstance(level, int) and 1 <= level <= 3 else 1
        return {"type": "heading", "level": level, "text": text}
    if t == "paragraph":
        text = _as_str(raw.get("text"))
        return {"type": "paragraph", "text": text} if text else None
    if t == "equation":
        latex = _as_str(raw.get("latex")) or _as_str(raw.get("text"))
        return {"type": "equation", "latex": latex} if latex else None
    if t == "list":
        items = []
        for it in raw.get("items") or []:
            if isinstance(it, str):
                items.append({"text": it, "level": 0})
            elif isinstance(it, dict) and _as_str(it.get("text")):
                lvl = it.get("level")
                items.append({"text": _as_str(it["text"]),
                              "level": lvl if isinstance(lvl, int) and lvl >= 0 else 0})
        return {"type": "list", "items": items} if items else None
    if t == "figure":
        # The VLM describes the figure now; a later step crops the page image and
        # fills `image`. The `image` slot is ALWAYS present (empty until then) so
        # every downstream step can carry it; `bbox` is an optional location hint.
        description = (_as_str(raw.get("description"))
                       or _as_str(raw.get("caption")) or _as_str(raw.get("text")))
        if not description:
            return None
        block = {"type": "figure", "description": description, "image": _as_str(raw.get("image"))}
        bbox = _clean_bbox(raw.get("bbox"))
        if bbox is not None:
            block["bbox"] = bbox
        return block
    return None


def _clean_bbox(v: Any) -> list[float] | None:
    """A figure's normalized [x, y, width, height] box, or None if not 4 numbers."""
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        return [float(n) for n in v]
    except (TypeError, ValueError):
        return None


def _empty(title: str) -> dict[str, Any]:
    return {"title": title, "blocks": [], "summary_topic_line": "", "summary_gist": ""}


def _as_str(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""
