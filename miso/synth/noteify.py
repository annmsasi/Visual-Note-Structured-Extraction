"""Rewrite clean structured content into telegraphic 'student note' text.

Surface-only and intended LOSSLESS: abbreviations, fragments, bullets — but keep
every fact, heading, and list item and all technical terms, so the clean
structured source stays a fair extraction gold (eval_design_v1.md §3.2). The
returned string is BOTH what gets rendered to the page AND the verbatim
transcription gold, so they are consistent by construction.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def _blocks_to_plain(title: str, blocks: list[dict]) -> str:
    lines = [title]
    for b in blocks:
        t = b.get("type")
        if t in ("heading", "paragraph"):
            lines.append(b.get("text", ""))
        elif t == "equation":
            lines.append(b.get("latex", ""))
        elif t == "list":
            lines.extend("  - " + it.get("text", "") for it in b.get("items", []))
    return "\n".join(ln for ln in lines if ln.strip())


_SYS = (
    "Rewrite the lecture content as a STUDENT'S HANDWRITTEN NOTES page: terse and "
    "telegraphic — fragments not full sentences, common abbreviations (w/, ->, esp., "
    "b/c, e.g., vs), bullet points, indentation for sub-points. STRICT: keep EVERY "
    "fact, heading, and list item; do not add, drop, merge, or reorder information; "
    "keep all technical terms and numbers EXACTLY as written (never abbreviate a "
    "technical term). Two leading spaces per indent level; one item/idea per line. "
    "Output ONLY the note text."
)


def noteify_stub(title: str, blocks: list[dict]) -> str:
    """Deterministic, lossless: headings/paragraphs as lines, list items as bullets.
    (The LLM path makes it genuinely telegraphic; this keeps the pipeline runnable
    without an API key.)"""
    return _blocks_to_plain(title, blocks)


def noteify_llm(title: str, blocks: list[dict], model: str = "claude-sonnet-4-6") -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    plain = _blocks_to_plain(title, blocks)
    msg = client.messages.create(
        model=model, max_tokens=2000, system=_SYS,
        messages=[{"role": "user", "content": plain}],
    )
    out = "\n".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
    return out or plain


def noteify(title: str, blocks: list[dict], use_llm: bool = True,
            model: str = "claude-sonnet-4-6") -> str:
    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return noteify_llm(title, blocks, model)
        except Exception as e:  # noqa: BLE001
            log.warning("noteify_llm failed (%s); using stub", e)
    return noteify_stub(title, blocks)
