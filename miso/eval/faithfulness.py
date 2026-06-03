"""LLM-as-judge faithfulness check — an optional secondary metric.

CER and term-recall cannot catch a *confident hallucination*: an extracted claim
with no support on the page. This module scores each note's extraction for
faithfulness against the OCR text. One call per note; opt-in
(`EvalConfig.enable_faithfulness_check`).

A deterministic `StubFaithfulnessJudge` keeps the path testable and runnable
without an API key; `AnthropicFaithfulnessJudge` is the real judge.
"""
from __future__ import annotations

import logging
import re
from typing import Protocol

log = logging.getLogger(__name__)


def document_claims(doc: dict) -> list[str]:
    """The atomic textual claims of a structured document (body only, no summary)."""
    claims: list[str] = []
    for block in doc.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        t = block.get("type")
        if t in ("heading", "paragraph") and block.get("text"):
            claims.append(block["text"])
        elif t == "equation" and block.get("latex"):
            claims.append(block["latex"])
        elif t == "list":
            for it in block.get("items") or []:
                if isinstance(it, dict) and it.get("text"):
                    claims.append(it["text"])
                elif isinstance(it, str):
                    claims.append(it)
    if doc.get("title"):
        claims.insert(0, doc["title"])
    return claims


class FaithfulnessJudge(Protocol):
    def judge(self, *, doc: dict, ocr_text: str) -> dict: ...


class StubFaithfulnessJudge:
    """Token-overlap proxy: a claim is 'supported' if at least `threshold` of its
    content tokens appear in the OCR text. Deterministic, no API — a coarse
    stand-in for the real judge, useful for tests and dry runs.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def judge(self, *, doc: dict, ocr_text: str) -> dict:
        haystack = set(_content_tokens(ocr_text))
        claims = document_claims(doc)
        unsupported: list[str] = []
        scored = 0
        for claim in claims:
            toks = _content_tokens(claim)
            if not toks:
                continue
            scored += 1
            overlap = sum(1 for t in toks if t in haystack) / len(toks)
            if overlap < self.threshold:
                unsupported.append(claim)
        score = 1.0 - (len(unsupported) / scored) if scored else 1.0
        return {"score": score, "n_claims": scored, "unsupported": unsupported}


_JUDGE_SYSTEM = (
    "You are a strict faithfulness judge for a handwritten-note extraction system. "
    "Given the OCR text of a page and the claims the extractor produced, decide for "
    "each claim whether it is SUPPORTED by the OCR (its content is present, allowing "
    "for spelling/normalization fixes) or UNSUPPORTED (added, hallucinated, or "
    "contradicted). Be conservative: if a claim asserts specifics absent from the "
    "OCR, mark it unsupported."
)

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "supported": {"type": "boolean"},
                },
                "required": ["index", "supported"],
            },
        }
    },
    "required": ["verdicts"],
}


class AnthropicFaithfulnessJudge:
    """One Claude call per note → a faithfulness score + the unsupported claims."""

    def __init__(self, api_key: str, model_id: str = "claude-haiku-4-5-20251001"):
        from anthropic import Anthropic
        self._client = Anthropic(api_key=api_key)
        self.model_id = model_id

    def judge(self, *, doc: dict, ocr_text: str) -> dict:
        claims = document_claims(doc)
        if not claims:
            return {"score": 1.0, "n_claims": 0, "unsupported": []}
        numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(claims))
        prompt = (
            f"OCR text of the page:\n---\n{ocr_text}\n---\n\n"
            f"Claims produced by the extractor:\n{numbered}\n\n"
            "Call emit_verdicts with exactly one verdict per claim index."
        )
        msg = self._client.messages.create(
            model=self.model_id,
            max_tokens=1024,
            system=_JUDGE_SYSTEM,
            tools=[{"name": "emit_verdicts",
                    "description": "Per-claim support verdicts.",
                    "input_schema": _VERDICT_SCHEMA}],
            tool_choice={"type": "tool", "name": "emit_verdicts"},
            messages=[{"role": "user", "content": prompt}],
        )
        verdicts = []
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "emit_verdicts":
                verdicts = block.input.get("verdicts", [])
                break
        unsupported = [
            claims[v["index"]] for v in verdicts
            if 0 <= v.get("index", -1) < len(claims) and not v.get("supported")
        ]
        score = 1.0 - (len(unsupported) / len(claims))
        return {"score": score, "n_claims": len(claims), "unsupported": unsupported}


def make_judge(model_id: str = "stub") -> FaithfulnessJudge:
    if model_id.startswith("stub"):
        return StubFaithfulnessJudge()
    if model_id.startswith("claude"):
        import os
        return AnthropicFaithfulnessJudge(
            api_key=os.environ["ANTHROPIC_API_KEY"], model_id=model_id,
        )
    raise ValueError(f"Unknown judge for model_id={model_id!r}")


def score_faithfulness(records: list[dict], judge: FaithfulnessJudge) -> dict[str, dict]:
    """Score each trace record's extraction against its OCR. Returns {note_id: verdict}."""
    out: dict[str, dict] = {}
    for r in records:
        ext = (r.get("extraction") or {}).get("structured_json") or {}
        ocr_text = ((r.get("corrected_ocr") or {}).get("corrected_text")
                    or (r.get("ocr_raw") or {}).get("raw_text", ""))
        out[r["note_id"]] = judge.judge(doc=ext, ocr_text=ocr_text)
    return out


def _content_tokens(s: str) -> list[str]:
    return [t for t in re.sub(r"[^0-9a-z]+", " ", s.lower()).split() if len(t) > 2]
