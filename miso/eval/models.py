"""Locked VL model suite for the figure / extraction eval (selected 2026-06-07).

All seven are vision + tool-calling capable. Tool-calling is the hard requirement —
the extractor forces the `emit_structured_note` function tool, and a model whose
OpenRouter providers don't support tool use 404s (this is what rules out
Qwen2.5-VL-72B). The four open models here were smoke-tested live and pass.

Routing (see miso.replay._make_extractor / miso.extraction.make_extractor):
  * `claude-*` ids   -> AnthropicExtractor  (native tool use, no OpenRouter markup)
  * every other id   -> OpenAIVisionExtractor via OPENROUTER_API_KEY

Spread: 4 lineages (Anthropic, Alibaba/Qwen, Zhipu/GLM, Mistral), ~70x cost range,
Qwen at two sizes for a within-family scaling signal. Prices are USD per 1M tokens
at selection time (image tokens billed separately) — informational only.
"""
from __future__ import annotations

EVAL_MODELS: list[dict] = [
    {"id": "claude-opus-4-8",                       "label": "Opus 4.8",            "lineage": "anthropic", "route": "anthropic",  "tier": "frontier",            "price_in": 5.00, "price_out": 25.00},
    {"id": "claude-sonnet-4-6",                     "label": "Sonnet 4.6",          "lineage": "anthropic", "route": "anthropic",  "tier": "frontier-mid",        "price_in": 3.00, "price_out": 15.00},
    {"id": "claude-haiku-4-5",                      "label": "Haiku 4.5",           "lineage": "anthropic", "route": "anthropic",  "tier": "frontier-light",      "price_in": 1.00, "price_out": 5.00},
    {"id": "qwen/qwen3-vl-235b-a22b-instruct",      "label": "Qwen3-VL-235B",       "lineage": "qwen",      "route": "openrouter", "tier": "open-flagship",       "price_in": 0.20, "price_out": 0.88},
    {"id": "qwen/qwen3-vl-8b-instruct",             "label": "Qwen3-VL-8B",         "lineage": "qwen",      "route": "openrouter", "tier": "open-light",          "price_in": 0.08, "price_out": 0.50},
    {"id": "z-ai/glm-4.6v",                         "label": "GLM-4.6V",            "lineage": "zhipu",     "route": "openrouter", "tier": "open-doc-specialist", "price_in": 0.30, "price_out": 0.90},
    {"id": "mistralai/mistral-small-3.2-24b-instruct", "label": "Mistral Small 3.2 24B", "lineage": "mistral", "route": "openrouter", "tier": "open-western",   "price_in": 0.07, "price_out": 0.20},
]

EVAL_MODEL_IDS: list[str] = [m["id"] for m in EVAL_MODELS]

# Smoke-test notes (2026-06-07, pure-image extract on cse138-002):
#   Qwen3-VL-235B  strong (clean title, figure+bbox)         — verified end-to-end via pipeline
#   GLM-4.6V       strong; bbox landed near the hand gold     — possible localization standout
#   Mistral 3.2    works, mostly readable, 1 figure (low box)
#   Qwen3-VL-8B    tool-calling OK but weak: 0 blocks on pure image — lightweight floor
# Not in the suite: InternVL3 is the open doc/chart specialist per the research but is
#   NOT on OpenRouter (self-host via vLLM only). Gemini 2.5/3-Flash is a strong closed
#   alternative for bbox grounding if a Google frontier is wanted.
