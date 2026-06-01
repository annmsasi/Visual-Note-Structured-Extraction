# `miso` — cache / RAG subsystem

Implementation of the cache design from `cache_design_v1.md` (Proposal B + the token/lexicon layer). The pipeline runs end-to-end with stubs; setting env vars and installing extras swaps in real components.

## Layout

| File | What it holds |
|---|---|
| `config.py` | `RunConfig` and the four ablation-config factories. |
| `types.py` | Dataclasses passed between components. |
| `schema.sql` / `db.py` | SQLite schema and connection management (with sqlite-vec fallback). |
| `trace.py` | Per-note JSONL writer. |
| `pipeline.py` | The ordered OCR → lexicon → retrieval → extraction → write-back loop. |
| `lexicon.py` | Shape-aware edit-distance, sighting + promotion, soft confidence reweight. |
| `wordlists.py` | wordfreq-backed common-words filter for lexicon admission. |
| `summary_store.py` | Per-note summary storage and read primitives. |
| `retrieval.py` | Hybrid BM25 + dense, RRF, reranker, two-part gate. |
| `encoders.py` | sentence-transformers embedder + cross-encoder reranker. |
| `ocr.py` | `StubOCR` and `AzureOCR`. |
| `extraction.py` | `StubExtractor` and `AnthropicExtractor`. |
| `augment.py` | Build the text portion of the extraction prompt. |
| `replay.py` | Eval-mode driver + `demo` / `ablation` CLI; auto-wires real vs stub. |
| `eval/` | Read traces, compute CER/WER/structural F1/correction P/R, bootstrap CIs, ramp curves, 2×2 attribution. |
| `tests/test_metrics.py` | Unit tests for the metric functions. |

## Quick start

```bash
python -m miso.replay demo                                 # stubs only
python -m miso.replay ablation                             # 4 cache configs
python -m miso.eval analyze --runs runs/ --synth-gold      # smoke-test eval
python -m miso.eval analyze --runs runs/ --gold gold/      # real eval
python -m unittest miso.tests.test_metrics                 # unit tests
```

## Install

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .[all]
```

Or selectively: `[embeddings,lexicon,vectordb,fuzzy,ocr-azure,llm-anthropic]`.

## Wiring real services

The replay driver auto-selects real vs stub based on env vars and installed packages:

| Component | Real wired when | Else |
|---|---|---|
| OCR | `cfg.ocr.engine="azure"` and the two Azure env vars set | `StubOCR` |
| Embedder | sentence-transformers installed | None (BM25 only) |
| Reranker | sentence-transformers installed | Token-overlap stub |
| Extractor | `cfg.extraction.model_id` starts `"claude"` and `ANTHROPIC_API_KEY` set | `StubExtractor` |

```bash
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://<name>.cognitiveservices.azure.com/"
export AZURE_DOCUMENT_INTELLIGENCE_KEY="<key1-or-key2>"
export ANTHROPIC_API_KEY="sk-ant-..."
export MISO_EXTRACTOR=claude-haiku-4-5-20251001   # optional; default when key is set
```
