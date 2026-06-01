# miso

Pipeline package: OCR -> lexicon correction -> retrieval -> extraction -> export.
Setup and usage are in the repo-root README; this lists the modules.

| Module | Holds |
|---|---|
| `config.py` | `RunConfig` and the ablation-config factories. |
| `types.py` | Dataclasses passed between components. |
| `db.py` / `schema.sql` | SQLite schema and connection management. |
| `trace.py` | Per-note JSONL trace writer. |
| `pipeline.py` | The OCR, lexicon, retrieval, extraction, write-back loop. |
| `ocr.py` | `StubOCR`, `AzureOCR`, and a content-addressed OCR cache. |
| `layout.py` | Reconstruct lines and indentation from OCR word boxes. |
| `lexicon.py` | Per-course term cache: shape-aware matching, promotion, reweighting. |
| `wordlists.py` | Common-word filter for lexicon admission. |
| `retrieval.py` | Hybrid BM25 + dense retrieval, RRF, reranker, gate. |
| `encoders.py` | sentence-transformers embedder and cross-encoder reranker. |
| `summary_store.py` | Per-note summary storage and reads. |
| `extraction.py` | `StubExtractor` and `AnthropicExtractor` (schema-forced IR). |
| `augment.py` | Assemble the extraction prompt. |
| `document.py` | Document IR schema and validation. |
| `export.py` | Render the IR to HTML and upload to Google Docs. |
| `replay.py` | Pipeline driver and `demo` / `ablation` / `note` CLI. |
| `eval/` | CER, WER, structural F1, correction precision/recall, bootstrap CIs. |
| `tests/` | Unit tests. |

Run tests:
```bash
python -m unittest discover -s miso/tests
```
