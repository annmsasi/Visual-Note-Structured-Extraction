# miso (base pipeline)

Pipeline package: OCR -> schema-forced LLM extraction -> export. Stateless, no cache.
Setup and usage are in the repo-root README; this lists the modules.

| Module | Holds |
|---|---|
| `config.py` | `RunConfig` and the OCR / extraction config. |
| `types.py` | Dataclasses passed between components. |
| `trace.py` | Per-note JSONL trace writer. |
| `pipeline.py` | The per-note OCR -> extraction loop. |
| `ocr.py` | `StubOCR`, `AzureOCR`, `PaddleOCR`, `TesseractOCR`, and a content-addressed OCR cache. |
| `layout.py` | Reconstruct lines and indentation from OCR word boxes. |
| `extraction.py` | `StubExtractor`, `AnthropicExtractor`, `OpenAIVisionExtractor` (schema-forced IR). |
| `augment.py` | Assemble the extraction prompt (image + OCR hint). |
| `document.py` | Document IR schema and validation. |
| `export.py` | Render the IR to HTML and upload to Google Docs. |
| `replay.py` | Pipeline driver and `demo` / `note` CLI. |
| `eval/` | CER, WER, structural F1, faithfulness, bootstrap CIs. |
| `tests/` | Unit tests. |

Run tests:
```bash
python -m unittest discover -s miso/tests
```
