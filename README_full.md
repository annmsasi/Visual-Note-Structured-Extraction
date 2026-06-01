# Visual-Note-Structured-Extraction — `full-pipeline` branch

The complete **miso** pipeline, vendored into this repo as `miso/`. The
original team scripts (`preprocess_test.py`, `ocr_test.py`, `extract_test.py`)
are kept intact; this branch adds the full pipeline alongside them.

## Pipeline

```
preprocess (downscale/normalize)
  → Azure Document Intelligence OCR  (per-word confidence, bboxes, line/indent layout)
  → lexicon correction               (shape-aware, per-course term cache)
  → retrieval / RAG                  (BM25 + optional dense, reranked)
  → Claude extraction                (schema-forced document IR; fixes OCR errors)
  → write-back to miso_cache.db
  → export                           (HTML, and Google Docs with --drive)
```

Math in equations renders to inline Unicode (e.g. `∑ᵢ₌₁ⁿ xᵢ`), never images or
raw LaTeX.

## Setup

```bash
pip install -r requirements.txt
```

`.env` (gitignored) needs:

```
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
ANTHROPIC_API_KEY=...
# optional
MISO_EXTRACTOR=claude-sonnet-4-6
```

For `--drive`, also drop a GCP OAuth desktop-client `credentials.json` in the
repo root (Drive API enabled). First run opens a browser to consent once.

## Run

```bash
python run_full_pipeline.py                      # first image in data/inbox
python run_full_pipeline.py data/inbox/notes.jpg
python run_full_pipeline.py --drive              # also create a Google Doc
```

Outputs `<note_id>.html` and, with `--drive`, a Google Doc named after the
note's title.

## Tests

```bash
python -m unittest discover -s miso/tests
```

## Notes

- `miso/` is a **vendored snapshot** of the standalone miso repo, so this branch
  is self-contained. The source of truth is the separate `miso_text_extraction`
  project; re-sync if it changes.
- Dense retrieval + cross-encoder rerank need `sentence-transformers` (heavy);
  without it, retrieval runs BM25-only. For a single note from a cold cache,
  lexicon and retrieval are effectively no-ops anyway.
