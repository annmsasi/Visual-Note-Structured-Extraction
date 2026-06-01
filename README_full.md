# Visual-Note-Structured-Extraction — full-pipeline

The full miso pipeline (vendored in `miso/`): Azure OCR -> lexicon correction ->
retrieval -> Claude (schema-forced document IR) -> HTML / Google Docs. The
original team scripts (`preprocess_test.py`, `ocr_test.py`, `extract_test.py`)
are kept alongside.

## Install
```bash
./install.sh        # creates .venv and installs requirements
```

## Configure
`.env` in the repo root:
```
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=...
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
ANTHROPIC_API_KEY=...
```
For `--drive`, also place a GCP OAuth desktop `credentials.json` in the repo root.

## Run
```bash
.venv/bin/python run_full_pipeline.py data/inbox/notes.jpg
.venv/bin/python run_full_pipeline.py --drive     # also create a Google Doc
```

## Test
```bash
.venv/bin/python -m unittest discover -s miso/tests
```

Equations render to inline Unicode. Dense retrieval/rerank needs the optional
`sentence-transformers` (BM25-only without it). Eval/experiment scripts live on
the `eval` branch.
