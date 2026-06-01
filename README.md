# Visual-Note-Structured-Extraction — full-pipeline

The full miso pipeline, vendored in `miso/`. It turns a handwritten note image
into a structured document (HTML, and optionally a Google Doc):

    preprocess (downscale) -> Azure OCR -> lexicon correction -> retrieval
    -> Claude (schema-forced document IR) -> write-back to miso_cache.db
    -> HTML / Google Docs export

`run_full_pipeline.py` is the entry point.

## Install
```bash
./install.sh        # creates .venv and installs requirements
```
Dense retrieval and the cross-encoder reranker are optional and need
`sentence-transformers`; without it, retrieval falls back to BM25:
```bash
.venv/bin/python -m pip install sentence-transformers
```

## Configure
`.env` in the repo root:
```
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
ANTHROPIC_API_KEY=...
MISO_EXTRACTOR=claude-sonnet-4-6        # optional, this is the default
```
Google Docs export (`--drive`) also needs an OAuth client:
1. In a GCP project, enable the Google Drive API.
2. Create an OAuth client ID of type "Desktop app".
3. Download it as `credentials.json` in the repo root.

The first `--drive` run opens a browser for one-time consent and caches a
`token.json`; later runs are silent. Each Doc is placed in a Drive folder named
after the note's course.

## Run
```bash
.venv/bin/python run_full_pipeline.py data/inbox/notes.jpg
.venv/bin/python run_full_pipeline.py --drive          # also create a Google Doc
```
Options:
```
image            note image (default: first in data/inbox)
--course NAME     course id, used as the Drive folder name (default: adhoc)
--model ID        extraction model (default: claude-sonnet-4-6)
--out PATH        HTML output path (default: <note_id>.html)
--drive           upload the result to Google Docs
```
Output is an HTML file and, with `--drive`, a Google Doc. Equations render as
inline Unicode (e.g. `∑ᵢ₌₁ⁿ xᵢ`), never images.

## Test
```bash
.venv/bin/python -m unittest discover -s miso/tests
```

## Layout
```
miso/                  the pipeline package (ocr, layout, lexicon, retrieval,
                       extraction, export, eval harness)
run_full_pipeline.py   entry point
data/inbox/            sample input image
```
The eval/experiment scripts (corpus runs, bilingual cleanup, CER/WER ablations)
live on the `eval` branch.
