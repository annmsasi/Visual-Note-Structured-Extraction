# Visual-Note-Structured-Extraction — full pipeline + eval harness

The full miso pipeline, vendored in `miso/`. It turns a handwritten note image
(or a multi-page PDF) into a structured document (Markdown, HTML, or a Google Doc):

    preprocess (downscale) -> OCR (Azure | Tesseract | PaddleOCR)
    -> flag-mode lexicon (candidate course terms) + retrieval
    -> Claude or an open VLM (schema-forced document IR) -> miso_cache.db
    -> Markdown / HTML / Google Docs export

Multi-page PDFs are mapped page-by-page (the cache warms across the pages) then
merged into one note. `run_full_pipeline.py` is the entry point; `miso.eval.run_grid`
runs the ablation grid (see Evaluate).

## Install
```bash
./install.sh        # creates .venv and installs requirements
```

## Configure
`.env` in the repo root:
```
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...                   # optional, to run open VLMs (e.g. qwen2.5-vl)
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
.venv/bin/python run_full_pipeline.py lecture.pdf --ocr tesseract \
    --model qwen/qwen2.5-vl-72b-instruct        # free OCR + open VLM
.venv/bin/python run_full_pipeline.py --drive   # also create a Google Doc
```
Options:
```
image            note image OR multi-page PDF (default: first in data/inbox)
--course NAME     course id (cache namespace) + Drive folder name (default: adhoc)
--ocr ENGINE      azure | paddle | tesseract | stub  (paddle/tesseract free + local)
--model ID        a claude-* id, or an open VLM id like qwen/qwen2.5-vl-72b-instruct
                  (served via OPENROUTER_API_KEY)
--format FMT      md | html  (default: md)
--out PATH        output path (default: <note_id>.<format>)
--drive           upload the result to Google Docs
```
Output is a Markdown file (plus the IR as JSON) and, with `--drive`, a Google Doc.
Equations render as inline Unicode (e.g. `∑ᵢ₌₁ⁿ xᵢ`), never images.

## Evaluate
The ablation grid over a labelled corpus. The cache cells (C3–C6: lexicon × retrieval)
run inside each (modality, OCR, model) sub-grid, so attribution holds OCR + model fixed.
It reports term-recall (headline), term-restricted CER, a 2×2 cache attribution with
bootstrap CIs, and a cross-grid summary.
```bash
.venv/bin/python -m miso.eval.run_grid corpora/tim172a --course tim172a \
    --gold corpora/tim172a_gold --ocr-engines azure --models claude-sonnet-4-6

# does the cache help MORE as the recognizer degrades? sweep both ladders:
.venv/bin/python -m miso.eval.run_grid corpora/tim172a --course tim172a \
    --gold corpora/tim172a_gold --ocr-engines azure tesseract \
    --models claude-sonnet-4-6 qwen/qwen2.5-vl-72b-instruct
```
Axes: `--modalities ocr+vlm vlm-only ocr-only`, `--ocr-engines`, `--models`,
`--lexicon-mode flag|replace`. Flag-mode (default) feeds candidate course terms to the
LLM, so it needs one: `ocr-only` forces `replace`, and `vlm-only` skips the lexicon cells.

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
The eval harness lives in `miso/eval/` (`run_grid` ablations, metrics, gold tools)
and synthetic corpora in `miso/synth/`.
