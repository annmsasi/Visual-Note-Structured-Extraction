# Visual-Note-Structured-Extraction — base pipeline

Turns a handwritten note image into a structured document (HTML, and optionally a
Google Doc). No cache, no database — a stateless four-stage pipeline:

    preprocess (downscale) -> OCR -> Claude / open VLM (schema-forced document IR)
    -> HTML / Google Docs export

The page **image is the source of truth**; OCR is a weak text hint the model can
lean on or ignore. `run_full_pipeline.py` is the entry point.

## Install
```bash
./install.sh        # creates .venv and installs requirements
```

## Configure
`.env` in the repo root:
```
# OCR (choose one path)
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
#   (or use a free local OCR instead: --ocr tesseract / --ocr paddle)

# LLM (choose one)
ANTHROPIC_API_KEY=...              # claude-* models (premium)
OPENROUTER_API_KEY=...             # open VLMs, e.g. qwen/qwen2.5-vl-72b-instruct (free/cheap)
```
Google Docs export (`--drive`) also needs an OAuth client: enable the Drive API in
a GCP project, create an OAuth client ID of type "Desktop app", and download it as
`credentials.json` in the repo root. The first `--drive` run opens a browser for a
one-time consent and caches `token.json`.

## Run
```bash
# premium path (Azure OCR + Claude)
.venv/bin/python run_full_pipeline.py data/inbox/notes.jpg

# fully free/local path (Tesseract OCR + an open VLM via OpenRouter)
.venv/bin/python run_full_pipeline.py notes.jpg --ocr tesseract --model qwen/qwen2.5-vl-72b-instruct

.venv/bin/python run_full_pipeline.py --drive          # also create a Google Doc
```
Options:
```
image            note image (default: first in data/inbox)
--course NAME     course id, used as the Drive folder name (default: adhoc)
--ocr ENGINE      azure | tesseract | paddle | stub   (default: azure)
--model ID        claude-* id, or an open VLM id (e.g. qwen/qwen2.5-vl-72b-instruct)
--out PATH        HTML output path (default: <note_id>.html)
--drive           upload the result to Google Docs
```
Equations render as inline Unicode (e.g. `∑ᵢ₌₁ⁿ xᵢ`), never images.

### OCR engines
| engine | cost | notes |
|---|---|---|
| `azure` | paid | Azure Document Intelligence `prebuilt-read`; best handwriting OCR. |
| `tesseract` | free, local | needs the system `tesseract` binary; weak reader, fine as a hint. |
| `paddle` | free, local | PaddleOCR; better than Tesseract, finicky to install on Apple Silicon. |
| `stub` | free | fixed fake page, for smoke tests. |

## Test
```bash
.venv/bin/python -m unittest discover -s miso/tests
```

## Layout
```
miso/                  the pipeline package (ocr, layout, extraction, export, eval)
run_full_pipeline.py   entry point (image -> structured doc -> HTML / Google Doc)
data/inbox/            sample input image
```
The cache/RAG add-on (per-course lexicon + retrieval) lives on the `full-pipeline`
branch; this branch is the base system without it.
