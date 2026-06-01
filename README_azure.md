# Visual-Note-Structured-Extraction — `azure-preprocessing` branch

Same shape as `main`, but the OCR step uses **Azure Document Intelligence**
(`prebuilt-read`) instead of Tesseract. Nothing from the original pipeline is
removed — `ocr_test.py` (Tesseract) still works; this branch just adds an Azure
alternative and connects the OCR output to extraction.

## Pipeline

1. `preprocess_test.py` — OpenCV grayscale + contrast → `data/output/*_clean.png`
2. `azure_ocr_test.py`  — Azure `prebuilt-read` → prints text + confidence, writes `ocr_output.txt`
3. `extract_test.py`    — OpenAI GPT-4.1-mini: image + `ocr_output.txt` → structured JSON

> Note: `extract_test.py` reads `ocr_output.txt`. The Tesseract step
> (`ocr_test.py`) only printed to stdout, so that file was never created;
> `azure_ocr_test.py` writes it, closing the gap.

## Setup

```bash
pip install -r requirements.txt
```

`.env` (gitignored) must contain:

```
OPENAI_API_KEY=...
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
```

## Run

```bash
python preprocess_test.py && python azure_ocr_test.py && python extract_test.py
```

The full miso pipeline (lexicon correction, RAG retrieval, schema-forced
extraction, HTML/Google-Docs export) lives on the `full-pipeline` branch.
