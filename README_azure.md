# Visual-Note-Structured-Extraction — `azure-preprocessing` branch

Same shape as `main`, but the OCR step uses **Azure Document Intelligence**
(`prebuilt-read`) instead of Tesseract. Nothing from the original pipeline is
removed — `ocr_test.py` (Tesseract) still works; this branch just adds an Azure
alternative and connects the OCR output to extraction.

## Pipeline (Azure)

1. `azure_ocr_test.py` — Azure `prebuilt-read` on the **raw** `data/inbox` image → prints text + confidence, writes `ocr_output.txt`
2. `extract_test.py`   — OpenAI GPT-4.1-mini: image + `ocr_output.txt` → structured JSON

> **No grayscale step for Azure.** Azure does its own deskew / binarization /
> contrast normalization, so feeding it the OpenCV `equalizeHist` output
> measurably *hurt* (mean confidence 0.88 → 0.84, low-confidence words 10% →
> 19% on this repo's note). Azure reads the raw image; the only prep is a Pillow
> downscale/format safeguard inside `azure_ocr_test.py`.
>
> The Tesseract path (`preprocess_test.py` → `ocr_test.py`) is left intact for
> comparison — grayscale + equalize helps there.
>
> `extract_test.py` reads `ocr_output.txt`, which the Tesseract step never wrote;
> `azure_ocr_test.py` writes it, closing that gap.

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
python azure_ocr_test.py && python extract_test.py
```

The full miso pipeline (lexicon correction, RAG retrieval, schema-forced
extraction, HTML/Google-Docs export) lives on the `full-pipeline` branch.
