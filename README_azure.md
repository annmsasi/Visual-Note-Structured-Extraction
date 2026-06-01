# Visual-Note-Structured-Extraction — azure-preprocessing

OCR variant of the team pipeline that uses Azure Document Intelligence
(`prebuilt-read`) instead of Tesseract. The original Tesseract path is kept for
comparison.

## Pipeline (Azure)
1. `azure_ocr_test.py` — Azure prebuilt-read on the raw `data/inbox` image;
   prints the text with a per-word confidence summary and writes `ocr_output.txt`.
2. `extract_test.py` — OpenAI GPT-4.1-mini reads the image + `ocr_output.txt`
   and returns structured JSON.

`extract_test.py` consumes `ocr_output.txt`. The Tesseract step (`ocr_test.py`)
only printed to stdout, so `azure_ocr_test.py` is what now fills that file.

## Why no grayscale step for Azure
Azure does its own deskew, binarization, and contrast normalization. Feeding it
the OpenCV `equalizeHist` output measurably hurt on the sample note (mean
confidence 0.88 -> 0.84, low-confidence words 10% -> 19%), so the Azure path
reads the raw image and only downscales/re-encodes to meet Azure's size limits.
The Tesseract path keeps the OpenCV preprocessing, where it helps.

## Install
```bash
./install.sh        # creates .venv and installs requirements
```
The Tesseract path also needs the tesseract binary (`brew install tesseract`);
the Azure path does not.

## Configure
`.env` in the repo root:
```
OPENAI_API_KEY=...
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
```

## Run
```bash
.venv/bin/python azure_ocr_test.py     # writes ocr_output.txt
.venv/bin/python extract_test.py       # structured JSON from image + OCR
```
Run `azure_ocr_test.py` before `extract_test.py` — the second reads the file the
first writes.

## Tesseract path (original, for comparison)
```bash
.venv/bin/python preprocess_test.py    # OpenCV grayscale + contrast -> data/output
.venv/bin/python ocr_test.py           # Tesseract on the cleaned image
```
