#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Installed. Next:"
echo "  1. Create .env with OPENAI_API_KEY + Azure keys (see README_azure.md)."
echo "  2. Run: .venv/bin/python azure_ocr_test.py && .venv/bin/python extract_test.py"
echo "  (Tesseract path also needs: brew install tesseract)"
