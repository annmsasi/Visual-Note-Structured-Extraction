#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Installed. Next:"
echo "  1. Create .env with your Azure + Anthropic keys (see README_full.md)."
echo "  2. Run: .venv/bin/python run_full_pipeline.py data/inbox/notes.jpg"
