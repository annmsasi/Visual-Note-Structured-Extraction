#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Installed. Next:"
echo "  1. Create .env with Azure + Anthropic keys (see README_eval.md)."
echo "  2. Stage a corpus, e.g.:"
echo "     .venv/bin/python stage_corpus.py HumynLabs/Handwritten-Computer-Science-Notes-Dataset corpora/cs cs"
echo "  3. Run it: .venv/bin/python run_corpus.py corpora/cs --course cs --config full"
