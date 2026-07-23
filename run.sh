#!/bin/bash
# Process everything waiting in the inbox RIGHT NOW (don't wait for the 15-min job).
#
# Usage:
#   ./run.sh [INPUT_FOLDER] [OUTPUT_FOLDER]
#
# Same folders as ./setup.sh; both default to the repo's data/inbox and data/output.
# A subfolder of the input folder is a course (INPUT/cse138/ -> course "cse138").

# Repo root (where this script lives) — cd in so .env, credentials and the cache resolve.
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO" || exit 1

# Create a folder if needed and print its absolute path (portable: no realpath).
abspath() { mkdir -p "$1" && ( cd "$1" && pwd ); }

INBOX="$(abspath "${1:-$REPO/data/inbox}")"
OUTPUT="$(abspath "${2:-$REPO/data/output}")"

# Prefer the project venv (install.sh builds it with the pipeline deps).
if [ -x "$REPO/.venv/bin/python" ]; then
  PYTHON_PATH="$REPO/.venv/bin/python"
else
  PYTHON_PATH="$(command -v python3)"
fi

exec "$PYTHON_PATH" process_inbox.py --inbox "$INBOX" --output "$OUTPUT"
