#!/bin/bash
# Set up a recurring job that turns dropped notes into Markdown.
#
# Usage:
#   ./setup.sh [INPUT_FOLDER] [OUTPUT_FOLDER]
#
# INPUT_FOLDER / OUTPUT_FOLDER can live ANYWHERE (relative or absolute); this
# script creates them and resolves them to absolute paths. They default to the
# repo's data/inbox and data/output.
#
# Drop a note (PDF/image) into a COURSE subfolder of the input folder, e.g.
#   INPUT/cse138/lecture1.pdf   ->  processed under course "cse138"
# and every 15 minutes process_inbox.py drains it into
#   OUTPUT/cse138/lecture1.md
#
# Works on macOS and Linux — both ship `crontab`.

# Repo root (where this script lives).
REPO="$(cd "$(dirname "$0")" && pwd)"

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

# The cron line: every 15 minutes, cd into the repo (so .env, credentials.json,
# the cache and the log resolve), drain the inbox, append to the log.
CRON_LINE="*/15 * * * * cd $REPO && $PYTHON_PATH process_inbox.py --inbox $INBOX --output $OUTPUT >> data/process.log 2>&1"

# Install it, preserving any other cron entries and replacing only our own line
# (so re-running this script never duplicates the job or wipes unrelated jobs).
existing="$(crontab -l 2>/dev/null | grep -v 'process_inbox.py' || true)"
printf '%s\n%s\n' "$existing" "$CRON_LINE" | grep -v '^[[:space:]]*$' | crontab -

echo "$CRON_LINE" > crontab.txt

echo "Cron job set up! Runs every 15 minutes."
echo "  drop notes in : $INBOX/<course>/    (one subfolder per course)"
echo "  markdown out  : $OUTPUT/<course>/"
echo "  logs          : $REPO/data/process.log"
echo
echo "Add a course any time with:  mkdir -p \"$INBOX/<course>\""
echo
echo "macOS only: grant Full Disk Access to /usr/sbin/cron (System Settings ->"
echo "  Privacy & Security -> Full Disk Access), or cron can't read the repo."
