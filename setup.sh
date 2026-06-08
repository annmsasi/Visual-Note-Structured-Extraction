#!/bin/bash
# Install the daily inbox cron job for THIS machine — paths + Python resolved
# automatically, so it works wherever the repo is checked out.
set -u

# Repo root = the folder this script lives in.
REPO="$(cd "$(dirname "$0")" && pwd)"

# Prefer the project venv (it has the pipeline deps); fall back to system python3.
if [ -x "$REPO/.venv/bin/python" ]; then
  PYTHON="$REPO/.venv/bin/python"
else
  PYTHON="$(command -v python3)"
  echo "WARNING: $REPO/.venv not found — using $PYTHON." >&2
  echo "         Run ./install.sh first so the pipeline's dependencies are available." >&2
fi

# process_inbox.py is the scheduled entry point: it drains data/inbox/ -> Markdown +
# Google Doc and moves each source to data/processed/. cd into the repo so .env,
# credentials.json, token.json, and the data/ paths all resolve.
CRON_LINE="0 9 * * * cd $REPO && $PYTHON process_inbox.py >> data/process.log 2>&1"

# Install idempotently: keep any other crontab entries, replace only our line.
# `grep -v ... || true` so an empty/all-filtered crontab doesn't abort the install.
EXISTING="$(crontab -l 2>/dev/null | grep -v 'process_inbox.py' || true)"
printf '%s\n' "$EXISTING" "$CRON_LINE" | grep -v '^$' | crontab -

echo "Installed cron job (runs daily at 9am):"
echo "  $CRON_LINE"
echo
echo "  view:   crontab -l"
echo "  remove: crontab -l | grep -v process_inbox.py | crontab -"
echo "  logs:   $REPO/data/process.log"
echo
echo "macOS note: the cron daemon needs Full Disk Access to run a job under ~/Documents."
echo "  System Settings -> Privacy & Security -> Full Disk Access -> enable /usr/sbin/cron"
echo "  (without this, cron is silently blocked from reading the repo)."
