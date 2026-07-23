# Visual-Note-Structured-Extraction

Drop a handwritten note into a folder; a few minutes later a clean Markdown
version appears in another folder. Under the hood: OCR → a per-course vocabulary
cache → Claude → Markdown.

## 1. Installation

```bash
./install.sh
```

Create a `.env` in the repo root with your keys:

```
ANTHROPIC_API_KEY=...
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
```

## 2. Scheduling

```bash
./setup.sh                       # uses ./data/inbox and ./data/output
./setup.sh ~/notes ~/markdown    # or put the folders anywhere you like
```

This schedules a job that checks the input folder **every 15 minutes**. Works on
macOS and Linux. On macOS, grant Full Disk Access to `/usr/sbin/cron` (System
Settings → Privacy & Security → Full Disk Access) so it can read your files.

## 3. Usage

Make a folder per course inside your input folder, and drop notes in:

```
~/notes/cse138/lecture1.pdf      # PDF or image; PDFs can be multi-page
```

Within 15 minutes the Markdown shows up mirrored by course:

```
~/markdown/cse138/lecture1.md
```

The source file moves to `data/processed/` when it's done (or `data/failed/` if
something went wrong — see `data/process.log`).

That's it. Add more courses by adding more subfolders.

---

### Running on demand

```bash
./run.sh                       # process the inbox once, right now
./run.sh ~/notes ~/markdown    # ...with the same folders you gave setup.sh
```

To also upload each note to Google Drive as a Google Doc:

```bash
.venv/bin/python process_inbox.py --inbox ~/notes --output ~/markdown --drive
```

`--drive` needs a Google OAuth client (`credentials.json` in the repo root, Drive
API enabled); the first run opens a browser once to create `token.json`.
