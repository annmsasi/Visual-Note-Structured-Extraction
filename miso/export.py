"""Render extracted notes (the document IR) to HTML and, optionally, push them
to Google Docs.

Two layers, deliberately decoupled:
  - `render_*` turns the IR into HTML. Pure, no network, no creds — always works.
  - `upload_html_to_drive` converts that HTML into a Google Doc via the Drive API.

HTML (not markdown) is the rendering target: Drive's HTML->Doc import maps
headings, nested lists, and inline styles far more faithfully than markdown,
and keeps a path open to tables/images later. The IR stays canonical, so a
native Docs `batchUpdate` renderer can be added without touching extraction.

One Google Doc per note. A course just exports each of its notes as a separate
doc (looped), so the unit of export is always a single note.

CLI:
    python -m miso.export --db ./miso_cache.db --note <id>   [--out doc.html] [--drive]
    python -m miso.export --db ./miso_cache.db --course <id> [--drive]   # one doc per note
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


# ----------------------------------------------------------------------------- math

# Equation blocks become inline Unicode text — never images, never raw LaTeX.
# Unicode super/subscript characters; anything not representable degrades to a
# parenthesised linear form (e.g. ^(n+1)), so the output is always editable text.
_SUP = dict(zip("0123456789+-=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ"))
_SUB = dict(zip("0123456789+-=()aeioxjhklmnpst", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢₒₓⱼₕₖₗₘₙₚₛₜ"))
_SENT = ""  # private-use sentinel; survives pylatexenc untouched


def _script_to_unicode(kind: str, body: str) -> str:
    table = _SUP if kind == "^" else _SUB
    if body and all(c in table for c in body):
        return "".join(table[c] for c in body)
    return f"{kind}({body})"


def latex_to_unicode(latex: str) -> str:
    """LaTeX -> editable Unicode text. pylatexenc handles symbols/structure; we
    protect super/subscript groups with a sentinel first (so their grouping
    survives pylatexenc dropping the braces) and substitute Unicode after.
    """
    latex = latex.strip()
    for d in ("$$", "$", r"\(", r"\)", r"\[", r"\]"):
        latex = latex.replace(d, "")
    latex = latex.strip()

    store: list[tuple[str, str]] = []

    def stash(m: re.Match) -> str:
        store.append((m.group(1), m.group(2)))
        return f"{_SENT}{len(store) - 1}{_SENT}"

    latex = re.sub(r"([_^])\{([^{}]*)\}", stash, latex)   # braced: ^{...}, _{...}
    latex = re.sub(r"([_^])([A-Za-z0-9])", stash, latex)  # single char: ^x, _2

    try:
        from pylatexenc.latex2text import LatexNodes2Text
        text = LatexNodes2Text().latex_to_text(latex)
    except Exception:  # pylatexenc absent: strip commands/braces rather than leak LaTeX
        text = re.sub(r"\\[a-zA-Z]+|[{}]", "", latex)

    text = re.sub(f"{_SENT}(\\d+){_SENT}",
                  lambda m: _script_to_unicode(*store[int(m.group(1))]), text)
    # leftover command-bodied scripts pylatexenc converted, e.g. ^\infty -> ^∞
    text = re.sub(r"([_^])(\S)", lambda m: _script_to_unicode(m.group(1), m.group(2)), text)
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------------------------------------------------------- rendering

def render_note_html(doc: dict[str, Any]) -> str:
    """Render one note's document IR to a full HTML page (one doc per note)."""
    title = html.escape(doc.get("title") or "(untitled)")
    parts = [f"<h1>{title}</h1>"]
    for block in _coalesce_lists(doc.get("blocks") or []):
        parts.append(_render_block(block))
    return _page(title, "\n".join(p for p in parts if p))


def _coalesce_lists(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive `list` blocks into one so adjacent bullets share a single
    <ul>. The model sometimes splits one logical list across several blocks;
    separate <ul>s render with extra inter-list margin, which reads as uneven
    spacing in the doc.
    """
    out: list[dict[str, Any]] = []
    for b in blocks:
        if b.get("type") == "list" and out and out[-1].get("type") == "list":
            out[-1] = {"type": "list", "items": out[-1]["items"] + (b.get("items") or [])}
        else:
            out.append(b)
    return out


def _render_block(block: dict[str, Any]) -> str:
    t = block.get("type")
    if t == "heading":
        lvl = min(6, int(block.get("level", 1)) + 1)  # +1: the note title owns <h1>
        return f"<h{lvl}>{html.escape(block.get('text', ''))}</h{lvl}>"
    if t == "paragraph":
        return f"<p>{html.escape(block.get('text', ''))}</p>"
    if t == "equation":
        return f'<p>{html.escape(latex_to_unicode(block.get("latex", "")))}</p>'
    if t == "list":
        return _render_list(block.get("items") or [])
    return ""


def _render_list(items: list[dict[str, Any]]) -> str:
    """Nested <ul> driven by each item's `level`. A level-N item lives inside
    N+1 nested <ul>s (level 0 = one list deep).
    """
    out: list[str] = []
    depth = 0
    for it in items:
        target = max(0, int(it.get("level", 0))) + 1
        while depth < target:
            out.append("<ul>"); depth += 1
        while depth > target:
            out.append("</ul>"); depth -= 1
        out.append(f"<li>{html.escape(it.get('text', ''))}</li>")
    out.append("</ul>" * depth)
    return "".join(out)


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title></head><body>\n{body}\n</body></html>"
    )


# ----------------------------------------------------------------------------- data access

def load_notes(db_path: Path, *, note_id: str | None = None,
               course_id: str | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Read note IR(s) as (note_id, doc) pairs. One of note_id / course_id required.

    A course returns its notes in processing_order — each becomes its own doc.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    if note_id is not None:
        rows = conn.execute(
            "SELECT note_id, extracted_json FROM notes WHERE note_id = ?", (note_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT note_id, extracted_json FROM notes WHERE course_id = ? "
            "ORDER BY processing_order", (course_id,),
        ).fetchall()
    conn.close()
    out: list[tuple[str, dict[str, Any]]] = []
    for r in rows:
        try:
            out.append((r["note_id"], json.loads(r["extracted_json"])))
        except (TypeError, json.JSONDecodeError):
            continue
    return out


# ----------------------------------------------------------------------------- Google Docs

def _drive_credentials():
    """OAuth installed-app flow, drive.file scope. Caches a token next to the
    client secrets. Paths come from env so nothing is hard-coded.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    secrets = Path(os.environ.get("GOOGLE_OAUTH_CLIENT_SECRETS", "credentials.json"))
    token = Path(os.environ.get("GOOGLE_OAUTH_TOKEN", "token.json"))
    creds = None
    if token.exists():
        creds = Credentials.from_authorized_user_file(str(token), _DRIVE_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not secrets.exists():
                raise FileNotFoundError(
                    f"Google OAuth client secrets not found at {secrets}. "
                    "Create an OAuth client in a GCP project with the Drive API "
                    "enabled, download it, and set GOOGLE_OAUTH_CLIENT_SECRETS."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(secrets), _DRIVE_SCOPES)
            creds = flow.run_local_server(port=0)
        token.write_text(creds.to_json())
    return creds


def upload_html_to_drive(html_doc: str, name: str) -> str:
    """Create a Google Doc from HTML and return its web link."""
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaInMemoryUpload

    service = build("drive", "v3", credentials=_drive_credentials())
    media = MediaInMemoryUpload(html_doc.encode("utf-8"), mimetype="text/html")
    created = service.files().create(
        body={"name": name, "mimeType": "application/vnd.google-apps.document"},
        media_body=media,
        fields="id,webViewLink",
    ).execute()
    # NOTE: creates a new doc each run. For idempotent updates, persist this id
    # (e.g. a `course_id -> drive_file_id` table) and call files().update instead.
    return created.get("webViewLink", created.get("id", ""))


# ----------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export extracted notes to HTML / Google Docs.")
    ap.add_argument("--db", type=Path, default=Path("./miso_cache.db"))
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--note", help="export a single note_id")
    grp.add_argument("--course", help="export every note in a course as its own doc")
    ap.add_argument("--out", type=Path, help="write a single note's HTML here (--note only)")
    ap.add_argument("--drive", action="store_true", help="also upload each note to Google Docs")
    args = ap.parse_args(argv)

    notes = load_notes(args.db, note_id=args.note, course_id=args.course)
    if not notes:
        print(f"No notes found for {'note ' + args.note if args.note else 'course ' + args.course!r}")
        return 1

    for note_id, doc in notes:
        html_doc = render_note_html(doc)
        out = args.out if (args.note and args.out) else Path(f"{note_id}.html")
        out.write_text(html_doc)
        line = f"wrote {out}"
        if args.drive:
            name = doc.get("title") or note_id
            line += f"  →  Google Doc: {upload_html_to_drive(html_doc, name=name)}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
