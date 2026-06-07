"""Render extracted notes to HTML and optionally upload them to Google Docs.

CLI:
    python -m miso.export --db ./miso_cache.db --note <id>   [--out doc.html] [--drive]
    python -m miso.export --db ./miso_cache.db --course <id> [--drive]
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import mimetypes
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


# ----------------------------------------------------------------------------- math

_SUP = dict(zip("0123456789+-=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ"))
_SUB = dict(zip("0123456789+-=()aeioxjhklmnpst", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢₒₓⱼₕₖₗₘₙₚₛₜ"))
_SENT = ""  # private-use sentinel; survives pylatexenc untouched


def _script_to_unicode(kind: str, body: str) -> str:
    table = _SUP if kind == "^" else _SUB
    if body and all(c in table for c in body):
        return "".join(table[c] for c in body)
    return f"{kind}({body})"


def latex_to_unicode(latex: str) -> str:
    """Convert LaTeX to editable Unicode text."""
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
    except Exception:
        text = re.sub(r"\\[a-zA-Z]+|[{}]", "", latex)

    text = re.sub(f"{_SENT}(\\d+){_SENT}",
                  lambda m: _script_to_unicode(*store[int(m.group(1))]), text)
    text = re.sub(r"([_^])(\S)", lambda m: _script_to_unicode(m.group(1), m.group(2)), text)
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------------------------------------------------------- rendering

def render_note_html(doc: dict[str, Any]) -> str:
    """Render one note's document IR to a full HTML page."""
    title = html.escape(doc.get("title") or "(untitled)")
    parts = [f"<h1>{title}</h1>"]
    for block in _coalesce_lists(doc.get("blocks") or []):
        parts.append(_render_block(block))
    return _page(title, "\n".join(p for p in parts if p))


def _coalesce_lists(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive `list` blocks so adjacent bullets share one <ul>."""
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
        lvl = min(6, int(block.get("level", 1)) + 1)  # note title owns <h1>
        return f"<h{lvl}>{html.escape(block.get('text', ''))}</h{lvl}>"
    if t == "paragraph":
        return f"<p>{html.escape(block.get('text', ''))}</p>"
    if t == "equation":
        return f'<p>{html.escape(latex_to_unicode(block.get("latex", "")))}</p>'
    if t == "list":
        return _render_list(block.get("items") or [])
    if t == "figure":
        desc = html.escape(block.get("description", ""))
        img = (block.get("image") or "").strip()
        img_tag = f'<img src="{html.escape(img)}" alt="{desc}">' if img else ""
        cap = f"<figcaption>{desc}</figcaption>" if desc else ""
        return f"<figure>{img_tag}{cap}</figure>"
    return ""


def _render_list(items: list[dict[str, Any]]) -> str:
    """Render items as nested <ul>s driven by each item's `level`."""
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


# ----------------------------------------------------------------------------- markdown

def render_note_markdown(doc: dict[str, Any]) -> str:
    """Render one note's document IR to Markdown."""
    lines: list[str] = [f"# {doc.get('title') or '(untitled)'}", ""]
    for block in _coalesce_lists(doc.get("blocks") or []):
        md = _block_to_md(block)
        if md:
            lines.append(md)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _block_to_md(block: dict[str, Any]) -> str:
    t = block.get("type")
    if t == "heading":
        lvl = min(6, int(block.get("level", 1)) + 1)  # the note title owns `#`
        return f"{'#' * lvl} {(block.get('text') or '').strip()}"
    if t == "paragraph":
        return (block.get("text") or "").strip()
    if t == "equation":
        latex = (block.get("latex") or "").strip()
        return f"$$\n{latex}\n$$" if latex else ""
    if t == "list":
        items = block.get("items") or []
        return "\n".join(
            f"{'  ' * max(0, int(it.get('level', 0)))}- {(it.get('text') or '').strip()}"
            for it in items
        )
    if t == "figure":
        desc = (block.get("description") or "").strip()
        img = (block.get("image") or "").strip()
        if img:
            return f"![{desc}]({img})"
        return f"*[Figure: {desc}]*" if desc else "*[Figure]*"
    return ""


# ----------------------------------------------------------------------------- data access

def load_notes(db_path: Path, *, note_id: str | None = None,
               course_id: str | None = None) -> list[tuple[str, str, dict[str, Any]]]:
    """Read note IR(s) as (note_id, course_id, doc) tuples; one of note_id or course_id required."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    if note_id is not None:
        rows = conn.execute(
            "SELECT note_id, course_id, extracted_json FROM notes WHERE note_id = ?",
            (note_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT note_id, course_id, extracted_json FROM notes WHERE course_id = ? "
            "ORDER BY processing_order", (course_id,),
        ).fetchall()
    conn.close()
    out: list[tuple[str, str, dict[str, Any]]] = []
    for r in rows:
        try:
            out.append((r["note_id"], r["course_id"], json.loads(r["extracted_json"])))
        except (TypeError, json.JSONDecodeError):
            continue
    return out


# ----------------------------------------------------------------------------- Google Docs

def _drive_credentials():
    """Return Drive OAuth credentials, caching a token to disk."""
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


def _ensure_folder(service, name: str) -> str:
    """Return the id of an app-created folder named `name`, creating it if needed."""
    safe = name.replace("\\", "\\\\").replace("'", "\\'")
    res = service.files().list(
        q=("mimeType='application/vnd.google-apps.folder' and trashed=false "
           f"and name='{safe}'"),
        fields="files(id,name)", spaces="drive",
    ).execute()
    found = res.get("files", [])
    if found:
        return found[0]["id"]
    folder = service.files().create(
        body={"name": name, "mimeType": "application/vnd.google-apps.folder"},
        fields="id",
    ).execute()
    return folder["id"]


def _create_doc_with(service, content: str, src_mimetype: str,
                     name: str, folder: str | None) -> tuple[str, str]:
    """Upload `content`, let Drive convert it to a Doc; return (doc_id, web_link)."""
    from googleapiclient.http import MediaInMemoryUpload
    body: dict[str, Any] = {"name": name, "mimeType": "application/vnd.google-apps.document"}
    if folder:
        body["parents"] = [_ensure_folder(service, folder)]
    media = MediaInMemoryUpload(content.encode("utf-8"), mimetype=src_mimetype)
    created = service.files().create(
        body=body, media_body=media, fields="id,webViewLink",
    ).execute()
    return created.get("id", ""), created.get("webViewLink", created.get("id", ""))


def _create_doc(content: str, src_mimetype: str, name: str, folder: str | None) -> str:
    """Create a Google Doc by uploading `content` and letting Drive convert it."""
    from googleapiclient.discovery import build
    service = build("drive", "v3", credentials=_drive_credentials())
    return _create_doc_with(service, content, src_mimetype, name, folder)[1]


def upload_html_to_drive(html_doc: str, name: str, folder: str | None = None) -> str:
    """Create a Google Doc from HTML and return its web link."""
    return _create_doc(html_doc, "text/html", name, folder)


def upload_markdown_to_drive(md_doc: str, name: str, folder: str | None = None) -> str:
    """Create a Google Doc from Markdown. Drive's markdown importer maps headings,
    lists, and spacing to native Doc styles — cleaner than the HTML importer."""
    return _create_doc(md_doc, "text/markdown", name, folder)


# ----------------------------------------------------------------------------- figures → inline images

# A unique sentinel that survives the Drive markdown/HTML importer as plain text, so
# we can find it again in the converted Doc and swap it for an inline image. The
# guillemet-style brackets never occur in normal note text.
_FIG_TOKEN = "⟦MISO_FIGURE_{}⟧"   # ⟦MISO_FIGURE_0⟧


def _stage_figures(doc: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Swap every figure block that HAS an image for a paragraph holding a unique
    token, returning the rewritten doc and an ordered (token, image_path) list.

    Figures whose `image` slot is still empty are left untouched (they render as
    their caption). This is the seam the crop step plugs into: once it fills
    `image`, that figure starts embedding with no further export changes.
    """
    pending: list[tuple[str, str]] = []
    blocks_out: list[dict[str, Any]] = []
    for b in doc.get("blocks") or []:
        if b.get("type") == "figure" and (b.get("image") or "").strip():
            token = _FIG_TOKEN.format(len(pending))
            pending.append((token, b["image"].strip()))
            blocks_out.append({"type": "paragraph", "text": token})
        else:
            blocks_out.append(b)
    return {**doc, "blocks": blocks_out}, pending


def _token_ranges(document: dict[str, Any], tokens: list[str]) -> dict[str, tuple[int, int]]:
    """Locate each token's (startIndex, endIndex) in a Docs API `documents.get`
    response, by walking the body's paragraph textRuns. A token carries no markup,
    so it lives inside a single run; we match the substring within the run."""
    ranges: dict[str, tuple[int, int]] = {}
    for el in document.get("body", {}).get("content", []):
        for pe in (el.get("paragraph") or {}).get("elements", []):
            tr = pe.get("textRun")
            start = pe.get("startIndex")
            if not tr or start is None:
                continue
            content = tr.get("content", "")
            for tok in tokens:
                pos = content.find(tok)
                if pos >= 0:
                    ranges[tok] = (start + pos, start + pos + len(tok))
    return ranges


def _inline_image_requests(ranges: dict[str, tuple[int, int]],
                           token_to_url: dict[str, str]) -> list[dict[str, Any]]:
    """Build batchUpdate requests that replace each token with an inline image.

    Processed bottom-to-top (highest index first) so each edit leaves the indices of
    not-yet-processed tokens valid. Per token: delete its text, then insert the image
    where it started.
    """
    reqs: list[dict[str, Any]] = []
    for tok, (start, end) in sorted(ranges.items(), key=lambda kv: -kv[1][0]):
        url = token_to_url.get(tok)
        if not url:
            continue
        reqs.append({"deleteContentRange": {"range": {"startIndex": start, "endIndex": end}}})
        reqs.append({"insertInlineImage": {"location": {"index": start}, "uri": url}})
    return reqs


def _upload_drive_image(service, path: Path, parent_id: str | None) -> str:
    """Upload an image to Drive, make it link-readable, and return a URL the Docs
    image fetcher can reach. Uses `drive.file`, valid here because the app creates
    the file. NOTE: Google fetches this URL server-side, so the file must be shared
    `anyone/reader`; some `uc?export=view` links fetch unreliably — if an image
    fails to appear, host it somewhere with a direct image content-type instead."""
    from googleapiclient.http import MediaFileUpload
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    body: dict[str, Any] = {"name": path.name}
    if parent_id:
        body["parents"] = [parent_id]
    created = service.files().create(
        body=body, media_body=MediaFileUpload(str(path), mimetype=mime), fields="id",
    ).execute()
    fid = created["id"]
    service.permissions().create(fileId=fid, body={"type": "anyone", "role": "reader"}).execute()
    return f"https://drive.google.com/uc?export=view&id={fid}"


def upload_note_to_drive(doc: dict[str, Any], name: str, folder: str | None = None,
                         *, fmt: str = "markdown") -> str:
    """Create a Google Doc from the note IR and embed any figure images inline.

    Figures whose `image` slot is filled (a local crop path) are uploaded to Drive
    and inserted as inline images via the Docs API; the rest of the note imports as
    before. With no filled figure images this is exactly the old text upload, so it
    is a safe drop-in for `upload_markdown_to_drive` / `upload_html_to_drive`.
    """
    from googleapiclient.discovery import build
    render_doc, pending = _stage_figures(doc)
    content, mime = ((render_note_html(render_doc), "text/html") if fmt == "html"
                     else (render_note_markdown(render_doc), "text/markdown"))
    creds = _drive_credentials()
    drive = build("drive", "v3", credentials=creds)
    doc_id, link = _create_doc_with(drive, content, mime, name, folder)
    if not pending:
        return link

    parent = _ensure_folder(drive, folder) if folder else None
    token_to_url: dict[str, str] = {}
    for token, image_path in pending:
        try:
            token_to_url[token] = _upload_drive_image(drive, Path(image_path), parent)
        except Exception as e:  # one bad crop shouldn't sink the whole upload
            log.warning("figure image %s not embedded: %s", image_path, e)

    docs = build("docs", "v1", credentials=creds)
    document = docs.documents().get(documentId=doc_id).execute()
    requests = _inline_image_requests(_token_ranges(document, list(token_to_url)), token_to_url)
    if requests:
        docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()
    return link


# ----------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export extracted notes to HTML / Google Docs.")
    ap.add_argument("--db", type=Path, default=Path("./miso_cache.db"))
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--note", help="export a single note_id")
    grp.add_argument("--course", help="export every note in a course as its own doc")
    ap.add_argument("--out", type=Path, help="write a single note's HTML here (--note only)")
    ap.add_argument("--drive", action="store_true", help="also upload each note to Google Docs")
    ap.add_argument("--folder", help="Drive folder name (default: the note's course_id)")
    args = ap.parse_args(argv)

    notes = load_notes(args.db, note_id=args.note, course_id=args.course)
    if not notes:
        print(f"No notes found for {'note ' + args.note if args.note else 'course ' + args.course!r}")
        return 1

    for note_id, course_id, doc in notes:
        html_doc = render_note_html(doc)
        out = args.out if (args.note and args.out) else Path(f"{note_id}.html")
        out.write_text(html_doc)
        line = f"wrote {out}"
        if args.drive:
            name = doc.get("title") or note_id
            folder = args.folder or course_id
            url = upload_note_to_drive(doc, name=name, folder=folder, fmt="html")
            line += f"  →  [{folder}] Google Doc: {url}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
