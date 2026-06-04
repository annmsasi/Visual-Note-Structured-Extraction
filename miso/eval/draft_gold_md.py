"""Draft per-page gold (PDF + Markdown) for human correction.

For each staged page in a manifest, create a directory containing:
  - <note_id>.pdf : the single source page (for side-by-side correction)
  - <note_id>.md  : an LLM-drafted gold doc — verbatim transcription + structured
                    markdown + distinctive terms — to be hand-corrected.

OCR hint: Azure Document Intelligence (cached, so re-runs never re-bill).
Drafter: a Claude vision model (default claude-opus-4-8).

    python -m miso.eval.draft_gold_md --limit 1     # test one page first
    python -m miso.eval.draft_gold_md               # all pages in the manifest
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

DRAFT_SYSTEM = (
    "You draft a GOLD-STANDARD reference from a handwritten lecture-notes page image, "
    "for a human to correct. The IMAGE is the source of truth; the OCR text is a weak, "
    "error-prone hint.\n\n"
    "Output a Markdown document with EXACTLY these three sections and nothing else:\n\n"
    "## Transcription (verbatim)\n"
    "A faithful, verbatim transcription of the handwriting EXACTLY as written. Keep the "
    "writer's own abbreviations, spelling, symbols, capitalization, and line breaks (one "
    "line per written line). Do NOT expand abbreviations or fix spelling. Mark anything "
    "illegible as [?]. Mark non-text figures/diagrams as [fig: short description].\n\n"
    "## Structured note\n"
    "The same content as clean structured Markdown mirroring the page's layout: '# ' for "
    "the page title, '## '/'### ' for section headings, '- ' bullet lists (indent two "
    "spaces per nesting level), plain text for paragraphs, and $...$ or $$...$$ (LaTeX) "
    "for equations. Be faithful to the page's own structure — keep outlines as lists; "
    "only use paragraphs where prose was actually written.\n\n"
    "## Distinctive terms\n"
    "A bullet list of the technical / course-specific terms appearing on the page (not "
    "general-English words).\n\n"
    'Output only the Markdown, starting with "## Transcription (verbatim)".'
)


def _load_env(root: Path) -> None:
    env = root / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _azure_ocr_hint(image_path: Path) -> str:
    from miso.ocr import AzureOCR, CachedOCR
    ocr = CachedOCR(AzureOCR(
        os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"],
        os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"],
    ))
    r = ocr.run(image_path)
    return r.layout_text or r.raw_text


def _draft_markdown(image_path: Path, ocr_hint: str, model: str) -> str:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        system=DRAFT_SYSTEM,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
            {"type": "text", "text": f"OCR (weak hint):\n{ocr_hint}\n\nProduce the Markdown gold draft."},
        ]}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Draft per-page gold (PDF + Markdown).")
    ap.add_argument("--manifest", default="corpora/tim172a/manifest.json")
    ap.add_argument("--images", default="corpora/tim172a")
    ap.add_argument("--src", default="corpora/tim172a_src")
    ap.add_argument("--out", default="corpora/tim172a_gold")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--limit", type=int, default=None, help="only draft the first N pages")
    args = ap.parse_args(argv)

    _load_env(Path("."))
    manifest = json.loads(Path(args.manifest).read_text())
    images, src, out = Path(args.images), Path(args.src), Path(args.out)
    n = 0
    for entry in manifest:
        if args.limit and n >= args.limit:
            break
        nid, page, pdf = entry["note_id"], entry["page"], entry["source_pdf"]
        d = out / nid
        d.mkdir(parents=True, exist_ok=True)

        # 1. single source page as its own PDF
        subprocess.run(
            ["pdfseparate", "-f", str(page), "-l", str(page), str(src / pdf), str(d / f"{nid}.pdf")],
            check=True,
        )
        # 2. Azure OCR hint (cached)
        hint = _azure_ocr_hint(images / f"{nid}.jpg")
        # 3. Opus draft -> markdown
        body = _draft_markdown(images / f"{nid}.jpg", hint, args.model)
        header = (
            f"# {nid}\n\n"
            f"> Source: {pdf} p{page} · drafted by {args.model} + Azure OCR.\n"
            f"> CORRECT this against {nid}.pdf. The transcription must be VERBATIM — keep the\n"
            f"> writer's exact spelling, abbreviations, symbols, and line breaks.\n\n"
        )
        (d / f"{nid}.md").write_text(header + body + "\n")
        n += 1
        log.info("drafted %s -> %s/", nid, d)

    print(f"\ndrafted {n} gold dir(s) under {out}/  (each has <note_id>.pdf + <note_id>.md)")
    print("Correct each .md against its .pdf, then we parse the markdown back to GoldNote JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
