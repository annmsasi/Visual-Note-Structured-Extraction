"""Build a synthetic handwritten-notes course.

Clean open-courseware source -> page-sized chunks -> telegraphic note text ->
degraded handwritten page images + per-note GoldNote JSON (transcription +
structured extraction gold + distinctive terms). Each chunk is one page, so the
transcription gold matches exactly what is rendered. Output is consumed by
run_corpus.py (images) and `python -m miso.eval analyze --gold` (gold).

    python -m miso.synth --course biology --limit 20
    python -m miso.synth --course biology --limit 5 --no-llm   # no API needed
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from miso.synth.noteify import noteify
from miso.synth.openstax import COURSES, iter_course
from miso.synth.render import render_note
from miso.synth.terms import course_distinctive_terms

log = logging.getLogger(__name__)

_SENT = re.compile(r"(?<=[.!?])\s+")


def _block_len(b: dict) -> int:
    return (len(b.get("text", "")) + len(b.get("latex", ""))
            + sum(len(i.get("text", "")) for i in b.get("items", [])))


def _split_paragraph(b: dict, budget: int) -> list[dict]:
    """Split an over-budget paragraph on sentence boundaries so no single block
    overflows a page."""
    txt = b.get("text", "")
    if len(txt) <= budget:
        return [b]
    out, cur = [], ""
    for s in _SENT.split(txt):
        if cur and len(cur) + len(s) + 1 > budget:
            out.append({"type": "paragraph", "text": cur.strip()})
            cur = ""
        cur = f"{cur} {s}".strip()
    if cur:
        out.append({"type": "paragraph", "text": cur})
    return out


def _chunk_blocks(blocks: list[dict], budget: int) -> list[list[dict]]:
    """Group blocks into ~page-sized chunks; one chunk = one note page."""
    flat: list[dict] = []
    for b in blocks:
        flat.extend(_split_paragraph(b, budget) if b.get("type") == "paragraph" else [b])
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    n = 0
    for b in flat:
        bl = _block_len(b)
        if cur and n + bl > budget:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(b)
        n += bl
    if cur:
        chunks.append(cur)
    return chunks


def _gold_body(title: str, blocks: list[dict]) -> str:
    parts = [title]
    for b in blocks:
        if b.get("type") in ("heading", "paragraph"):
            parts.append(b.get("text", ""))
        elif b.get("type") == "equation":
            parts.append(b.get("latex", ""))
        elif b.get("type") == "list":
            parts.extend(i.get("text", "") for i in b.get("items", []))
    return " ".join(p for p in parts if p)


def _summary_gist(blocks: list[dict]) -> str:
    for b in blocks:
        if b.get("type") == "paragraph" and b.get("text"):
            return b["text"][:300]
    return ""


def _load_env() -> None:
    """Best-effort .env load so LLM noteify can find the API key."""
    env = Path("miso/.env")
    if not env.exists():
        return
    import os
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _load_env()
    ap = argparse.ArgumentParser(description="Build a synthetic handwritten course corpus.")
    ap.add_argument("--course", default="biology", choices=sorted(COURSES))
    ap.add_argument("--limit", type=int, default=20, help="number of note pages")
    ap.add_argument("--budget", type=int, default=800, help="approx chars of source text per page")
    ap.add_argument("--out", type=Path, default=Path("corpora"))
    ap.add_argument("--fonts-dir", type=Path, default=Path("assets/fonts"))
    ap.add_argument("--no-llm", action="store_true", help="use the deterministic noteify stub")
    ap.add_argument("--no-degrade", action="store_true", help="skip degradation")
    ap.add_argument("--degrade-strength", type=float, default=1.0, help="0=clean .. 1=hard (Arm-C knob)")
    ap.add_argument("--min-recurrence", type=int, default=2)
    args = ap.parse_args(argv)

    fonts = sorted(p for p in args.fonts_dir.glob("*.ttf"))
    if not fonts:
        print(f"No .ttf fonts in {args.fonts_dir}.", file=sys.stderr)
        return 1
    font = fonts[sum(map(ord, args.course)) % len(fonts)]  # deterministic per-course "hand"

    # Fetch + chunk lazily until we have `limit` page-sized notes.
    log.info("fetching %s source ...", args.course)
    pages: list[tuple[str, list[dict]]] = []
    for sn in iter_course(args.course):
        for ci, chunk in enumerate(_chunk_blocks(sn.blocks, args.budget)):
            pages.append((sn.title if ci == 0 else f"{sn.title} (cont.)", chunk))
            if len(pages) >= args.limit:
                break
        if len(pages) >= args.limit:
            break
    if not pages:
        print("No pages produced.", file=sys.stderr)
        return 1
    log.info("built %d note pages", len(pages))

    img_dir = args.out / args.course
    gold_dir = args.out / f"{args.course}_gold"

    records: list[tuple[str, dict, str]] = []
    bodies: list[str] = []
    for i, (title, blocks) in enumerate(pages):
        nid = f"{args.course}-{i:03d}"
        extracted = {
            "title": title, "blocks": blocks,
            "summary_topic_line": title, "summary_gist": _summary_gist(blocks),
        }
        transcription = noteify(title, blocks, use_llm=not args.no_llm)
        render_note(transcription, str(font), str(img_dir / f"{nid}.jpg"),
                    seed=i, degrade=not args.no_degrade, strength=args.degrade_strength)
        bodies.append(_gold_body(title, blocks))
        records.append((nid, extracted, transcription))
        log.info("[%2d/%d] %s  %-38s  %d note-chars", i + 1, len(pages), nid,
                 title[:38], len(transcription))

    distinctive, per_note_terms = course_distinctive_terms(
        bodies, min_recurrence=args.min_recurrence,
    )
    log.info("course distinctive terms: %d (e.g. %s)",
             len(distinctive), ", ".join(sorted(distinctive)[:18]))

    gold_dir.mkdir(parents=True, exist_ok=True)
    for (nid, extracted, transcription), terms in zip(records, per_note_terms):
        (gold_dir / f"{nid}.json").write_text(json.dumps({
            "note_id": nid, "extracted_json": extracted,
            "transcription": transcription, "distinctive_terms": terms,
        }, indent=2, ensure_ascii=False))

    print(f"\nWrote {len(records)} notes (hand={font.name}, llm={not args.no_llm}, "
          f"degrade={not args.no_degrade}@{args.degrade_strength})")
    print(f"  images: {img_dir}/")
    print(f"  gold:   {gold_dir}/")
    print(f"  next:   python run_corpus.py {img_dir} --course {args.course}   "
          f"# (eval/full-pipeline branch)")
    print(f"          python -m miso.eval analyze --runs runs/ --gold {gold_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
