"""Arm-B: real handwritten notes (HuggingFace) + LLM-drafted gold for human correction.

No public dataset of real handwritten technical notes ships transcription/structured
gold, so we draft it: a strong model reads each real page and proposes a verbatim
transcription + structured extraction + distinctive terms, which you then correct.
(eval_design_v1.md §4 — the "LLM-draft + human-correct" pass.)

    # 1. pull real note images
    python -m miso.eval.armb stage HumynLabs/English-Handwritten-Math-Notes-Dataset corpora/math_real --course math_real --limit 30
    # 2. auto-draft gold (one Claude call per page; bills API)
    python -m miso.eval.armb draft corpora/math_real --course math_real
    # 3. you hand-correct corpora/math_real_gold/*.json, then:
    python -m miso.eval.run_grid corpora/math_real --course math_real --gold corpora/math_real_gold
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import sys
from pathlib import Path

from miso.eval.ocr_runner import _load_env

log = logging.getLogger(__name__)

_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

_GOLD_SCHEMA = {
    "type": "object",
    "properties": {
        "transcription": {"type": "string",
                          "description": "Faithful verbatim transcription of the handwriting AS WRITTEN "
                                         "(keep the page's own abbreviations, casing, line breaks)."},
        "title": {"type": "string"},
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["heading", "paragraph", "list", "equation"]},
                    "level": {"type": "integer"},
                    "text": {"type": "string"},
                    "latex": {"type": "string"},
                    "items": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}, "level": {"type": "integer"}},
                        "required": ["text"]}},
                },
                "required": ["type"],
            },
        },
        "summary_topic_line": {"type": "string"},
        "summary_gist": {"type": "string"},
        "distinctive_terms": {"type": "array", "items": {"type": "string"},
                              "description": "Technical / course-specific terms on the page "
                                             "(not general English)."},
    },
    "required": ["transcription", "title", "blocks", "distinctive_terms"],
}

_DRAFT_PROMPT = (
    "This is a real handwritten study-notes page. Produce a GOLD-STANDARD DRAFT for a human "
    "to correct. Call emit_gold with: (1) transcription — a faithful verbatim transcription of "
    "the handwriting exactly as written (keep the page's abbreviations, symbols, spelling, and "
    "line order); (2) a clean structured extraction (title, blocks = headings/paragraphs/lists/"
    "equations, and the two summary fields); (3) distinctive_terms — the technical/course-specific "
    "terms that appear (not general-English words). Be accurate and complete."
)


def stage(repo: str, out_dir: str, course: str, limit: int | None) -> int:
    from datasets import load_dataset
    from PIL import Image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(repo, split="train")
    if "pdf" in ds.column_names:  # get raw PDF bytes instead of decoding (no pdfplumber needed)
        try:
            from datasets import Pdf
        except ImportError:
            from datasets.features import Pdf
        ds = ds.cast_column("pdf", Pdf(decode=False))
    i = 0
    for row in ds:
        if limit and i >= limit:
            break
        im = row.get("image")
        if im is not None:
            if not isinstance(im, Image.Image):
                im = Image.fromarray(im)
            im.convert("RGB").save(out / f"{course}-{i:03d}.jpg", quality=92)
            i += 1
            continue
        cell = row.get("pdf")
        if cell is not None:  # render PDF pages with PyMuPDF
            import fitz
            data = (cell["bytes"] if isinstance(cell, dict) and cell.get("bytes")
                    else open(cell["path"], "rb").read())
            doc = fitz.open(stream=data, filetype="pdf")
            for pno in range(len(doc)):
                if limit and i >= limit:
                    break
                pix = doc[pno].get_pixmap(dpi=200)
                Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(
                    out / f"{course}-{i:03d}.jpg", quality=92)
                i += 1
    print(f"staged {i} real note images to {out}/")
    return i


def draft(corpus_dir: str, course: str, model: str) -> int:
    _load_env()
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    corpus = Path(corpus_dir)
    gold_dir = corpus.parent / f"{corpus.name}_gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in corpus.glob("*")
                  if p.suffix.lower() in _EXTS and ".prepared" not in p.name)
    n = 0
    for i, img in enumerate(imgs):
        nid = f"{course}-{i:03d}"
        mime = mimetypes.guess_type(str(img))[0] or "image/jpeg"
        b64 = base64.b64encode(img.read_bytes()).decode()
        msg = client.messages.create(
            model=model, max_tokens=4096,
            tools=[{"name": "emit_gold", "description": "Draft gold for a handwritten page.",
                    "input_schema": _GOLD_SCHEMA}],
            tool_choice={"type": "tool", "name": "emit_gold"},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": _DRAFT_PROMPT},
            ]}],
        )
        payload = next((b.input for b in msg.content
                        if getattr(b, "type", None) == "tool_use" and b.name == "emit_gold"), None)
        if payload is None:
            log.warning("%s: no draft returned (stop_reason=%s)", nid, msg.stop_reason)
            continue
        gold = {
            "note_id": nid,
            "extracted_json": {
                "title": payload.get("title", ""),
                "blocks": payload.get("blocks", []),
                "summary_topic_line": payload.get("summary_topic_line", ""),
                "summary_gist": payload.get("summary_gist", ""),
            },
            "transcription": payload.get("transcription", ""),
            "distinctive_terms": payload.get("distinctive_terms", []),
            "_draft": True,  # remove after human correction
        }
        (gold_dir / f"{nid}.json").write_text(json.dumps(gold, indent=2, ensure_ascii=False))
        n += 1
        log.info("drafted %s (%d terms, %d blocks)", nid,
                 len(gold["distinctive_terms"]), len(gold["extracted_json"]["blocks"]))
    print(f"\ndrafted {n} gold files in {gold_dir}/  (each marked \"_draft\": true)")
    print("Correct them by hand (fix transcription to match the page exactly, fix structure/terms),")
    print(f"then: python -m miso.eval.run_grid {corpus_dir} --course {course} --gold {gold_dir}")
    return n


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Arm-B real-notes staging + gold drafting.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage", help="Download real note images from a HF dataset.")
    s.add_argument("repo")
    s.add_argument("out_dir")
    s.add_argument("--course", required=True)
    s.add_argument("--limit", type=int, default=30)
    d = sub.add_parser("draft", help="LLM-draft gold for the staged images.")
    d.add_argument("corpus_dir")
    d.add_argument("--course", required=True)
    d.add_argument("--model", default="claude-sonnet-4-6")
    args = ap.parse_args(argv)
    if args.cmd == "stage":
        return 0 if stage(args.repo, args.out_dir, args.course, args.limit) else 1
    if args.cmd == "draft":
        return 0 if draft(args.corpus_dir, args.course, args.model) else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
