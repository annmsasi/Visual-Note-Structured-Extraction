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
import os
import sys
from pathlib import Path

from miso.document import validate
from miso.eval.ocr_runner import _load_env

log = logging.getLogger(__name__)

_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

_GOLD_SCHEMA = {
    "type": "object",
    "properties": {
        "transcription": {"type": "string",
                          "description": "Faithful verbatim transcription of the handwriting AS WRITTEN "
                                         "(keep the page's own abbreviations, casing, line breaks). "
                                         "Replace any figure/diagram/chart/circuit/plot/drawing with the "
                                         "single token [figure]; never transcribe a figure's contents."},
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
    "to correct. Call emit_gold with: (1) transcription — a faithful transcription using STANDARD "
    "spelling (silently fix obvious misspellings), keeping the page's abbreviations, symbols, and "
    "line order; (2) a clean structured extraction (title, blocks = headings/paragraphs/lists/"
    "equations, and the two summary fields); (3) distinctive_terms — the technical/course-specific "
    "terms that appear (not general-English words). "
    "Figures are handled by a SEPARATE step: do NOT transcribe or describe the contents of any "
    "figure/diagram/chart/circuit/plot/drawing. Put the single token [figure] where each appears "
    "(in the transcription, and as its own paragraph block whose text is exactly [figure]); transcribe surrounding handwritten "
    "text normally, but never convert a figure's contents into text, a list, or an equation. "
    "Be accurate and complete."
)


def stage_local(pdfs: list[str], out_dir: str, course: str, dpi: int) -> int:
    """Render local handwritten-note PDFs to page images — the local counterpart
    of `stage` (which only reads HuggingFace datasets).

    PDFs are rendered in the order given and pages numbered sequentially, so the
    chronological course order is preserved (note_id = ``{course}-{NNN}``). A
    manifest.json maps each image back to its source PDF + page, so hand-correction
    of the drafted gold stays traceable.
    """
    import shutil
    from pdf2image import convert_from_path

    poppler = shutil.which("pdfinfo")
    poppler_dir = os.path.dirname(poppler) if poppler else None

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    i = 0
    for pdf in pdfs:
        pages = convert_from_path(pdf, dpi=dpi, poppler_path=poppler_dir)
        for pno, page in enumerate(pages, start=1):
            nid = f"{course}-{i:03d}"
            page.convert("RGB").save(out / f"{nid}.jpg", quality=92)
            manifest.append({"note_id": nid, "source_pdf": os.path.basename(pdf), "page": pno})
            i += 1
        log.info("rendered %s (%d pages)", os.path.basename(pdf), len(pages))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"staged {i} page images to {out}/  (manifest.json written)")
    print(f"next (bills API): python -m miso.eval.armb draft {out_dir} --course {course}")
    return i


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


def _img_b64(path: Path, max_edge: int = 4000) -> str:
    """Base64 JPEG, downscaled so the long edge <= max_edge. The staged page scans are
    kept high-res for OCR, but Anthropic rejects images over 8000 px (and downsamples
    large ones anyway), so the draft call sends a bounded copy."""
    from io import BytesIO

    from PIL import Image
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_edge:
        s = max_edge / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def draft(corpus_dir: str, course: str, model: str, limit: int | None = None) -> int:
    _load_env()
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    corpus = Path(corpus_dir)
    gold_dir = corpus.parent / f"{corpus.name}_gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in corpus.glob("*")
                  if p.suffix.lower() in _EXTS and ".prepared" not in p.name)
    if limit:
        imgs = imgs[:limit]
    n = 0
    for i, img in enumerate(imgs):
        nid = f"{course}-{i:03d}"
        b64 = _img_b64(img)
        msg = client.messages.create(
            model=model, max_tokens=4096,
            tools=[{"name": "emit_gold", "description": "Draft gold for a handwritten page.",
                    "input_schema": _GOLD_SCHEMA}],
            tool_choice={"type": "tool", "name": "emit_gold"},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
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
            "extracted_json": validate({
                "title": payload.get("title", ""),
                "blocks": payload.get("blocks", []),
                "summary_topic_line": payload.get("summary_topic_line", ""),
                "summary_gist": payload.get("summary_gist", ""),
            }),
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
    sl = sub.add_parser("stage-local", help="Render local note PDFs to page images (chronological).")
    sl.add_argument("out_dir")
    sl.add_argument("--pdf", action="append", required=True, dest="pdfs",
                    metavar="PDF", help="a source PDF (repeat in chronological order)")
    sl.add_argument("--course", required=True)
    sl.add_argument("--dpi", type=int, default=200)
    d = sub.add_parser("draft", help="LLM-draft gold for the staged images.")
    d.add_argument("corpus_dir")
    d.add_argument("--course", required=True)
    d.add_argument("--model", default="claude-sonnet-4-6")
    d.add_argument("--limit", type=int, default=None, help="only draft the first N pages (smoke test)")
    args = ap.parse_args(argv)
    if args.cmd == "stage":
        return 0 if stage(args.repo, args.out_dir, args.course, args.limit) else 1
    if args.cmd == "stage-local":
        return 0 if stage_local(args.pdfs, args.out_dir, args.course, args.dpi) else 1
    if args.cmd == "draft":
        return 0 if draft(args.corpus_dir, args.course, args.model, args.limit) else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
