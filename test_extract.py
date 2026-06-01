"""Run AnthropicExtractor on the note (OCR from ocr_dump.json) and write HTML.

Needs ANTHROPIC_API_KEY from .env.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    _load_dotenv(HERE / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY missing from .env")
        return 2

    from miso.config import ExtractionConfig
    from miso.extraction import AnthropicExtractor
    from miso.export import render_note_html
    from miso.types import CorrectedOCR, Note, OCRResult, OCRWord

    p = json.load(open(HERE / "ocr_dump.json"))["pages"][0]
    words = []
    for w in p["words"]:
        xs, ys = w["polygon"][0::2], w["polygon"][1::2]
        words.append(OCRWord(w["text"], w["confidence"],
                             bbox=(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))))
    res = OCRResult.from_words(words)
    corrected = CorrectedOCR(words=res.words, corrected_text=res.raw_text,
                             corrections=[], touched_terms=[], layout_text=res.layout_text)
    note = Note(note_id="HIST101-n1", course_id="HIST101",
                image_path=HERE / "Journal-example-1-2378482629.jpg",
                processing_order=1, timestamp=datetime(2019, 2, 7))

    model = os.environ.get("MISO_EXTRACTOR", "claude-sonnet-4-6")
    print(f"extracting with {model} (image + structured OCR)...\n")
    ext = AnthropicExtractor(api_key=os.environ["ANTHROPIC_API_KEY"], model_id=model)
    out = ext.extract(note=note, corrected_ocr=corrected, retrieved=[], glossary=[],
                      cfg=ExtractionConfig())

    doc = out.structured_json
    print("title:", doc.get("title"))
    print("blocks:")
    for b in doc.get("blocks", []):
        if b["type"] == "heading":
            print(f"  H{b['level']}  {b['text']}")
        elif b["type"] == "paragraph":
            print(f"  P    {b['text'][:90]}")
        elif b["type"] == "list":
            for it in b["items"]:
                print(f"     {'  ' * it.get('level', 0)}- {it['text'][:84]}")
        elif b["type"] == "equation":
            print(f"  EQ   {b['latex']}")
    print("\nsummary_topic_line:", doc.get("summary_topic_line"))
    print("summary_gist:", doc.get("summary_gist"))

    (HERE / "HIST101-n1.real.html").write_text(render_note_html(doc))
    print("\nwrote HIST101-n1.real.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
