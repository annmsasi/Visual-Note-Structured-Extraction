"""Delete bilingual (Devanagari-containing) notes and their images from a course.

  python clean_bilingual.py --db cache_cs_full.db --course cs --corpus corpora/cs
  python clean_bilingual.py --db cache_cs_full.db --course cs --corpus corpora/cs --apply
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def has_devanagari(s: str | None) -> bool:
    return any("ऀ" <= c <= "ॿ" for c in (s or ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--course", required=True)
    ap.add_argument("--corpus", required=True, help="corpus image directory")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    args = ap.parse_args()

    imgs = sorted(p for p in Path(args.corpus).glob("*")
                  if p.suffix.lower() in _EXTS and ".prepared" not in p.name)
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT note_id, processing_order, extracted_json FROM notes "
        "WHERE course_id=? ORDER BY processing_order", (args.course,),
    ).fetchall()

    bilingual = [r for r in rows if has_devanagari(r["extracted_json"])]
    print(f"{args.course}: {len(bilingual)}/{len(rows)} bilingual"
          f"{'' if args.apply else '  (dry run — pass --apply to delete)'}")
    for r in bilingual:
        order = r["processing_order"]
        img = imgs[order] if order < len(imgs) else None
        print(f"  {r['note_id']}  ->  {img.name if img else '(image not found)'}")
        if args.apply:
            if img and img.exists():
                img.unlink()
                prep = img.with_name(img.stem + ".prepared.jpg")
                if prep.exists():
                    prep.unlink()
            conn.execute("DELETE FROM notes WHERE note_id=?", (r["note_id"],))
            conn.execute("DELETE FROM summaries WHERE note_id=?", (r["note_id"],))
    if args.apply:
        conn.commit()
        print(f"deleted {len(bilingual)} bilingual notes from {args.course}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
