"""Compare raw-OCR CER vs lexicon-corrected CER per note against Bentham gold.

  python bentham_eval.py
"""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

from miso.eval.metrics import cer, correction_precision_recall, wer

GOLD_DIR = Path("corpora/bentham_sub_gold")
TAG = "bentham_lexicon"


def main() -> int:
    gold = {p.stem: json.loads(p.read_text())["transcription"] for p in GOLD_DIR.glob("*.json")}

    runs = []
    for f in glob.glob("runs/*/trace.jsonl"):
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if recs and recs[0].get("config_tag") == TAG:
            runs.append((Path(f).parent.name, recs))
    if not runs:
        print(f"No run with config_tag={TAG} found.")
        return 1
    run_id, recs = runs[-1]
    print(f"Run {run_id} — {len(recs)} notes\n")

    raw_cers, corr_cers, precisions, recalls = [], [], [], []
    rows = []
    for r in recs:
        nid = r["note_id"]
        g = gold.get(nid)
        if not g:
            continue
        raw = (r.get("ocr_raw") or {}).get("raw_text", "")
        corrected = (r.get("corrected_ocr") or {}).get("corrected_text", "")
        corrections = (r.get("corrected_ocr") or {}).get("corrections", []) or []
        cr, cc = cer(g, raw), cer(g, corrected)
        p, rc, _ = correction_precision_recall(corrections, g, raw)
        raw_cers.append(cr); corr_cers.append(cc)
        if corrections:
            precisions.append(p); recalls.append(rc)
        rows.append((r.get("processing_order", 0), nid, cr, cc, len(corrections)))

    rows.sort()
    print(f'{"note":13}{"CER_raw":>9}{"CER_corr":>10}{"delta":>9}{"corrections":>13}')
    for _, nid, cr, cc, nc in rows:
        print(f"{nid:13}{cr:9.4f}{cc:10.4f}{cr - cc:+9.4f}{nc:13}")

    mr, mc = statistics.mean(raw_cers), statistics.mean(corr_cers)
    print(f"\nmean CER  raw={mr:.4f}  corrected={mc:.4f}  "
          f"reduction={mr - mc:+.4f} ({(mr - mc) / mr * 100:+.1f}%)")
    if precisions:
        print(f"lexicon correction precision={statistics.mean(precisions):.2f} "
              f"recall={statistics.mean(recalls):.2f} (over notes with corrections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
