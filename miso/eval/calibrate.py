"""Arm-C calibration: is the synthetic corpus's OCR difficulty realistic?

Runs (cached) Azure OCR on a synthetic corpus and a real-handwriting corpus and
compares CER distribution, error-type mix, top character confusions, and OCR
confidence. If synthetic is much *easier* than real, raise the generator's
--degrade-strength; if much harder, lower it. (eval_design_v1.md §5)

    python -m miso.eval.calibrate --synthetic biology --real iam
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path

from miso.eval.gold import load_gold
from miso.eval.metrics import align_tokens, bootstrap_ci, cer
from miso.eval.ocr_runner import run_ocr_dir

log = logging.getLogger(__name__)


def _corpus_stats(name: str, root: str, engine: str) -> dict:
    gold = load_gold(Path(root) / f"{name}_gold")
    ocr = run_ocr_dir(Path(root) / name, engine=engine)
    cers: list[float] = []
    sub = ins = dele = 0
    conf: list[float] = []
    confusions: Counter[str] = Counter()
    for nid, g in sorted(gold.items()):
        rec = ocr.get(nid)
        if not rec:
            continue
        ref, hyp = g.transcription, rec["text"]
        cers.append(cer(ref, hyp))
        for a, b in align_tokens(list(ref.lower()), list(hyp.lower())):
            if a is None:
                ins += 1
            elif b is None:
                dele += 1
            elif a != b:
                sub += 1
                confusions[f"{a if a.strip() else '·'}->{b if b.strip() else '·'}"] += 1
        conf.extend(w["c"] for w in rec["words"])
    return {"name": name, "n": len(cers), "cers": cers,
            "sub": sub, "ins": ins, "del": dele, "conf": conf, "confusions": confusions}


def _report(s: dict) -> float:
    m, lo, hi = bootstrap_ci(s["cers"]) if s["cers"] else (0.0, 0.0, 0.0)
    tot = max(1, s["sub"] + s["ins"] + s["del"])
    print(f"\n### {s['name']}  (n={s['n']})")
    print(f"  CER: mean={m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  error mix: sub={s['sub'] / tot:.0%}  ins={s['ins'] / tot:.0%}  del={s['del'] / tot:.0%}")
    if s["conf"]:
        cs = sorted(s["conf"])
        print(f"  OCR confidence: mean={statistics.mean(cs):.3f}  p10={cs[len(cs) // 10]:.3f}")
    print(f"  top confusions: {', '.join(f'{k}({v})' for k, v in s['confusions'].most_common(8))}")
    return m


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Arm-C OCR-difficulty calibration.")
    ap.add_argument("--synthetic", default="biology")
    ap.add_argument("--real", default="iam")
    ap.add_argument("--root", default="corpora")
    ap.add_argument("--engine", default="azure")
    args = ap.parse_args(argv)

    syn = _corpus_stats(args.synthetic, args.root, args.engine)
    real = _corpus_stats(args.real, args.root, args.engine)

    print("\n# Arm-C calibration report")
    ms = _report(syn)
    mr = _report(real)

    print("\n## Verdict")
    if not syn["cers"] or not real["cers"]:
        print("  insufficient data (missing gold or OCR).")
        return 1
    ratio = ms / mr if mr else float("inf")
    print(f"  synthetic CER {ms:.3f} vs real CER {mr:.3f}  →  ratio {ratio:.2f}")
    if ratio < 0.6:
        print("  → synthetic is TOO CLEAN. RAISE --degrade-strength (re-run miso.synth).")
    elif ratio > 1.6:
        print("  → synthetic is HARDER than real. LOWER --degrade-strength.")
    else:
        print("  → synthetic CER is in a realistic band (~0.6–1.6× real). Good to proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
