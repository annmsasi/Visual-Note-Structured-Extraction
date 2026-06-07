"""CLI for the eval harness.

    python -m miso.eval analyze --runs runs/                    # all runs under runs/
    python -m miso.eval analyze --runs runs/<id1> runs/<id2>     # specific runs
    python -m miso.eval analyze --runs runs/ --gold gold/        # real gold
    python -m miso.eval analyze --runs runs/ --synth-gold        # smoke-test gold
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from miso.eval.analyze import compare_attribution, compute_run_report, ramp_curve
from miso.eval.faithfulness import make_judge, score_faithfulness
from miso.eval.gold import load_gold, synthesize_gold_from_traces
from miso.eval.loader import discover_runs, load_trace


def _fmt_opt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def cmd_analyze(args) -> int:
    runs: list[Path] = []
    for r in args.runs:
        p = Path(r)
        if (p / "trace.jsonl").exists():
            runs.append(p)
        elif p.is_dir():
            runs.extend(discover_runs(p))
    if not runs:
        print("No runs found.", file=sys.stderr)
        return 1

    all_records: list[dict] = []
    per_tag: dict[str, list[dict]] = {}
    for r in runs:
        records = load_trace(r)
        if not records:
            continue
        per_tag.setdefault(records[0]["config_tag"], []).extend(records)
        all_records.extend(records)

    if args.synth_gold or not args.gold:
        gold = synthesize_gold_from_traces(all_records)
        gold_source = "<synthesized from traces (smoke-test mode)>"
    else:
        gold = load_gold(Path(args.gold))
        gold_source = args.gold

    judge = make_judge(args.faithfulness) if args.faithfulness != "off" else None
    reports = {
        tag: compute_run_report(
            records, gold,
            faithfulness=score_faithfulness(records, judge) if judge else None,
        )
        for tag, records in per_tag.items()
    }

    print(f"# Eval report\n")
    print(f"- Gold source: `{gold_source}` ({len(gold)} notes)")
    print(f"- Configs analysed: {len(reports)} ({', '.join(reports)})\n")

    print("## Per-config means\n")
    print("_Headline = term-recall (end-to-end) and term CER (intrinsic lexicon); "
          "global CER is secondary — it barely moves even when the cache helps._\n")
    print("| config_tag | n | term-recall | term CER | mean CER | raw-OCR CER "
          "| structural F1 | faith | over-correction |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag, report in reports.items():
        print(
            f"| `{tag}` | {report.n_notes} | {_fmt_opt(report.mean_term_recall)} "
            f"| {_fmt_opt(report.mean_term_restricted_cer)} | {report.mean_cer:.4f} "
            f"| {report.mean_ocr_cer:.4f} | {report.mean_structural_f1:.4f} "
            f"| {_fmt_opt(report.mean_faithfulness)} | {report.mean_over_correction:.4f} |"
        )
    print()

    print("## Ramp curves (processing_order → CER)\n")
    for tag, report in reports.items():
        pairs = ramp_curve(report)
        pretty = ", ".join(f"({o},{c:.3f})" for o, c in pairs)
        print(f"- **{tag}**: {pretty}")
    print()

    needed = {"C3_llm_ocr_only", "C4_lexicon_only", "C5_retrieval_only", "C6_full"}
    if needed.issubset(reports.keys()):
        attribution = compare_attribution(
            reports["C3_llm_ocr_only"],
            reports["C4_lexicon_only"],
            reports["C5_retrieval_only"],
            reports["C6_full"],
        )
        print("## 2×2 attribution — CER reduction vs C3 (mean [95% CI bootstrap])\n")
        print("| effect | mean | 95% CI |")
        print("|---|---:|---:|")
        for k, (mean, lo, hi) in attribution.items():
            print(f"| {k} | {mean:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
        print()
    else:
        missing = sorted(needed - reports.keys())
        print(f"_2×2 attribution skipped — missing: {', '.join(missing)}._\n")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="miso-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    analyze = sub.add_parser("analyze", help="Compute metrics from JSONL traces.")
    analyze.add_argument("--runs", nargs="+", required=True,
                         help="Run dirs (or a parent dir containing run dirs).")
    analyze.add_argument("--gold", default=None,
                         help="Directory of gold JSON files, one per note.")
    analyze.add_argument("--synth-gold", action="store_true",
                         help="Synthesize gold from traces (smoke-test only).")
    analyze.add_argument("--faithfulness", default="off",
                         help="hallucination judge: off | stub | a judge model id (e.g. claude-haiku-4-5-20251001)")
    args = parser.parse_args(argv)
    if args.cmd == "analyze":
        return cmd_analyze(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
