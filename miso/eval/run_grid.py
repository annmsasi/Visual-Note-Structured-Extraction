"""Run the ablation grid over a real corpus, then print the eval report
(term-recall, term CER, raw-OCR CER, faithfulness, 2x2 attribution, longitudinal ramp, cross-grid headline).

Three crossed axes — the cache cells (C3-C6) are run inside each (modality, OCR, model)
sub-grid, so attribution is always computed with OCR + model held fixed:

  * modality   : ocr+vlm (default) | vlm-only | ocr-only
  * OCR ladder : --ocr-engines azure paddle tesseract      (strong -> weak)
  * LLM ladder : --models claude-sonnet-4-6 qwen/qwen2.5-vl-72b-instruct  (via OPENROUTER_API_KEY)

    python -m miso.eval.run_grid corpora/tim172a --course tim172a \
        --gold corpora/tim172a_gold --models claude-sonnet-4-6 --ocr-engines azure --limit 8

    # the headline experiment: does the cache help MORE as the recognizer gets worse?
    python -m miso.eval.run_grid corpora/tim172a --course tim172a --gold corpora/tim172a_gold \
        --ocr-engines azure tesseract --models claude-sonnet-4-6 qwen/qwen2.5-vl-72b-instruct

Each cell is a separate run with its own cache (from empty), processed in chronological
filename order so the cache warms across the note sequence. OCR + extraction run once
per cell. Cells that are semantically impossible are skipped (logged), not silently
dropped: flag-mode needs an LLM to feed, and vlm-only has no OCR to flag from.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from miso.config import RunConfig
from miso.eval.analyze import compare_attribution, compute_run_report
from miso.eval.faithfulness import make_judge, score_faithfulness
from miso.eval.gold import load_gold
from miso.eval.loader import load_trace
from miso.eval.ocr_runner import _load_env  # reads miso/.env explicitly
from miso.replay import _configure_logging, _prepare_image, run
from miso.types import Note

_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# The four cache cells (the 2x2: lexicon x retrieval).
_CACHE_CELLS = {
    "C3_llm_ocr_only": RunConfig.config_3_llm_ocr_only,
    "C4_lexicon_only": RunConfig.config_4_lexicon_only,
    "C5_retrieval_only": RunConfig.config_5_retrieval_only,
    "C6_full": RunConfig.config_6_full,
}

# Which cache cells are meaningful per modality.
#   vlm-only : no OCR -> no flag-mode lexicon; keep the pure image->LLM baseline only.
#   ocr-only : no LLM -> flag-mode can't feed anything, so only the no-cache raw-OCR
#              baseline and the replace-mode lexicon (which mutates the text) differ.
_MODALITY_CELLS = {
    "ocr+vlm": ["C3_llm_ocr_only", "C4_lexicon_only", "C5_retrieval_only", "C6_full"],
    "vlm-only": ["C3_llm_ocr_only"],
    "ocr-only": ["C3_llm_ocr_only", "C4_lexicon_only"],
}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "-", s).strip("-")


def _short_model(model: str) -> str:
    """Compact, filesystem-safe label for a model id (drops the vendor path prefix)."""
    return _slug(model.split("/")[-1])


def build_notes(corpus_dir: str, course: str, limit: int | None) -> list[Note]:
    imgs = sorted(p for p in Path(corpus_dir).glob("*")
                  if p.suffix.lower() in _EXTS and ".prepared" not in p.name)
    if limit:
        imgs = imgs[:limit]
    base = datetime(2026, 1, 1)
    return [
        Note(note_id=f"{course}-{i:03d}", course_id=course,
             image_path=_prepare_image(img), processing_order=i,
             timestamp=base + timedelta(days=i))
        for i, img in enumerate(imgs)
    ]


def _apply_modality(cfg: RunConfig, modality: str, *, model: str, engine: str,
                    lexicon_mode: str) -> None:
    """Mutate cfg for a (modality, model, engine) combination, in place."""
    cfg.extraction.model_id = model
    cfg.ocr.engine = engine
    if cfg.lexicon.enabled:
        cfg.lexicon.mode = lexicon_mode
    if modality == "ocr+vlm":
        cfg.extraction.use_image = True
        cfg.extraction.use_ocr_hint = True
    elif modality == "vlm-only":
        cfg.extraction.use_image = True
        cfg.extraction.use_ocr_hint = False
        cfg.ocr.engine = "stub"  # OCR output is unused here; don't pay to run it
    elif modality == "ocr-only":
        cfg.extraction.model_id = "stub"  # no LLM: deterministic OCR -> IR
        cfg.extraction.use_image = False
        if cfg.lexicon.enabled:
            cfg.lexicon.mode = "replace"  # flag-mode only feeds an LLM; replace mutates the text
    else:
        raise SystemExit(f"unknown modality: {modality}")


def _axis_values(modality: str, ocr_engines: list[str], models: list[str]) -> tuple[list[str], list[str]]:
    """Collapse the irrelevant axis per modality so we don't launch redundant runs."""
    engines = ["stub"] if modality == "vlm-only" else ocr_engines
    mods = ["stub"] if modality == "ocr-only" else models
    return engines, mods


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def _ramp(rep, attr: str) -> str:
    """Per-note metric in processing (chronological) order — the F4 longitudinal series."""
    pts = sorted(rep.per_note, key=lambda p: p.processing_order)
    return ", ".join(f"({p.processing_order},{_fmt(getattr(p, attr))})" for p in pts)


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    _load_env()
    ap = argparse.ArgumentParser(description="Run the cache ablation grid across modality/OCR/model axes.")
    ap.add_argument("corpus_dir")
    ap.add_argument("--course", required=True)
    ap.add_argument("--gold", required=True, help="gold dir (GoldNote JSON per note)")
    ap.add_argument("--models", nargs="+", default=["claude-sonnet-4-6"],
                    help="LLM/VLM ladder: claude-* ids and/or open VLM ids (via OPENROUTER_API_KEY)")
    ap.add_argument("--ocr-engines", nargs="+", default=["azure"],
                    choices=["stub", "azure", "paddle", "tesseract"],
                    help="OCR ladder, strong -> weak")
    ap.add_argument("--modalities", nargs="+", default=["ocr+vlm"],
                    choices=["ocr+vlm", "vlm-only", "ocr-only"])
    ap.add_argument("--lexicon-mode", default="flag", choices=["flag", "replace"],
                    help="behaviour of the enabled lexicon cells (ocr+vlm); ocr-only forces replace")
    ap.add_argument("--faithfulness", default="stub",
                    help="hallucination judge: off | stub (free token-overlap proxy) | a judge model id "
                         "like claude-haiku-4-5-20251001 (pick a DIFFERENT family than --models)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cold-start", type=int, default=2)
    args = ap.parse_args(argv)

    notes = build_notes(args.corpus_dir, args.course, args.limit)
    if not notes:
        print("No images found.", file=sys.stderr)
        return 1
    gold = load_gold(Path(args.gold))

    # Enumerate sub-grids up front so the user sees the run count (no silent fan-out).
    plan: list[tuple[str, str, str, list[str]]] = []
    for modality in args.modalities:
        engines, models = _axis_values(modality, args.ocr_engines, args.models)
        for engine in engines:
            for model in models:
                plan.append((modality, engine, model, _MODALITY_CELLS[modality]))
    n_runs = sum(len(cells) for *_, cells in plan)
    print(f"Plan: {len(plan)} sub-grid(s), {n_runs} runs "
          f"({len(notes)} notes each), gold={args.gold} ({len(gold)} notes)\n")

    summary_rows: list[dict] = []
    for modality, engine, model, cells in plan:
        subgrid = f"{modality} | ocr={engine} | llm={_short_model(model)}"
        tag_suffix = _slug(f"{modality}_{engine}_{_short_model(model)}")
        print(f"\n{'=' * 70}\n### sub-grid: {subgrid}\n{'=' * 70}")
        reports = {}
        for cell in cells:
            cfg = _CACHE_CELLS[cell]()
            _apply_modality(cfg, modality, model=model, engine=engine,
                            lexicon_mode=args.lexicon_mode)
            cfg.config_tag = f"{cell}__{tag_suffix}"
            cfg.retrieval.cold_start_note_count = args.cold_start
            cfg.cache_path = Path(f"./cache_{args.course}_{tag_suffix}_{cell}.db")
            cfg.traces_dir = Path("runs")
            print(f"  -- {cfg.config_tag} (model={cfg.extraction.model_id}, ocr={cfg.ocr.engine}) --")
            run_dir = run(cfg, notes)
            records = load_trace(run_dir)
            faith = (score_faithfulness(records, make_judge(args.faithfulness))
                     if args.faithfulness != "off" else None)
            reports[cell] = compute_run_report(records, gold, faithfulness=faith)

        # per-cell table
        print("\n| cell | n | term-recall | term CER | mean CER | raw-OCR CER | struct F1 | faith | over-corr |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for cell, rep in reports.items():
            print(f"| {cell} | {rep.n_notes} | {_fmt(rep.mean_term_recall)} "
                  f"| {_fmt(rep.mean_term_restricted_cer)} | {rep.mean_cer:.4f} "
                  f"| {rep.mean_ocr_cer:.4f} | {rep.mean_structural_f1:.4f} "
                  f"| {_fmt(rep.mean_faithfulness)} | {rep.mean_over_correction:.4f} |")

        # 2x2 attribution when the full set is present
        if set(_CACHE_CELLS).issubset(reports):
            attribution = compare_attribution(
                reports["C3_llm_ocr_only"], reports["C4_lexicon_only"],
                reports["C5_retrieval_only"], reports["C6_full"],
            )
            print("\n2x2 attribution — CER reduction vs C3 (mean [95% CI bootstrap]):")
            for k, (mean, lo, hi) in attribution.items():
                print(f"  {k}: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]")

        base = reports.get("C3_llm_ocr_only")
        cache_cell = ("C6_full" if "C6_full" in reports
                      else "C4_lexicon_only" if "C4_lexicon_only" in reports else None)

        # longitudinal ramp (F4) — no-cache vs cache by processing order; the cache's
        # distinctive claim is that it IMPROVES across the note sequence as vocab accrues.
        ramp_cells = [c for c in ("C3_llm_ocr_only", cache_cell) if c and c in reports]
        if ramp_cells:
            print("\nramp (processing_order → term-recall | CER):")
            for cell in ramp_cells:
                print(f"  {cell} term-recall: {_ramp(reports[cell], 'term_recall')}")
                print(f"  {cell} CER:         {_ramp(reports[cell], 'cer')}")

        # row for the cross-grid headline (no-cache baseline vs the richest cache cell present)
        if base and cache_cell:
            cache = reports[cache_cell]
            summary_rows.append({
                "subgrid": subgrid, "cache_cell": cache_cell,
                "tr_base": base.mean_term_recall, "tr_cache": cache.mean_term_recall,
                "cer_base": base.mean_cer, "cer_cache": cache.mean_cer,
            })

    # cross-grid headline: does the cache help MORE as the recognizer gets worse?
    if summary_rows:
        print(f"\n\n{'=' * 70}\n### cross-grid headline — no-cache (C3) vs cache\n{'=' * 70}")
        print("| sub-grid | cache | term-recall C3→cache | mean CER C3→cache |")
        print("|---|---|---:|---:|")
        for r in summary_rows:
            tr = f"{_fmt(r['tr_base'])} → {_fmt(r['tr_cache'])}"
            ce = f"{r['cer_base']:.4f} → {r['cer_cache']:.4f}"
            print(f"| {r['subgrid']} | {r['cache_cell']} | {tr} | {ce} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
