"""Per-config aggregates, ramp curves, and the 2×2 cache attribution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from miso.eval.gold import GoldNote
from miso.eval.metrics import (
    bootstrap_ci, cer, correction_precision_recall, structural_f1, wer,
)


@dataclass
class PerNote:
    note_id: str
    processing_order: int
    cer: float
    wer: float
    structural_f1: float
    correction_precision: float
    correction_recall: float
    over_correction_rate: float


@dataclass
class RunReport:
    config_tag: str
    n_notes: int
    per_note: list[PerNote] = field(default_factory=list)

    @property
    def mean_cer(self) -> float:
        return _mean(p.cer for p in self.per_note)

    @property
    def mean_wer(self) -> float:
        return _mean(p.wer for p in self.per_note)

    @property
    def mean_structural_f1(self) -> float:
        return _mean(p.structural_f1 for p in self.per_note)

    @property
    def mean_over_correction(self) -> float:
        return _mean(p.over_correction_rate for p in self.per_note)


def compute_run_report(records: list[dict], gold: dict[str, GoldNote]) -> RunReport:
    tag = records[0]["config_tag"] if records else "unknown"
    report = RunReport(config_tag=tag, n_notes=len(records))
    for r in records:
        g = gold.get(r["note_id"])
        if g is None:
            continue
        ext = r.get("extraction") or {}
        ext_json = ext.get("structured_json", {})
        # Stub stores flat text under `ocr_text`; otherwise flatten the JSON.
        hyp_text = ext_json.get("ocr_text") or _flatten_strings(ext_json)

        corrected = r.get("corrected_ocr") or {}
        raw_text = (r.get("ocr_raw") or {}).get("raw_text", "")
        corrections = corrected.get("corrections", []) or []

        precision, recall, over_correction = correction_precision_recall(
            corrections, g.transcription, raw_text,
        )
        report.per_note.append(PerNote(
            note_id=r["note_id"],
            processing_order=r.get("processing_order", 0),
            cer=cer(g.transcription, hyp_text),
            wer=wer(g.transcription, hyp_text),
            structural_f1=structural_f1(g.extracted_json, ext_json),
            correction_precision=precision,
            correction_recall=recall,
            over_correction_rate=over_correction,
        ))
    return report


def ramp_curve(report: RunReport) -> list[tuple[int, float]]:
    """CER vs processing_order."""
    pairs = [(p.processing_order, p.cer) for p in report.per_note]
    pairs.sort()
    return pairs


def compare_attribution(
    baseline: RunReport,
    lexicon_only: RunReport,
    retrieval_only: RunReport,
    full: RunReport,
) -> dict[str, tuple[float, float, float]]:
    """CER reduction vs baseline for each cache piece, with bootstrap CIs."""
    def cer_by_id(rep: RunReport) -> dict[str, float]:
        return {p.note_id: p.cer for p in rep.per_note}

    baseline_cer = cer_by_id(baseline)
    lexicon_cer = cer_by_id(lexicon_only)
    retrieval_cer = cer_by_id(retrieval_only)
    full_cer = cer_by_id(full)
    common = sorted(set(baseline_cer) & set(lexicon_cer)
                    & set(retrieval_cer) & set(full_cer))

    lexicon_delta = [baseline_cer[n] - lexicon_cer[n] for n in common]
    retrieval_delta = [baseline_cer[n] - retrieval_cer[n] for n in common]
    both_delta = [baseline_cer[n] - full_cer[n] for n in common]
    interaction = [b - (l + r) for b, l, r in zip(both_delta, lexicon_delta, retrieval_delta)]

    return {
        "lexicon_delta": bootstrap_ci(lexicon_delta),
        "retrieval_delta": bootstrap_ci(retrieval_delta),
        "both_delta": bootstrap_ci(both_delta),
        "interaction": bootstrap_ci(interaction),
    }


def _mean(it: Iterable[float]) -> float:
    vals = list(it)
    return sum(vals) / len(vals) if vals else 0.0


def _flatten_strings(value) -> str:
    parts: list[str] = []
    _collect(value, parts)
    return " ".join(parts).strip()


def _collect(value, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _collect(v, out)
    elif isinstance(value, list):
        for v in value:
            _collect(v, out)
