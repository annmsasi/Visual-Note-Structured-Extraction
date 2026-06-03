"""Per-config aggregates, ramp curves, and the 2×2 cache attribution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from miso.eval.gold import GoldNote
from miso.eval.metrics import (
    bootstrap_ci, cer, correction_precision_recall, structural_f1,
    term_recall, term_restricted_cer, wer,
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
    term_recall: float | None = None          # end-to-end: terms surfaced in the extraction
    term_restricted_cer: float | None = None   # intrinsic: OCR-corrected term spans vs verbatim


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

    @property
    def mean_term_recall(self) -> float | None:
        return _mean_opt(p.term_recall for p in self.per_note)

    @property
    def mean_term_restricted_cer(self) -> float | None:
        return _mean_opt(p.term_restricted_cer for p in self.per_note)


def compute_run_report(records: list[dict], gold: dict[str, GoldNote]) -> RunReport:
    tag = records[0]["config_tag"] if records else "unknown"
    report = RunReport(config_tag=tag, n_notes=len(records))
    for r in records:
        g = gold.get(r["note_id"])
        if g is None:
            continue
        ext = r.get("extraction") or {}
        ext_json = ext.get("structured_json", {})
        # Stub stores flat text under `ocr_text`; real extractions: flatten the
        # document BODY only (title + blocks), excluding the piggybacked summary
        # so the ~150-token abstract doesn't pollute the body CER/WER.
        hyp_text = ext_json.get("ocr_text") or flatten_document_body(ext_json)

        corrected = r.get("corrected_ocr") or {}
        corrected_text = corrected.get("corrected_text", "")
        raw_text = (r.get("ocr_raw") or {}).get("raw_text", "")
        corrections = corrected.get("corrections", []) or []

        precision, recall, over_correction = correction_precision_recall(
            corrections, g.transcription, raw_text,
        )
        terms = g.distinctive_terms
        report.per_note.append(PerNote(
            note_id=r["note_id"],
            processing_order=r.get("processing_order", 0),
            cer=cer(g.transcription, hyp_text),
            wer=wer(g.transcription, hyp_text),
            structural_f1=structural_f1(g.extracted_json, ext_json),
            correction_precision=precision,
            correction_recall=recall,
            over_correction_rate=over_correction,
            # End-to-end: did the final extraction surface the recurring terms?
            term_recall=term_recall(terms, hyp_text) if terms else None,
            # Intrinsic lexicon: OCR-corrected text vs verbatim, over term spans.
            term_restricted_cer=(
                term_restricted_cer(g.transcription, corrected_text, terms) if terms else None
            ),
        ))
    return report


def ramp_curve(report: RunReport) -> list[tuple[int, float]]:
    """CER vs processing_order. The cache's claim is `improves over time` —
    a ramp curve is more honest than a single before/after.
    """
    pairs = [(p.processing_order, p.cer) for p in report.per_note]
    pairs.sort()
    return pairs


def compare_attribution(
    baseline: RunReport,        # C3: no cache
    lexicon_only: RunReport,    # C4
    retrieval_only: RunReport,  # C5
    full: RunReport,            # C6: both
) -> dict[str, tuple[float, float, float]]:
    """CER reduction vs baseline for each cache piece, with bootstrap CIs.

    Aligned per-note by note_id; only notes appearing in all four reports
    contribute. Interaction = both − (lexicon + retrieval); a non-zero value
    means the two pieces aren't strictly additive.
    """
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


def _mean_opt(it: Iterable[float | None]) -> float | None:
    vals = [v for v in it if v is not None]
    return sum(vals) / len(vals) if vals else None


_SUMMARY_KEYS = {"summary_topic_line", "summary_gist", "summary"}


def flatten_document_body(value) -> str:
    """Flatten a structured extraction to its body text, EXCLUDING summary fields,
    so the body CER/WER isn't polluted by the ~150-token piggybacked abstract.
    """
    parts: list[str] = []
    _collect_body(value, parts)
    return " ".join(parts).strip()


def _collect_body(value, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            if k in _SUMMARY_KEYS:
                continue
            _collect_body(v, out)
    elif isinstance(value, list):
        for v in value:
            _collect_body(v, out)
