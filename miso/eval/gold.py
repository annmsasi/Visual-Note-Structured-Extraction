"""Load gold extractions for evaluation, or synthesise them for smoke tests.

Expected on-disk layout: a directory of JSON files, one per note. Filename
stem = note_id. Each file contains:

    {
        "note_id": "...",
        "extracted_json": {...},
        "transcription": "..."   # optional; derived from extracted_json otherwise
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GoldNote:
    note_id: str
    extracted_json: dict
    transcription: str  # flat text used as the CER/WER reference
    # Recurring course-distinctive vocabulary the cache targets; drives the
    # term-recall and term-restricted-CER metrics. Empty → those are reported n/a.
    distinctive_terms: list[str] = field(default_factory=list)


def load_gold(gold_dir: Path) -> dict[str, GoldNote]:
    out: dict[str, GoldNote] = {}
    if not gold_dir.exists():
        return out
    for path in sorted(gold_dir.glob("*.json")):
        data = json.loads(path.read_text())
        note_id = data.get("note_id") or path.stem
        ej = data.get("extracted_json", {})
        transcription = data.get("transcription") or _flatten_strings(ej)
        out[note_id] = GoldNote(
            note_id=note_id,
            extracted_json=ej,
            transcription=transcription,
            distinctive_terms=data.get("distinctive_terms") or [],
        )
    return out


def synthesize_gold_from_traces(traces: list[dict]) -> dict[str, GoldNote]:
    """Fabricate gold from each trace's raw OCR — smoke-test only.

    "Fixes" the demo's deliberate `eigenvecter → eigenvector` mis-OCR so the
    lexicon's effect is at least measurable end-to-end. Replace with real
    `load_gold(...)` once annotated data exists.
    """
    out: dict[str, GoldNote] = {}
    for r in traces:
        note_id = r["note_id"]
        raw = (r.get("ocr_raw") or {}).get("raw_text", "")
        gold_text = raw.replace("eigenvecter", "eigenvector")
        out[note_id] = GoldNote(
            note_id=note_id,
            extracted_json={"ocr_text": gold_text},
            transcription=gold_text,
            # The demo's deliberate mis-OCR is on this term, so the term metrics
            # have something to measure in smoke-test mode.
            distinctive_terms=["eigenvector"],
        )
    return out


def _flatten_strings(value) -> str:
    parts: list[str] = []
    _walk(value, parts)
    return " ".join(parts).strip()


def _walk(value, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _walk(v, out)
    elif isinstance(value, list):
        for v in value:
            _walk(v, out)
