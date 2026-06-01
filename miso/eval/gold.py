"""Load gold extractions, or synthesize them for smoke tests.

On-disk layout: a directory of JSON files, one per note, filename stem = note_id.
Each file contains note_id, extracted_json, and optional transcription.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoldNote:
    note_id: str
    extracted_json: dict
    transcription: str  # CER/WER reference


def load_gold(gold_dir: Path) -> dict[str, GoldNote]:
    out: dict[str, GoldNote] = {}
    if not gold_dir.exists():
        return out
    for path in sorted(gold_dir.glob("*.json")):
        data = json.loads(path.read_text())
        note_id = data.get("note_id") or path.stem
        extracted = data.get("extracted_json", {})
        transcription = data.get("transcription") or _flatten_strings(extracted)
        out[note_id] = GoldNote(note_id=note_id, extracted_json=extracted, transcription=transcription)
    return out


def synthesize_gold_from_traces(traces: list[dict]) -> dict[str, GoldNote]:
    """Fabricate gold from each trace's raw OCR for smoke tests."""
    out: dict[str, GoldNote] = {}
    for r in traces:
        note_id = r["note_id"]
        raw = (r.get("ocr_raw") or {}).get("raw_text", "")
        gold_text = raw.replace("eigenvecter", "eigenvector")
        out[note_id] = GoldNote(
            note_id=note_id,
            extracted_json={"ocr_text": gold_text},
            transcription=gold_text,
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
