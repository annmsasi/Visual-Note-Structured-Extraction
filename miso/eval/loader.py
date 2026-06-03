"""Read per-note JSONL traces produced by replay runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_trace(run_dir: Path) -> list[dict]:
    path = run_dir / "trace.jsonl"
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def discover_runs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    return sorted(
        p for p in runs_dir.iterdir() if p.is_dir() and (p / "trace.jsonl").exists()
    )


def load_runs(run_dirs: Iterable[Path]) -> dict[str, list[dict]]:
    """Concatenate runs that share a config_tag — useful for pooling re-runs."""
    out: dict[str, list[dict]] = {}
    for d in run_dirs:
        records = load_trace(d)
        if not records:
            continue
        tag = records[0]["config_tag"]
        out.setdefault(tag, []).extend(records)
    return out
