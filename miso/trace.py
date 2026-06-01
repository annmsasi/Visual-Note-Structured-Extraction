"""Append-only JSONL writer — one record per note, one file per run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import IO

from miso.types import TraceRecord


class TraceWriter:
    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = run_dir / "trace.jsonl"
        self._fh: IO[str] = open(self.trace_path, "a", buffering=1)

    def write(self, record: TraceRecord) -> None:
        self._fh.write(json.dumps(record.to_dict()) + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
