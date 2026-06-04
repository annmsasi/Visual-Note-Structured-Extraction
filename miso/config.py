"""Run-time configuration threaded through the base (no-cache) pipeline."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OCRConfig:
    engine: str = "stub"   # "stub" | "azure" | "paddle" | "tesseract" (last two free, local)


@dataclass
class ExtractionConfig:
    model_id: str = "claude-sonnet-4-6"
    use_image: bool = True
    use_ocr_hint: bool = True
    # emit summary fields in the extraction JSON, no separate summariser call
    piggyback_summary: bool = True


@dataclass
class EvalConfig:
    enable_faithfulness_check: bool = False
    bootstrap_samples: int = 1000


@dataclass
class RunConfig:
    config_tag: str
    pipeline_version: str = "v1.0.0"
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    ocr: OCRConfig = field(default_factory=OCRConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    traces_dir: Path = field(default_factory=lambda: Path("./runs"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def base(cls, tag: str = "base") -> "RunConfig":
        return cls(config_tag=tag)
