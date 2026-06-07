"""Run-time configuration threaded through the pipeline."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LexiconConfig:
    enabled: bool = True
    # admit a term only after this many sightings
    n_recurrence: int = 2

    # How the lexicon feeds the LLM:
    #   "flag"    — annotate uncertain OCR words with candidate course terms (default)
    #   "replace" — hard-swap the OCR token for the top candidate (ablation arm)
    #   "off"     — pass OCR through untouched
    mode: str = "flag"

    # Continuous candidate score (flag mode) — a bounded pseudo-posterior, so no
    # hard edit-distance band is needed: every factor is <= 1, frequency can only
    # down-weight, and the likelihood term makes distance dominate the tail.
    #   relevance(T) = (1 - confidence)                       # suspicion      (<= 1)
    #                  * exp(-distance_decay * d_norm(word, T))# OCR likelihood (<= 1)
    #                  * count(T) / (count(T) + freq_prior_k)  # frequency prior(<= 1)
    # A saturating prior (not count/sum-count): robust to Zipfian skew, never
    # crushes a rare-but-close term below the floor — only gently prefers common ones.
    distance_decay: float = 4.0       # OCR channel temperature: how fast likelihood falls with distance
    freq_prior_k: float = 5.0         # frequency saturation: prior = count/(count+k); higher k trusts frequency less
    relevance_floor: float = 0.05     # absolute floor: surface a candidate only above this posterior
    relative_gate: float = 0.0        # optional: also drop candidates below this fraction of the best (0 = off)
    max_candidates: int = 3           # cap on candidates offered per uncertain word
    search_ceiling: float = 0.95      # skip the vocab search for words the OCR is this sure of

    # replace-mode only: hard-correct words below this confidence, boosting the
    # corrected word's confidence by this magnitude.
    confidence_threshold: float = 0.7
    boost_magnitude: float = 0.3

    also_feed_llm_glossary: bool = True


@dataclass
class RetrievalConfig:
    enabled: bool = True
    top_k_candidates: int = 10
    top_k_inject: int = 3
    reranker_enabled: bool = True
    reranker_threshold: float = 0.5
    cold_start_note_count: int = 5
    recency_tie_break: bool = True
    rrf_k: int = 60


@dataclass
class AugmentationConfig:
    inject_position: str = "reverse"       # "reverse" | "forward" | "sides"
    inject_token_budget: int = 700
    framing: str = "weak_hint"
    inject_glossary: bool = True


@dataclass
class OCRConfig:
    engine: str = "stub"                   # "stub" | "azure" | "paddle" | "tesseract" (last two free, local)
    stub_inject_errors: bool = True


@dataclass
class ExtractionConfig:
    model_id: str = "claude-opus-4-7"
    use_image: bool = True
    use_ocr_hint: bool = True
    use_retrieved_summaries: bool = True
    use_glossary: bool = True
    # emit summary fields in the extraction JSON (per-page); the stored whole-note
    # summary is produced separately in finalize via extractor.summarize()
    piggyback_summary: bool = True
    # pages of THIS note's already-extracted markdown to prepend to each page's
    # prompt for cross-page continuity (0 = off, -1 = all). The model still
    # transcribes only the current page.
    prior_page_context: int = 10


@dataclass
class EvalConfig:
    enable_faithfulness_check: bool = False
    bootstrap_samples: int = 1000
    # seed the cache from gold extractions instead of self-extractions
    cache_from_corrected_ground_truth: bool = False


@dataclass
class RunConfig:
    config_tag: str
    pipeline_version: str = "v1.0.0"
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    lexicon: LexiconConfig = field(default_factory=LexiconConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    from_empty_cache: bool = True
    cache_path: Path = field(default_factory=lambda: Path("./miso_cache.db"))
    traces_dir: Path = field(default_factory=lambda: Path("./runs"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    # Factories for the ablation grid: C3 is the no-cache baseline, C6 is the full system.

    @classmethod
    def config_3_llm_ocr_only(cls, tag: str = "C3_llm_ocr_only") -> "RunConfig":
        cfg = cls(config_tag=tag)
        cfg.lexicon.enabled = False
        cfg.retrieval.enabled = False
        return cfg

    @classmethod
    def config_4_lexicon_only(cls, tag: str = "C4_lexicon_only") -> "RunConfig":
        cfg = cls(config_tag=tag)
        cfg.retrieval.enabled = False
        return cfg

    @classmethod
    def config_5_retrieval_only(cls, tag: str = "C5_retrieval_only") -> "RunConfig":
        cfg = cls(config_tag=tag)
        cfg.lexicon.enabled = False
        return cfg

    @classmethod
    def config_6_full(cls, tag: str = "C6_full") -> "RunConfig":
        return cls(config_tag=tag)
