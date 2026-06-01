"""OCR adapter — a Protocol + a deterministic stub + an Azure-backed implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from miso.types import OCRResult, OCRWord


class OCRAdapter(Protocol):
    def run(self, image_path: Path) -> OCRResult: ...


class StubOCR:
    """Deterministic stub. Returns a fixed page with one deliberately mis-OCR'd
    distinctive token, so the lexicon layer has something to correct.
    """

    _DEFAULT_PAGE: list[OCRWord] = [
        OCRWord("lecture", 0.98),
        OCRWord("notes", 0.97),
        OCRWord("on", 0.99),
        OCRWord("eigenvecter", 0.55),
        OCRWord("decomposition", 0.91),
        OCRWord("and", 0.99),
        OCRWord("the", 0.99),
        OCRWord("spectral", 0.86),
        OCRWord("theorem", 0.93),
    ]

    def __init__(self, fixtures: dict[str, list[OCRWord]] | None = None):
        self._fixtures = fixtures or {}

    def run(self, image_path: Path) -> OCRResult:
        words = self._fixtures.get(image_path.name, list(self._DEFAULT_PAGE))
        return OCRResult.from_words(words)


class AzureOCR:
    """Azure AI Document Intelligence (`prebuilt-read`). Returns per-word confidence."""

    def __init__(self, endpoint: str, key: str):
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
        self._client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
        self.endpoint = endpoint

    def run(self, image_path: Path) -> OCRResult:
        with open(image_path, "rb") as fh:
            poller = self._client.begin_analyze_document(
                "prebuilt-read",
                body=fh,
                content_type="application/octet-stream",
            )
        result = poller.result()
        words: list[OCRWord] = []
        for page in (result.pages or []):
            for w in (page.words or []):
                bbox = _polygon_to_bbox(getattr(w, "polygon", None))
                conf = float(getattr(w, "confidence", 0.0) or 0.0)
                words.append(OCRWord(text=w.content, confidence=conf, bbox=bbox))
        return OCRResult.from_words(words)


def _polygon_to_bbox(polygon) -> tuple[float, float, float, float] | None:
    """Reduce Azure's 8-point polygon to an axis-aligned (x, y, w, h)."""
    if not polygon or len(polygon) < 8:
        return None
    xs = polygon[0::2]
    ys = polygon[1::2]
    return (float(min(xs)), float(min(ys)),
            float(max(xs) - min(xs)), float(max(ys) - min(ys)))


def make_ocr(engine: str) -> OCRAdapter:
    if engine == "stub":
        return StubOCR()
    if engine == "azure":
        import os
        endpoint = os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"]
        key = os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"]
        return AzureOCR(endpoint, key)
    raise ValueError(f"Unknown OCR engine: {engine!r}")
