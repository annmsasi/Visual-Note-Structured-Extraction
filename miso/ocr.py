"""OCR adapters: a Protocol, a stub, and an Azure-backed implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from miso.types import OCRResult, OCRWord


class OCRAdapter(Protocol):
    def run(self, image_path: Path) -> OCRResult: ...


class StubOCR:
    """Return a fixed page with one mis-OCR'd token for the lexicon layer to correct."""

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
    """Run Azure AI Document Intelligence (`prebuilt-read`) with per-word confidence."""

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
    """Reduce an 8-point polygon to an axis-aligned (x, y, w, h)."""
    if not polygon or len(polygon) < 8:
        return None
    xs = polygon[0::2]
    ys = polygon[1::2]
    return (float(min(xs)), float(min(ys)),
            float(max(xs) - min(xs)), float(max(ys) - min(ys)))


class PaddleOCR:
    """Free, local OCR via PaddleOCR — a drop-in for `AzureOCR` (no cloud, no key,
    no per-word confidence needed now the cache is optional).

    PaddleOCR detects text *lines*; we split each line into words with proportional
    boxes so `miso.layout` can still recover line breaks and indentation from the
    same `(x, y, w, h)` geometry it gets from Azure.
    """

    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR as _Engine
        try:
            self._engine = _Engine(use_angle_cls=True, lang=lang)
        except TypeError:  # newer releases dropped/renamed kwargs
            self._engine = _Engine(lang=lang)

    def run(self, image_path: Path) -> OCRResult:
        try:
            result = self._engine.ocr(str(image_path), cls=True)
        except TypeError:
            result = self._engine.ocr(str(image_path))
        lines = result[0] if (result and result[0]) else []
        words: list[OCRWord] = []
        for entry in lines:
            try:
                box, (text, conf) = entry
            except (ValueError, TypeError):
                continue
            text = (text or "").strip()
            if text:
                words.extend(_line_to_words(box, text, float(conf)))
        return OCRResult.from_words(words)


class TesseractOCR:
    """Free, local OCR via Tesseract — a drop-in for `AzureOCR`. The weakest reader
    of the free options, but it only needs the system `tesseract` binary (no Python
    deps) and returns word-level boxes + confidence, so `miso.layout` recovers
    structure exactly as with Azure. We call the binary directly and parse its TSV
    output, sidestepping `pytesseract`'s fragile stderr decoding.
    """

    def __init__(self, lang: str = "eng", binary: str = "tesseract"):
        self.lang = lang
        self.binary = binary

    def run(self, image_path: Path) -> OCRResult:
        import subprocess
        proc = subprocess.run(
            [self.binary, str(image_path), "stdout", "-l", self.lang, "tsv"],
            capture_output=True,
        )
        rows = proc.stdout.decode("utf-8", errors="replace").splitlines()
        words: list[OCRWord] = []
        for row in rows[1:]:  # skip the TSV header
            cols = row.split("\t")
            if len(cols) < 12:
                continue
            text = cols[11].strip()
            if not text:
                continue
            try:
                left, top, width, height, conf = (float(cols[i]) for i in (6, 7, 8, 9, 10))
            except ValueError:
                continue
            words.append(OCRWord(
                text=text,
                confidence=conf / 100.0 if conf >= 0 else 0.0,  # tesseract: 0-100, -1 = none
                bbox=(left, top, width, height),
            ))
        return OCRResult.from_words(words)


def _line_to_words(box, text: str, conf: float) -> list[OCRWord]:
    """Spread a recognised line's words across its bounding box by character length."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    left, top = float(min(xs)), float(min(ys))
    width = max(float(max(xs)) - left, 1.0)
    height = max(float(max(ys)) - top, 1.0)
    toks = text.split()
    if not toks:
        return []
    span = sum(len(t) for t in toks) + (len(toks) - 1)  # chars + inter-word spaces
    out: list[OCRWord] = []
    cursor = 0
    for t in toks:
        x = left + (cursor / span) * width
        w = max((len(t) / span) * width, 1.0)
        out.append(OCRWord(text=t, confidence=conf, bbox=(x, top, w, height)))
        cursor += len(t) + 1
    return out


class CachedOCR:
    """Content-addressed disk cache around any OCRAdapter."""

    def __init__(self, inner: "OCRAdapter", cache_dir: Path = Path(".ocr_cache")):
        self._inner = inner
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def run(self, image_path: Path) -> OCRResult:
        import hashlib
        import json
        digest = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
        cache_file = self._dir / f"{digest}.json"
        if cache_file.exists():
            return _ocr_from_dict(json.loads(cache_file.read_text()))
        result = self._inner.run(image_path)
        cache_file.write_text(json.dumps(_ocr_to_dict(result)))
        return result


def _ocr_to_dict(r: OCRResult) -> dict:
    return {
        "raw_text": r.raw_text,
        "layout_text": r.layout_text,
        "words": [{"text": w.text, "confidence": w.confidence,
                   "bbox": list(w.bbox) if w.bbox else None, "line_id": w.line_id}
                  for w in r.words],
    }


def _ocr_from_dict(d: dict) -> OCRResult:
    words = [OCRWord(text=w["text"], confidence=w["confidence"],
                     bbox=tuple(w["bbox"]) if w["bbox"] else None,
                     line_id=w.get("line_id"))
             for w in d["words"]]
    return OCRResult(words=words, raw_text=d["raw_text"],
                     layout_text=d.get("layout_text", ""))


def make_ocr(engine: str) -> OCRAdapter:
    if engine == "stub":
        return StubOCR()
    if engine == "azure":
        import os
        endpoint = os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"]
        key = os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"]
        return AzureOCR(endpoint, key)
    if engine == "paddle":
        return PaddleOCR()
    if engine == "tesseract":
        return TesseractOCR()
    raise ValueError(f"Unknown OCR engine: {engine!r}")
