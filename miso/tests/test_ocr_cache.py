import tempfile
import unittest
from pathlib import Path

from miso.ocr import CachedOCR
from miso.types import OCRResult, OCRWord


class _FakeOCR:
    def __init__(self):
        self.calls = 0

    def run(self, path):
        self.calls += 1
        return OCRResult.from_words([OCRWord("hi", 0.9, bbox=(0.0, 0.0, 1.0, 1.0))])


class CachedOCRTests(unittest.TestCase):
    def test_same_image_hits_cache(self):
        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / "a.png"
            img.write_bytes(b"IMAGE-BYTES")
            inner = _FakeOCR()
            ocr = CachedOCR(inner, cache_dir=Path(d) / "cache")
            r1 = ocr.run(img)
            r2 = ocr.run(img)              # served from disk cache
            self.assertEqual(inner.calls, 1)
            self.assertEqual(r1.raw_text, r2.raw_text)
            self.assertEqual(r2.words[0].text, "hi")
            self.assertEqual(r2.words[0].bbox, (0.0, 0.0, 1.0, 1.0))

    def test_different_content_misses_cache(self):
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / "a.png"; a.write_bytes(b"A")
            b = Path(d) / "b.png"; b.write_bytes(b"B")
            inner = _FakeOCR()
            ocr = CachedOCR(inner, cache_dir=Path(d) / "cache")
            ocr.run(a)
            ocr.run(b)
            self.assertEqual(inner.calls, 2)


if __name__ == "__main__":
    unittest.main()
