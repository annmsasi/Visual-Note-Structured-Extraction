"""Tests for the deterministic faithfulness stub (no API needed).

    python -m unittest miso.tests.test_faithfulness
"""
from __future__ import annotations

import unittest

from miso.eval.faithfulness import StubFaithfulnessJudge, document_claims


class DocumentClaimsTests(unittest.TestCase):
    def test_extracts_body_excludes_summary(self):
        doc = {
            "title": "Topic",
            "blocks": [
                {"type": "heading", "text": "Heading one"},
                {"type": "list", "items": [{"text": "first item"}, {"text": "second item"}]},
                {"type": "paragraph", "text": "a running sentence"},
            ],
            "summary_gist": "should be ignored",
        }
        claims = document_claims(doc)
        self.assertIn("Topic", claims)
        self.assertIn("Heading one", claims)
        self.assertIn("first item", claims)
        self.assertIn("a running sentence", claims)
        self.assertNotIn("should be ignored", claims)


class StubJudgeTests(unittest.TestCase):
    def test_supported_claims_score_high(self):
        doc = {"title": "eigenvector decomposition",
               "blocks": [{"type": "paragraph", "text": "the eigenvector basis vectors"}]}
        ocr = "notes about the eigenvector decomposition and eigenvector basis vectors"
        v = StubFaithfulnessJudge().judge(doc=doc, ocr_text=ocr)
        self.assertEqual(v["unsupported"], [])
        self.assertEqual(v["score"], 1.0)

    def test_hallucination_is_flagged(self):
        doc = {"title": "topic",
               "blocks": [{"type": "paragraph", "text": "quantum entanglement teleportation protocol"}]}
        ocr = "basic arithmetic addition and subtraction only"
        v = StubFaithfulnessJudge().judge(doc=doc, ocr_text=ocr)
        self.assertGreaterEqual(len(v["unsupported"]), 1)
        self.assertLess(v["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
