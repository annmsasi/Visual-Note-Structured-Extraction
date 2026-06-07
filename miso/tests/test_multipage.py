import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from miso.augment import assemble_prompt
from miso.config import ExtractionConfig, RunConfig
from miso.db import open_db
from miso.eval.gold import GoldNote
from miso.eval.metrics import normalized_wer
from miso.eval.ocr_eval import score_pages
from miso.lexicon import LexiconLayer
from miso.pipeline import finalize_note
from miso.replay import _prior_window
from miso.summary_store import SummaryStore
from miso.types import CorrectedOCR, Note, OCRWord


def _corrected(text="alpha beta"):
    return CorrectedOCR(
        words=[OCRWord(text=t, confidence=0.9) for t in text.split()],
        corrected_text=text, corrections=[], touched_terms=[],
        layout_text=text, flags=[],
    )


class FakeExtractor:
    """Records the whole-note doc it was asked to summarize."""
    def __init__(self):
        self.summarize_calls = []

    def summarize(self, note_doc):
        self.summarize_calls.append(note_doc)
        return "TopicLine", "A gist."


class PriorWindowTests(unittest.TestCase):
    def test_window_semantics(self):
        md = [f"p{i}" for i in range(15)]
        self.assertEqual(_prior_window(md, 10), md[-10:])   # cap to last 10
        self.assertEqual(len(_prior_window(md, 10)), 10)
        self.assertEqual(_prior_window(md, 0), [])          # 0 = off
        self.assertEqual(_prior_window(md, -1), md)         # -1 = all
        self.assertEqual(_prior_window(["a"], 10), ["a"])   # fewer than k


class PriorPagesPromptTests(unittest.TestCase):
    def test_block_present_and_instructs_current_page_only(self):
        prompt = assemble_prompt(
            corrected_ocr=_corrected(), retrieved=[], glossary=[],
            cfg=ExtractionConfig(), prior_pages_md=["# Page one\n- earlier point"],
        )
        self.assertIn("Earlier pages of THIS note", prompt)
        self.assertIn("earlier point", prompt)
        self.assertIn("only the current page", prompt.lower())

    def test_block_absent_when_no_prior_pages(self):
        prompt = assemble_prompt(
            corrected_ocr=_corrected(), retrieved=[], glossary=[],
            cfg=ExtractionConfig(), prior_pages_md=None,
        )
        self.assertNotIn("Earlier pages of THIS note", prompt)


class FinalizeNoteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "t.db"
        self.conn = open_db(self.tmp)
        self.summary_store = SummaryStore(self.conn, embedder=None)
        self.lex = LexiconLayer(self.conn, common_words=set())
        self.cfg = RunConfig(config_tag="t")

    def tearDown(self):
        self.conn.close()

    def test_stores_whole_note_summary_at_note_granularity(self):
        note = Note(note_id="doc-001", course_id="CS101",
                    image_path=Path("x.jpg"), processing_order=0,
                    timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))
        note_doc = {"title": "Eigenvalues",
                    "blocks": [{"type": "paragraph", "text": "eigenvalue manifold"}],
                    "summary_topic_line": "", "summary_gist": ""}
        fake = FakeExtractor()
        ext = finalize_note(note_doc, note, extractor=fake,
                            summary_store=self.summary_store,
                            lexicon_layer=self.lex, cfg=self.cfg)

        # exactly one dedicated summary call, over the whole note
        self.assertEqual(len(fake.summarize_calls), 1)
        # summary written back into the doc (for export) and returned
        self.assertEqual(note_doc["summary_topic_line"], "TopicLine")
        self.assertEqual(ext.summary_gist, "A gist.")
        # stored under the note id (no -pNNN), retrievable
        row = self.conn.execute(
            "SELECT topic_line FROM summaries WHERE note_id = ?", ("doc-001",)
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["topic_line"], "TopicLine")
        # harvest recorded term sightings from the whole-note body
        n = self.conn.execute(
            "SELECT COUNT(*) AS c FROM lexicon_sightings WHERE course_id = ?",
            ("CS101",)).fetchone()["c"]
        self.assertGreater(n, 0)

    def test_skips_summary_call_when_retrieval_off(self):
        note = Note(note_id="doc-001", course_id="CS101",
                    image_path=Path("x.jpg"), processing_order=0,
                    timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))
        note_doc = {"title": "Eigenvalues",
                    "blocks": [{"type": "paragraph", "text": "eigenvalue manifold"}],
                    "summary_topic_line": "", "summary_gist": ""}
        fake = FakeExtractor()
        cfg = RunConfig(config_tag="t")
        cfg.retrieval.enabled = False
        finalize_note(note_doc, note, extractor=fake,
                      summary_store=self.summary_store,
                      lexicon_layer=self.lex, cfg=cfg)
        # the dedicated (billed) summary call is skipped; a cheap title is used
        self.assertEqual(fake.summarize_calls, [])
        self.assertEqual(note_doc["summary_topic_line"], "Eigenvalues")
        # nothing stored, since only retrieval reads summaries
        self.assertIsNone(self.conn.execute(
            "SELECT 1 FROM summaries WHERE note_id = ?", ("doc-001",)).fetchone())
        # but the lexicon harvest still ran
        self.assertGreater(self.conn.execute(
            "SELECT COUNT(*) AS c FROM lexicon_sightings WHERE course_id = ?",
            ("CS101",)).fetchone()["c"], 0)


class NormalizedWerTests(unittest.TestCase):
    def test_formatting_does_not_count(self):
        # case, extra spaces, line breaks, trailing punctuation — all ignored
        self.assertEqual(normalized_wer("The cat sat.", "the   CAT\nsat"), 0.0)

    def test_word_choice_counts(self):
        self.assertAlmostEqual(normalized_wer("the cat sat", "the dog sat"), 1 / 3)
        self.assertAlmostEqual(normalized_wer("the cat sat", "the cat"), 1 / 3)


class OcrEvalScoringTests(unittest.TestCase):
    def test_word_choice_only_aligned_to_gold(self):
        imgs = [Path("tim172a-000.jpg"), Path("tim172a-001.jpg"),
                Path("tim172a-002.jpg")]  # 002 has no gold -> skipped
        gold = {
            "tim172a-000": GoldNote("tim172a-000", {}, "The eigenvalue of the matrix",
                                    ["eigenvalue", "matrix"]),
            "tim172a-001": GoldNote("tim172a-001", {}, "gradient descent converges",
                                    ["gradient descent"]),
        }
        ocr_by_stem = {
            # same words, different layout/case/punctuation -> perfect word choice
            "tim172a-000": {"text": "the  EIGENVALUE of the Matrix."},
            # 'gradient' misread -> the term is missed (strict spelling) and 1 word wrong
            "tim172a-001": {"text": "grodient descent converges"},
            "tim172a-002": {"text": "ignored, no gold"},
        }
        out = score_pages(imgs, "tim172a", ocr_by_stem, gold)
        self.assertEqual(out["n"], 2)
        self.assertAlmostEqual(out["word_wer"], (0.0 + 1 / 3) / 2)
        self.assertAlmostEqual(out["term_recall"], (1.0 + 0.0) / 2)


if __name__ == "__main__":
    unittest.main()
