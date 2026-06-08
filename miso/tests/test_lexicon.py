import tempfile
import unittest
from pathlib import Path

from miso.config import LexiconConfig
from miso.db import open_db
from miso.lexicon import LexiconLayer
from miso.types import OCRResult, OCRWord


def _ocr(*words):
    """Build an OCRResult; one word per line so layout is well-defined."""
    ws = [OCRWord(text=t, confidence=c, bbox=(100.0, 100.0 * (i + 1), 40.0, 30.0))
          for i, (t, c) in enumerate(words)]
    return OCRResult.from_words(ws)


class LexiconCandidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "t.db"
        self.conn = open_db(self.tmp)
        now = "2026-01-01T00:00:00"
        # Course vocabulary with frequencies (the relevance prior).
        for term, freq in [("derivative", 47), ("derivation", 12),
                           ("eigenvalue", 30), ("manifold", 8),
                           ("cat", 5), ("cot", 80)]:
            self.conn.execute(
                "INSERT INTO lexicon_terms(course_id, term, frequency, "
                "context_snippet, first_seen, last_seen) VALUES (?,?,?,?,?,?)",
                ("CS101", term, freq, None, now, now),
            )
        self.conn.commit()
        self.lex = LexiconLayer(self.conn, common_words=set())

    def tearDown(self):
        self.conn.close()

    def test_flag_surfaces_candidates_for_low_confidence_word(self):
        cfg = LexiconConfig(mode="flag")
        ocr = _ocr(("deriative", 0.40), ("the", 0.99))
        out = self.lex.correct(ocr, "CS101", cfg)
        # The misread is flagged; the confident word is not searched.
        self.assertEqual(len(out.flags), 1)
        flag = out.flags[0]
        self.assertEqual(flag.original, "deriative")
        self.assertEqual(flag.candidates[0].term, "derivative")
        # Flag mode never mutates the OCR.
        self.assertEqual(out.corrected_text, ocr.raw_text)
        self.assertEqual([w.text for w in out.words], ["deriative", "the"])

    def test_high_confidence_word_not_searched(self):
        cfg = LexiconConfig(mode="flag", search_ceiling=0.95)
        ocr = _ocr(("derivatve", 0.98))  # close to a term, but the OCR is sure
        out = self.lex.correct(ocr, "CS101", cfg)
        self.assertEqual(out.flags, [])

    def test_frequency_prior_ranks_common_term_first(self):
        cfg = LexiconConfig(mode="flag")
        # 'cit' is one edit from both 'cat' (freq 5) and 'cot' (freq 80).
        out = self.lex.correct(_ocr(("cit", 0.30)), "CS101", cfg)
        terms = [c.term for c in out.flags[0].candidates]
        self.assertEqual(set(terms), {"cot", "cat"})
        self.assertEqual(terms[0], "cot")  # frequency breaks the distance tie

    def test_relevance_is_monotone_in_confidence(self):
        cfg = LexiconConfig(mode="flag")
        low = self.lex.correct(_ocr(("deriative", 0.20)), "CS101", cfg)
        high = self.lex.correct(_ocr(("deriative", 0.80)), "CS101", cfg)
        self.assertGreater(low.flags[0].candidates[0].relevance,
                           high.flags[0].candidates[0].relevance)

    def test_replace_mode_swaps_token(self):
        cfg = LexiconConfig(mode="replace", confidence_threshold=0.7)
        out = self.lex.correct(_ocr(("deriative", 0.40)), "CS101", cfg)
        self.assertEqual(out.words[0].text, "derivative")
        self.assertEqual(out.corrected_text, "derivative")
        self.assertEqual(out.flags, [])

    def test_off_mode_passthrough(self):
        cfg = LexiconConfig(mode="off")
        out = self.lex.correct(_ocr(("deriative", 0.40)), "CS101", cfg)
        self.assertEqual(out.flags, [])
        self.assertEqual(out.words[0].text, "deriative")

    def test_empty_vocab_is_passthrough(self):
        cfg = LexiconConfig(mode="flag")
        out = self.lex.correct(_ocr(("deriative", 0.40)), "NO_SUCH_COURSE", cfg)
        self.assertEqual(out.flags, [])
        self.assertEqual(out.corrected_text, "deriative")


if __name__ == "__main__":
    unittest.main()
