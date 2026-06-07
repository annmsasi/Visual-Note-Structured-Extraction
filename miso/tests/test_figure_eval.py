import unittest

from miso.eval.figure_eval import (
    count_gold_figures, count_sys_figures, detection_scores,
)

GOLD_MD = """# cse138-000

## Transcription (verbatim)

intro text
[figure]
more text

## Structured note

# Title

[figure]

- a bullet

## Distinctive terms

- Lamport
"""


class CountTests(unittest.TestCase):
    def test_gold_counts_structured_section_only(self):
        # one [figure] in transcription + one in structured note -> count must be 1
        self.assertEqual(count_gold_figures(GOLD_MD), 1)

    def test_gold_zero_when_no_figure(self):
        md = "## Structured note\n\n# T\n\n- bullet\n\n## Distinctive terms\n- x\n"
        self.assertEqual(count_gold_figures(md), 0)

    def test_robust_to_duplicate_headers_and_nested_headings(self):
        # mirrors real drafts: a repeated transcription header + a `## ` content
        # heading inside the structured note. Both figures must still be counted.
        md = (
            "# p\n\n## Transcription (verbatim)\n## Transcription\n"
            "notes\n[figure]\nmore\n[figure]\n\n"
            "## Structured note\n# Title\n[figure]\n## State & Events\n[figure]\n\n"
            "## Distinctive terms\n- x\n"
        )
        # transcription has 2, structured has 2 (one after a nested `## ` heading) -> max 2
        self.assertEqual(count_gold_figures(md), 2)

    def test_sys_counts_figure_blocks(self):
        doc = {"blocks": [
            {"type": "paragraph", "text": "p"},
            {"type": "figure", "description": "d", "image": ""},
            {"type": "figure", "description": "e", "image": "x.png"},
        ]}
        self.assertEqual(count_sys_figures(doc), 2)


class DetectionTests(unittest.TestCase):
    def test_perfect_match(self):
        d = detection_scores({"a": 1, "b": 0}, {"a": 1, "b": 0})
        self.assertEqual((d["tp"], d["fp"], d["fn"]), (1, 0, 0))
        self.assertEqual(d["f1"], 1.0)
        self.assertEqual(d["count_mae"], 0.0)

    def test_miss_and_hallucination(self):
        # a: figure missed (FN); b: hallucinated figure (FP); c: correct (TP)
        gold = {"a": 1, "b": 0, "c": 2}
        sys = {"a": 0, "b": 1, "c": 2}
        d = detection_scores(gold, sys)
        self.assertEqual((d["tp"], d["fp"], d["fn"]), (1, 1, 1))
        self.assertAlmostEqual(d["precision"], 0.5)
        self.assertAlmostEqual(d["recall"], 0.5)
        self.assertAlmostEqual(d["count_mae"], (1 + 1 + 0) / 3)
        self.assertEqual(d["gold_total"], 3)
        self.assertEqual(d["sys_total"], 3)

    def test_count_error_tracked_even_when_page_detected(self):
        # page detected (both >0) but count differs -> TP, but nonzero MAE
        d = detection_scores({"a": 3}, {"a": 1})
        self.assertEqual((d["tp"], d["fp"], d["fn"]), (1, 0, 0))
        self.assertEqual(d["count_mae"], 2.0)


if __name__ == "__main__":
    unittest.main()
