"""Tests for `miso.eval.metrics`. Stdlib unittest, no pytest required.

    python -m unittest miso.tests.test_metrics
"""
from __future__ import annotations

import unittest

from miso.eval.metrics import (
    align_tokens,
    bootstrap_ci,
    cer,
    correction_precision_recall,
    levenshtein,
    structural_f1,
    term_recall,
    term_restricted_cer,
    wer,
)


class LevenshteinTests(unittest.TestCase):
    def test_classic_example(self):
        self.assertEqual(levenshtein("kitten", "sitting"), 3)

    def test_empty_strings(self):
        self.assertEqual(levenshtein("", ""), 0)
        self.assertEqual(levenshtein("abc", ""), 3)
        self.assertEqual(levenshtein("", "abc"), 3)

    def test_identical(self):
        self.assertEqual(levenshtein("hello", "hello"), 0)

    def test_works_on_token_lists(self):
        self.assertEqual(levenshtein(["a", "b", "c"], ["a", "x", "c"]), 1)
        self.assertEqual(levenshtein(["a", "b"], ["a", "b", "c"]), 1)


class CERTests(unittest.TestCase):
    def test_perfect_match_is_zero(self):
        self.assertEqual(cer("hello world", "hello world"), 0.0)

    def test_single_substitution(self):
        self.assertAlmostEqual(cer("abc", "abd"), 1 / 3)

    def test_empty_reference(self):
        self.assertEqual(cer("", ""), 0.0)
        self.assertEqual(cer("", "abc"), 1.0)

    def test_can_exceed_one(self):
        self.assertGreater(cer("ab", "abcdef"), 1.0)


class WERTests(unittest.TestCase):
    def test_perfect_match(self):
        self.assertEqual(wer("a b c", "a b c"), 0.0)

    def test_one_word_substituted(self):
        self.assertAlmostEqual(wer("a b c", "a x c"), 1 / 3)

    def test_insertion(self):
        self.assertAlmostEqual(wer("a b c", "a b c d"), 1 / 3)

    def test_empty_reference(self):
        self.assertEqual(wer("", ""), 0.0)
        self.assertEqual(wer("", "hello"), 1.0)


class StructuralF1Tests(unittest.TestCase):
    def test_identical_returns_one(self):
        d = {"a": 1, "b": {"c": 2}}
        self.assertEqual(structural_f1(d, d), 1.0)

    def test_both_empty_returns_one(self):
        self.assertEqual(structural_f1({}, {}), 1.0)

    def test_disjoint_returns_zero(self):
        self.assertEqual(structural_f1({"a": 1}, {"b": 2}), 0.0)

    def test_partial_overlap(self):
        self.assertAlmostEqual(structural_f1({"a": 1, "b": 2}, {"a": 1, "c": 3}), 0.5)


class CorrectionPrecisionRecallTests(unittest.TestCase):
    def test_no_corrections_returns_zeros(self):
        self.assertEqual(correction_precision_recall([], "gold", "raw"), (0.0, 0.0, 0.0))

    def test_helpful_correction(self):
        corrections = [{"original": "eigenvecter", "suggested": "eigenvector", "accepted": True}]
        precision, recall, over = correction_precision_recall(
            corrections,
            gold_text="lecture notes on eigenvector",
            raw_text="lecture notes on eigenvecter",
        )
        self.assertEqual(precision, 1.0)
        self.assertEqual(over, 0.0)
        self.assertEqual(recall, 1.0)

    def test_harmful_correction(self):
        corrections = [{"original": "eigenvector", "suggested": "eigenvecter", "accepted": True}]
        precision, _, over = correction_precision_recall(
            corrections,
            gold_text="lecture notes on eigenvector",
            raw_text="lecture notes on eigenvector",
        )
        self.assertEqual(precision, 0.0)
        self.assertEqual(over, 1.0)

    def test_neutral_correction_excluded(self):
        corrections = [{"original": "foo", "suggested": "bar", "accepted": True}]
        precision, _, over = correction_precision_recall(
            corrections,
            gold_text="completely unrelated text",
            raw_text="foo here",
        )
        self.assertEqual(precision, 0.0)
        self.assertEqual(over, 0.0)

    def test_position_aware_correct_token_made_wrong(self):
        # 'note' is already correct at its position; changing it away from gold hurts —
        # the set-membership version could miss this. Alignment catches it.
        corrections = [{"original": "note", "suggested": "bar", "accepted": True}]
        precision, _, over = correction_precision_recall(
            corrections, gold_text="foo note", raw_text="foo note",
        )
        self.assertEqual(precision, 0.0)
        self.assertEqual(over, 1.0)


class AlignTokensTests(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(align_tokens(["a", "b"], ["a", "b"]), [("a", "a"), ("b", "b")])

    def test_substitution(self):
        self.assertEqual(
            align_tokens(["a", "b", "c"], ["a", "x", "c"]),
            [("a", "a"), ("b", "x"), ("c", "c")],
        )

    def test_insertion_and_deletion(self):
        self.assertEqual(align_tokens(["a", "c"], ["a", "b", "c"]),
                         [("a", "a"), (None, "b"), ("c", "c")])
        self.assertEqual(align_tokens(["a", "b", "c"], ["a", "c"]),
                         [("a", "a"), ("b", None), ("c", "c")])


class TermRecallTests(unittest.TestCase):
    def test_none_when_no_terms(self):
        self.assertIsNone(term_recall([], "anything at all"))

    def test_all_present_case_insensitive(self):
        self.assertEqual(term_recall(["eigenvector", "matrix"], "the eigenvector of a Matrix."), 1.0)

    def test_partial(self):
        self.assertAlmostEqual(term_recall(["eigenvector", "kernel"], "only the eigenvector here"), 0.5)

    def test_multiword_needs_adjacency(self):
        self.assertEqual(term_recall(["dynamic programming"], "we use Dynamic Programming today"), 1.0)
        self.assertEqual(term_recall(["dynamic programming"], "dynamic and programming apart"), 0.0)

    def test_plural_matches_but_misspelling_misses(self):
        # regular plurals count (the LLM's legitimate morphology)
        self.assertEqual(term_recall(["eigenvector"], "the eigenvectors shown here"), 1.0)
        self.assertEqual(term_recall(["spectral theorem"], "by the spectral theorems above"), 1.0)
        # an unrelated similar word does NOT count
        self.assertEqual(term_recall(["kernel"], "only the colonel spoke"), 0.0)
        # the crucial property: a misspelling the cache is meant to FIX stays a miss,
        # else term-recall couldn't see the cache's benefit
        self.assertEqual(term_recall(["eigenvector"], "the eigenvecter here"), 0.0)


class TermRestrictedCERTests(unittest.TestCase):
    def test_none_when_term_absent_from_reference(self):
        self.assertIsNone(term_restricted_cer("plain words here", "plain words here", ["eigenvector"]))

    def test_perfect_recognition_is_zero(self):
        self.assertEqual(
            term_restricted_cer("the eigenvector x", "the eigenvector x", ["eigenvector"]), 0.0,
        )

    def test_misrecognised_term(self):
        # 'eigenvector' (11 chars) → 'eigenvecter' is one edit.
        self.assertAlmostEqual(
            term_restricted_cer("the eigenvector x", "the eigenvecter x", ["eigenvector"]), 1 / 11,
        )

    def test_only_term_spans_count(self):
        # Non-term words differ but don't count; the term matches → 0.
        self.assertEqual(
            term_restricted_cer("foo eigenvector bar", "XXX eigenvector YYY", ["eigenvector"]), 0.0,
        )


class BootstrapCITests(unittest.TestCase):
    def test_empty_returns_zeros(self):
        self.assertEqual(bootstrap_ci([]), (0.0, 0.0, 0.0))

    def test_constant_input(self):
        mean, lo, hi = bootstrap_ci([0.5] * 20)
        self.assertEqual(mean, 0.5)
        self.assertEqual(lo, 0.5)
        self.assertEqual(hi, 0.5)

    def test_ci_brackets_mean(self):
        values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        mean, lo, hi = bootstrap_ci(values, n_samples=500)
        self.assertAlmostEqual(mean, 0.45)
        self.assertLess(lo, mean)
        self.assertGreater(hi, mean)

    def test_deterministic_with_seed(self):
        values = [0.1, 0.2, 0.3, 0.4]
        a = bootstrap_ci(values, seed=42, n_samples=200)
        b = bootstrap_ci(values, seed=42, n_samples=200)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
