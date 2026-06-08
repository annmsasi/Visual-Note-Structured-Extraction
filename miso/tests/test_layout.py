import unittest

from miso.layout import group_into_lines, iter_lines, render_layout_text
from miso.types import OCRResult, OCRWord


def _w(text, x, y, w=40.0, h=30.0):
    return OCRWord(text=text, confidence=0.9, bbox=(x, y, w, h))


class LineGroupingTests(unittest.TestCase):
    def test_words_on_same_y_form_one_line(self):
        words = [_w("hello", 100, 100), _w("world", 160, 102)]
        group_into_lines(words)
        self.assertEqual(words[0].line_id, words[1].line_id)

    def test_y_jump_starts_new_line(self):
        words = [_w("top", 100, 100), _w("bottom", 100, 200)]
        group_into_lines(words)
        self.assertNotEqual(words[0].line_id, words[1].line_id)

    def test_reading_order_is_top_then_left(self):
        # second line word listed first; left/top ordering must be recovered
        words = [_w("b2", 200, 200), _w("a1", 100, 100), _w("b1", 100, 200)]
        text = render_layout_text(words)
        self.assertEqual(text.splitlines()[0].strip(), "a1")
        self.assertEqual(text.splitlines()[1].strip(), "b1 b2")

    def test_indentation_depth_increases_with_left_margin(self):
        words = [_w("heading", 100, 100), _w("indented", 220, 200)]
        lines = iter_lines(words)
        self.assertEqual(lines[0].depth, 0)
        self.assertGreater(lines[1].depth, 0)

    def test_no_geometry_collapses_to_flat_join(self):
        words = [OCRWord("a", 0.9), OCRWord("b", 0.9)]
        self.assertEqual(render_layout_text(words), "a b")

    def test_from_words_populates_layout_text(self):
        res = OCRResult.from_words([_w("x", 100, 100), _w("y", 100, 200)])
        self.assertEqual(res.raw_text, "x y")            # flat path unchanged
        self.assertEqual(res.layout_text, "x\ny")        # structure preserved


class SoftWrapTests(unittest.TestCase):
    def test_full_width_line_merges_with_continuation(self):
        # Line 1 runs to the right margin; line 2 starts at left -> a soft wrap,
        # not a real line break. Expect one merged logical line.
        words = [
            _w("aaaa", 100, 100, w=200), _w("bbbb", 320, 100, w=200), _w("cccc", 540, 100, w=200),
            _w("dddd", 100, 200, w=200),  # next physical line, leftmost -> would not fit at x=760
        ]
        text = render_layout_text(words)
        self.assertEqual(text, "aaaa bbbb cccc dddd")
        self.assertEqual(len(text.splitlines()), 1)

    def test_short_line_does_not_absorb_next(self):
        # Line 1 ends far from the right margin (room to spare) -> a real break.
        words = [
            _w("short", 100, 100, w=100),
            _w("verylongword", 100, 200, w=100), _w("filling", 220, 200, w=100),
            _w("therow", 340, 200, w=2000),  # establishes a far-right margin
        ]
        text = render_layout_text(words)
        self.assertEqual(text.splitlines()[0].strip(), "short")
        self.assertEqual(len(text.splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
