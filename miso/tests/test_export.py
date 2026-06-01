import unittest

from miso.document import validate
from miso.export import latex_to_unicode, render_note_html

try:
    import pylatexenc  # noqa: F401
    _HAS_PYLATEXENC = True
except ImportError:
    _HAS_PYLATEXENC = False

DOC = {
    "title": "Chapter 3 <1763>",        # contains a char that must be escaped
    "blocks": [
        {"type": "heading", "level": 2, "text": "Imperial Reforms"},
        {"type": "paragraph", "text": "Debt mushroomed."},
        {"type": "list", "items": [
            {"text": "Sugar Act", "level": 0},
            {"text": "stop smuggling", "level": 1},
        ]},
        {"type": "equation", "latex": "E = mc^2"},
    ],
    "summary_topic_line": "Imperial reforms",
    "summary_gist": "Taxes rose.",
}


class ValidateTests(unittest.TestCase):
    def test_drops_unknown_block_types(self):
        out = validate({"title": "t", "blocks": [{"type": "bogus"}, {"type": "paragraph", "text": "ok"}]})
        self.assertEqual(len(out["blocks"]), 1)
        self.assertEqual(out["blocks"][0]["type"], "paragraph")

    def test_coerces_string_list_items(self):
        out = validate({"title": "t", "blocks": [{"type": "list", "items": ["a", "b"]}]})
        self.assertEqual(out["blocks"][0]["items"], [{"text": "a", "level": 0}, {"text": "b", "level": 0}])

    def test_non_dict_payload_is_safe(self):
        out = validate("not a dict")
        self.assertEqual(out["blocks"], [])
        self.assertTrue(out["title"])

    def test_required_summary_fields_always_present(self):
        out = validate({"title": "t", "blocks": []})
        self.assertIn("summary_topic_line", out)
        self.assertIn("summary_gist", out)


class RenderTests(unittest.TestCase):
    def test_note_html_is_a_full_page(self):
        h = render_note_html(DOC)
        self.assertTrue(h.lstrip().startswith("<!doctype"))   # one self-contained doc per note

    def test_note_html_has_title_and_blocks(self):
        h = render_note_html(DOC)
        self.assertIn("<h1>Chapter 3 &lt;1763&gt;</h1>", h)   # escaped
        self.assertIn("<h3>Imperial Reforms</h3>", h)         # level 2 -> +1 under title h1
        self.assertIn("<li>Sugar Act</li>", h)

    def test_nested_list_opens_and_closes_tags(self):
        h = render_note_html(DOC)
        self.assertEqual(h.count("<ul>"), h.count("</ul>"))   # balanced

    def test_level_one_item_actually_nests(self):
        h = render_note_html(DOC)
        # 'stop smuggling' is level 1 -> must sit inside a second, nested <ul>
        self.assertIn("<li>Sugar Act</li><ul><li>stop smuggling</li></ul>", h)

    def test_adjacent_list_blocks_coalesce(self):
        doc = {"title": "t", "blocks": [
            {"type": "list", "items": [{"text": "a", "level": 0}]},
            {"type": "list", "items": [{"text": "b", "level": 0}]},
        ], "summary_topic_line": "", "summary_gist": ""}
        h = render_note_html(doc)
        self.assertNotIn("</ul><ul>", h)                      # one shared list, not two
        self.assertIn("<li>a</li><li>b</li>", h)


@unittest.skipUnless(_HAS_PYLATEXENC, "pylatexenc not installed")
class LatexUnicodeTests(unittest.TestCase):
    def test_superscript(self):
        self.assertEqual(latex_to_unicode("E = mc^2"), "E = mc²")

    def test_sum_symbol_survives_with_scripts(self):
        self.assertEqual(latex_to_unicode(r"\sum_{i=1}^{n} x_i"), "∑ᵢ₌₁ⁿ xᵢ")

    def test_chemistry_subscript(self):
        self.assertEqual(latex_to_unicode("H_2O"), "H₂O")

    def test_stray_digit_not_treated_as_placeholder(self):
        self.assertEqual(latex_to_unicode("2x^2"), "2x²")  # leading 2 is not a script

    def test_never_leaks_raw_latex_or_dollars(self):
        out = latex_to_unicode(r"$\frac{\alpha+1}{\beta} \leq \gamma$")
        self.assertNotIn("\\", out)
        self.assertNotIn("$", out)

    def test_equation_block_renders_as_plain_text(self):
        doc = {"title": "t", "summary_topic_line": "", "summary_gist": "",
               "blocks": [{"type": "equation", "latex": "E=mc^2"}]}
        h = render_note_html(doc)
        self.assertIn("<p>E=mc²</p>", h)
        self.assertNotIn("[equation]", h)
        self.assertNotIn("<code>", h)


if __name__ == "__main__":
    unittest.main()
