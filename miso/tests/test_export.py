import unittest

from miso.document import validate
from miso.export import latex_to_unicode, render_note_html, render_note_markdown

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

    def test_strips_nbsp_layout_artifacts(self):
        # a VLM may pad text with the &nbsp; entity or the literal NBSP char to fake
        # horizontal layout; validate() must collapse both so they don't pollute scoring
        out = validate({"title": "t", "blocks": [
            {"type": "paragraph", "text": "Safety Properties&nbsp;&nbsp;&nbsp;is something"},
        ]})
        self.assertEqual(out["blocks"][0]["text"], "Safety Properties is something")


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


class FigureTests(unittest.TestCase):
    """A figure block carries the VLM's description now and an `image` slot a later
    crop step fills — so every step supports an image before the cropping exists."""

    def test_validate_keeps_figure_with_empty_image_slot(self):
        out = validate({"title": "t", "blocks": [
            {"type": "figure", "description": "A state diagram", "bbox": [0.1, 0.2, 0.3, 0.4]},
        ]})
        b = out["blocks"][0]
        self.assertEqual(b["type"], "figure")
        self.assertEqual(b["description"], "A state diagram")
        self.assertEqual(b["image"], "")            # slot present, empty until the crop step
        self.assertEqual(b["bbox"], [0.1, 0.2, 0.3, 0.4])

    def test_validate_drops_figure_without_description(self):
        out = validate({"title": "t", "blocks": [{"type": "figure", "bbox": [0, 0, 1, 1]}]})
        self.assertEqual(out["blocks"], [])

    def test_validate_drops_malformed_bbox_but_keeps_figure(self):
        out = validate({"title": "t", "blocks": [
            {"type": "figure", "description": "d", "bbox": [1, 2, 3]},   # wrong length
        ]})
        self.assertEqual(out["blocks"][0]["description"], "d")
        self.assertNotIn("bbox", out["blocks"][0])

    def test_figure_html_renders_image_when_slot_filled(self):
        doc = {"title": "t", "summary_topic_line": "", "summary_gist": "", "blocks": [
            {"type": "figure", "description": "A circuit", "image": "fig1.png"},
        ]}
        h = render_note_html(doc)
        self.assertIn("<figcaption>A circuit</figcaption>", h)
        self.assertIn('<img src="fig1.png"', h)

    def test_figure_html_caption_only_until_image_exists(self):
        doc = {"title": "t", "summary_topic_line": "", "summary_gist": "", "blocks": [
            {"type": "figure", "description": "A circuit"},
        ]}
        h = render_note_html(doc)
        self.assertIn("<figcaption>A circuit</figcaption>", h)
        self.assertNotIn("<img", h)

    def test_figure_markdown_with_and_without_image(self):
        base = {"title": "t", "summary_topic_line": "", "summary_gist": ""}
        no_img = render_note_markdown({**base, "blocks": [
            {"type": "figure", "description": "A circuit"}]})
        self.assertIn("*[Figure: A circuit]*", no_img)
        with_img = render_note_markdown({**base, "blocks": [
            {"type": "figure", "description": "A circuit", "image": "f.png"}]})
        self.assertIn("![A circuit](f.png)", with_img)


class FigureEmbedTests(unittest.TestCase):
    """Pure logic behind embedding figure images into a Google Doc (no network)."""

    def test_stage_figures_tokenizes_only_imaged_figures(self):
        from miso.export import _stage_figures
        doc = {"title": "t", "blocks": [
            {"type": "paragraph", "text": "p"},
            {"type": "figure", "description": "d1", "image": "a.png"},
            {"type": "figure", "description": "d2", "image": ""},   # empty slot -> untouched
        ]}
        render_doc, pending = _stage_figures(doc)
        self.assertEqual(len(pending), 1)
        token, path = pending[0]
        self.assertEqual(path, "a.png")
        self.assertEqual(render_doc["blocks"][1], {"type": "paragraph", "text": token})
        self.assertEqual(render_doc["blocks"][2]["type"], "figure")   # caption-only figure kept
        self.assertEqual(doc["blocks"][1]["type"], "figure")          # original not mutated

    def test_token_ranges_offsets_within_run(self):
        from miso.export import _token_ranges
        document = {"body": {"content": [
            {"paragraph": {"elements": [{"startIndex": 5, "textRun": {"content": "hi X\n"}}]}},
        ]}}
        self.assertEqual(_token_ranges(document, ["X"]), {"X": (8, 9)})  # X at run offset 3 -> 5+3

    def test_image_requests_delete_then_insert_bottom_up(self):
        from miso.export import _inline_image_requests
        ranges = {"A": (8, 9), "B": (20, 21)}
        reqs = _inline_image_requests(ranges, {"A": "urlA", "B": "urlB"})
        # higher index (B) processed first so A's indices stay valid
        self.assertEqual(reqs[0]["deleteContentRange"]["range"]["startIndex"], 20)
        self.assertEqual(reqs[1]["insertInlineImage"], {"location": {"index": 20}, "uri": "urlB"})
        self.assertEqual(reqs[2]["deleteContentRange"]["range"]["startIndex"], 8)
        self.assertEqual(reqs[3]["insertInlineImage"], {"location": {"index": 8}, "uri": "urlA"})


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
