"""End-to-end test of figure extraction: a page image + a model payload carrying a
figure bbox → a real cropped PNG → the block's `image` slot filled → that crop
flowing into both local Markdown rendering and the Docs API embed staging.

Everything the live pipeline does is exercised except the VLM call itself (replaced
by a fixed payload) and the Drive/Docs network calls.
"""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from miso.export import _stage_figures, render_note_markdown
from miso.extraction import _to_note
from miso.figures import _pixel_box, _valid_bbox, crop_figures
from miso.types import Note

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def _page_with_red_box(path: Path) -> tuple[int, int]:
    """A 400×300 white page with a red rectangle at pixels (200,60)–(360,180).
    That region is the normalized bbox [0.5, 0.2, 0.4, 0.4]."""
    img = Image.new("RGB", (400, 300), "white")
    for x in range(200, 360):
        for y in range(60, 180):
            img.putpixel((x, y), (255, 0, 0))
    img.save(path, "JPEG", quality=95)
    return img.size


class PixelBoxTests(unittest.TestCase):
    def test_valid_bbox_requires_positive_extent(self):
        self.assertTrue(_valid_bbox([0.1, 0.1, 0.2, 0.2]))
        self.assertFalse(_valid_bbox([0.1, 0.1, 0.0, 0.2]))   # zero width
        self.assertFalse(_valid_bbox([0.1, 0.1, 0.2]))        # wrong length
        self.assertFalse(_valid_bbox("nope"))

    def test_pixel_box_maps_and_clamps(self):
        # no padding -> exact mapping
        self.assertEqual(_pixel_box([0.5, 0.2, 0.4, 0.4], 400, 300, 0.0), (200, 60, 360, 180))
        # padding that would overflow the page is clamped to its edges
        self.assertEqual(_pixel_box([0.0, 0.0, 1.0, 1.0], 400, 300, 0.1), (0, 0, 400, 300))

    def test_pixel_box_never_empty(self):
        left, top, right, bottom = _pixel_box([0.5, 0.5, 0.0001, 0.0001], 10, 10, 0.0)
        self.assertGreater(right, left)
        self.assertGreater(bottom, top)


@unittest.skipUnless(_HAS_PIL, "Pillow not installed")
class CropFiguresTests(unittest.TestCase):
    def test_crop_writes_png_and_fills_image_with_right_region(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            page = tmp / "page.jpg"
            _page_with_red_box(page)
            doc = {"title": "t", "blocks": [
                {"type": "figure", "description": "red box", "image": "",
                 "bbox": [0.5, 0.2, 0.4, 0.4]},
            ]}
            crop_figures(doc, page, tmp / "figs", note_id="n1", pad=0.0)
            img_path = Path(doc["blocks"][0]["image"])
            self.assertTrue(img_path.exists())
            self.assertEqual(img_path.parent.name, "n1")            # namespaced by note id
            crop = Image.open(img_path)
            self.assertEqual(crop.size, (160, 120))                 # exact cropped region
            r, g, b = crop.convert("RGB").getpixel((80, 60))        # it's the red box
            self.assertTrue(r > 200 and g < 60 and b < 60, f"center pixel not red: {(r, g, b)}")

    def test_figure_without_bbox_is_left_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            page = tmp / "page.jpg"
            _page_with_red_box(page)
            doc = {"title": "t", "blocks": [{"type": "figure", "description": "d", "image": ""}]}
            crop_figures(doc, page, tmp / "figs", note_id="n1")
            self.assertEqual(doc["blocks"][0]["image"], "")

    def test_missing_page_image_is_a_safe_noop(self):
        with tempfile.TemporaryDirectory() as td:
            doc = {"title": "t", "blocks": [
                {"type": "figure", "description": "d", "image": "", "bbox": [0, 0, 1, 1]}]}
            crop_figures(doc, Path(td) / "nope.jpg", Path(td) / "figs", note_id="n1")
            self.assertEqual(doc["blocks"][0]["image"], "")        # nothing crashed, no image


@unittest.skipUnless(_HAS_PIL, "Pillow not installed")
class EndToEndTests(unittest.TestCase):
    def test_payload_to_cropped_image_to_rendered_output(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            page = tmp / "lecture.jpg"
            _page_with_red_box(page)
            note = Note(note_id="cse138-001", course_id="cse138", image_path=page,
                        processing_order=1, timestamp=datetime(2026, 1, 1))
            # what a VLM extractor would hand to _to_note
            payload = {"title": "Vector Clocks", "blocks": [
                {"type": "paragraph", "text": "A causal-order diagram:"},
                {"type": "figure", "description": "vector-clock lattice",
                 "bbox": [0.5, 0.2, 0.4, 0.4]},
            ], "summary_topic_line": "", "summary_gist": ""}

            extracted = _to_note(note, payload, model_id="test",
                                 figures_dir=tmp / "figures")
            fig = extracted.structured_json["blocks"][1]

            # 1. the figure now carries a real image file
            self.assertEqual(fig["type"], "figure")
            self.assertTrue(Path(fig["image"]).exists())
            self.assertIn("cse138-001", fig["image"])              # namespaced by note id

            # 2. local Markdown renders it as an image reference (not just a caption)
            md = render_note_markdown(extracted.structured_json)
            self.assertIn(f"![vector-clock lattice]({fig['image']})", md)

            # 3. the Docs-embed staging picks the filled slot up as pending work
            _render_doc, pending = _stage_figures(extracted.structured_json)
            self.assertEqual([p[1] for p in pending], [fig["image"]])


if __name__ == "__main__":
    unittest.main()
