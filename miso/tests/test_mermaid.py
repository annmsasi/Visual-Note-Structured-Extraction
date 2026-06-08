"""Tests for the figure → Mermaid pass (miso/mermaid.py).

The model call is faked with a plain callable, so these run with no API key and no
network. `mmdc` is not assumed present — the rendering tests assert the graceful
degradation path (Mermaid source kept, no PNG).
"""
import shutil
import tempfile
import unittest
from pathlib import Path

from miso.mermaid import _clean, add_mermaid, render_mermaid

_HAS_MMDC = shutil.which("mmdc") is not None


def _figure_doc(*descriptions):
    return {"title": "t", "summary_topic_line": "", "summary_gist": "", "blocks": [
        {"type": "figure", "description": d, "mermaid": "", "image": ""} for d in descriptions
    ]}


class CleanTests(unittest.TestCase):
    def test_strips_mermaid_fence(self):
        self.assertEqual(_clean("```mermaid\nflowchart TD\n A-->B\n```"), "flowchart TD\n A-->B")

    def test_strips_bare_fence(self):
        self.assertEqual(_clean("```\nflowchart TD\n```"), "flowchart TD")

    def test_no_mermaid_sentinel_becomes_empty(self):
        self.assertEqual(_clean("NO_MERMAID"), "")
        self.assertEqual(_clean("  no_mermaid  "), "")

    def test_empty_reply_is_empty(self):
        self.assertEqual(_clean(""), "")


class AddMermaidTests(unittest.TestCase):
    def test_fills_mermaid_source_without_rendering(self):
        doc = _figure_doc("A flowchart")
        calls = []

        def fake(image_path, prompt):
            calls.append(prompt)
            return "```mermaid\nflowchart TD\n A-->B\n```"

        add_mermaid(doc, "page.png", fake, out_dir=None, note_id="n1")
        self.assertEqual(doc["blocks"][0]["mermaid"], "flowchart TD\n A-->B")
        self.assertEqual(doc["blocks"][0]["image"], "")          # out_dir=None → never rendered
        self.assertEqual(len(calls), 1)
        self.assertIn("A flowchart", calls[0])                   # caption fed to the pass

    def test_skips_figure_without_description(self):
        doc = {"title": "t", "summary_topic_line": "", "summary_gist": "", "blocks": [
            {"type": "figure", "description": "", "mermaid": "", "image": ""}]}
        add_mermaid(doc, "page.png", lambda i, p: "flowchart TD", out_dir=None)
        self.assertEqual(doc["blocks"][0]["mermaid"], "")

    def test_no_mermaid_reply_leaves_caption_only(self):
        doc = _figure_doc("A messy photo")
        add_mermaid(doc, "page.png", lambda i, p: "NO_MERMAID", out_dir=None)
        self.assertEqual(doc["blocks"][0]["mermaid"], "")
        self.assertEqual(doc["blocks"][0]["image"], "")

    def test_model_error_does_not_break_the_note(self):
        doc = _figure_doc("A flowchart")

        def boom(image_path, prompt):
            raise RuntimeError("api down")

        add_mermaid(doc, "page.png", boom, out_dir=None)          # must not raise
        self.assertEqual(doc["blocks"][0]["mermaid"], "")

    def test_writes_mmd_source_when_mmdc_absent(self):
        if _HAS_MMDC:
            self.skipTest("mmdc installed; this asserts the no-CLI degradation path")
        doc = _figure_doc("A flowchart")
        with tempfile.TemporaryDirectory() as td:
            add_mermaid(doc, "page.png", lambda i, p: "flowchart TD\n A-->B",
                        out_dir=Path(td), note_id="n1")
            self.assertEqual(doc["blocks"][0]["image"], "")       # no PNG without the CLI
            self.assertTrue((Path(td) / "n1" / "figure_0.mmd").exists())


class RenderMermaidTests(unittest.TestCase):
    def test_missing_cli_is_a_clean_false(self):
        if _HAS_MMDC:
            self.skipTest("mmdc installed")
        with tempfile.TemporaryDirectory() as td:
            ok, err = render_mermaid("flowchart TD\n A-->B", Path(td) / "f.png")
            self.assertFalse(ok)
            self.assertIn("mmdc", err)
            self.assertTrue((Path(td) / "f.mmd").exists())        # source written for inspection

    @unittest.skipUnless(_HAS_MMDC, "mmdc not installed")
    def test_renders_png_when_cli_present(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "f.png"
            ok, err = render_mermaid("flowchart TD\n A-->B", out)
            self.assertTrue(ok, err)
            self.assertTrue(out.exists() and out.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
