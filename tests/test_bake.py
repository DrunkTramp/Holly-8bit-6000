"""Bake-time tests: the manifest must describe the buffers that were actually made.

A scaled bake is a generated asset nothing in the offline pipeline ever loaded, so its
self-consistency was unpinned -- and wrong, at exactly one scale. `resize()` rounds the
product of a dimension, while a rect built from two independently rounded edges can
disagree with it; `Renderer` then fails to broadcast. These tests load every scale the
same way a host would: through `Renderer`.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from holly.render import Renderer
from holly.vise import BASE_VISEME, VISEMES

ROOT = Path(__file__).resolve().parents[1]
PSD = ROOT / "viseme.psd"


def _load_exporter():
    spec = importlib.util.spec_from_file_location("export_visemes", ROOT / "tools" / "export_visemes.py")
    if spec is None or spec.loader is None:
        raise unittest.SkipTest(f"cannot import {ROOT / 'tools' / 'export_visemes.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(PSD.exists(), f"{PSD} missing")
class TestBakeScales(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ev = _load_exporter()
        from psd_tools import PSDImage

        cls.psd = PSDImage.open(PSD)
        cls.layers = cls.ev.read_layers(cls.psd)

    def bake(self, scale: float, resample: str = "bilinear"):
        return self.ev.bake(self.psd, self.layers, scale, 2, resample)

    def test_mouth_rect_matches_the_buffers(self):
        for scale in (1.0, 0.5, 1 / 3, 0.7, 2.0):
            with self.subTest(scale=scale):
                baked = self.bake(scale)
                rect = baked["manifest"]["mouth_rect"]
                canvas = baked["manifest"]["render_canvas"]
                base = baked["base"]
                patches = baked["patches"]

                self.assertEqual((canvas["width"], canvas["height"]), (base.shape[1], base.shape[0]))
                self.assertEqual((rect["w"], rect["h"]), (patches.shape[2], patches.shape[1]),
                                 "manifest mouth rect disagrees with the patch buffers")
                self.assertEqual(base[rect["y"] : rect["y"] + rect["h"],
                                    rect["x"] : rect["x"] + rect["w"]].shape[:2],
                                 (rect["h"], rect["w"]), "mouth rect does not fit the base canvas")
                self.assertLessEqual(rect["x"] + rect["w"], canvas["width"])
                self.assertLessEqual(rect["y"] + rect["h"], canvas["height"])

    def test_renderer_accepts_every_scale(self):
        """The host's actual entry condition: Renderer(npz) must not raise or mis-blend."""
        import json
        import tempfile

        for scale in (1.0, 0.5, 1 / 3):
            with self.subTest(scale=scale):
                baked = self.bake(scale, "nearest")
                with tempfile.TemporaryDirectory() as tmp:
                    np.savez_compressed(Path(tmp) / "visemes.npz",
                                        base=baked["base"], patches=baked["patches"])
                    Path(tmp, "manifest.json").write_text(json.dumps(baked["manifest"]))
                    renderer = Renderer(Path(tmp) / "visemes.npz")
                    row = np.zeros(len(VISEMES), dtype=np.float64)
                    row[VISEMES.index(BASE_VISEME)] = 0.5
                    row[VISEMES.index("aa")] = 0.5
                    frame = renderer.draw_row(row)
                    self.assertEqual(frame.shape, (renderer.size[1], renderer.size[0], 3))
                    self.assertEqual(frame.dtype, np.uint8)

    def test_resample_is_recorded_and_nearest_adds_no_new_values(self):
        baked = self.bake(0.5, "nearest")
        self.assertEqual(baked["manifest"]["resample"], "nearest")
        source = np.array(self.psd.composite().convert("RGBA"), dtype=np.uint8)

        def values(image):
            return {int(v) for v in np.unique(image[..., :3])}

        # Nearest copies pixels, so it cannot invent an intensity the source does not
        # already have. (Bilinear happens to stay inside the source's value set on this
        # asset too -- it is pixel art with a coarse palette -- so the interesting claim
        # is simply that the two filters are not the same operation.)
        self.assertTrue(values(baked["base"]) <= values(source))
        blended = self.bake(0.5, "bilinear")["base"]
        self.assertEqual(self.bake(0.5, "bilinear")["manifest"]["resample"], "bilinear")
        self.assertFalse(np.array_equal(blended, baked["base"]),
                         "--resample nearest and --resample bilinear produced identical buffers")

    def test_native_bake_records_no_resample(self):
        self.assertEqual(self.bake(1.0)["manifest"]["resample"], "none")

    def test_unknown_resample_is_rejected(self):
        with self.assertRaises(ValueError):
            self.ev.resize_premultiplied(np.zeros((4, 4, 4), dtype=np.float32), 0.5, "catmull")


if __name__ == "__main__":
    unittest.main()
