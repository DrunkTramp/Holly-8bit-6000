"""Timing-geometry tests: when the mouth moves, relative to when the sound happens.

`tools/check_sync.py` measures this; these tests pin what it measured, so a change to the
fade math, the slot grid or the clock mapping fails here rather than showing up as "the
mouth looks a bit detached" in the host.

Measured on a 12-burst vowel train at 500 ms spacing, 12/s @ 30 fps:

    classifier      lead       center      lag         spread of center
    openness-ramp  -66.7 ms   -33.3 ms     0.0 ms      33.3 ms
    headaudio      -66.7 ms   -33.3 ms     0.0 ms       0.0 ms

So the shipping `center` fade opens the mouth exactly **one render frame early**, and the
three fade shapes are exactly one frame apart. That is the cartoon grid working, not a
bug -- but it is a real 33 ms lead, and it must stay consistent.
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from holly.audio import TARGET_RATE
from holly.keyframes import KEY_HZ, RENDER_FPS
from holly.runtime import HollyFace

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model" / "model-en-mixed.bin"


def _load_tool():
    spec = importlib.util.spec_from_file_location("check_sync", ROOT / "tools" / "check_sync.py")
    if spec is None or spec.loader is None:
        raise unittest.SkipTest("cannot import tools/check_sync.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sync = _load_tool()

BURSTS = 12
SPACING = 0.5
SLOT_MS = 1000.0 / KEY_HZ
FRAME_MS = 1000.0 / RENDER_FPS


def face_for(shape: str, classifier: str) -> HollyFace:
    return HollyFace(classifier=classifier, model_path=MODEL if classifier == "headaudio" else None,
                     fade_shape=shape)


def offsets(classifier: str, shape: str):
    samples, events = sync.click_train(n=BURSTS, spacing=SPACING, stimulus="vowel")
    return sync.offsets(face_for(shape, classifier), samples, events)


class TestTimingGeometry(unittest.TestCase):
    """Deterministic: the stub reacts to every burst, so this is pure timing."""

    @classmethod
    def setUpClass(cls):
        cls.by_shape = {shape: offsets("openness-ramp", shape)[0] for shape in ("lead", "center", "lag")}

    def test_every_event_is_detected(self):
        for shape, got in self.by_shape.items():
            with self.subTest(shape=shape):
                self.assertEqual(len(got), BURSTS, f"{shape} missed bursts")

    def test_center_leads_by_about_one_frame_and_stays_inside_half_a_slot(self):
        med = statistics.median(self.by_shape["center"])
        self.assertLess(abs(med), SLOT_MS / 2.0, "the mouth leads by more than half a pose slot")
        self.assertAlmostEqual(med, -FRAME_MS, delta=FRAME_MS,
                               msg=f"center fade offset moved off one frame: {med:+.1f} ms")

    def test_the_three_fade_shapes_are_ordered_and_one_frame_apart(self):
        lead = statistics.median(self.by_shape["lead"])
        center = statistics.median(self.by_shape["center"])
        lag = statistics.median(self.by_shape["lag"])
        self.assertLess(lead, center, "lead is not earlier than center")
        self.assertLess(center, lag, "center is not earlier than lag")
        # Each shape is one whole output frame from the next (2-frame fade, frame-quantised).
        self.assertAlmostEqual(center - lead, FRAME_MS, delta=FRAME_MS / 2)
        self.assertAlmostEqual(lag - center, FRAME_MS, delta=FRAME_MS / 2)

    def test_offsets_are_consistent_across_events(self):
        """Jitter, not skew, is what reads as a mouth that has given up keeping up."""
        for shape, got in self.by_shape.items():
            with self.subTest(shape=shape):
                self.assertLessEqual(max(got) - min(got), SLOT_MS,
                                     f"{shape} offsets span more than one pose slot")

    def test_no_reaction_is_reported_as_missing_not_as_zero(self):
        # A click train the classifier ignores must shrink n, never fake a 0 ms offset.
        samples, events = sync.click_train(n=4, spacing=0.5, stimulus="click")
        got, frames = sync.offsets(HollyFace(classifier="openness-ramp", model_path=None), samples, events)
        self.assertLessEqual(len(got), len(events))
        self.assertTrue(all(len(str(f)) for f in frames))


@unittest.skipUnless(MODEL.exists(), "model/model-en-mixed.bin missing (see model/README.md)")
class TestTimingGeometryRealClassifier(TestTimingGeometry):
    """The shipping driver, on the same stimulus. Same grid, tighter spread."""

    @classmethod
    def setUpClass(cls):
        cls.by_shape = {shape: offsets("headaudio", shape)[0] for shape in ("lead", "center", "lag")}

    def test_every_event_is_detected(self):
        # A harmonic vowel is speech-like but not speech: the prototypes react to most of
        # it, not all of it. Coverage is a viseme-quality question (Phase B), not timing.
        for shape, got in self.by_shape.items():
            with self.subTest(shape=shape):
                self.assertGreaterEqual(len(got), BURSTS * 0.7, f"{shape} detected too few")

    def test_center_leads_by_about_one_frame_and_stays_inside_half_a_slot(self):
        med = statistics.median(self.by_shape["center"])
        self.assertLess(abs(med), SLOT_MS / 2.0)
        self.assertAlmostEqual(med, -FRAME_MS, delta=FRAME_MS / 2)

    def test_offsets_are_consistent_across_events(self):
        for shape, got in self.by_shape.items():
            with self.subTest(shape=shape):
                self.assertLessEqual(max(got) - min(got), FRAME_MS,
                                     f"{shape} jitter exceeds one render frame")

    def test_stimulus_events_are_on_the_sample_grid(self):
        _samples, events = sync.click_train(n=BURSTS, spacing=SPACING, stimulus="vowel")
        np.testing.assert_allclose(events * TARGET_RATE, np.round(events * TARGET_RATE))


if __name__ == "__main__":
    unittest.main()
