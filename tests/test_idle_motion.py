"""Phase E idle micro-motion: a throttled breath overlay that never lies to the dirty check.

The claims under test are the ones the design rests on:
* at most one draw per `step` seconds (the plan's ~1-2 draws/s ceiling), and no duplicate
  draws -- the renderer has no dirty check, so the overlay keeps its own;
* every drawn pose is `sil` plus one low-weight viseme, and the overlay always restores
  the pure idle pose the moment speech is queued -- so the renderer's persistent frame
  matches what the component's dirty check believes it shows;
* the cycle restarts after a turn without a wasted draw.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holly.runtime import HollyFace
from holly.vise import BASE_VISEME, VISEME_INDEX

from hollyhost.idle_motion import IdleMotion

BUFFERS = Path(__file__).resolve().parents[1] / "holly-8bit-6000" / "build" / "visemes_pixel" / "visemes.npz"


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def overlay_face():
    """A renderer-backed face without the classifier model: the overlay only draws rows."""
    from holly.render import Renderer

    return HollyFace(classifier="openness-ramp", model_path=None, renderer=Renderer(BUFFERS))


def motion(**kwargs):
    clock = FakeClock()
    face = overlay_face()
    rows: list[np.ndarray] = []
    original = face.renderer.draw_row

    def spy(row):
        rows.append(np.array(row, copy=True))
        return original(row)

    face.renderer.draw_row = spy
    kwargs.setdefault("period", 4.0)
    kwargs.setdefault("step", 0.5)
    kwargs.setdefault("max_weight", 0.12)
    return IdleMotion(face, clock=clock, **kwargs), clock, rows


class TestIdleMotion(unittest.TestCase):
    def setUp(self):
        if not BUFFERS.exists():
            raise unittest.SkipTest(f"host bake missing: {BUFFERS}")

    def test_breathing_happens_but_stays_under_the_throttle(self):
        motion_, clock, _rows = motion()
        seconds, poll = 12.0, 1.0 / 60.0
        elapsed = 0.0
        while elapsed < seconds:
            clock.t += poll
            motion_.poll(active=False)
            elapsed += poll
        # At most one draw per step (0.5 s) -- plus slack for the very first/last window.
        self.assertLessEqual(motion_.draws, seconds / 0.5 + 2, "the overlay exceeds 2 draws/s")
        self.assertGreater(motion_.draws, 0, "the overlay never breathed")

    def test_every_pose_is_subtle_and_blended_against_sil(self):
        motion_, clock, rows = motion(max_weight=0.12)
        for _ in range(int(4.0 / (1.0 / 60.0))):  # one whole cycle
            clock.t += 1.0 / 60.0
            motion_.poll(active=False)
        self.assertTrue(rows)
        base = VISEME_INDEX[BASE_VISEME]
        for row in rows:
            extras = [(i, w) for i, w in enumerate(row) if i != base and w > 0.0]
            self.assertLessEqual(len(extras), 1, "the overlay must only ever use one viseme")
            for _i, w in extras:
                self.assertLessEqual(w, 0.12 + 1e-9, "the breath outstayed its subtlety")
            self.assertGreaterEqual(row[base], 0.0)
            self.assertAlmostEqual(float(row.sum()), 1.0, places=6)

    def test_speech_restores_the_idle_pose_then_the_overlay_goes_quiet(self):
        motion_, clock, rows = motion()
        # Breathe until mid-cycle (a non-idle pose is on screen).
        deadline = 4.0
        while not motion_._active and clock.t < deadline:
            clock.t += 1.0 / 60.0
            motion_.poll(active=False)
        self.assertTrue(motion_._active, "the overlay never drew a breath pose")

        self.assertTrue(motion_.poll(active=True), "the restore must count as a draw")
        last = rows[-1]
        self.assertEqual(int(np.argmax(last)), VISEME_INDEX[BASE_VISEME])
        self.assertAlmostEqual(float(last[VISEME_INDEX[BASE_VISEME]]), 1.0,
                               msg="the restore did not draw the pure idle row")
        self.assertFalse(motion_._active)
        draws_after = motion_.draws
        for _ in range(10):
            clock.t += 1.0 / 60.0
            self.assertFalse(motion_.poll(active=True), "the overlay breathed during speech")
        self.assertEqual(motion_.draws, draws_after)

    def test_no_duplicate_draws_within_a_step(self):
        motion_, clock, _rows = motion()
        clock.t = 1.25  # mid-step, phase gives a non-zero weight
        first = motion_.poll(active=False)
        second = motion_.poll(active=False)
        self.assertTrue(first)
        self.assertFalse(second, "the overlay drew twice inside one step")

    def test_the_cycle_restarts_after_speech_without_a_wasted_draw(self):
        motion_, clock, _rows = motion()
        clock.t = 1.25
        motion_.poll(active=False)          # a breath pose
        motion_.poll(active=True)           # speech: restore
        before = motion_.draws
        self.assertFalse(motion_.poll(active=False), "phase 0 of the new cycle drew idle again")
        self.assertEqual(motion_.draws, before)

    def test_constructor_rejects_gaudy_or_unknown_settings(self):
        if not BUFFERS.exists():
            self.skipTest("needs the bake")
        face = overlay_face()
        with self.assertRaises(ValueError):
            IdleMotion(face, viseme="xx")
        with self.assertRaises(ValueError):
            IdleMotion(face, viseme="sil")
        with self.assertRaises(ValueError):
            IdleMotion(face, max_weight=0.9)
        with self.assertRaises(ValueError):
            IdleMotion(face, period=0.6, step=0.5)


if __name__ == "__main__":
    unittest.main()
