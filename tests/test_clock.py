"""Phase A0's one job: prove the face clock and the audio clock are the same number."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holly.runtime import HollyFace

from hollyhost.audio import FakeDevice
from hollyhost.session import SAMPLE_RATE, build_session

AUDIO = Path(__file__).resolve().parents[1] / "holly-8bit-6000" / "test_audio.flac"
BUFFERS = Path(__file__).resolve().parents[1] / "holly-8bit-6000" / "build" / "visemes_pixel" / "visemes.npz"


def backed_face():
    """A HollyFace with a renderer, so frame_at() (the dirty check) is exercisable."""
    from holly.render import Renderer

    return HollyFace(renderer=Renderer(BUFFERS))


def synthetic(seconds: float, f0: float = 130.0) -> np.ndarray:
    t = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    body = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3, 4))
    return (body / np.abs(body).max() * 0.5).astype(np.float32)


class TestClock(unittest.TestCase):
    def setUp(self):
        if not BUFFERS.exists():
            raise unittest.SkipTest(f"host bake missing: {BUFFERS}")
        self.buffers = [synthetic(0.6), synthetic(0.9), synthetic(0.4)]
        self.gap = 0.25
        self.face, self.sources, self.playback = build_session(self.buffers, face=backed_face(), gap=self.gap)

    def test_audio_length_equals_the_face_clock(self):
        """The invariant the whole design rests on. If this drifts, mouth and voice part."""
        self.assertAlmostEqual(self.face.queued_duration, self.playback.size / SAMPLE_RATE, places=6)

    def test_gaps_are_rendered_as_silence_so_the_two_stay_together(self):
        expected = sum(b.size for b in self.buffers) + round(self.gap * SAMPLE_RATE) * 2
        self.assertEqual(self.playback.size, expected)

    def test_sources_match_the_component_queue(self):
        for source, utterance in zip(self.sources, self.face.utterances):
            self.assertAlmostEqual(source.start, utterance.start)
            self.assertAlmostEqual(source.duration, utterance.audio_duration)

    def test_device_position_is_the_only_clock(self):
        device = FakeDevice(self.playback)
        seen = []
        while not device.finished:
            device.advance(512)
            seen.append((device.t, self.face.frame_index_at(device.t)))
        # monotonic, and never samples outside the audio
        self.assertEqual(seen, sorted(seen))
        self.assertLessEqual(max(t for t, _ in seen), self.playback.size / SAMPLE_RATE + 1e-6)

    def test_dirty_check_skips_most_polls(self):
        device = FakeDevice(self.playback)
        draws = 0
        polls = 0
        while not device.finished:
            device.advance(round(SAMPLE_RATE / 60))
            polls += 1
            draws += self.face.frame_at(device.t)[2]
        self.assertLess(draws / polls, 0.5, "the dirty check is not skipping")

    def test_the_mouth_returns_to_rest_and_stops_drawing(self):
        device = FakeDevice(self.playback)
        while not device.finished:
            device.advance(round(SAMPLE_RATE / 60))
            self.face.frame_at(device.t)
        before = self.face.stats["draws"]
        self.face.frame_at(device.t)  # the one restore
        after_restore = self.face.stats["draws"]
        self.assertLessEqual(after_restore - before, 1)
        for _ in range(60):
            device.advance(round(SAMPLE_RATE / 60))
            self.assertFalse(self.face.frame_at(device.t)[2], "idle is drawing")
        self.assertEqual(self.face.viseme_at(device.t), "sil")


@unittest.skipUnless(AUDIO.exists(), f"{AUDIO} missing")
class TestWithRealAudio(unittest.TestCase):
    def test_file_sources_queue_and_play_in_step(self):
        from hollyhost.session import from_file

        buffers = [from_file(AUDIO), from_file(AUDIO)]
        face, sources, playback = build_session(buffers, face=backed_face(), gap=0.3)
        self.assertEqual(len(sources), 2)
        self.assertAlmostEqual(playback.size / SAMPLE_RATE, face.queued_duration, places=6)
        device = FakeDevice(playback)
        while not device.finished:
            device.advance(1024)
            face.frame_at(device.t)
        self.assertGreater(face.stats["draws"], 50)


if __name__ == "__main__":
    unittest.main()
