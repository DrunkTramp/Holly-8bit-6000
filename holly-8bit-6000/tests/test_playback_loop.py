"""End-to-end playback-loop tests: a queue of utterances driven by a fake output clock.

Phase A's host is going to do one thing repeatedly: take the frames the audio device has
consumed, divide by the sample rate, and ask the face for a pose. The queue, the clock
mapping and the idle boundaries are each unit-tested separately; what has not been tested
is their *composition*, which is where off-by-one errors at utterance seams live.

This simulates the device callback (a monotonic frame counter, polled at 60 Hz like the
reference player) and asserts the properties the host is relying on:

    - the pose at time t always belongs to the utterance whose audio is playing at t
    - no frame is sampled twice, none is skipped, none arrives out of order
    - gaps between utterances are idle, and so is everything after the queue drains
    - the face clock and the audio clock never drift apart
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from holly.audio import TARGET_RATE
from holly.runtime import HollyFace, IDLE_ROW
from holly.vise import BASE_VISEME, VISEME_INDEX

BASE = VISEME_INDEX[BASE_VISEME]
POLL_HZ = 60.0


def burst(seconds: float, f0: float, tail_s: float = 0.08) -> np.ndarray:
    """A speech-ish harmonic burst with a short closed tail, deterministic and
    distinguishable per utterance (different pitch and length)."""
    t = np.arange(round(seconds * TARGET_RATE)) / TARGET_RATE
    body = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3, 4))
    body = (body / np.abs(body).max() * 0.5).astype(np.float32)
    tail = np.zeros(round(tail_s * TARGET_RATE), dtype=np.float32)
    return np.concatenate([body, tail])


class FakeOutput:
    """The accounting a sounddevice output callback gives the host.

    `frames_consumed` only ever grows, by exactly what the device took, so `t` is the
    audio output position and never a wall clock.
    """

    def __init__(self, rate: int = TARGET_RATE):
        self.rate = rate
        self.frames_consumed = 0

    @property
    def time(self) -> float:
        return self.frames_consumed / self.rate

    def consume(self, frames: int) -> float:
        self.frames_consumed += frames
        return self.time


class TestPlaybackLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Three utterances, different pitch and length, laid back-to-back with a gap.
        cls.gap = 0.3
        cls.buffers = [burst(0.6, 110.0), burst(1.0, 165.0), burst(0.4, 220.0)]

        cls.face = HollyFace(classifier="openness-ramp", model_path=None)
        clock = 0.0
        cls.queued = []
        for i, samples in enumerate(cls.buffers):
            u = cls.face.speak(samples, TARGET_RATE, at=clock if clock > 0 else None)
            cls.queued.append(u)
            clock = u.end + cls.gap
        cls.total = cls.queued[-1].end

    def poll(self):
        device = FakeOutput()
        frames_per_poll = round(TARGET_RATE / POLL_HZ)
        seen = []
        while device.time < self.total + 1.0:
            t = device.consume(frames_per_poll)
            seen.append((t, self.face.row_at(t), self.face.frame_index_at(t)))
        return seen

    def test_the_clock_and_the_audio_agree(self):
        """No drift: the queue's clock span equals the audio it was built from."""
        audio_total = sum(b.size / TARGET_RATE for b in self.buffers) + self.gap * (len(self.buffers) - 1)
        self.assertAlmostEqual(self.face.queued_duration, audio_total, places=9)

    def test_each_pose_belongs_to_the_utterance_playing_at_that_time(self):
        for t, row, index in self.poll():
            owner = [u for u in self.queued if u.covers(t)]
            if not owner:
                self.assertIs(row, IDLE_ROW, f"t={t:.3f} is outside every utterance but not idle")
                self.assertEqual(index, -1)
                continue
            self.assertEqual(len(owner), 1, f"t={t:.3f} is covered by {len(owner)} utterances")
            u = owner[0]
            self.assertEqual(index, min(int((t - u.start) * u.render_fps), u.frames - 1))
            np.testing.assert_array_equal(row, u.weights[index])

    def test_frames_are_monotonic_within_and_across_utterances(self):
        last = (-1, -1)  # (utterance index, frame index)
        for t, _row, index in self.poll():
            u = self.face.utterance_at(t)
            if u is None:
                continue
            pair = (u.index, index)
            self.assertGreaterEqual(pair, last, f"clock went backwards at t={t:.3f}: {last} -> {pair}")
            last = pair

    def test_the_60hz_poll_reaches_every_reachable_frame(self):
        """At 60 Hz over a 30 fps timeline, no frame the clock can reach is skipped.

        Frames *past* the audio are unreachable by design: `expand_keys` rounds the final
        pose slot up, so a timeline can run past its audio by up to one slot plus one output
        frame (`Utterance.tail`; 87 ms on test_audio.flac). The queue advances by audio
        duration, because that is what the clock is, so those tail frames simply never play --
        which is the right outcome, since the tail is a held final pose and latching it is
        forbidden.
        """
        reachable = {(u.index, k) for u in self.queued for k in range(u.frames)
                     if k / u.render_fps < u.audio_duration}
        hit = set()
        for t, _row, index in self.poll():
            u = self.face.utterance_at(t)
            if u is not None and index >= 0:
                hit.add((u.index, index))
        self.assertEqual(hit - reachable, set(), "the clock sampled a frame past its audio")
        self.assertEqual(reachable - hit, set(),
                         f"reachable frames never sampled: {sorted(reachable - hit)[:5]}")

    def test_the_tail_is_bounded_by_one_slot_plus_one_frame(self):
        """tail <= 1/key_hz + 1/render_fps, not one slot alone.

        Two round-ups stack: `expand_keys` first rounds the key count up to whole slots
        (1/key_hz), then the timeline up to whole output frames (1/render_fps). At 12/s @
        30 fps that is 83.3 + 33.3 = 116.7 ms worst case, and 87 ms on test_audio.flac --
        which is why the bound cannot be one slot.
        """
        bound = 1.0 / self.face.key_hz + 1.0 / self.face.render_fps
        for u in self.queued:
            with self.subTest(utterance=u.index):
                self.assertGreaterEqual(u.tail, 0.0)
                self.assertLessEqual(u.tail, bound + 1e-9)

    def test_gaps_and_the_tail_are_idle(self):
        for i, u in enumerate(self.queued[:-1]):
            mid = u.end + self.gap / 2
            np.testing.assert_array_equal(self.face.row_at(mid), IDLE_ROW, f"gap after {i} not idle")
            self.assertEqual(self.face.viseme_at(mid), BASE_VISEME)
        # Past the end: the no-latch rule, forever.
        for extra in (0.0, 0.5, 5.0, 60.0):
            np.testing.assert_array_equal(self.face.row_at(self.total + extra), IDLE_ROW)

    def test_the_drawn_set_is_a_subset_of_the_queued_rows(self):
        """Whatever the host blits must come from the timelines it was given."""
        allowed = {tuple(np.round(row, 6)) for u in self.queued for row in u.weights}
        allowed.add(tuple(np.round(IDLE_ROW, 6)))
        for _t, row, _i in self.poll():
            self.assertIn(tuple(np.round(row, 6)), allowed)

    def test_pruning_played_utterances_does_not_move_the_clock(self):
        face = HollyFace(classifier="openness-ramp", model_path=None)
        clock = 0.0
        utterances = []
        for samples in self.buffers:
            u = face.speak(samples, TARGET_RATE, at=clock if clock > 0 else None)
            utterances.append(u)
            clock = u.end + self.gap
        mid = utterances[1].start + 0.05
        before = face.row_at(mid)
        face.prune(utterances[0].end)
        self.assertEqual(face.pending, 2)
        np.testing.assert_array_equal(face.row_at(mid), before, "pruning shifted the clock")


if __name__ == "__main__":
    unittest.main()
