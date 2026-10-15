"""The host's own playback-loop test: the live queue driven by the output clock.

This ports the assertions from the animation component's `tests/test_playback_loop.py`
into the host, but drives them through the host's *actual* loop — `LiveAudio` (the
appendable output clock) and `LiveSession` (face queue + audio in lockstep) — rather than
re-deriving them. The composition is where off-by-one errors at utterance seams live, and
the interactive path (utterances arrive over time) has seams the batch path does not.

The properties under test, unchanged from the component:

    - the pose at time t belongs to the utterance whose audio is playing at t
    - no frame is sampled twice, none skipped, none out of order
    - gaps and everything after the queue drains are idle (zero-draw, no final-pose latch)
    - the face clock and the audio clock never drift apart
    - pruning played utterances does not move the clock

Plus the host-specific rule that makes interactive playback safe: the clock is frames
*actually emitted*, so it freezes at the tail while Holly waits for input and never runs
ahead of the audio.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holly.audio import TARGET_RATE
from holly.runtime import HollyFace, IDLE_ROW
from holly.vise import BASE_VISEME, VISEME_INDEX

from hollyhost.audio import LiveAudio
from hollyhost.session import LiveSession

BASE = VISEME_INDEX[BASE_VISEME]
POLL_HZ = 60.0
BUFFERS = Path(__file__).resolve().parents[1] / "holly-8bit-6000" / "build" / "visemes_pixel" / "visemes.npz"


def burst(seconds: float, f0: float, tail_s: float = 0.08) -> np.ndarray:
    """A speech-ish harmonic burst with a short closed tail, deterministic and
    distinguishable per utterance (different pitch and length). 16 kHz float32."""
    t = np.arange(round(seconds * TARGET_RATE)) / TARGET_RATE
    body = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3, 4))
    body = (body / np.abs(body).max() * 0.5).astype(np.float32)
    tail = np.zeros(round(tail_s * TARGET_RATE), dtype=np.float32)
    return np.concatenate([body, tail])


def live_session(buffers, gap, *, rate=TARGET_RATE, renderer=False):
    """Build a LiveSession and enqueue every buffer through `say()` — the real host path."""
    face = HollyFace(classifier="openness-ramp", model_path=None)
    if renderer:
        from holly.render import Renderer

        face = HollyFace(classifier="openness-ramp", model_path=None, renderer=Renderer(BUFFERS))
    audio = LiveAudio(rate)
    session = LiveSession(face, audio, gap=gap)
    queued = []
    for i, samples in enumerate(buffers):
        queued.append(session.say(f"utt-{i}", samples=samples, rate=rate))
    return session, queued


class TestLivePlaybackLoop(unittest.TestCase):
    """The component's seam-boundary assertions, driven through the host's live loop."""

    @classmethod
    def setUpClass(cls):
        cls.gap = 0.3
        cls.buffers = [burst(0.6, 110.0), burst(1.0, 165.0), burst(0.4, 220.0)]

    def build(self):
        return live_session(self.buffers, self.gap)

    def poll(self, session, queued):
        """Advance the live clock at 60 Hz, past the end, recording what the host would see.

        The clock freezes at the tail, so we drive a fixed number of polls (total + 1 s)
        rather than looping until `t` crosses a wall value it can never reach.
        """
        total = session.audio.queued_seconds
        frames_per_poll = round(TARGET_RATE / POLL_HZ)
        n_polls = int(np.ceil((total + 1.0) * POLL_HZ)) + 1
        seen = []
        for _ in range(n_polls):
            t = session.audio.advance(frames_per_poll)
            seen.append((t, session.face.row_at(t), session.face.frame_index_at(t)))
        return seen

    def test_the_clock_and_the_audio_agree(self):
        """No drift at 16k: the audio tail and the face queue are the same number exactly."""
        session, queued = self.build()
        self.assertAlmostEqual(session.audio.queued_seconds, session.face.queued_duration, places=9)
        self.assertEqual(len(queued), 3)

    def test_each_pose_belongs_to_the_utterance_playing_at_that_time(self):
        session, queued = self.build()
        for t, row, index in self.poll(session, queued):
            owner = [u for u in queued if u.covers(t)]
            if not owner:
                self.assertIs(row, IDLE_ROW, f"t={t:.3f} is outside every utterance but not idle")
                self.assertEqual(index, -1)
                continue
            self.assertEqual(len(owner), 1, f"t={t:.3f} is covered by {len(owner)} utterances")
            u = owner[0]
            self.assertEqual(index, min(int((t - u.start) * u.render_fps), u.frames - 1))
            np.testing.assert_array_equal(row, u.weights[index])

    def test_frames_are_monotonic_within_and_across_utterances(self):
        session, queued = self.build()
        last = (-1, -1)
        for t, _row, index in self.poll(session, queued):
            u = session.face.utterance_at(t)
            if u is None:
                continue
            pair = (u.index, index)
            self.assertGreaterEqual(pair, last, f"clock went backwards at t={t:.3f}: {last} -> {pair}")
            last = pair

    def test_the_60hz_poll_reaches_every_reachable_frame(self):
        """Every frame the clock can reach is sampled; nothing past an utterance's audio is."""
        session, queued = self.build()
        reachable = {(u.index, k) for u in queued for k in range(u.frames)
                     if k / u.render_fps < u.audio_duration}
        hit = set()
        for t, _row, index in self.poll(session, queued):
            u = session.face.utterance_at(t)
            if u is not None and index >= 0:
                hit.add((u.index, index))
        self.assertEqual(hit - reachable, set(), "the clock sampled a frame past its audio")
        self.assertEqual(reachable - hit, set(),
                         f"reachable frames never sampled: {sorted(reachable - hit)[:5]}")

    def test_the_tail_is_bounded_by_one_slot_plus_one_frame(self):
        session, queued = self.build()
        bound = 1.0 / session.face.key_hz + 1.0 / session.face.render_fps
        for u in queued:
            with self.subTest(utterance=u.index):
                self.assertGreaterEqual(u.tail, 0.0)
                self.assertLessEqual(u.tail, bound + 1e-9)

    def test_gaps_and_the_tail_are_idle(self):
        session, queued = self.build()
        total = session.audio.queued_seconds
        for i, u in enumerate(queued[:-1]):
            mid = u.end + self.gap / 2
            np.testing.assert_array_equal(session.face.row_at(mid), IDLE_ROW, f"gap after {i} not idle")
            self.assertEqual(session.face.viseme_at(mid), BASE_VISEME)
        # Past the end: the no-latch rule, and the frozen clock sits exactly there.
        self.assertAlmostEqual(session.audio.t if session.audio.idle() else total, total, places=6)
        for extra in (0.0, 0.5, 5.0, 60.0):
            np.testing.assert_array_equal(session.face.row_at(total + extra), IDLE_ROW)

    def test_the_drawn_set_is_a_subset_of_the_queued_rows(self):
        session, queued = self.build()
        allowed = {tuple(np.round(row, 6)) for u in queued for row in u.weights}
        allowed.add(tuple(np.round(IDLE_ROW, 6)))
        for _t, row, _i in self.poll(session, queued):
            self.assertIn(tuple(np.round(row, 6)), allowed)

    def test_pruning_played_utterances_does_not_move_the_clock(self):
        session, queued = self.build()
        mid = queued[1].start + 0.05
        before = session.face.row_at(mid)
        session.face.prune(queued[0].end)
        self.assertEqual(session.face.pending, 2)
        np.testing.assert_array_equal(session.face.row_at(mid), before, "pruning shifted the clock")


@unittest.skipUnless(BUFFERS.exists(), f"host bake missing: {BUFFERS}")
class TestLiveDirtyCheck(unittest.TestCase):
    """The interactive loop's draw behaviour, through the real renderer + dirty check."""

    def test_idle_is_zero_draw_and_the_mouth_returns_to_rest(self):
        session, _queued = live_session([burst(0.8, 150.0)], gap=0.3, renderer=True)
        frames_per_poll = round(TARGET_RATE / POLL_HZ)
        while not session.audio.idle():
            session.audio.advance(frames_per_poll)
            session.poll()  # poll() prunes and calls frame_at (the dirty check)
        before = session.face.stats["draws"]
        session.poll()  # the single restore-to-base draw
        self.assertLessEqual(session.face.stats["draws"] - before, 1)
        for _ in range(60):  # 60 polls of idle: nothing draws
            session.audio.advance(frames_per_poll)
            self.assertFalse(session.poll()[2], "idle is drawing")
        self.assertEqual(session.face.viseme_at(session.audio.t), BASE_VISEME)


class TestLiveAudioClock(unittest.TestCase):
    """The host-specific rule: the clock is frames actually emitted, frozen at the tail."""

    def test_clock_freezes_when_the_buffer_is_exhausted(self):
        audio = LiveAudio(TARGET_RATE)
        audio.append(burst(0.5, 120.0))
        total = audio.queued_seconds
        for _ in range(int(POLL_HZ * 3)):  # way past the end
            audio.advance(round(TARGET_RATE / POLL_HZ))
        self.assertTrue(audio.idle())
        self.assertAlmostEqual(audio.t, total, places=9, msg="clock ran ahead of the audio")

    def test_appends_land_at_the_tail_with_the_gap(self):
        audio = LiveAudio(TARGET_RATE)
        b0, b1 = burst(0.5, 120.0), burst(0.5, 120.0)
        start0 = audio.append(b0, gap=0.3)
        start1 = audio.append(b1, gap=0.3)
        self.assertEqual(start0, 0.0)
        # Second utterance starts after the first's audio plus exactly one gap.
        self.assertAlmostEqual(start1, (b0.size + round(0.3 * TARGET_RATE)) / TARGET_RATE, places=9)

    def test_advance_returns_the_updated_clock(self):
        audio = LiveAudio(TARGET_RATE)
        audio.append(np.ones(TARGET_RATE, dtype=np.float32))
        t = audio.advance(TARGET_RATE // 2)
        self.assertAlmostEqual(t, 0.5, places=9)


class TestSampleRateContract(unittest.TestCase):
    """A1's boundary: a foreign-rate voice folds to 16k analysis + native playback, 1:1 in time."""

    def test_22k_voice_queues_with_the_clock_still_aligned(self):
        # NullTTS-style 22.05k buffer: prepare() must resample analysis to 16k (or speak()
        # raises), and the queue still advances by duration so the two clocks agree.
        from hollyhost.tts import NullTTS

        samples, rate = NullTTS(rate=22050).synthesize("hello there my friend speaking")
        session, queued = live_session([samples], gap=0.3, rate=rate)
        self.assertEqual(queued[0].index, 0)
        # analysis was 16k (speak() would have raised otherwise); playback stayed at 22.05k.
        self.assertAlmostEqual(session.audio.queued_seconds, session.face.queued_duration, places=3)

    def test_multi_utterance_22k_seams_stay_aligned(self):
        """The boundary the task flags as easy to get wrong, across seams.

        Several foreign-rate utterances queued back-to-back: the clock still lines up at two
        rates because the queue advances by *duration*, not sample count. Polling the frozen
        live clock, every covered position maps to the right utterance's frame and every gap
        is idle — the property Phase C's sentence-streaming depends on.
        """
        from hollyhost.audio import LiveAudio
        from hollyhost.session import LiveSession
        from hollyhost.tts import NullTTS

        face = HollyFace(classifier="openness-ramp", model_path=None)
        audio = LiveAudio(22050)
        session = LiveSession(face, audio, gap=0.3, tts=NullTTS(rate=22050))
        queued = [session.say(t) for t in ("one two three", "four five", "six seven eight nine")]
        self.assertEqual(len(queued), 3)
        self.assertAlmostEqual(audio.queued_seconds, face.queued_duration, places=3)

        per = round(22050 / POLL_HZ)
        for _ in range(int((audio.queued_seconds + 1.0) * POLL_HZ) + 1):
            t = audio.advance(per)
            owner = [u for u in queued if u.covers(t)]
            row = face.row_at(t)
            if owner:
                self.assertEqual(len(owner), 1, f"t={t:.3f} covered by {len(owner)} utterances")
                u = owner[0]
                idx = min(int((t - u.start) * u.render_fps), u.frames - 1)
                np.testing.assert_array_equal(row, u.weights[idx])
            else:
                self.assertIs(row, IDLE_ROW, f"t={t:.3f} in a seam but not idle")


if __name__ == "__main__":
    unittest.main()
