"""Phase E barge-in: cut playback, drop the queue, back to idle -- without breaking the clock.

The invariant survives the cut: `LiveAudio.flush()` truncates at the *consumed* position,
so the clock freezes where the device is and the next `append` pins to that same position
(the rule `LiveSession.say` is built on). The face half is the component's own
`HollyFace.reset()` -- queue cleared, dirty check cleared, so the rest pose redraws once
and idle goes back to zero-draw. The pump half abandons the turn without blocking the UI
thread, and nothing from a cut turn may land on the truncated clock.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holly.runtime import HollyFace

from hollyhost.audio import LiveAudio
from hollyhost.brain import SentencePump
from hollyhost.session import FACE_RATE, SAMPLE_RATE, LiveSession
from hollyhost.tts import NullTTS

BUFFERS = Path(__file__).resolve().parents[1].parent / "Holly" / "build" / "visemes_pixel" / "visemes.npz"


def backed_face():
    from holly.render import Renderer

    return HollyFace(renderer=Renderer(BUFFERS))


class TestLiveAudioFlush(unittest.TestCase):
    def test_flush_freezes_the_clock_at_the_cut(self):
        audio = LiveAudio()
        audio.append(np.ones(SAMPLE_RATE, dtype=np.float32))
        audio.append(np.ones(SAMPLE_RATE, dtype=np.float32))
        audio.advance(round(0.4 * SAMPLE_RATE))
        t = audio.flush()
        self.assertAlmostEqual(t, 0.4, places=6)
        self.assertTrue(audio.idle(), "flush left unplayed audio behind")
        self.assertAlmostEqual(audio.queued_seconds, t)
        block, n = audio.emit(512)
        self.assertIsNone(block)
        self.assertEqual(n, 0, "the callback must emit nothing real after a cut")
        self.assertAlmostEqual(audio.t, t, places=9, msg="the clock ran past the cut")

    def test_flush_is_a_noop_at_the_tail(self):
        audio = LiveAudio()
        audio.append(np.ones(SAMPLE_RATE // 2, dtype=np.float32))
        audio.advance(SAMPLE_RATE)  # consumed everything
        t = audio.flush()
        self.assertAlmostEqual(t, audio.t)
        self.assertAlmostEqual(audio.t, 0.5, places=6)

    def test_next_append_pins_to_the_frozen_clock(self):
        audio = LiveAudio()
        audio.append(np.ones(SAMPLE_RATE, dtype=np.float32))
        audio.advance(round(0.4 * SAMPLE_RATE))
        t = audio.flush()
        start = audio.append(np.ones(SAMPLE_RATE // 2, dtype=np.float32))
        self.assertAlmostEqual(start, t, places=9,
                               msg="post-cut audio was not pinned to the frozen clock")
        start = audio.append(np.ones(SAMPLE_RATE // 2, dtype=np.float32), gap=0.25)
        self.assertAlmostEqual(start, t + 0.5 + 0.25, places=9)


class TestSessionInterrupt(unittest.TestCase):
    def setUp(self):
        if not BUFFERS.exists():
            raise unittest.SkipTest(f"host bake missing: {BUFFERS}")
        self.face = backed_face()
        self.audio = LiveAudio()
        self.session = LiveSession(self.face, self.audio, gap=0.25, tts=NullTTS())

    def test_interrupt_returns_the_mouth_to_rest_and_redraws_once(self):
        self.session.say("Hello. This is a longer sentence to chew on before cutting off.")
        self.session.say("A second sentence queued behind the first.")
        self.audio.advance(round(0.3 * self.audio.rate))
        self.assertIsNotNone(self.face.utterance_at(self.audio.t), "not mid-speech")
        queued_before = self.face.pending
        self.assertGreater(queued_before, 0)

        t = self.session.interrupt()
        self.assertAlmostEqual(t, self.audio.t, places=9)
        self.assertEqual(self.face.pending, 0, "the face queue survived the cut")
        self.assertTrue(self.audio.idle())

        _frame, _rect, changed = self.session.poll()
        self.assertTrue(changed, "the rest pose must be redrawn once after a cut")
        self.assertEqual(self.face.viseme_at(self.audio.t), "sil")
        _frame, _rect, changed = self.session.poll()
        self.assertFalse(changed, "idle must be back to zero-draw")

    def test_speech_after_a_cut_is_still_pinned_to_the_audio(self):
        self.session.say("Cut me off before you finish.")
        self.audio.advance(round(0.2 * self.audio.rate))
        self.session.interrupt()

        self.session.say("And here we speak again, in sync.")
        utterance = self.face.utterances[-1]
        # places=3: the analysis (22.05k NullTTS) and the resampled playback buffer agree to
        # sub-sample duration rounding -- the settled two-rate tolerance.
        self.assertAlmostEqual(utterance.start, self.audio.queued_seconds - utterance.audio_duration,
                               places=3, msg="post-cut face `at` drifted off the audio position")
        # The two clocks still agree at the end of everything.
        self.assertAlmostEqual(self.face.queued_duration, self.audio.queued_seconds, places=3)
        while not self.audio.idle():
            self.audio.advance(round(0.05 * self.audio.rate))
        self.session.poll()
        self.assertEqual(self.face.viseme_at(self.audio.t), "sil")


class _SlowBrain:
    """A brain that drips words with a pause, so a turn is still running when the cut lands."""

    name = "slow"

    def __init__(self, words, *, delay: float = 0.05):
        self.words = list(words)
        self.delay = delay

    def reply_stream(self, messages):
        for word in self.words:
            time.sleep(self.delay)
            yield word + " "


class _SlowTTS:
    """Synthesis slow enough that one sentence is in flight across the cut."""

    name = "slowtest"
    rate = FACE_RATE

    def __init__(self, *, delay: float = 0.25):
        self.delay = delay
        self.calls = 0

    def synthesize(self, text):
        self.calls += 1
        time.sleep(self.delay)
        return np.zeros(FACE_RATE // 5, dtype=np.float32), FACE_RATE  # 0.2 s of quiet


class TestPumpInterrupt(unittest.TestCase):
    def test_interrupt_abandons_the_turn_and_drains_the_queue(self):
        pump = SentencePump(_SlowBrain(["One.", "Two.", "Three.", "Four."]), _SlowTTS(delay=0.1))
        self.assertTrue(pump.submit("hello"))
        deadline = time.monotonic() + 3.0
        while not pump.sentences() and time.monotonic() < deadline:
            time.sleep(0.01)
        pump.interrupt()
        self.assertTrue(pump.interrupted)
        time.sleep(0.5)  # outlives every remaining delta and any in-flight synthesis
        self.assertEqual(pump.sentences(), [], "sentences arrived after the barge-in")
        self.assertEqual(pump.pending, 0)
        self.assertFalse(pump.busy, "the worker did not exit after the interrupt")

    def test_a_sentence_synthesising_during_the_cut_never_lands(self):
        tts = _SlowTTS(delay=0.3)
        pump = SentencePump(_SlowBrain(["One.", "Two.", "Three."], delay=0.01), tts)
        pump.submit("hello")
        deadline = time.monotonic() + 3.0
        while not pump.sentences() and time.monotonic() < deadline:
            time.sleep(0.01)  # first sentence drained; the worker is inside synthesis again
        pump.interrupt()
        time.sleep(0.6)  # the in-flight sentence has long since finished -- and must be dropped
        self.assertEqual(pump.sentences(), [], "an in-flight sentence landed after the cut")
        self.assertFalse(pump.busy)

    def test_a_new_turn_after_a_barge_in_works(self):
        pump = SentencePump(_SlowBrain(["One.", "Two."], delay=0.02), _SlowTTS(delay=0.05))
        pump.submit("first")
        time.sleep(0.15)
        pump.interrupt()
        deadline = time.monotonic() + 2.0
        while pump.busy and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(pump.interrupted is True and pump.busy)
        self.assertTrue(pump.submit("second"), "submit refused after the worker exited")
        self.assertFalse(pump.interrupted, "the cancel outlived submit()")
        deadline = time.monotonic() + 3.0
        got = []
        while time.monotonic() < deadline:
            got.extend(pump.sentences())
            if not pump.busy and pump.pending == 0 and len(got) >= 2:
                break
            time.sleep(0.01)
        pump.stop()
        self.assertGreaterEqual(len(got), 1, "the post-barge turn produced nothing")


if __name__ == "__main__":
    unittest.main()
