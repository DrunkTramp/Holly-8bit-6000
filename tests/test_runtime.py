"""Phase 3a runtime tests: the host-facing component API.

These pin the four properties the live model rests on -- clock->frame mapping,
dirty-check, zero-draw idle, no final-pose latch -- plus the sample-rate contract the
host's TTS buffers have to satisfy. Structural tests use a synthetic buffer and the
stub classifier so they do not depend on the vendored model; one test exercises the
shipping headaudio path end to end.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from holly.audio import TARGET_RATE, decode
from holly.keyframes import RENDER_FPS
from holly.render import Renderer
from holly.runtime import (
    DEFAULT_BUFFERS,
    DEFAULT_MODEL,
    HollyFace,
    IDLE_ROW,
    Utterance,
    make_idle_row,
)
from holly.vise import BASE_VISEME, VISEMES, VISEME_INDEX

ROOT = Path(__file__).resolve().parents[1]
BUDDERS = ROOT / "build" / "visemes" / "visemes.npz"
AUDIO = ROOT / "test_audio.flac"
MODEL_EXISTS = (ROOT / "model" / "model-en-mixed.bin").exists()

BASE = VISEME_INDEX[BASE_VISEME]


def speech_like(bursts: int = 4, burst_s: float = 0.3, gap_s: float = 0.2) -> np.ndarray:
    """Loud harmonic bursts separated by near-silence.

    The auto VAD gate derives its thresholds from the buffer's own energy, so a
    bimodal signal is what makes the stub produce both mouth poses and closures.
    """
    out = []
    t = np.arange(0, burst_s, 1.0 / TARGET_RATE)
    for i in range(bursts):
        f = 110.0 * (1.0 + i % 3)
        burst = 0.35 * (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 2.3 * f * t))
        out.append(burst.astype(np.float32))
        out.append((1e-4 * np.sin(2 * np.pi * 400 * np.arange(0, gap_s, 1.0 / TARGET_RATE))).astype(np.float32))
    return np.concatenate(out)


def stub_face(**kwargs) -> HollyFace:
    kwargs.setdefault("classifier", "openness-ramp")
    kwargs.setdefault("model_path", None)
    return HollyFace(**kwargs)


def hand_utterance(face: HollyFace, rows: list[np.ndarray], start: float | None = None) -> Utterance:
    """Append an utterance with an explicit weight matrix -- no analysis involved.

    `start` defaults to the end of the queue, exactly as `speak()` behaves.
    """
    weights = np.asarray(rows, dtype=np.float64)
    if start is None:
        start = face.utterances[-1].end if face.utterances else 0.0
    utterance = Utterance(
        index=face._n,
        start=start,
        audio_duration=weights.shape[0] / face.render_fps,
        weights=weights,
        keys=(),
        render_fps=face.render_fps,
        analysis_ms=0.0,
    )
    face._n += 1
    face.utterances.append(utterance)
    return utterance


def one_hot(viseme: str) -> np.ndarray:
    row = np.zeros(len(VISEMES), dtype=np.float64)
    row[VISEME_INDEX[viseme]] = 1.0
    return row


class TestSampleRateContract(unittest.TestCase):
    def setUp(self):
        self.face = stub_face()

    def test_requires_the_analysis_rate(self):
        with self.assertRaises(ValueError) as ctx:
            self.face.analyse(speech_like(), 24000)
        message = str(ctx.exception)
        self.assertIn("16000", message)
        self.assertIn("Resample", message)

    def test_requires_a_1d_buffer(self):
        with self.assertRaises(ValueError):
            self.face.analyse(np.zeros((1600, 2), dtype=np.float32), TARGET_RATE)

    def test_rejects_an_empty_buffer(self):
        with self.assertRaises(ValueError):
            self.face.analyse(np.zeros(0, dtype=np.float32), TARGET_RATE)

    def test_speak_accepts_the_same_buffer_as_analyse(self):
        samples = speech_like()
        utterance = self.face.speak(samples)
        weights, _keys, _ms, _fps = self.face.analyse(samples)
        np.testing.assert_array_equal(utterance.weights, weights)


class TestAnalysisShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.face = stub_face()
        cls.samples = speech_like()
        cls.utterance = cls.face.speak(cls.samples)

    def test_weight_matrix_is_dense_and_normalised(self):
        weights = self.utterance.weights
        self.assertEqual(weights.ndim, 2)
        self.assertEqual(weights.shape[1], len(VISEMES))
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-9)

    def test_timeline_covers_the_audio(self):
        audio_s = self.samples.size / TARGET_RATE
        self.assertLessEqual(audio_s, self.utterance.timeline_duration + 1e-9)
        # expand_keys rounds the last slot up, so the overshoot is under one pose slot.
        self.assertLess(self.utterance.tail, 1.0 / self.face.key_hz)
        self.assertAlmostEqual(self.utterance.audio_duration, audio_s, places=9)

    def test_the_buffer_actually_moves_the_mouth(self):
        rows = self.utterance.weights
        self.assertTrue(np.any(rows[:, BASE] < 1.0 - 1e-9), "stub produced only closures")

    def test_analysis_is_deterministic(self):
        again = stub_face().speak(self.samples)
        np.testing.assert_array_equal(again.weights, self.utterance.weights)

    def test_pose_slots_are_the_key_grid(self):
        self.face.reset()
        keys = self.face.speak(self.samples).keys
        audio_s = self.samples.size / TARGET_RATE
        self.assertEqual(len(keys), max(1, int(np.ceil(audio_s * self.face.key_hz))))
        np.testing.assert_allclose([k.t for k in keys], [j / self.face.key_hz for j in range(len(keys))])


class TestClockMapping(unittest.TestCase):
    def setUp(self):
        self.face = stub_face()
        self.rows = [one_hot("aa"), one_hot("aa"), one_hot("E"), one_hot("ou"), one_hot("ou")]
        self.utterance = hand_utterance(self.face, self.rows)

    def test_frame_is_audio_time_times_render_fps(self):
        fps = self.face.render_fps
        for k in range(len(self.rows)):
            for frac in (0.0, 0.4, 0.9):
                t = (k + frac) / fps
                self.assertEqual(self.face.frame_index_at(t), k, f"t={t}")

    def test_row_at_returns_the_matching_row(self):
        fps = self.face.render_fps
        for k, expected in enumerate(self.rows):
            np.testing.assert_allclose(self.face.row_at((k + 0.5) / fps), expected)

    def test_negative_time_is_idle(self):
        np.testing.assert_array_equal(self.face.row_at(-0.5), IDLE_ROW)
        self.assertEqual(self.face.viseme_at(-0.5), BASE_VISEME)
        self.assertEqual(self.face.frame_index_at(-0.5), -1)

    def test_viseme_at_is_the_argmax_of_row_at(self):
        fps = self.face.render_fps
        for k in range(len(self.rows)):
            self.assertEqual(
                self.face.viseme_at((k + 0.5) / fps),
                VISEMES[int(np.argmax(self.face.row_at((k + 0.5) / fps)))],
            )

    def test_no_final_pose_latch(self):
        # The last slot is "ou"; past the end the mouth must be back at base.
        np.testing.assert_array_equal(self.face.row_at(self.utterance.end - 1e-6), one_hot("ou"))
        for t in (self.utterance.end, self.utterance.end + 1e-6, self.utterance.end + 3.0):
            np.testing.assert_array_equal(self.face.row_at(t), IDLE_ROW)
        self.assertEqual(self.face.viseme_at(self.utterance.end + 1.0), BASE_VISEME)

    def test_idle_row_is_the_base_pose_alone(self):
        np.testing.assert_array_equal(IDLE_ROW, make_idle_row())
        self.assertEqual(IDLE_ROW[BASE], 1.0)
        self.assertEqual(np.count_nonzero(IDLE_ROW), 1)
        self.assertFalse(IDLE_ROW.flags["WRITEABLE"])


class TestUtteranceQueue(unittest.TestCase):
    def setUp(self):
        self.face = stub_face()

    def test_speaks_land_back_to_back(self):
        a = hand_utterance(self.face, [one_hot("aa")] * 3)
        b = hand_utterance(self.face, [one_hot("E")] * 3)
        self.assertAlmostEqual(b.start, a.end)
        self.assertAlmostEqual(self.face.queued_duration, a.end + b.audio_duration)

    def test_gap_between_utterances_is_idle(self):
        a = hand_utterance(self.face, [one_hot("aa")] * 3)
        gap = 0.5
        b = hand_utterance(self.face, [one_hot("E")] * 3, start=a.end + gap)
        fps = self.face.render_fps
        np.testing.assert_array_equal(self.face.row_at(a.end - 1.0 / fps), one_hot("aa"))
        np.testing.assert_array_equal(self.face.row_at(a.end + gap / 2), IDLE_ROW)
        np.testing.assert_array_equal(self.face.row_at(b.start), one_hot("E"))

    def test_speak_at_places_the_next_utterance(self):
        samples = speech_like(bursts=1, burst_s=0.2, gap_s=0.05)
        first = self.face.speak(samples)
        second = self.face.speak(samples, at=first.end + 1.0)
        self.assertAlmostEqual(second.start, first.end + 1.0)
        self.assertIsNone(self.face.utterance_at(first.end + 0.5))

    def test_overlap_is_rejected(self):
        samples = speech_like(bursts=1, burst_s=0.2, gap_s=0.05)
        first = self.face.speak(samples)
        with self.assertRaises(ValueError):
            self.face.speak(samples, at=first.end - 0.05)
        with self.assertRaises(ValueError):
            self.face.speak(samples, at=-1.0)

    def test_prune_drops_only_finished_utterances(self):
        a = hand_utterance(self.face, [one_hot("aa")] * 3)
        b = hand_utterance(self.face, [one_hot("E")] * 3)
        self.assertEqual(self.face.prune(a.end), 1)
        self.assertEqual(self.face.pending, 1)
        self.assertIs(self.face.utterances[0], b)

    def test_reset_clears_the_queue_and_the_dirty_check(self):
        hand_utterance(self.face, [one_hot("aa")] * 3)
        face = self.face
        face._last_row = one_hot("aa")
        face.reset()
        self.assertEqual(face.pending, 0)
        self.assertEqual(face.queued_duration, 0.0)
        self.assertIsNone(face._last_row)
        np.testing.assert_array_equal(face.row_at(0.0), IDLE_ROW)


@unittest.skipUnless(DEFAULT_BUFFERS.exists(),
                     "host bake missing: tools/export_visemes.py --scale 0.5 --resample nearest "
                     f"--out {DEFAULT_BUFFERS.parent}")
class TestHostBake(unittest.TestCase):
    """The host canvas is settled at 576x432: the face is a corner element, and the mouth
    rect is the only thing a host ever blits, so halving the linear size quarters the draw."""

    def test_default_buffers_are_the_host_bake(self):
        renderer = Renderer(DEFAULT_BUFFERS)
        self.assertEqual(renderer.size, (576, 432))
        _x, _y, w, h = renderer.mouth_rect
        self.assertEqual((w, h), (168, 132))

    def test_manifest_records_a_nearest_half_scale_bake(self):
        manifest = json.loads(DEFAULT_BUFFERS.with_name("manifest.json").read_text())
        self.assertEqual(manifest["resample"], "nearest")
        self.assertEqual(manifest["render_scale"], 0.5)
        self.assertEqual(manifest["alpha"], "premultiplied")

    def test_mouth_rect_is_quarter_the_native_cost(self):
        if not BUDDERS.exists():
            self.skipTest(f"{BUDDERS} missing")
        native = Renderer(BUDDERS)
        host = Renderer(DEFAULT_BUFFERS)
        self.assertEqual(native.size, (1152, 864))
        ratio = (native.mouth_rect[2] * native.mouth_rect[3]) / (host.mouth_rect[2] * host.mouth_rect[3])
        self.assertAlmostEqual(ratio, 4.0, delta=0.3)

    def test_the_runtime_draws_it(self):
        face = HollyFace(renderer=Renderer(DEFAULT_BUFFERS))
        utterance = face.speak(speech_like())
        frame, rect, changed = face.frame_at(0.0)
        self.assertEqual(frame.shape, (432, 576, 3))
        self.assertEqual(frame.dtype, np.uint8)

        # The rect a host blits is the buffer's own, and stable across polls. Find a
        # moment where the mouth is actually open, then poll past the end: that must cost
        # exactly one draw, the one that returns the mouth to rest (no final-pose latch).
        open_at = next(t for t in np.arange(0.0, utterance.audio_duration, 1 / 30.0)
                       if face.viseme_at(t) != BASE_VISEME)
        face.frame_at(open_at)
        past_frame, same_rect, changed = face.frame_at(utterance.end)
        self.assertEqual(same_rect, rect)
        self.assertTrue(changed, "returning to rest did not redraw the mouth rect")
        self.assertEqual(face.viseme_at(utterance.end), BASE_VISEME)
        self.assertFalse(face.frame_at(utterance.end + 0.1)[2], "idle polls are redrawing")
        self.assertEqual(past_frame.shape, (432, 576, 3))


class TestDirtyCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BUDDERS.exists():
            raise unittest.SkipTest(f"run tools/export_visemes.py: {BUDDERS} missing")
        cls.renderer = Renderer(BUDDERS)

    def setUp(self):
        self.face = HollyFace(renderer=self.renderer, classifier="openness-ramp", model_path=None)

    def test_frame_at_needs_a_renderer(self):
        with self.assertRaises(RuntimeError):
            HollyFace(classifier="openness-ramp", model_path=None).frame_at(0.0)

    def test_equal_rows_are_not_redrawn(self):
        hand_utterance(self.face, [one_hot("aa")] * 6)
        fps = self.face.render_fps
        _frame, _rect, changed = self.face.frame_at(0.0)
        self.assertTrue(changed)
        for k in range(1, 6):
            _frame, _rect, changed = self.face.frame_at((k + 0.5) / fps)
            self.assertFalse(changed, f"frame {k} redrew an identical pose")
        self.assertEqual(self.face.stats["draws"], 1)

    def test_a_changed_row_draws_once(self):
        hand_utterance(self.face, [one_hot("aa"), one_hot("E")])
        fps = self.face.render_fps
        self.face.frame_at(0.0)
        _frame, _rect, changed = self.face.frame_at((1 + 0.5) / fps)
        self.assertTrue(changed)
        self.assertEqual(self.face.stats["draws"], 2)

    def test_idle_is_zero_draw_once_the_mouth_is_at_rest(self):
        utterance = hand_utterance(self.face, [one_hot("aa")] * 3)
        fps = self.face.render_fps
        self.face.frame_at(0.0)  # mouth pose
        self.face.frame_at(utterance.end)  # the single draw that restores base
        draws = self.face.stats["draws"]
        for i in range(30):
            _frame, _rect, changed = self.face.frame_at(utterance.end + (i + 1) / fps)
            self.assertFalse(changed)
        self.assertEqual(self.face.stats["draws"], draws, "idle polls cost draws")
        self.assertEqual(self.face.stats["skips"], 30)

    def test_only_the_mouth_rect_ever_changes(self):
        x, y, w, h = self.renderer.mouth_rect
        hand_utterance(self.face, [one_hot("aa"), one_hot("E"), one_hot("ou")])
        fps = self.face.render_fps
        previous = self.face.frame_at(0.0)[0].copy()
        for k in (1, 2):
            current = self.face.frame_at((k + 0.5) / fps)[0]
            np.testing.assert_array_equal(current[:y], previous[:y])
            np.testing.assert_array_equal(current[y + h :], previous[y + h :])
            np.testing.assert_array_equal(current[:, :x], previous[:, :x])
            np.testing.assert_array_equal(current[:, x + w :], previous[:, x + w :])
            previous = current.copy()

    def test_mouth_rect_matches_the_baked_canvas(self):
        x, y, w, h = self.renderer.mouth_rect
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + w, self.renderer.size[0])
        self.assertLessEqual(y + h, self.renderer.size[1])

    def tearDown(self):
        self.face.attach_renderer(None)


class TestShippingPath(unittest.TestCase):
    """The real classifier on the real clip, through the component API."""

    @classmethod
    def setUpClass(cls):
        if not MODEL_EXISTS or not AUDIO.exists():
            raise unittest.SkipTest("model/model-en-mixed.bin or test_audio.flac missing")
        cls.face = HollyFace(model_path=DEFAULT_MODEL)
        cls.samples = decode(AUDIO)
        cls.utterance = cls.face.speak(cls.samples)

    def test_headaudio_is_the_default_classifier(self):
        import inspect

        default = inspect.signature(HollyFace.__init__).parameters["classifier"].default
        self.assertEqual(default, "headaudio")

    def test_analysis_of_a_real_utterance_is_cheap(self):
        # HANDOFF: ~30 ms per 7.7 s utterance, done before playback.
        self.assertLess(self.utterance.analysis_ms, 400.0)
        self.assertGreater(self.utterance.analysis_ms, 0.0)

    def test_the_real_clip_produces_poses(self):
        self.assertGreater(len(self.utterance.keys), 10)
        rows = self.utterance.weights
        self.assertEqual(rows.shape[1], len(VISEMES))
        np.testing.assert_allclose(rows.sum(axis=1), 1.0, atol=1e-9)
        self.assertTrue(np.any(rows[:, BASE] < 1.0 - 1e-9))

    def test_the_queue_covers_the_whole_clip(self):
        self.assertAlmostEqual(self.utterance.audio_duration, self.samples.size / TARGET_RATE, places=6)
        self.assertAlmostEqual(self.face.queued_duration, self.utterance.end)

    def test_the_auto_gate_is_what_closes_the_mouth(self):
        """Regression pin for the shipping look.

        HeadAudio's fixed -40/-50 dBFS gate assumes a noise floor under -50 dBFS. Clean
        TTS does not have that: on test_audio.flac the quiet frames sit around le -4.9,
        so the fixed gate never closes and the mouth hangs open through every pause.
        `vad="auto"` derives the same hysteresis from the utterance's own energy, which is
        what makes the runtime match `visemes.mp4`.
        """
        absolute = HollyFace(vad="absolute")
        absolute.speak(self.samples)
        idle = lambda face: sum(1 for i in range(face.utterances[0].frames)
                                if face.utterances[0].weights[i][BASE] >= 1.0 - 1e-9)
        self.assertGreater(idle(self.face), idle(absolute),
                           "the auto gate produced no more closures than the fixed one")

    def test_every_frame_of_a_real_utterance_samples(self):
        fps = RENDER_FPS
        for k in range(self.utterance.frames):
            row = self.face.row_at(self.utterance.start + (k + 0.5) / fps)
            self.assertEqual(row.shape, (len(VISEMES),))
            self.assertAlmostEqual(float(row.sum()), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
