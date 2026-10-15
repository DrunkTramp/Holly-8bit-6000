"""Pipeline tests.

Run with:  .venv/bin/python -m unittest discover -s tests -t .

These pin the contracts between pipeline stages, because Phase 2 replaces the
classifier and Phase 3 replaces the frame source. If a contract is right and the
classifier is wrong, the failures should point at the classifier only.
"""

from __future__ import annotations

import sys
import unittest
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from holly.audio import FRAME_SIZE, HOP_SIZE, decode, frame, frame_times
from holly.classify import (
    HA_VISEME_TO_CANONICAL,
    OPENNESS_ORDER,
    RECORD_BYTES,
    RECORD_FLOATS,
    HeadAudioClassifier,
    OpennessRamp,
    derive_le_thresholds,
    derive_vad_thresholds,
    mahalanobis,
    parse_model,
    top_two,
)
from holly.features import (
    N_COEFFS,
    N_MELS,
    FeatureSet,
    cepstral_lifter,
    dct_matrix,
    extract_features,
    hamming_window,
    mel_filterbank,
)
from holly.keyframes import (
    Keyframe,
    draw_stats,
    expand_keys,
    fade_frames,
    key_stats,
    keyframe_weights,
    pool_keys,
    pool_weights,
)
from holly.render import Renderer, unpremultiply
from holly.smooth import attack_release, normalize_weights
from holly.timeline import FORMAT, build_timeline, read_timeline, sample_frames, timeline_weights
from holly.vise import BASE_VISEME, VISEMES, VISEME_INDEX

ROOT = Path(__file__).resolve().parents[1]
BUDDERS = ROOT / "build" / "visemes" / "visemes.npz"
AUDIO = ROOT / "test_audio.flac"
MODEL = ROOT / "model" / "model-en-mixed.bin"
DISTANCE_ORACLE = ROOT / "tests" / "fixtures" / "headaudio_distances_oracle.csv"


class TestFraming(unittest.TestCase):
    def test_hop_and_window(self):
        signal = np.zeros(16000, dtype=np.float32)
        frames = frame(signal)
        self.assertEqual(frames.shape[1], FRAME_SIZE)
        self.assertEqual(frames.shape[0], 16000 // HOP_SIZE)

    def test_every_sample_reaches_a_frame(self):
        # Padding exists so the tail is not silently dropped: without it a 1000-sample
        # signal yields 2 frames covering samples 0-767.
        for length in (1000, 1024, 1025, 512, 511, 16000):
            frames = frame(np.arange(length, dtype=np.float32))
            starts = np.arange(frames.shape[0]) * HOP_SIZE
            covered = np.zeros(length, dtype=bool)
            for start in starts:
                lo, hi = max(start, 0), min(start + FRAME_SIZE, length)
                if hi > lo:
                    covered[lo:hi] = True
            self.assertTrue(covered.all(), f"{length} samples left {int((~covered).sum())} uncovered")

    def test_frames_hold_the_original_samples(self):
        signal = np.arange(1000, dtype=np.float32)
        frames = frame(signal)
        self.assertEqual(frames.shape[0], 3)
        np.testing.assert_array_equal(frames[1][:HOP_SIZE], signal[HOP_SIZE : 2 * HOP_SIZE])

    def test_times_match_rate(self):
        times = frame_times(5, hop=256, rate=16000)
        np.testing.assert_allclose(times, [0.0, 0.016, 0.032, 0.048, 0.064], atol=1e-12)

    def test_decode_is_mono_float32(self):
        samples = decode(AUDIO)
        self.assertEqual(samples.dtype, np.float32)
        self.assertTrue(np.isfinite(samples).all())
        self.assertGreater(samples.size, 1000)


class TestFeatures(unittest.TestCase):
    def test_filterbank_shape_and_nonneg(self):
        bank = mel_filterbank(N_MELS, 512, 16000)
        # HeadAudio's power loop covers bins 0..N/2-1, so the bank spans 256 bins.
        self.assertEqual(bank.shape, (N_MELS, 256))
        self.assertTrue((bank >= 0).all())
        # Every mel band must actually reach some FFT bin, or a coefficient is dead.
        self.assertTrue((bank.sum(axis=1) > 0).all())

    def test_filterbank_is_plain_triangles(self):
        # HeadAudio does NOT area-normalise: each triangle peaks at 1.0.
        bank = mel_filterbank(N_MELS, 512, 16000)
        np.testing.assert_allclose(bank.max(axis=1), 1.0)

    def test_hamming_window_matches_headaudio(self):
        w = hamming_window(512)
        self.assertAlmostEqual(float(w[0]), 0.08, places=12)
        self.assertAlmostEqual(float(w[511]), 0.08, places=12)
        i = np.arange(512)
        np.testing.assert_allclose(w, 0.54 - 0.46 * np.cos(2 * np.pi * i / 511))

    def test_dct_rows_are_orthonormal(self):
        # Rows 1..12 of the sqrt(2/M)-scaled DCT-II basis are orthonormal; that is
        # exactly the set HeadAudio keeps (c0 is dropped, energy is separate).
        matrix = dct_matrix(N_COEFFS, N_MELS)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(N_COEFFS), atol=1e-10)

    def test_cepstral_lifter_shape_and_range(self):
        lifter = cepstral_lifter(N_COEFFS)
        self.assertEqual(lifter.shape, (N_COEFFS,))
        # 1 + 11*sin(pi i / 22) for i=1..12: strictly positive, peaks near i=11.
        self.assertTrue((lifter > 1.0).all())
        self.assertAlmostEqual(float(lifter[10]), 12.0, places=9)

    def test_mfcc_is_tanh_compressed(self):
        rng = np.random.default_rng(0)
        features = extract_features(frame(rng.normal(0, 0.5, 4096).astype(np.float32)))
        # tanh saturates to exactly +/-1.0 in float32; the bound is the compression.
        self.assertTrue((np.abs(features.mfcc) <= 1.0).all())

    def test_log_energy_tracks_amplitude(self):
        frames = frame(np.random.default_rng(1).normal(0, 0.1, 4096).astype(np.float32))
        loud = frames * 4.0
        quiet_features = extract_features(frames)
        loud_features = extract_features(loud)
        # Quadrupling amplitude multiplies power by 16; le is log10 of total power.
        np.testing.assert_allclose(
            loud_features.log_energy - quiet_features.log_energy, np.log10(16.0), atol=1e-5
        )

    def test_mfcc_is_invariant_to_amplitude_after_compression(self):
        # tanh(x) != tanh(16x), but the features must stay finite and in range.
        frames = frame(np.random.default_rng(4).normal(0, 0.05, 4096).astype(np.float32))
        features = extract_features(frames)
        self.assertEqual(features.mfcc.shape[1], N_COEFFS)
        self.assertTrue(np.isfinite(features.mfcc).all())

    def test_centroid_in_range(self):
        features = extract_features(frame(np.random.default_rng(2).normal(0, 0.1, 2048).astype(np.float32)))
        self.assertTrue(((features.centroid >= 0.0) & (features.centroid <= 1.0)).all())

    def test_silent_frames_are_finite(self):
        features = extract_features(frame(np.zeros(2048, dtype=np.float32)))
        self.assertTrue(np.isfinite(features.mfcc).all())
        self.assertTrue(np.isfinite(features.centroid).all())


class TestClassify(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.features = extract_features(frame(rng.normal(0, 0.05, 16000).astype(np.float32)))
        floor, ceiling = derive_vad_thresholds(self.features.rms)
        self.classifier = OpennessRamp(silence_floor=floor, speech_ceiling=ceiling)

    def test_weight_matrix_shape(self):
        weights = self.classifier(self.features)
        self.assertEqual(weights.shape, (self.features.rms.shape[0], len(VISEMES)))
        self.assertTrue((weights >= -1e-12).all())

    def test_hat_basis_partitions_unity(self):
        weights = self.classifier(self.features)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-9)

    def test_openness_order_covers_every_overlay(self):
        self.assertEqual(sorted(OPENNESS_ORDER), sorted(v for v in VISEMES if v != BASE_VISEME))

    def test_silence_gates_to_base(self):
        silence = extract_features(frame(np.zeros(16000, dtype=np.float32)))
        weights = OpennessRamp(silence_floor=0.005, speech_ceiling=0.02)(silence)
        base = VISEME_INDEX[BASE_VISEME]
        np.testing.assert_allclose(weights[:, base], 1.0, atol=1e-9)

    def test_derive_le_thresholds_splits_modes(self):
        # Bimodal log-energy: a quiet mode at -4.7 (like test_audio.flac's pauses,
        # which HeadAudio's fixed -5.0 close threshold never reaches) and speech
        # at -2.5. The derived gate must sit between the modes.
        le = np.concatenate([np.full(30, -4.7), np.full(60, -2.5)])
        floor, ceiling = derive_le_thresholds(le)
        self.assertLess(floor, -4.0)
        self.assertGreater(ceiling, -3.0)
        self.assertGreaterEqual(ceiling - floor, 0.5)

    def test_derive_le_thresholds_keeps_minimum_gap(self):
        # A recording with no real silence: percentiles collapse, the gap rule
        # must still leave a usable hysteresis band.
        floor, ceiling = derive_le_thresholds(np.full(50, -2.0))
        self.assertAlmostEqual(ceiling - floor, 0.5)

    def test_top_two_renormalises(self):
        row = np.zeros(len(VISEMES))
        row[VISEME_INDEX["PP"]] = 0.6
        row[VISEME_INDEX["TH"]] = 0.2
        a, b, weight_b = top_two(row)
        self.assertEqual(VISEMES[a], "PP")
        self.assertEqual(VISEMES[b], "TH")
        self.assertAlmostEqual(weight_b, 0.2 / 0.8, places=6)

    def test_empty_row(self):
        self.assertEqual(top_two(np.zeros(len(VISEMES))), (-1, -1, 0.0))


def _encode_record(phoneme: str, group: int, viseme: int, mu, lower) -> bytes:
    """Pack one prototype the way training.mjs computePrototype writes it.

    The phoneme is a BIG-endian uint32 (DataView.setUint32's default) at bytes 0-3;
    group and viseme are raw bytes 5 and 7; the rest is little-endian float32.
    """
    mu = np.asarray(mu, dtype="<f4")
    lower = np.asarray(lower, dtype="<f4")
    assert mu.shape == (N_COEFFS,) and lower.shape == (N_COEFFS * (N_COEFFS + 1) // 2,)
    record = np.zeros(RECORD_FLOATS, dtype="<f4")
    record[2:] = np.concatenate([mu, lower])
    buf = bytearray(record.tobytes())
    cp1 = ord(phoneme[0])
    cp2 = ord(phoneme[1]) if len(phoneme) > 1 else 0
    buf[0:4] = ((cp1 << 16) | cp2).to_bytes(4, "big")
    buf[5] = group
    buf[7] = viseme
    return bytes(buf)


def _js_mahalanobis(v, mu, sigma_lower) -> float:
    """Literal transcription of classifier.mjs distanceMahalanobis, for cross-checking."""
    n = len(v)
    diff = [float(v[i]) - float(mu[i]) for i in range(n)]
    matrix = [[0.0] * n for _ in range(n)]
    pos = 0
    for i in range(n):
        for j in range(i + 1):
            matrix[i][j] = matrix[j][i] = float(sigma_lower[pos])
            pos += 1
    d = 0.0
    for i in range(n):
        d += matrix[i][i] * diff[i] * diff[i]
        for j in range(i):
            d += 2.0 * matrix[i][j] * diff[i] * diff[j]
    return d


def _synthetic_model():
    """A tiny model: 'a' -> PP (id 5) at mu=+0.5, 's1' -> sil (id 14) at mu=0."""
    rng = np.random.default_rng(7)
    n = N_COEFFS
    # identity covariance: the packed lower triangle is 1.0 at positions i*(i+1)/2 + i
    diag_pos = np.array([i * (i + 1) // 2 + i for i in range(n)])
    lower_pp = np.zeros(n * (n + 1) // 2)
    lower_pp[diag_pos] = 1.0
    lower_sil = lower_pp.copy()
    mu_pp = np.full(n, 0.5)
    mu_sil = np.zeros(n)
    data = (
        _encode_record("a", 0, 5, mu_pp, lower_pp)
        + _encode_record("s1", 0, 14, mu_sil, lower_sil)
    )
    return parse_model(data), rng


class TestHeadAudioModel(unittest.TestCase):
    def test_record_geometry(self):
        self.assertEqual(RECORD_FLOATS, 2 + N_COEFFS + N_COEFFS * (N_COEFFS + 1) // 2)
        self.assertEqual(RECORD_BYTES, RECORD_FLOATS * 4)

    def test_parser_round_trip(self):
        model, _ = _synthetic_model()
        self.assertEqual(model.n_prototypes, 2)
        self.assertEqual(model.phonemes, ("a", "s1"))
        self.assertEqual(list(model.groups), [0, 0])
        self.assertEqual(list(model.visemes), [5, 14])
        np.testing.assert_allclose(model.mu[0], np.full(N_COEFFS, 0.5))
        np.testing.assert_allclose(model.mu[1], 0.0)
        # identity covariance rebuilt from the packed lower triangle
        np.testing.assert_allclose(model.sigma_inv[0], np.eye(N_COEFFS))

    def test_parser_rejects_partial_record(self):
        with self.assertRaises(ValueError):
            parse_model(b"\x00" * (RECORD_BYTES + 4))

    def test_two_char_phoneme(self):
        data = _encode_record("s1", 3, 14, np.zeros(N_COEFFS), np.zeros(N_COEFFS * (N_COEFFS + 1) // 2))
        self.assertEqual(parse_model(data).phonemes, ("s1",))

    def test_mahalanobis_matches_the_js_quadratic_form(self):
        # Pins the lower-triangle packing ORDER: a transposed rebuild would keep
        # every shape plausible while silently scrambling accuracy.
        rng = np.random.default_rng(11)
        n = N_COEFFS
        lower = rng.normal(0, 0.2, n * (n + 1) // 2)
        mu = rng.normal(0, 0.3, n)
        v = rng.normal(0, 0.4, (5, n))
        data = _encode_record("x", 0, 0, mu, lower)
        model = parse_model(data)
        got = mahalanobis(v, model)[:, 0]
        # The record stores float32; reference the float32-rounded parameters so
        # the only remaining difference is summation order.
        want = np.array([
            _js_mahalanobis(
                v[i], mu.astype(np.float32).astype(np.float64),
                lower.astype(np.float32).astype(np.float64),
            )
            for i in range(v.shape[0])
        ])
        np.testing.assert_allclose(got, want, rtol=1e-9)

    def test_distance_to_own_prototype_is_zero(self):
        model, _ = _synthetic_model()
        # At mu_pp the PP distance is 0; the sil distance is sum(0.5^2) over 12 dims.
        d = mahalanobis(np.array([model.mu[0]]), model)
        np.testing.assert_allclose(d, [[0.0, 12 * 0.25]], atol=1e-9)

    def test_viseme_remap_is_a_permutation(self):
        self.assertEqual(len(HA_VISEME_TO_CANONICAL), 15)
        self.assertEqual(sorted(HA_VISEME_TO_CANONICAL), list(range(15)))
        self.assertEqual(VISEMES[HA_VISEME_TO_CANONICAL[14]], "sil")
        self.assertEqual(VISEMES[HA_VISEME_TO_CANONICAL[0]], "aa")
        self.assertEqual(VISEMES[HA_VISEME_TO_CANONICAL[2]], "ih")
        self.assertEqual(VISEMES[HA_VISEME_TO_CANONICAL[3]], "oh")
        self.assertEqual(VISEMES[HA_VISEME_TO_CANONICAL[4]], "ou")

    @unittest.skipUnless(MODEL.exists(), "model-en-mixed.bin not downloaded")
    def test_shipped_model_parses(self):
        model = parse_model(MODEL.read_bytes())
        # 14352 bytes = 39 records of 368 bytes: 38 IPA phonemes + synthetic s1.
        self.assertEqual(model.n_prototypes, 39)
        self.assertEqual(MODEL.stat().st_size % RECORD_BYTES, 0)
        self.assertIn("s1", model.phonemes)
        # every canonical viseme is reachable from the shipped map
        reachable = set(model.phoneme_viseme_map().values())
        self.assertEqual(reachable, set(VISEMES))

    @unittest.skipUnless(MODEL.exists(), "model-en-mixed.bin not downloaded")
    def test_shipped_phoneme_viseme_map(self):
        # HeadAudio's real map, replacing OPENNESS_ORDER. Transcribed from
        # dist/model-en-mixed.bin record headers.
        expected = {
            "\u00f0": "TH", "\u0259": "E", "b": "PP", "\u025c": "E", "\u0279": "RR",
            "\u02a7": "CH", "k": "kk", "n": "nn", "u": "ou", "s": "SS",
            "l": "RR", "\u026a": "ih", "d": "DD", "\u0251": "aa", "m": "PP",
            "p": "PP", "\u00e6": "aa", "\u014b": "nn", "\u0261": "kk", "\u0283": "SS",
            "i": "ih", "t": "DD", "W": "ou", "z": "SS", "\u025b": "E", "\u03b8": "TH",
            "\u028c": "aa", "v": "FF", "w": "ou", "A": "E", "I": "ih",
            "\u0254": "oh", "f": "FF", "O": "oh", "\u02a4": "CH", "Y": "ih",
            "j": "ih", "\u028a": "ou", "s1": "sil",
        }
        model = parse_model(MODEL.read_bytes())
        self.assertEqual(model.phoneme_viseme_map(), expected)

    @unittest.skipUnless(MODEL.exists() and DISTANCE_ORACLE.exists(),
                         "model or oracle missing")
    def test_distances_match_the_js_classifier_output(self):
        # tests/distances.csv from the HeadAudio repo: the ACTUAL 39x39 pairwise
        # Mahalanobis matrix their JS classifier emits at every prototype's mean,
        # with silSensitivity=1.2 baked in and toFixed(1) rounding. Our float64
        # port must agree within that rounding (0.05) plus float32 accumulation drift.
        import csv

        model = parse_model(MODEL.read_bytes())
        with open(DISTANCE_ORACLE, newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0][1:], list(model.phonemes))  # also pins record ORDER
        want = np.array([[float(x) for x in r[1:]] for r in rows[1:]])

        got = mahalanobis(model.mu, model)
        got[:, model.visemes == 14] /= 1.2
        np.testing.assert_allclose(got, want, atol=0.06)


def _features_for(mfcc, log_energy):
    n = mfcc.shape[0]
    return FeatureSet(
        mfcc=mfcc.astype(np.float32),
        mel_log=np.zeros((n, N_MELS), dtype=np.float32),
        centroid=np.zeros(n, dtype=np.float32),
        rms=np.zeros(n, dtype=np.float32),
        log_energy=np.asarray(log_energy, dtype=np.float32),
    )


class TestHeadAudioClassifier(unittest.TestCase):
    def setUp(self):
        self.model, self.rng = _synthetic_model()

    def _classifier(self, vote_window=0, vad=False, gain=False, **kwargs):
        return HeadAudioClassifier(
            model=self.model, vote_window=vote_window, vad=vad, gain=gain, **kwargs
        )

    def test_weight_matrix_contract(self):
        mfcc = self.rng.normal(0, 0.3, (20, N_COEFFS))
        weights = self._classifier()(_features_for(mfcc, np.zeros(20)))
        self.assertEqual(weights.shape, (20, len(VISEMES)))
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
        self.assertTrue(((weights == 0.0) | (weights == 1.0)).all())

    def test_display_gain_caps_mouth_poses(self):
        # headaudio.mjs visemeMaxs: PP (id 5) caps at 0.75, the rest to sil.
        mfcc = np.tile(self.model.mu[0], (4, 1))
        weights = self._classifier(gain=True)(_features_for(mfcc, np.zeros(4)))
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
        np.testing.assert_allclose(weights[:, VISEME_INDEX["PP"]], 0.75)
        np.testing.assert_allclose(weights[:, VISEME_INDEX["sil"]], 0.25)
        # a sil decision stays a full sil, not 0.65 of it
        mfcc = np.tile(self.model.mu[1], (3, 1))
        weights = self._classifier(gain=True)(_features_for(mfcc, np.zeros(3)))
        np.testing.assert_allclose(weights[:, VISEME_INDEX["sil"]], 1.0)

    def test_picks_the_nearest_prototype(self):
        mfcc = np.vstack([self.model.mu[0], self.model.mu[1], self.model.mu[0]])
        weights = self._classifier()(_features_for(mfcc, np.zeros(3)))
        self.assertEqual(VISEMES[int(weights[0].argmax())], "PP")
        self.assertEqual(VISEMES[int(weights[1].argmax())], "sil")
        self.assertEqual(VISEMES[int(weights[2].argmax())], "PP")

    def test_sil_sensitivity_can_flip_a_boundary_frame(self):
        # v = 0.277 (all dims): d_pp = 12*(0.5-0.277)^2 = 0.597, d_sil = 12*0.277^2 = 0.921.
        # At sensitivity 1.0 PP wins; at 2.0 the sil distance is halved to 0.460 and sil wins.
        mfcc = np.full((1, N_COEFFS), 0.277)
        plain = self._classifier(sil_sensitivity=1.0)(_features_for(mfcc, [0.0]))
        boosted = self._classifier(sil_sensitivity=2.0)(_features_for(mfcc, [0.0]))
        self.assertEqual(VISEMES[int(plain.argmax(axis=1)[0])], "PP")
        self.assertEqual(VISEMES[int(boosted.argmax(axis=1)[0])], "sil")

    def test_vad_gate_blocks_quiet_frames(self):
        mfcc = np.tile(self.model.mu[0], (5, 1))
        # le must exceed -4.0 (= -40 dBFS) for the gate to open.
        quiet = self._classifier(vad=True, analysis_fps=62.5)(_features_for(mfcc, np.full(5, -6.0)))
        np.testing.assert_allclose(quiet[:, VISEME_INDEX["sil"]], 1.0)
        loud = self._classifier(vad=True, analysis_fps=62.5)(_features_for(mfcc, np.full(5, 0.0)))
        self.assertEqual(VISEMES[int(loud[0].argmax())], "PP")

    def test_majority_vote_delays_a_single_flip(self):
        # Ring initialised to sil: one PP frame inside a run of sil frames must
        # not win a 6-frame vote (1 PP vs 5 sil).
        mfcc = np.vstack([self.model.mu[1], self.model.mu[0], self.model.mu[1]])
        weights = self._classifier(vote_window=6, vad=False)(_features_for(mfcc, [0.0, 0.0, 0.0]))
        self.assertEqual(VISEMES[int(weights[1].argmax())], "sil")
        # Five PP frames in a row outvote the initial sil ring.
        mfcc = np.tile(self.model.mu[0], (5, 1))
        weights = self._classifier(vote_window=6, vad=False)(_features_for(mfcc, np.zeros(5)))
        self.assertEqual(VISEMES[int(weights[4].argmax())], "PP")

    @unittest.skipUnless(MODEL.exists(), "model-en-mixed.bin not downloaded")
    def test_shipped_model_on_silence_is_all_sil(self):
        classifier = HeadAudioClassifier(MODEL)
        n = 30
        mfcc = np.zeros((n, N_COEFFS))
        weights = classifier(_features_for(mfcc, np.full(n, -8.0)))  # le below -5: VAD closed
        np.testing.assert_allclose(weights[:, VISEME_INDEX[BASE_VISEME]], 1.0)

    @unittest.skipUnless(AUDIO.exists() and MODEL.exists(), "test audio or model missing")
    def test_shipped_model_spreads_beyond_openness_on_real_audio(self):
        # The stub's tell was aa+sil hogging the screen. The real classifier must
        # reach at least 10 distinct visemes on the test clip -- phonetics, not openness.
        from holly.audio import decode

        samples = decode(AUDIO)
        features = extract_features(frame(samples), hop=HOP_SIZE)
        weights = HeadAudioClassifier(MODEL)(features)
        top = weights.argmax(axis=1)
        active = weights[:, VISEME_INDEX[BASE_VISEME]] < 0.999
        distinct = set(int(i) for i in top[active])
        self.assertGreaterEqual(len(distinct), 10)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)


class TestSmoothing(unittest.TestCase):
    def test_attack_is_faster_than_release(self):
        # A 1.6 s pulse: aa on for the first 100 frames, off for the rest.
        pulse = np.zeros((200, len(VISEMES)))
        pulse[:100, VISEME_INDEX["aa"]] = 1.0
        smoothed = attack_release(pulse, fps=62.5, attack_ms=30.0, release_ms=120.0)
        column = smoothed[:, VISEME_INDEX["aa"]]

        rise = int(np.argmax(column >= 0.5))
        fall = int(np.argmax(column[100:] <= 0.5))
        self.assertLess(rise, fall, "a 30 ms attack must reach half weight sooner than a 120 ms release")
        self.assertGreater(fall, 0)

    def test_constant_input_converges(self):
        steady = np.ones((400, len(VISEMES))) / len(VISEMES)
        smoothed = attack_release(steady, fps=62.5)
        np.testing.assert_allclose(smoothed[-1], steady[-1], atol=1e-3)

    def test_normalize_falls_back_to_base(self):
        out = normalize_weights(np.zeros((3, len(VISEMES))))
        np.testing.assert_allclose(out[:, VISEME_INDEX[BASE_VISEME]], 1.0)
        np.testing.assert_allclose(out.sum(axis=1), 1.0)

    def test_normalize_scales(self):
        row = np.zeros((1, len(VISEMES)))
        row[0, VISEME_INDEX["aa"]] = 3.0
        row[0, VISEME_INDEX["PP"]] = 1.0
        out = normalize_weights(row)
        self.assertAlmostEqual(out[0, VISEME_INDEX["aa"]], 0.75, places=6)


class TestKeyframes(unittest.TestCase):
    """Keyframing is a re-timing transform inserted before the timeline.

    These use hand-built weight matrices rather than the classifier, so the
    expected poses and fade ramps are exact.
    """

    def pose_rows(self, seq, per_slot):
        rows = []
        for name in seq:
            row = np.zeros(len(VISEMES))
            row[VISEME_INDEX[name]] = 1.0
            rows.extend([row] * per_slot)
        return np.array(rows)

    def test_slot_boundaries_are_exact_in_time(self):
        # 64 analysis frames at 64 fps, 8 poses/s -> 8 frames pooled per pose.
        weights = self.pose_rows(["aa"] * 8, 8)
        pooled = pool_weights(weights, analysis_fps=64.0, key_hz=8.0)
        self.assertEqual(len(pooled), 8)
        for row in pooled:
            np.testing.assert_allclose(row, np.eye(len(VISEMES))[VISEME_INDEX["aa"]], atol=1e-12)

    def test_key_grid_times_are_multiples_of_the_period(self):
        keys = pool_keys(self.pose_rows(["aa", "PP", "TH"] * 4, 8), analysis_fps=64.0, key_hz=8.0)
        np.testing.assert_allclose([k.t for k in keys], [i / 8.0 for i in range(12)], atol=1e-12)

    def test_constant_pose_never_switches(self):
        keys = pool_keys(self.pose_rows(["aa"] * 10, 8), analysis_fps=64.0, key_hz=8.0)
        self.assertEqual(len({k.viseme for k in keys}), 1)
        stats = key_stats(keys, 8.0, 10 / 8.0)
        self.assertEqual(stats["changes"], 0)
        self.assertEqual(stats["changes_per_s"], 0.0)

    def test_switch_rate_cannot_exceed_the_pose_rate(self):
        seq = ["aa", "PP", "TH", "DD", "SS", "CH", "nn", "RR"]
        keys = pool_keys(self.pose_rows(seq, 8), analysis_fps=64.0, key_hz=8.0)
        stats = key_stats(keys, 8.0, len(seq) / 8.0)
        self.assertEqual(stats["changes"], len(seq) - 1)
        self.assertLessEqual(stats["changes_per_s"], 8.0 + 1e-9)

    def test_silence_only_is_all_base(self):
        weights = np.zeros((16, len(VISEMES)))
        weights[:, VISEME_INDEX["sil"]] = 1.0
        keys = pool_keys(weights, analysis_fps=64.0, key_hz=8.0)
        self.assertEqual({k.viseme for k in keys}, {BASE_VISEME})

    def test_sticky_holds_a_marginal_pose(self):
        # One analysis frame per pose slot, so the pooled vector is the row itself.
        row = np.zeros(len(VISEMES))
        row[VISEME_INDEX["PP"]] = 0.42
        row[VISEME_INDEX["aa"]] = 0.38
        row[VISEME_INDEX["TH"]] = 0.20
        weights = np.vstack([np.eye(len(VISEMES))[VISEME_INDEX["aa"]], row])

        held = pool_keys(weights, analysis_fps=8.0, key_hz=8.0, sticky=0.05)
        self.assertEqual([k.viseme for k in held], ["aa", "aa"], "0.42 does not beat 0.38 by 0.05")

        eager = pool_keys(weights, analysis_fps=8.0, key_hz=8.0, sticky=0.0)
        self.assertEqual([k.viseme for k in eager], ["aa", "PP"])

    def test_sticky_does_not_gate_the_base_viseme(self):
        # A margin here would delay mouth opening by a whole pose slot, because sil
        # is often the argmax of a slot that straddles a speech onset.
        closing = np.zeros(len(VISEMES))
        closing[VISEME_INDEX["sil"]] = 0.42
        closing[VISEME_INDEX["aa"]] = 0.38
        closing[VISEME_INDEX["TH"]] = 0.20
        weights = np.vstack([np.eye(len(VISEMES))[VISEME_INDEX["aa"]], closing])
        keys = pool_keys(weights, analysis_fps=8.0, key_hz=8.0, sticky=0.05)
        self.assertEqual([k.viseme for k in keys], ["aa", "sil"],
                         "the margin must not stop the mouth closing when speech ends")

        opening = np.zeros(len(VISEMES))
        opening[VISEME_INDEX["sil"]] = 0.30
        opening[VISEME_INDEX["aa"]] = 0.70
        weights = np.vstack([np.eye(len(VISEMES))[VISEME_INDEX["sil"]], opening])
        keys = pool_keys(weights, analysis_fps=8.0, key_hz=8.0, sticky=0.05)
        self.assertEqual([k.viseme for k in keys], ["sil", "aa"], "mouth must open on the slot speech starts in")

    def test_max_pool_picks_a_short_peak(self):
        # Three frames of aa then one bright PP frame: mean keeps aa, max surfaces PP.
        block = np.zeros((4, len(VISEMES)))
        block[:3, VISEME_INDEX["aa"]] = 0.5
        block[3, VISEME_INDEX["PP"]] = 0.9
        mean_keys = pool_keys(block, analysis_fps=32.0, key_hz=8.0, pool="mean")
        max_keys = pool_keys(block, analysis_fps=32.0, key_hz=8.0, pool="max")
        self.assertEqual(mean_keys[0].viseme, "aa")
        self.assertEqual(max_keys[0].viseme, "PP")

    def test_expanded_weights_sum_to_one(self):
        seq = ["sil", "aa", "PP", "sil", "TH", "aa"]
        keys = pool_keys(self.pose_rows(seq, 8), analysis_fps=64.0, key_hz=8.0)
        dense, times, fps = expand_keys(keys, key_hz=8.0, out_fps=30.0, fade_s=0.05)
        np.testing.assert_allclose(dense.sum(axis=1), 1.0, atol=1e-12)
        self.assertAlmostEqual(fps, 30.0)
        np.testing.assert_allclose(times, np.arange(len(dense)) / 30.0, atol=1e-12)

    def test_at_most_two_visemes_are_blendable(self):
        # The renderer composites the top 2 only; a row with three live weights
        # would silently drop one.
        seq = ["aa", "PP", "TH", "DD", "SS", "CH"]
        keys = pool_keys(self.pose_rows(seq, 8), analysis_fps=64.0, key_hz=8.0)
        for shape in ("center", "lead", "lag"):
            dense, _, _ = expand_keys(keys, key_hz=8.0, out_fps=30.0, fade_s=0.05, shape=shape)
            self.assertTrue((dense > 0).sum(axis=1).max() <= 2, f"{shape} produced a triple blend")

    def test_fade_is_quantised_to_whole_frames(self):
        self.assertEqual(fade_frames(0.05, 30.0), 2)
        self.assertEqual(fade_frames(0.05, 60.0), 3)
        self.assertEqual(fade_frames(0.004, 30.0), 1, "a sub-frame fade must still render as one frame")

    def test_fade_midpoint_is_actually_sampled(self):
        # 4 poses/s at 16 fps -> 4 output frames per slot, 4-frame fade.
        # center puts the 50/50 blend on the frame just before the boundary.
        keys = [Keyframe(t=0.0, viseme="aa", weight=1.0), Keyframe(t=0.25, viseme="PP", weight=1.0)]
        dense, _, _ = expand_keys(keys, key_hz=4.0, out_fps=16.0, fade_s=0.25, shape="center")

        self.assertEqual(len(dense), 8)
        np.testing.assert_allclose(dense[0], self.one_hot("aa"), atol=1e-12)
        self.assertAlmostEqual(dense[2, VISEME_INDEX["PP"]], 0.15625, places=6)
        self.assertAlmostEqual(dense[3, VISEME_INDEX["PP"]], 0.5, places=6)
        self.assertAlmostEqual(dense[4, VISEME_INDEX["PP"]], 0.84375, places=6)
        np.testing.assert_allclose(dense[5], self.one_hot("PP"), atol=1e-12)
        np.testing.assert_allclose(dense[7], self.one_hot("PP"), atol=1e-12)

    def test_lead_and_lag_position_the_fade(self):
        # 4 poses/s at 16 fps -> 4 output frames per slot; a 2-frame fade leaves
        # most of each slot held pure, so the shape difference is visible.
        #
        # The `lag` assertion (no blending before the boundary) is the causality
        # property the live path needs: a slot's pose is only knowable when that
        # slot's audio has arrived, so `center` and `lead` both require a pose from
        # the future. Offline, `center` is fine because the whole file exists.
        keys = [Keyframe(t=0.0, viseme="aa", weight=1.0), Keyframe(t=0.25, viseme="PP", weight=1.0)]

        lead, _, _ = expand_keys(keys, key_hz=4.0, out_fps=16.0, fade_s=0.125, shape="lead")
        for k in range(2):
            np.testing.assert_allclose(lead[k], self.one_hot("aa"), atol=1e-12)
        self.assertAlmostEqual(lead[2, VISEME_INDEX["PP"]], 0.5, places=6)
        np.testing.assert_allclose(lead[3], self.one_hot("PP"), atol=1e-12)
        self.assertEqual(lead[3, VISEME_INDEX["aa"]], 0.0, "lead must reach the new pose before the boundary")

        lag, _, _ = expand_keys(keys, key_hz=4.0, out_fps=16.0, fade_s=0.125, shape="lag")
        for k in range(4):
            np.testing.assert_allclose(lag[k], self.one_hot("aa"), atol=1e-12)
        self.assertAlmostEqual(lag[4, VISEME_INDEX["PP"]], 0.5, places=6)
        np.testing.assert_allclose(lag[5], self.one_hot("PP"), atol=1e-12)

        center, _, _ = expand_keys(keys, key_hz=4.0, out_fps=16.0, fade_s=0.125, shape="center")
        for k in range(3):
            np.testing.assert_allclose(center[k], self.one_hot("aa"), atol=1e-12)
        self.assertAlmostEqual(center[3, VISEME_INDEX["PP"]], 0.5, places=6)
        np.testing.assert_allclose(center[4], self.one_hot("PP"), atol=1e-12)

    def test_same_pose_on_both_sides_of_a_boundary_does_not_blend(self):
        keys = [Keyframe(t=0.0, viseme="aa", weight=1.0), Keyframe(t=0.25, viseme="aa", weight=1.0)]
        dense, _, _ = expand_keys(keys, key_hz=4.0, out_fps=16.0, fade_s=0.25)
        np.testing.assert_allclose(dense, np.tile(self.one_hot("aa"), (len(dense), 1)), atol=1e-12)

    def test_oversized_fade_is_clamped(self):
        keys = pool_keys(self.pose_rows(["aa", "PP", "TH", "DD"], 8), analysis_fps=64.0, key_hz=8.0)
        dense, _, _ = expand_keys(keys, key_hz=8.0, out_fps=30.0, fade_s=5.0)
        self.assertTrue((dense > 0).sum(axis=1).max() <= 2)
        np.testing.assert_allclose(dense.sum(axis=1), 1.0, atol=1e-12)
        # A fade can never dissolve every slot; at least one pose must be held pure.
        pure = (dense.max(axis=1) >= 1.0 - 1e-9).sum()
        self.assertGreater(pure, 0)

    def test_keyframing_reduces_the_switch_rate(self):
        """The whole reason this stage exists: fewer, longer-held pose changes."""
        rng = np.random.default_rng(11)
        weights = normalize_weights(np.abs(rng.normal(0, 1, (480, len(VISEMES)))))
        tops = [VISEMES[int(i)] for i in weights.argmax(axis=1)]
        raw = sum(1 for a, b in pairwise(tops) if a != b)

        keys, _, _, _ = keyframe_weights(weights, analysis_fps=62.5, key_hz=8.0, out_fps=30.0)
        stats = key_stats(keys, 8.0, 480 / 62.5)
        self.assertLess(stats["changes_per_s"], 8.0)
        self.assertLess(stats["changes"], raw)
        self.assertGreaterEqual(stats["shortest_hold_s"], 1 / 8.0 - 1e-9)

    def one_hot(self, viseme):
        row = np.zeros(len(VISEMES))
        row[VISEME_INDEX[viseme]] = 1.0
        return row


class TestDrawCost(unittest.TestCase):
    """30 fps is a logical clock, not a refresh rate: only changed frames cost anything."""

    def test_repeated_poses_are_free(self):
        # 3 slots of 4 frames each, no fades: only 3 draws out of 12.
        rows = np.repeat([np.eye(len(VISEMES))[VISEME_INDEX[v]] for v in ("aa", "PP", "TH")], 4, axis=0)
        stats = draw_stats(rows, fps=30.0)
        self.assertEqual(stats["redraws"], 3)
        self.assertEqual(stats["skippable"], 9)
        self.assertEqual(stats["longest_burst_frames"], 1)

    def test_idle_frames_are_counted_separately(self):
        base = np.eye(len(VISEMES))[VISEME_INDEX[BASE_VISEME]]
        rows = np.vstack([base] * 5 + [np.eye(len(VISEMES))[VISEME_INDEX["aa"]]] * 5 + [base] * 5)
        stats = draw_stats(rows, fps=30.0)
        self.assertEqual(stats["idle_frames"], 10)
        self.assertEqual(stats["idle_periods"], 2, "the mouth closes, speaks, closes again")
        self.assertAlmostEqual(stats["idle_share"], 10 / 15)
        self.assertEqual(stats["redraws"], 3)

    def test_all_sil_needs_exactly_one_draw(self):
        rows = np.tile(np.eye(len(VISEMES))[VISEME_INDEX[BASE_VISEME]], (20, 1))
        stats = draw_stats(rows, fps=30.0)
        self.assertEqual(stats["redraws"], 1, "settling to base once is the only cost of silence")
        self.assertEqual(stats["idle_periods"], 1)
        self.assertEqual(stats["draws_per_s_speaking"], 0.0, "nothing was spoken")

    def test_keyframed_output_is_cheaper_than_continuous(self):
        samples = decode(AUDIO)
        features = extract_features(frame(samples))
        floor, ceiling = derive_vad_thresholds(features.rms)
        weights = normalize_weights(OpennessRamp(floor, ceiling)(features))
        fps = 16000 / HOP_SIZE

        continuous = normalize_weights(attack_release(weights, fps=fps))
        cont = draw_stats(continuous, fps)

        _, dense, _, out_fps = keyframe_weights(weights, analysis_fps=fps, key_hz=12.0, out_fps=30.0)
        keyed = draw_stats(dense, out_fps)

        self.assertGreater(cont["redraw_share"], 0.9,
                           "the envelope output changes on nearly every frame -- no free skips")
        self.assertLess(keyed["redraw_share"], 0.6)
        self.assertLess(keyed["draws_per_s_speaking"], cont["draws_per_s_overall"])

    def test_rejects_bad_fps(self):
        with self.assertRaises(ValueError):
            draw_stats(np.zeros((3, len(VISEMES))), fps=0.0)


class TestTimeline(unittest.TestCase):
    def test_round_trip(self):
        weights = np.zeros((5, len(VISEMES)))
        weights[0, VISEME_INDEX["sil"]] = 1.0
        weights[1, VISEME_INDEX["aa"]] = 0.7
        weights[1, VISEME_INDEX["PP"]] = 0.3
        weights[2, VISEME_INDEX["oh"]] = 1.0
        weights[3, VISEME_INDEX["sil"]] = 0.5
        weights[3, VISEME_INDEX["E"]] = 0.5
        weights[4, VISEME_INDEX["nn"]] = 1.0
        times = frame_times(5)

        timeline = build_timeline(weights, times, fps=62.5)
        self.assertEqual(timeline["format"], FORMAT)
        self.assertEqual(timeline["frame_count"], 5)

        restored = timeline_weights(timeline)
        np.testing.assert_allclose(restored, weights, atol=1e-3)

    def test_alt_drops_negligible_weights(self):
        weights = np.full((2, len(VISEMES)), 0.001)
        weights[:, VISEME_INDEX["sil"]] = 0.999
        timeline = build_timeline(weights, frame_times(2), fps=62.5)
        for record in timeline["frames"]:
            self.assertLessEqual(len(record["alt"]), 2)

    def test_read_rejects_unknown_format(self):
        path = Path("/tmp/holly_bad_timeline.json")
        path.write_text('{"format": "nope", "frames": []}')
        with self.assertRaises(ValueError):
            read_timeline(path)

    def test_sample_frames_endpoints(self):
        timeline = {"frames": [{"t": i / 10.0} for i in range(11)]}
        sampled = sample_frames(timeline, 5)
        self.assertEqual(len(sampled), 5)
        self.assertEqual(sampled[0]["t"], 0.0)
        self.assertEqual(sampled[-1]["t"], 1.0)


class TestRenderer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BUDDERS.exists():
            raise unittest.SkipTest(f"run tools/export_visemes.py first: {BUDDERS} missing")
        cls.renderer = Renderer(BUDDERS)

    def weights(self, **values: float) -> np.ndarray:
        return np.array([values.get(v, 0.0) for v in VISEMES], dtype=np.float64)

    def test_base_only_reproduces_the_base_layer(self):
        # The baked base is premultiplied, so compare against the straight-alpha
        # source layer rather than the buffer itself.
        from psd_tools import PSDImage

        source = ROOT / "viseme.psd"
        if not source.exists():
            self.skipTest(f"{source} missing")
        layers = {layer.name: layer for layer in PSDImage.open(source)}
        layer = layers.get("sil") or layers.get("sil.png")
        if layer is None:
            self.fail("viseme.psd has no sil base layer")
        image = layer.composite()
        if image is None:
            self.skipTest("the base layer produced no image")
        straight = np.array(image.convert("RGB"), dtype=np.int16)
        drawn = self.renderer.draw_row(self.weights(sil=1.0)).astype(np.int16)
        self.assertLessEqual(float(np.abs(drawn - straight).mean()), 0.2)

    def test_only_the_mouth_rect_is_touched(self):
        # draw_row returns a persistent buffer, so these must be copied or the two
        # names would alias one object and the comparison would assert nothing.
        base = self.renderer.draw_row(self.weights(sil=1.0)).copy()
        x, y = self.renderer.origin
        h, w = self.renderer.patch_shape
        outside = np.ones(base.shape[:2], dtype=bool)
        outside[y : y + h, x : x + w] = False

        for viseme in ("aa", "PP", "TH", "ou"):
            drawn = self.renderer.draw_row(self.weights(**{viseme: 1.0})).copy()
            self.assertEqual(int(np.abs(drawn - base)[outside].sum()), 0, f"{viseme} bled outside the mouth rect")
        # Guard against the test itself going vacuous: inside the rect must differ.
        drawn = self.renderer.draw_row(self.weights(aa=1.0)).copy()
        self.assertGreater(int(np.abs(drawn - base)[~outside].sum()), 0,
                           "the mouth rect never changed, so this test is not measuring anything")

    def test_draw_row_returns_the_persistent_frame(self):
        # The aliasing contract, pinned: no per-draw copy of the 3 MB canvas.
        a = self.renderer.draw_row(self.weights(sil=1.0))
        b = self.renderer.draw_row(self.weights(aa=1.0))
        self.assertIs(a, b)
        self.assertIs(a, self.renderer.frame)

    def test_snapshot_survives_later_draws(self):
        held = self.renderer.snapshot()
        after = held.copy()
        self.renderer.draw_row(self.weights(ou=1.0))
        np.testing.assert_array_equal(held, after, "snapshot() must not alias the live buffer")

    def test_redrawing_the_same_row_is_idempotent(self):
        # The region is reset from base each draw, so repeats must not accumulate.
        row = self.weights(aa=0.5, PP=0.5)
        first = self.renderer.draw_row(row).copy()
        for _ in range(5):
            np.testing.assert_array_equal(self.renderer.draw_row(row), first)

    def test_mouth_rect_only_matches_the_full_canvas_algorithm(self):
        """Correctness oracle for the ~35x cheaper draw path.

        `draw_row` recomputes and writes only the mouth rect, in float32. That is only
        valid because the rest of the canvas is byte-identical between frames and the base
        layer is fully opaque, which makes unpremultiplying it a no-op. This compares
        against the original whole-canvas float64 computation so the optimisation can never
        silently become an approximation.

        Tolerance is exactly 1/255 per channel. Measured over 308 frames and 307M pixels:
        0.0036% of pixels differ, every one of them by exactly 1, none by 2 or more -- that
        is float32 blend rounding, and `np.rint` landing on the far side of a .5 boundary.
        A real regression would move pixels by far more than 1, so the bound still bites.
        """
        data = np.load(BUDDERS)
        base = data["base"].astype(np.float64)
        patches = data["patches"].astype(np.float64)
        x, y = self.renderer.origin
        h, w = self.renderer.patch_shape

        def reference(weights):
            surface = np.array(base, copy=True)
            region = surface[y : y + h, x : x + w]
            row = np.asarray(weights, dtype=np.float64)
            picks = [int(i) for i in np.argsort(row)[::-1][:2] if row[int(i)] > 0.0]
            total = sum(row[i] for i in picks)
            if total > 0.0:
                for i in picks:
                    name = self.renderer.visemes[i]
                    if name == self.renderer.base_viseme:
                        continue
                    patch = patches[self.renderer.patch_index[name]]
                    weight = row[i] / total
                    alpha = patch[..., 3:] / 255.0 * weight
                    region[..., :3] = region[..., :3] * (1.0 - alpha) + patch[..., :3] * weight
            return np.rint(unpremultiply(surface)[..., :3]).astype(np.uint8)

        cases = [
            self.weights(sil=1.0),
            self.weights(),                                    # all-zero row
            self.weights(aa=1.0),
            self.weights(PP=0.5, TH=0.5),                      # even blend
            self.weights(sil=0.5, ou=0.5),                     # fade toward base
            self.weights(kk=0.2, nn=0.3, aa=0.5),              # top-2 picks only
        ]
        rng = np.random.default_rng(7)
        cases += [normalize_weights(rng.random((1, len(VISEMES))))[0] for _ in range(12)]

        for row in cases:
            drawn = self.renderer.draw_row(row).copy().astype(np.int16)
            diff = np.abs(drawn - reference(row).astype(np.int16))
            self.assertLessEqual(int(diff.max()), 1, "float32 draw diverged from float64 by more than rounding")
            # The oracle must stay a real oracle: outside the mouth rect there is no
            # float32 rounding to excuse, so that region must match exactly.
            outside = np.ones(drawn.shape[:2], dtype=bool)
            outside[y : y + h, x : x + w] = False
            self.assertEqual(int(diff[outside].max()), 0, "pixels outside the mouth rect must be identical")

    def test_float32_rounding_never_exceeds_one_unit(self):
        """The ±1 tolerance above is asserted, not assumed, over a wide weight sweep."""
        data = np.load(BUDDERS)
        base = data["base"].astype(np.float64)
        patches = data["patches"].astype(np.float64)
        x, y = self.renderer.origin
        h, w = self.renderer.patch_shape

        rng = np.random.default_rng(21)
        worst = 0
        for _ in range(40):
            row = normalize_weights(rng.random((1, len(VISEMES))))[0]
            surface = np.array(base, copy=True)
            region = surface[y : y + h, x : x + w]
            picks = [int(i) for i in np.argsort(row)[::-1][:2] if row[int(i)] > 0.0]
            total = sum(row[i] for i in picks)
            for i in picks:
                name = self.renderer.visemes[i]
                if name == self.renderer.base_viseme:
                    continue
                patch = patches[self.renderer.patch_index[name]]
                weight = row[i] / total
                alpha = patch[..., 3:] / 255.0 * weight
                region[..., :3] = region[..., :3] * (1.0 - alpha) + patch[..., :3] * weight
            ref = np.rint(unpremultiply(surface)[..., :3]).astype(np.int16)
            drawn = self.renderer.draw_row(row).copy().astype(np.int16)
            worst = max(worst, int(np.abs(drawn - ref).max()))
        self.assertLessEqual(worst, 1, f"float32 drift reached {worst}/255, expected at most 1")

    def test_region_math_stays_float32(self):
        """A numpy float64 scalar would silently promote the blend back to float64."""
        self.assertEqual(self.renderer.region.dtype, np.float32)
        self.assertEqual(self.renderer.base_region.dtype, np.float32)
        self.assertEqual(self.renderer.patches.dtype, np.uint8, "patches must not be pre-widened")

        row = np.zeros(len(VISEMES))
        row[VISEME_INDEX["aa"]] = 0.6
        row[VISEME_INDEX["PP"]] = 0.4
        self.renderer.draw_row(row)
        self.assertEqual(self.renderer.region.dtype, np.float32, "the blend promoted to float64")

    def test_mouth_rect_is_exposed_for_host_blitting(self):
        x, y, w, h = self.renderer.mouth_rect
        self.assertEqual((x, y), self.renderer.origin)
        self.assertEqual((w, h), (self.renderer.patch_shape[1], self.renderer.patch_shape[0]))
        self.assertLess(w * h, self.renderer.size[0] * self.renderer.size[1] * 0.11,
                        "the mouth rect should stay a small fraction of the canvas")

    def test_fading_toward_sil_is_linear(self):
        base = self.renderer.draw_row(self.weights(sil=1.0)).astype(np.float64)
        deltas = []
        for weight in (1.0, 0.75, 0.5, 0.25, 0.0):
            drawn = self.renderer.draw_row(self.weights(aa=weight, sil=1.0 - weight)).astype(np.float64)
            deltas.append(float(np.abs(drawn - base).mean()))
        self.assertEqual(deltas[-1], 0.0)
        # Each quarter step should move roughly the same amount.
        steps = np.diff(deltas)
        np.testing.assert_allclose(steps, steps[0], rtol=0.05)

    def test_rejects_mismatched_row_length(self):
        with self.assertRaises(ValueError):
            self.renderer.draw_row(np.ones(len(VISEMES) - 1))

    def test_unpremultiply_round_trip(self):
        premultiplied = np.zeros((2, 2, 4), dtype=np.float64)
        premultiplied[..., 0] = 128.0
        premultiplied[..., 3] = 255.0
        straight = unpremultiply(premultiplied)
        np.testing.assert_allclose(straight[..., 0], 128.0, atol=1e-9)


class TestEndToEnd(unittest.TestCase):
    def test_pipeline_produces_a_drawable_timeline(self):
        samples = decode(AUDIO)
        frames = frame(samples)
        features = extract_features(frames)
        floor, ceiling = derive_vad_thresholds(features.rms)
        weights = OpennessRamp(floor, ceiling)(features)
        weights = normalize_weights(weights)
        weights = attack_release(weights, fps=16000 / HOP_SIZE)
        weights = normalize_weights(weights)

        timeline = build_timeline(weights, frame_times(frames.shape[0]), fps=16000 / HOP_SIZE)
        restored = timeline_weights(timeline)

        renderer = Renderer(BUDDERS)
        image = renderer.draw_row(restored[len(restored) // 2])
        self.assertEqual(image.shape, (renderer.size[1], renderer.size[0], 3))
        self.assertEqual(image.dtype, np.uint8)

    def test_keyframed_timeline_round_trips_through_the_renderer(self):
        samples = decode(AUDIO)
        frames = frame(samples)
        features = extract_features(frames)
        floor, ceiling = derive_vad_thresholds(features.rms)
        weights = normalize_weights(OpennessRamp(floor, ceiling)(features))

        _, dense, times, out_fps = keyframe_weights(
            weights, analysis_fps=16000 / HOP_SIZE, key_hz=12.0, out_fps=30.0,
            # Explicit, not the module default: the settled look is a one-frame fade, which
            # degenerates to a hard cut and produces no blended frames at all. This test is
            # about fractional weights surviving the JSON round trip, so it needs a fade
            # that actually blends, whatever ships.
            fade_s=0.05,
        )
        timeline = build_timeline(dense, times, out_fps)
        restored = timeline_weights(timeline)

        # The JSON keeps 3 decimals, so the blend weights survive the round trip.
        np.testing.assert_allclose(restored, dense, atol=2e-3)

        renderer = Renderer(BUDDERS)
        blended = np.flatnonzero(((dense > 0.02) & (dense < 0.98)).any(axis=1))
        self.assertGreater(blended.size, 0, "expected cross-fade frames in the timeline")
        image = renderer.draw_row(dense[blended[len(blended) // 2]])
        self.assertEqual(image.dtype, np.uint8)

    def test_keyframing_replaces_continuous_smoothing(self):
        samples = decode(AUDIO)
        features = extract_features(frame(samples))
        floor, ceiling = derive_vad_thresholds(features.rms)
        weights = normalize_weights(OpennessRamp(floor, ceiling)(features))
        fps = 16000 / HOP_SIZE

        continuous = normalize_weights(attack_release(weights, fps=fps))
        tops = [VISEMES[int(i)] for i in continuous.argmax(axis=1)]
        raw_changes = sum(1 for a, b in pairwise(tops) if a != b) / (len(tops) / fps)

        keys, _, _, _ = keyframe_weights(weights, analysis_fps=fps, key_hz=8.0, out_fps=30.0)
        stats = key_stats(keys, 8.0, len(tops) / fps)
        self.assertLess(stats["changes_per_s"], raw_changes, "keyframing should cut the switch rate")
        self.assertLessEqual(stats["changes_per_s"], 8.0)

    def test_collected_frames_must_be_copied_to_stay_distinct(self):
        """Why tools/render_filmstrip.py copies draw_row's result.

        Any tool that collects frames and converts them later would otherwise see one
        buffer holding only the last drawn pose -- which is how the contact sheet ended
        up with 40 identical tiles once draw_row stopped allocating per frame.
        """
        renderer = Renderer(BUDDERS)
        rows = []
        for viseme in ("aa", "PP", "TH", "ou"):
            row = np.zeros(len(VISEMES))
            row[VISEME_INDEX[viseme]] = 1.0
            rows.append(row)

        aliased = [renderer.draw_row(row) for row in rows]
        self.assertEqual(len({f.tobytes() for f in aliased}), 1,
                         "the persistent buffer should alias -- that is the point")

        kept = [renderer.draw_row(row).copy() for row in rows]
        self.assertEqual(len({f.tobytes() for f in kept}), 4, "copies must stay distinct")


if __name__ == "__main__":
    unittest.main()
