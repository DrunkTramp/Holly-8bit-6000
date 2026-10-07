"""The sample-rate contract: resample once, keep both, never re-derive."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hollyhost.session import FACE_RATE
from hollyhost.tts import NullTTS, Prepared, prepare, resample, split_sentences


def tone(freq: float, seconds: float, rate: int, amp: float = 0.6) -> np.ndarray:
    t = np.arange(round(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_freq(samples: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    bins = np.fft.rfftfreq(samples.size, 1.0 / rate)
    return float(bins[int(np.argmax(spectrum))])


class TestResample(unittest.TestCase):
    def test_same_rate_is_a_no_op(self):
        x = tone(440.0, 0.2, 16000)
        np.testing.assert_allclose(resample(x, 16000, 16000), x, atol=1e-6)

    def test_length_is_the_same_duration(self):
        for src, dst in ((22050, 16000), (24000, 16000), (16000, 24000)):
            with self.subTest(src=src, dst=dst):
                x = tone(500.0, 1.0, src)
                y = resample(x, src, dst)
                self.assertAlmostEqual(y.size / dst, x.size / src, places=4)

    def test_pitch_survives_downsampling(self):
        """A sloppy resampler moves formant energy; the classifier then picks different
        visemes and it looks like a lip-sync bug that isn't."""
        for src in (22050, 24000):
            with self.subTest(src=src):
                y = resample(tone(1000.0, 0.5, src), src, FACE_RATE)
                self.assertAlmostEqual(dominant_freq(y, FACE_RATE), 1000.0, delta=40.0)

    def _peak_near(self, samples, rate, freq, width=500.0):
        spec = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
        bins = np.fft.rfftfreq(samples.size, 1.0 / rate)
        band = (bins >= freq - width) & (bins <= freq + width)
        return float(spec[band].max()) if band.any() else 0.0

    def test_stopband_attenuation_is_measured_against_the_input(self):
        """Attenuation in dB relative to the same tone at the input rate.

        Dividing stopband energy by total energy is meaningless once the filter has done
        its job: the total *is* the residue, so the ratio stays near 1 no matter how deep
        the rejection. Compare against what went in instead.
        """
        rate, freq = 22050, 11500.0
        x = tone(freq, 0.4, rate)
        y = resample(x, rate, FACE_RATE)
        in_peak = self._peak_near(x, rate, freq)
        out_peak = self._peak_near(y, FACE_RATE, freq)
        attenuation_db = 20.0 * np.log10(max(in_peak, 1e-12) / max(out_peak, 1e-12))
        self.assertGreater(attenuation_db, 60.0, f"only {attenuation_db:.1f} dB of rejection")

    def test_content_near_the_cutoff_does_not_fold_down(self):
        # 9 kHz is in the skirt. The failure mode that matters is not its amplitude but
        # where it lands: decimating to 16 kHz folds 9 kHz onto 7 kHz.
        rate, freq = 22050, 9000.0
        x = tone(freq, 0.4, rate)
        y = resample(x, rate, FACE_RATE)
        in_peak = self._peak_near(x, rate, freq)
        folded = self._peak_near(y, FACE_RATE, 7000.0, width=400.0)
        attenuation_db = 20.0 * np.log10(max(in_peak, 1e-12) / max(folded, 1e-12))
        self.assertGreater(attenuation_db, 30.0,
                           f"9 kHz folded to 7 kHz with only {attenuation_db:.1f} dB rejection")

    def test_energy_is_preserved(self):
        x = tone(800.0, 0.5, 24000)
        y = resample(x, 24000, FACE_RATE)
        self.assertAlmostEqual(float(np.sqrt(np.mean(y**2))) / float(np.sqrt(np.mean(x**2))),
                               1.0, delta=0.05)

    def test_degenerate_inputs(self):
        self.assertEqual(resample(np.zeros(0, np.float32), 24000, 16000).size, 0)
        self.assertEqual(resample(np.ones(1, np.float32), 24000, 16000).size, 1)


class TestPrepared(unittest.TestCase):
    def test_analysis_is_16k_and_playback_stays_native(self):
        p = prepare("hi", tone(600.0, 0.5, 24000), 24000)
        self.assertEqual(p.analysis.dtype, np.float32)
        self.assertEqual(p.rate, 24000)
        self.assertEqual(p.playback.size, 0.5 * 24000)
        self.assertAlmostEqual(p.analysis.size / FACE_RATE, p.duration, places=4)

    def test_native_16k_is_not_resampled_at_all(self):
        x = tone(600.0, 0.3, FACE_RATE)
        p = prepare("hi", x, FACE_RATE)
        np.testing.assert_array_equal(p.analysis, x)

    def test_the_two_buffers_correspond_in_time(self):
        """The rule the component depends on: one second of analysis is one second of audio."""
        for rate in (16000, 22050, 24000):
            with self.subTest(rate=rate):
                p = prepare("x", tone(700.0, 0.75, rate), rate)
                self.assertIsInstance(p, Prepared)
                self.assertAlmostEqual(p.analysis.size / FACE_RATE, p.playback.size / rate, places=3)

    def test_bad_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare("x", tone(400.0, 0.1, 16000), 0)

    def test_the_face_accepts_what_prepare_produces(self):
        from holly.runtime import HollyFace

        p = prepare("hello there", tone(500.0, 0.8, 22050), 22050)
        u = HollyFace(classifier="openness-ramp", model_path=None).speak(p.analysis, FACE_RATE)
        self.assertAlmostEqual(u.audio_duration, p.duration, places=4)


class TestSentenceSplit(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(split_sentences("One. Two! Three?"), ["One.", "Two!", "Three?"])

    def test_abbreviations_and_decimals_are_not_split(self):
        self.assertEqual(split_sentences("He weighed 12.5 kg."), ["He weighed 12.5 kg."])
        self.assertEqual(split_sentences("Dr. Who said no."), ["Dr. Who said no."])

    def test_whitespace_and_empty(self):
        self.assertEqual(split_sentences("  \n  "), [])
        self.assertEqual(split_sentences("a\nb"), ["a", "b"])

    def test_long_run_without_punctuation_is_cut(self):
        out = split_sentences("word " * 200, max_chars=120)
        self.assertGreater(len(out), 1)
        self.assertTrue(all(len(s) <= 120 for s in out))

    def test_sentences_join_back_to_the_original_words(self):
        text = "Hello there. How are you? I am fine! Really."
        self.assertEqual(" ".join(split_sentences(text)).split(), text.split())


class TestNullTTS(unittest.TestCase):
    def test_produces_a_usable_buffer(self):
        samples, rate = NullTTS().synthesize("hello there friend")
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(samples.ndim, 1)
        self.assertLess(np.abs(samples).max(), 1.0)
        p = prepare("hello there friend", samples, rate)
        self.assertGreater(p.analysis.size, FACE_RATE // 10)

    def test_length_tracks_word_count(self):
        tts = NullTTS()
        short, rate = tts.synthesize("hi")
        long, _ = tts.synthesize("many words here and they keep going on for a while")
        self.assertGreater(long.size, short.size)
        self.assertEqual(rate, 22050)


if __name__ == "__main__":
    unittest.main()
