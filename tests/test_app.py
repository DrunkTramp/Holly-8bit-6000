"""App-level smoke tests: the CLI dispatch and the --dry loop, end to end.

The clock/queue/dirty-check logic is covered by test_clock.py and test_playback_loop.py;
this guards the *glue* — that `app.run()` picks the right mode, wires build_session + the
renderer + the dry loop together, and returns 0 with the mouth back at rest. `--dry` needs
no audio device and no window, so it runs anywhere the bake and the model exist.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holly.runtime import DEFAULT_BUFFERS, DEFAULT_MODEL

from hollyhost import app
from hollyhost import brain as app_brain

ANIM = Path(__file__).resolve().parents[1].parent / "Holly"
AUDIO = ANIM / "test_audio.flac"


@unittest.skipUnless(DEFAULT_BUFFERS.exists() and DEFAULT_MODEL.exists() and AUDIO.exists(),
                     "needs the host bake, the model, and test_audio.flac")
class TestAppDry(unittest.TestCase):
    def _run(self, argv):
        # Both streams: the dry harness reports a cut-short rehearsal on stderr.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = app.run(argv)
        return rc, buf.getvalue()

    def test_batch_dry_returns_zero_and_rests(self):
        rc, log = self._run(["--audio", str(AUDIO), "--audio", str(AUDIO), "--gap", "0.3", "--dry"])
        self.assertEqual(rc, 0)
        self.assertIn("final pose 'sil'", log, "the batch dry loop did not return the mouth to rest")
        # Two utterances queued, and the draw share is the low dirty-checked number, not ~100%.
        self.assertIn("queued 2 utterances", log)

    def test_no_audio_selects_interactive(self):
        # Interactive --dry advances the clock in real time; the null tone is ~1.3 s, so give
        # the loop room to drain and rest. This only proves the dispatch + loop run.
        rc, log = self._run(["--tts", "null", "--dry", "--seconds", "5"])
        self.assertEqual(rc, 0)
        self.assertIn("interactive", log)

    def test_brain_chat_dry_returns_zero_and_rests(self):
        """Phase C wiring end to end, without a model: prompt -> chat thread -> queue -> clock.

        The script brain is patched to a two-sentence reply so the turn is ~1 s of null tone.
        What this pins is the *threading contract*: the pump does brain + TTS off the UI thread,
        and the dry loop (the UI thread) is the only thing that calls `session.say` -- so the
        face `at` stays pinned to the audio position and the mouth still returns to rest.
        """
        with unittest.mock.patch.object(app_brain.ScriptBrain, "DEFAULT_SCRIPT",
                                       ["Short one. Short two."]):
            rc, log = self._run(["--brain", "script", "--tts", "null", "--dry", "--seconds", "8"])
        self.assertEqual(rc, 0)
        self.assertIn("brain=script", log)
        self.assertIn("Short one.", log, "the chat thread never reached the utterance queue")
        self.assertIn("Short two.", log)
        self.assertIn("final pose 'sil'", log, "the mouth did not return to rest after the reply")

    def test_dry_budget_expiry_is_reported_not_asserted(self):
        """A --dry run that runs out of time mid-reply must say so, not fail the rest check.

        The mouth is legitimately mid-vowel when the budget expires; claiming "the mouth did not
        return to rest" would point at a sync bug that does not exist.
        """
        rc, log = self._run(["--brain", "script", "--tts", "null", "--dry", "--seconds", "0.6"])
        self.assertEqual(rc, 0)
        self.assertIn("TRUNCATED", log)
        self.assertIn("Raise --seconds", log)

    def test_interactive_dry_barge_in_cuts_and_rests(self):
        """Phase E barge-in end to end (headless): the cut stops the clock, drops the queue,
        and the mouth returns to rest -- the same assertions the live Tab key drives."""
        with unittest.mock.patch.object(app_brain.ScriptBrain, "DEFAULT_SCRIPT",
                                       ["First sentence here. Second sentence follows on the clock."]):
            rc, log = self._run(["--brain", "script", "--tts", "null", "--dry",
                                 "--barge-at", "0.8", "--seconds", "8"])
        self.assertEqual(rc, 0)
        self.assertIn("BARGE-IN", log, "the barge-in never fired")
        self.assertIn("final pose 'sil'", log, "the mouth did not return to rest after the cut")


@unittest.skipUnless(DEFAULT_MODEL.exists(), "needs the classifier model for analyse()")
class TestCalibrate(unittest.TestCase):
    """The first-run `--calibrate` flow: measure -> derive -> write, model-free (script + null)."""

    def _run(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = app.run(argv)
        return rc, buf.getvalue()

    def test_calibrate_measures_and_writes_a_config(self):
        with tempfile.TemporaryDirectory() as d:
            cfgp = Path(d) / "cfg.json"
            rc, log = self._run(["--calibrate", "--brain", "script", "--tts", "null",
                                 "--config", str(cfgp)])
            self.assertEqual(rc, 0)
            self.assertIn("recommended defaults", log)
            self.assertTrue(cfgp.exists(), "calibrate did not write the config")
            cfg = json.loads(cfgp.read_text())
            self.assertIn("max_chars", cfg)
            self.assertIn("max_tokens", cfg)
            self.assertFalse(cfg["measured"]["stall"], "script+null should keep the queue ahead")

    def test_calibrate_no_write_previews_without_writing(self):
        with tempfile.TemporaryDirectory() as d:
            cfgp = Path(d) / "cfg.json"
            rc, log = self._run(["--calibrate", "--brain", "script", "--tts", "null",
                                 "--no-write", "--config", str(cfgp)])
            self.assertEqual(rc, 0)
            self.assertIn("--no-write", log)
            self.assertFalse(cfgp.exists(), "--no-write still wrote the config")

    def test_calibrate_without_a_brain_is_rejected(self):
        rc, log = self._run(["--calibrate", "--brain", "none", "--tts", "null"])
        self.assertEqual(rc, 2)
        self.assertIn("needs a brain", log)


class TestConfigDefaults(unittest.TestCase):
    """A calibration config fills unset flags; an explicit flag always wins. Pure, no model."""

    def test_config_fills_unset_flags(self):
        with tempfile.TemporaryDirectory() as d:
            cfgp = Path(d) / "cfg.json"
            cfgp.write_text(json.dumps({"max_chars": 100, "max_tokens": 60, "gpu_layers": 20}))
            args = app.parse_args(["--config", str(cfgp)])
            self.assertIsNone(args.max_chars, "unset flags must start as None so config can fill")
            app._apply_config_defaults(args)
            self.assertEqual(args.max_chars, 100)
            self.assertEqual(args.max_tokens, 60)
            self.assertEqual(args.gpu_layers, 20)

    def test_explicit_flag_beats_the_config(self):
        with tempfile.TemporaryDirectory() as d:
            cfgp = Path(d) / "cfg.json"
            cfgp.write_text(json.dumps({"max_chars": 100}))
            args = app.parse_args(["--config", str(cfgp), "--max-chars", "150"])
            app._apply_config_defaults(args)
            self.assertEqual(args.max_chars, 150)

    def test_no_config_keeps_the_built_in_defaults(self):
        args = app.parse_args(["--config", "/nonexistent/holly-hoster.json"])
        app._apply_config_defaults(args)
        self.assertEqual(args.max_chars, 220)
        self.assertEqual(args.max_tokens, 200)
        self.assertEqual(args.gpu_layers, -1)


class TestUiSettings(unittest.TestCase):
    """The persistent settings file: saved panel state fills unset flags; explicit flags win."""

    def _file(self, d, payload):
        p = Path(d) / "ui.json"
        p.write_text(json.dumps(payload))
        return p

    def test_saved_settings_fill_unset_flags(self):
        from hollyhost.settings import load_settings, save_settings

        with tempfile.TemporaryDirectory() as d:
            sp = self._file(d, {"tts": "piper", "brain": "http", "host": "box.local",
                                "port": "1234", "http_model": "m1", "api_key": "k",
                                "llama_model": str(Path(d) / "m.gguf"),
                                "idle_motion": True, "idle_weight": 0.4})
            args = app.parse_args(["--settings", str(sp)])
            app._apply_ui_settings(args)
            self.assertEqual(args.tts, "piper")
            self.assertEqual(args.brain, "http")
            self.assertEqual(args.base_url, "http://box.local:1234")
            self.assertEqual(args.http_model, "m1")
            self.assertEqual(args.api_key, "k")
            self.assertEqual(args.model, Path(d) / "m.gguf")
            self.assertTrue(args.idle_motion)
            self.assertEqual(args.idle_weight, 0.4)

            save_settings({"tts": "null"}, sp)
            self.assertEqual(load_settings(sp), {"tts": "null"})  # round trip

    def test_explicit_flags_beat_saved_settings(self):
        with tempfile.TemporaryDirectory() as d:
            sp = self._file(d, {"tts": "piper", "brain": "llama", "idle_weight": 0.4})
            args = app.parse_args(["--settings", str(sp), "--tts", "null", "--brain", "none",
                                   "--idle-weight", "0.1"])
            app._apply_ui_settings(args)
            self.assertEqual(args.tts, "null")
            self.assertEqual(args.brain, "none")
            self.assertEqual(args.idle_weight, 0.1)
            self.assertTrue(args.model is None)

    def test_no_settings_file_keeps_the_built_in_defaults(self):
        args = app.parse_args(["--settings", "/nonexistent/holly-ui-settings.json"])
        app._apply_ui_settings(args)
        self.assertEqual(args.tts, "null")
        self.assertEqual(args.brain, "none")
        self.assertIsNone(args.base_url)
        self.assertFalse(args.idle_motion)
        self.assertEqual(args.idle_weight, 0.25)

    def test_saved_settings_outrank_the_calibration_config(self):
        """A model chosen in the panel beats the config's suggested one."""
        with tempfile.TemporaryDirectory() as d:
            cfgp = Path(d) / "cfg.json"
            cfgp.write_text(json.dumps({"model": "/config/suggested.gguf"}))
            sp = self._file(d, {"llama_model": "/panel/chosen.gguf"})
            args = app.parse_args(["--config", str(cfgp), "--settings", str(sp)])
            app._apply_ui_settings(args)
            app._apply_config_defaults(args)
            self.assertEqual(args.model, Path("/panel/chosen.gguf"))


if __name__ == "__main__":
    unittest.main()
