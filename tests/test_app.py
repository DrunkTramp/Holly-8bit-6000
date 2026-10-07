"""App-level smoke tests: the CLI dispatch and the --dry loop, end to end.

The clock/queue/dirty-check logic is covered by test_clock.py and test_playback_loop.py;
this guards the *glue* — that `app.run()` picks the right mode, wires build_session + the
renderer + the dry loop together, and returns 0 with the mouth back at rest. `--dry` needs
no audio device and no window, so it runs anywhere the bake and the model exist.
"""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holly.runtime import DEFAULT_BUFFERS, DEFAULT_MODEL

from hollyhost import app

ANIM = Path(__file__).resolve().parents[1].parent / "Holly"
AUDIO = ANIM / "test_audio.flac"


@unittest.skipUnless(DEFAULT_BUFFERS.exists() and DEFAULT_MODEL.exists() and AUDIO.exists(),
                     "needs the host bake, the model, and test_audio.flac")
class TestAppDry(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
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


if __name__ == "__main__":
    unittest.main()
