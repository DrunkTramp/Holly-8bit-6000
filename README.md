# Holly host

The application shell around the [Holly-8bit-6000](../Holly) animation component: a
keyboard-driven Holly that speaks text with a lip-synced, retro pixel-art face in the corner
of the window. You type, a voice synthesises it, Holly's mouth moves in time with the audio
*he himself is playing* — never a microphone.

The animation component is a finished library. This repo consumes it, it does not modify it.

## Integration model

The component is installed editable and imported as a package — no submodule, no `sys.path`
surgery. The host gets its **own venv** so it is not limited by whatever the animation
component's venv happens to have:

```sh
python3 -m venv .venv
.venv/bin/pip install -e ../Holly            # the animation component -> `import holly`
.venv/bin/pip install -e ".[ui,dev,piper]"   # host + pygame + pytest + piper-tts (+ sounddevice binding)
```

`holly.DEFAULT_BUFFERS` / `DEFAULT_MODEL` resolve against the animation repo's own root, so
the host runs from *here* and still finds the baked face and the vendored classifier model.

### Two generated assets (in the animation repo, gitignored)

A fresh clone of `../Holly` has neither, and the host will not render without them:

```sh
cd ../Holly
.venv/bin/python tools/export_visemes.py --scale 0.5 --resample nearest --out build/visemes_pixel
# model/model-en-mixed.bin — refetch per model/README.md
```

`build/visemes_pixel/visemes.npz` is the 576x432 nearest bake the host uses (the settled
corner-element size: the mouth rect is 168x132, ~0.72 ms/draw). Verify the component is
intact before starting: `.venv/bin/python -m unittest discover -s tests -t .` → **147 tests, OK**.

## Running it

The clock is always **frames the output callback has consumed** (`t = frames_played / rate`),
never `time.time()`. That single rule is what keeps mouth and voice locked; everything else in
the host is plumbing around it. Run everything with `.venv/bin/python`.

```sh
# A0 — batch: queue files, play once. Proves the clock, dirty check, zero-draw idle, return-to-rest.
.venv/bin/python -m hollyhost.app --audio ../Holly/test_audio.flac --audio ../Holly/test_audio.flac --gap 0.3

# A1 — interactive: type, Enter to speak. No --audio means the text line.
.venv/bin/python -m hollyhost.app --tts null                       # tone stand-in (no downloads)
.venv/bin/python -m hollyhost.app --tts piper                      # stock voice in voices/ (see below)
.venv/bin/python -m hollyhost.app --tts piper --voice path/to.onnx # a specific voice

# No audio device? Rehearse the identical loop with a hand-driven clock (no sounddevice needed):
.venv/bin/python -m hollyhost.app --audio ../Holly/test_audio.flac --dry
.venv/bin/python -m hollyhost.app --tts piper --dry
```

`sounddevice` is **output only** — there is no input path anywhere.

### Stock Piper voice

`--tts piper` with no `--voice` uses the first `.onnx` in `voices/` (gitignored — it is a
63 MB download). Fetch one with:

```sh
.venv/bin/python -m piper.download_voices en_US-lessac-medium --download-dir voices
```

Verified on `en_US-lessac-medium` (22.05 kHz): the shipping HeadAudio classifier drives the
mouth well on this synthetic speech (~14 pose-changes/s, 9–10 distinct visemes per sentence,
~14 ms analysis) — the HANDOFF's open "classifier on synthetic speech" question, answered
positively for a stock voice. Judge each new voice by eye with `../Holly/tools/reference_player.py`
/ `render_filmstrip.py`; if one gives poor visemes the fix is the classifier's gate/gain knobs,
never the renderer.

### Real audio output needs libportaudio (system package)

The `sounddevice` *binding* installs via pip, but opening a stream needs the native PortAudio
library, which is a system package (root):

```sh
sudo apt-get install -y libportaudio2
```

Without it, `import sounddevice` raises `OSError: PortAudio library not found` and the
interactive UI exits with a clear message. `--dry` runs the identical loop on a hand-driven
clock, so the whole pipeline is testable on a box with no audio hardware.

## The boundary the host owns

The component refuses to be the place where the sample-rate contract gets fudged:
`HollyFace.analyse()` **raises** unless the buffer is 16 kHz mono float32 (the HeadAudio front
end is pinned to it). So `hollyhost/tts.py` owns one rule — **resample once, keep both, never
re-derive**: `prepare()` returns the 16k *analysis* buffer and the native-rate *playback*
buffer. They correspond 1:1 in time, and the queue advances by **duration** (seconds), not
sample count, so 16k and 22.05k buffers of the same audio line up on the clock.

The dirty check, the zero-draw idle, and the no-final-pose-latch rule (`IDLE_ROW` outside
speech) all live in the component. The host must not reimplement them, and must not latch a
final pose when it thinks an utterance is over — `frame_at()` returns `changed=False` for idle,
so the host blits nothing.

## Layout

```
hollyhost/
  app.py       entry point: A0 batch player + A1 interactive text line (pygame, corner face)
  session.py   the pure part: build_session (batch) + LiveSession (interactive) — face queue
               and audio buffer laid out on one clock so they cannot drift
  audio.py     the clock: SounddeviceOutput (batch) + LiveAudio (appendable, freezes at the
               tail while Holly waits for input) + FakeDevice/LiveAudio.advance for tests
  tts.py       the seam: synthesize(text) -> (float32, rate); resample-once/keep-both contract;
               NullTTS / PiperTTS / ScriptTTS; sentence splitting for later streaming
tests/
  test_clock.py           the audio length == face clock invariant, dirty check, return-to-rest
  test_playback_loop.py   the component's seam assertions, driven through the live loop
  test_tts.py             the sample-rate contract + resampler quality
```

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -t .
```

The tests run without a working audio device: `FakeDevice` and `LiveAudio.advance` drive the
clock by hand, and the pure queue logic needs no window. (`sounddevice`'s binding may be
installed but unused by the tests.)

## Status — Phase A

- **A0 (clock proof):** done. Batch queue drives the corner face from the device position;
  measured ~11% of polls cost a draw, idle is zero-draw, the mouth returns to rest on drain.
- **A1 (text to speech):** done. The interactive line synthesises typed text through a real
  stock Piper voice (`--tts piper`), folds it to 16k analysis + native playback, and speaks it
  on the same clock. Verified headless with `--dry`.
- **Not in Phase A:** the LLM (Phase C), the fine-tuned/OmniVoice backends (Phase B), and the
  classifier-quality judgement on real TTS voices (run `../Holly/tools/reference_player.py` and
  `render_filmstrip.py` per voice; if a voice gives poor visemes the fix is the classifier's
  gate/gain knobs, never the renderer).
