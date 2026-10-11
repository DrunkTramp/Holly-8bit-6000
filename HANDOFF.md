# Handoff: Holly host — Phase A (skeleton, no LLM) is done and verified on real hardware

This file is the authoritative state record for the **host** repo. Read it before touching
this code; several decisions here cannot be inferred from the source alone.

- The **animation component** lives in `../Holly` and is **finished** — consume it as a
  library (`import holly`), never rebuild or re-architect it. Its own state record is
  `../Holly/HANDOFF.md`; its plan is `../Holly/viseme-mouth-plan.md`.
- The **whole-program plan** (this host, phases A–E) is `../Holly/holly-host-plan.md`.
- This repo's user-facing doc is `README.md`.

Paste the **Kickoff prompt** at the bottom into a new session.

## Status

| Phase | State |
|---|---|
| A0 — clock proof (batch files → corner face from the device position) | **Done.** `hollyhost/app.py` batch mode, `session.build_session`, `audio.SounddeviceOutput`/`FakeDevice`. |
| A1 — text → stock Piper voice → `speak()` → same sync (interactive text line) | **Done.** `audio.LiveAudio`, `session.LiveSession`, `tts.py` (Piper backend), interactive mode in `app.py`. |
| Real-hardware verification | **Done** (2026-10-07): batch UI 0 underruns; `LiveAudio` stream tracks audio length; `reference_player.py` run on a real device. |
| B — voice abstraction (fine-tuned Piper ONNX / OmniVoice GPU) | Not started. Seam exists (`tts.make_tts`, `synthesize(text)->(float32,rate)`). |
| C — LLM/brain + sentence-splitter → queue | Not started. `tts.split_sentences` and the streaming queue are ready to feed it. |
| D — persona / TTS text preprocessing | Not started. |
| E — polish (barge-in, idle micro-motion, per-viseme gain) | Not started. |

**42 host tests pass:** `.venv/bin/python -m unittest discover -s tests -t .`
**147 animation tests pass** (untouched): `cd ../Holly && .venv/bin/python -m unittest discover -s tests -t .`

## Environment (this box, verified)

- **Dedicated venv at `Holly-host/.venv`** — do NOT run the host under `../Holly/.venv`
  (that one lacks sounddevice/piper/setuptools). Python 3.12.3, pip 26.2.1.
- Installed editable: `holly-8bit-6000` (from `../Holly`), `holly-host[ui,dev,piper]`.
  Versions: numpy 2.5.3, pygame 2.6.1, pytest 9.1.1, sounddevice 0.5.6, piper-tts 1.8.0,
  onnxruntime 1.30.0.
- **`libportaudio2` is installed system-wide** (`/lib/x86_64-linux-gnu/libportaudio.so.2`) —
  the user ran `sudo apt-get install -y libportaudio2` (2026-10-07). Without it `import
  sounddevice` raises `OSError: PortAudio library not found`; the interactive UI then exits
  with a clear message and `--dry` still works.
- Default audio output is **PipeWire** (`sd.default.device[1] == 11`, "default", 44.1k). The
  host plays at 16k/22.05k; PortAudio/PipeWire resamples transparently (verified, 0 underruns).
- **Two stock Piper voices in `voices/`** (gitignored, 63 MB each, both 22050 Hz):
  `en_GB-alan-medium.onnx` (British male — apt for Holly; the alphabetical default for
  `--tts piper`) and `en_US-lessac-medium.onnx`. Fetch more with
  `.venv/bin/python -m piper.download_voices <name> --download-dir voices`.
- **Two generated assets in `../Holly` are required and gitignored there** — a fresh clone has
  neither: `build/visemes_pixel/visemes.npz` (the 576x432 nearest bake `holly.DEFAULT_BUFFERS`
  points at; re-create with `tools/export_visemes.py --scale 0.5 --resample nearest --out
  build/visemes_pixel`) and `model/model-en-mixed.bin` (refetch per `../Holly/model/README.md`).
- `ffmpeg` is present (the offline `decode` path shells out to it).

## The clock architecture — settled, do not re-litigate

The whole design rests on one rule, inherited from the component: **`t` is frames the output
callback has actually consumed (`t = frames_played / rate`), never `time.time()`.** Two
host-specific decisions make that work for *interactive* playback (utterances arriving over
time), and they are the subtle part:

1. **`LiveAudio` freezes the clock at the tail.** The callback emits real samples and advances
   `fill` by *only what was real*; when the buffer is exhausted it outputs silence but does NOT
   advance `fill`. So while Holly waits for the next utterance the clock sits at the tail
   (`row_at` → `IDLE_ROW`, zero-draw) and **never runs ahead of the audio**. This is why the
   clock cannot free-run on wall time in an interactive app.
2. **`LiveSession.say()` pins the face `at` to the audio buffer position.** Each utterance's
   face-clock start is exactly the byte position its playback audio was appended at, divided by
   the rate — so mouth and voice cannot drift *by construction*, not by luck. At 16k (batch,
   `build_session`) the audio tail and `face.queued_duration` are equal to 9 decimal places;
   at 22.05k (foreign-rate TTS) they agree to ~3 places (sub-sample duration rounding), which
   is fine because the queue advances by **duration**, not sample count.

**Threading:** the `HollyFace` queue is touched only from the UI thread (`say`/`poll`); the
audio callback runs on the PortAudio thread and only reads the `LiveAudio` buffer (guarded by
its own lock). That is why there is no lock around the face queue — do not add one, and do not
call face methods from the callback.

**The component owns the three runtime behaviours** — dirty check, zero-draw idle, and the
no-final-pose-latch rule (`IDLE_ROW` outside speech). `frame_at()` returns `changed=False` for
idle; the host blits nothing then. **The host must not reimplement these, and must not latch a
final pose when it thinks an utterance is over.**

## The sample-rate boundary (the thing that is easy to get wrong)

`HollyFace.analyse()` **raises** unless the buffer is 16 kHz mono float32 (the HeadAudio front
end is pinned to it). `tts.prepare()` owns the rule **resample once, keep both, never
re-derive**: it returns the 16k *analysis* buffer AND the native-rate *playback* buffer, which
correspond 1:1 in time. Playback may stay native because the queue advances by duration. The
resampler (`tts.resample`) is band-limited windowed-sinc, pure numpy (no scipy), block-wise
(~90 MB avoided per 7 s utterance), unit-DC-gain normalised.

## Bugs found and pinned (do not reintroduce)

- **numpy 2.5 will not hold a `.name` attribute** on an ndarray, and `np.reshape` drops it
  anyway. `build_session` therefore takes `names=[...]` explicitly (the old `samples.name`
  trick silently fell back to `utterance-N`).
- The animation component's `tools/render_filmstrip.py` **reads a timeline from `--timeline`**
  (default `build/debug/timeline.json`, i.e. the stale `test_audio` one) and uses `--audio`
  only for the waveform band + mux. To filmstrip a *new* utterance you must run
  `make_timeline.py --audio <wav> --out <t.json>` first, then `render_filmstrip.py --timeline
  <t.json> --audio <wav>`. (A wrong, mismatched strip is otherwise easy to produce.)

## Measured on this box (not assumed)

**Cross-voice classifier spot-check** (`tools/voice_check.py`, shipping config: headaudio +
auto `le` gate + gain, 12/s @ 30 fps) — this answers the animation HANDOFF's open
*"classifier on synthetic speech"* question for stock voices:

| voice | changes/s | idle | visemes/sentence | verdict |
|---|---|---|---|---|
| `en_GB-alan-medium` | 8.0 | 47% | 9–12 | OK |
| `en_US-lessac-medium` | 7.4 | 54% | 9–11 | OK |

Both match the component's real-speech baseline (~7 changes/s at 12/s). `mean top weight` is
1.000 (the settled one-frame fade is a hard cut → pure one-hot rows). Analysis ~4.6 ms per
second of speech.

**Real-device verification** (PipeWire default output):
- Batch UI: 119 polls / 12 draws (**10.1%**) / **0 underruns**, clock driven by the callback.
- `LiveAudio` stream: a 2.15 s Piper utterance tracked to within 2 ms, drained to idle,
  `last_status=None` (no PortAudio errors).
- `reference_player.py --audio /tmp/holly_alan.wav` on the real device: 295 polls / 44 draws
  (**14.9%**, higher than test_audio because the voice has more pose variety), return-to-rest.

## File map

```
README.md                 user-facing: setup, run commands, Phase A status
hollyhost/
  app.py                  entry point. `--audio …` = batch (A0); no --audio = interactive text
                          line (A1). `--dry` = hand-driven clock, no device/window.
  audio.py                the clock: SounddeviceOutput (batch), LiveAudio (appendable, freezes
                          at tail), FakeDevice + LiveAudio.advance (tests/--dry).
  session.py              pure part: build_session (batch) + LiveSession (interactive) — face
                          queue and audio tail laid out on one clock so they cannot drift.
  tts.py                  the seam: synthesize(text)->(float32,rate); resample-once/keep-both;
                          NullTTS / PiperTTS / ScriptTTS; make_tts factory; split_sentences.
tools/
  voice_check.py          cross-voice classifier spot-check (changes/s, idle %, visemes).
tests/
  test_clock.py           audio length == face clock invariant; dirty check; return-to-rest.
  test_playback_loop.py   the component's seam assertions, driven through the LIVE loop
                          (reachable frames, no sample past audio, gaps/tail idle,
                          prune-no-drift) + the two-rate multi-utterance seam test.
  test_tts.py             sample-rate contract + resampler quality + sentence splitting.
  test_app.py             app.run() --dry dispatch smoke (batch + interactive).
```

## Reproduce / verify everything

```sh
cd Holly-host
.venv/bin/python -m unittest discover -s tests -t .                 # 42 tests
.venv/bin/python tools/voice_check.py                               # cross-voice metrics
# batch clock proof, no device:
.venv/bin/python -m hollyhost.app --audio ../Holly/test_audio.flac --audio ../Holly/test_audio.flac --gap 0.3 --dry
# real device (needs libportaudio2): type, Enter to speak
.venv/bin/python -m hollyhost.app --tts piper
```

## Open items (in rough priority order)

1. **The look has not been judged by a human.** Sync geometry is measured and the code runs on
   real hardware, but whether **hard-cut pose changes read as Holly next to a real TTS voice**
   is a person's call. Run the two commands above on the box with speakers and decide. If it
   feels stiff, the levers are the classifier's gate/gain knobs and `--fade-shape`
   (`center`==`lag` now; `lead` anticipates one frame) — **never the renderer.**
2. **Phase B — voice abstraction.** Wire the fine-tuned Piper ONNX (CPU) and OmniVoice (GPU)
   behind `make_tts`/`synthesize`. Spot-check each new voice with `tools/voice_check.py` +
   a filmstrip before trusting it.
3. **Phase C — LLM/brain.** llama.cpp + chat loop + `split_sentences` feeding `LiveSession.say`
   per sentence. The queue already handles streamed multi-utterance input on two rates
   (pinned by `test_playback_loop.py`).
4. **3b streaming-TTS refactor** — only if the host adopts chunked TTS that plays as it arrives
   (analysis currently always runs ahead of playback, which is what keeps the auto gate +
   `center` fade valid live).
5. **Phase D persona; Phase E polish** (barge-in, idle micro-motion, per-viseme gain curves).

## Kickoff prompt

> Continue the **Holly host** at `/home/irreverend/Projects/Holly-host`. Read this repo's
> `HANDOFF.md` first (authoritative state), then `README.md`, then `../Holly/HANDOFF.md`
> (the finished animation component — consume it, never modify it). Phase A (skeleton, no LLM)
> is **done and verified on real hardware**; 42 host tests + 147 animation tests pass.
>
> Verify before anything else: `.venv/bin/python -m unittest discover -s tests -t .` → 42 OK,
> and `cd ../Holly && .venv/bin/python -m unittest discover -s tests -t .` → 147 OK. The host
> runs under its **own** `.venv` (not `../Holly/.venv`). `libportaudio2` is installed and two
> stock Piper voices are in `voices/` (gitignored).
>
> Do not re-litigate the settled clock architecture (see "The clock architecture"): `t` is the
> device's consumed-frame position; `LiveAudio` freezes at the tail; `LiveSession` pins each
> utterance's face `at` to its audio buffer position; the component owns dirty-check / zero-draw
> idle / no-final-pose-latch. Do not reimplement those in the host.
>
> Then pick up the first open item: either run the by-eye look check on the real device (needs
> a human), or start **Phase C** (the LLM/brain) — `tts.split_sentences` and the
> streaming-capable queue are ready to be fed.
