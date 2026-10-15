# Handoff: Holly host — Phases A, C, D and E done and verified (E accepted by the user 2026-10-13); B partly unblocked (OmniVoice GPU clone working), still blocked on a better voice corpus

This file is the authoritative state record for the **host** repo. Read it before touching
this code; several decisions here cannot be inferred from the source alone.

- The **animation component** lives in `holly-8bit-6000/` and is **finished** — consume it as a
  library (`import holly`), never rebuild or re-architect it. Its own state record is
  `holly-8bit-6000/HANDOFF.md`; its plan is `holly-8bit-6000/viseme-mouth-plan.md`.
- The **whole-program plan** (this host, phases A–E) is `holly-8bit-6000/holly-host-plan.md`.
- This repo's user-facing doc is `README.md`.

Paste the **Kickoff prompt** at the bottom into a new session.

## Status

| Phase | State |
|---|---|
| A0 — clock proof (batch files → corner face from the device position) | **Done.** `hollyhost/app.py` batch mode, `session.build_session`, `audio.SounddeviceOutput`/`FakeDevice`. |
| A1 — text → stock Piper voice → `speak()` → same sync (interactive text line) | **Done.** `audio.LiveAudio`, `session.LiveSession`, `tts.py` (Piper backend), interactive mode in `app.py`. |
| Real-hardware verification | **Done** (2026-10-07): batch UI 0 underruns; `LiveAudio` stream tracks audio length; `reference_player.py` run on a real device. |
| B — voice abstraction (fine-tuned Piper ONNX / OmniVoice GPU) | **Partly unblocked** (2026-10-11): the OmniVoice GPU clone backend is built and measured — `tts.OmniVoiceTTS`, `--tts omnivoice`, cloning `voices/holly_base.flac` (see "OmniVoice on the GPU, measured"). Still **blocked on data** for a better reference clip and for a fine-tuned Piper voice. Seam unchanged (`tts.make_tts`, `synthesize(text)->(float32,rate)`). |
| C — LLM/brain + sentence-splitter → queue | **Done** (2026-10-11, run by the user): `--brain llama` with a local GGUF + `--tts piper` — the model answered, the voice played, **the mouth was in sync**. The llama tier is **measured and settled**: `--max-chars 120`, 6.9 s queue margin, first audio ~0.3–0.5 s, no latency-driven `--max-tokens` cap needed (see "The llama tier, measured"). |
| D — persona / TTS text preprocessing | **Done and verified** (2026-10-11): `hollyhost/normalize.py` (`preprocess`) expands numbers/decimals/unit-slashes/symbols/abbreviations and strips leaked markdown **after** splitting, **before** synthesis; wired into `SentencePump` and the typed `LiveSession.say` path (caption keeps the original text), with `--no-normalize` to opt out. `DEFAULT_PERSONA` rewritten for spoken output. Preprocessing tested (28 cases). The user ran the persona with `--tts omnivoice` and judged it **sounded like Holly** — with the model not yet fine-tuned, the persona is good enough for now; a real fine-tune is the later lever (see Open items). |
| E — polish (barge-in, idle micro-motion, per-viseme gain) | **Done and accepted** (2026-10-13). Barge-in: Tab (or Space on an empty line) in the live UI; `--barge-at <s>` headless — `LiveAudio.flush()` + `HollyFace.reset()` + `SentencePump.interrupt()`; the pinning rule survives the cut. Idle micro-motion: opt-in `--idle-motion` (≤ 2 draws/s overlay, off by default; default peak weight **0.25** — 0.12 was measured invisible and the user reported seeing nothing; `--idle-weight` is the dial). Per-viseme gain: the component's `visemeMaxs` curve is exposed via `--no-gain`/`--key-hz`/`--fade-shape`; *custom* curves would be component-side work — not done. Ran on the real device with piper; the user accepted the look and behaviour as-is. |

**135 host tests pass** (~15 s): `.venv/bin/python -m pytest -q`
**147 animation tests pass** (untouched): `cd holly-8bit-6000 && ../.venv/bin/python -m pytest -q`

## Environment (this box, verified)

- **Dedicated venv at `Holly-host/.venv`** — the only venv in this repo; the component is
  installed into it editable from `holly-8bit-6000/`. Python 3.12.3, pip 26.2.1.
- Installed editable: `holly-8bit-6000` (from `./holly-8bit-6000`, with its `[build]` extra for the PSD bake), `holly-host[ui,dev,piper]`, plus
  **`llama-cpp-python` 0.3.36** in the host venv (user-installed 2026-10-11 with the CUDA
  `CMAKE_ARGS` below). Versions: numpy 2.5.3, pygame 2.6.1, pytest 9.1.1, sounddevice 0.5.6,
  piper-tts 1.8.0, onnxruntime 1.30.0.
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
- **Two generated assets in `holly-8bit-6000/` are required and gitignored there** — a fresh clone
  has neither: `build/visemes_pixel/visemes.npz` (the 576x432 nearest bake `holly.DEFAULT_BUFFERS`
  points at; re-create with `cd holly-8bit-6000 && ../.venv/bin/python tools/export_visemes.py
  --scale 0.5 --resample nearest --out build/visemes_pixel`) and `model/model-en-mixed.bin`
  (refetch per `holly-8bit-6000/model/README.md`).
- `ffmpeg` is present (the offline `decode` path shells out to it).
- **GPU / LLM side:** RTX 5060 Ti **16 GB**, driver 595.91.07, CUDA 13.0 toolkit at
  `/usr/local/cuda` with **`nvcc` NOT on `PATH`**, 31 GB RAM, 16 cores. `g++`, `make`, `curl`
  present; **no `cmake`** (pip build isolation supplies it). The GPU build of llama-cpp-python
  **works on this hardware**: `llama_cpp.llama_supports_gpu_offload()` is True and ggml reports
  the card as **compute capability 12.0** — which is why `-DCMAKE_CUDA_ARCHITECTURES=120` is the
  right flag, not a guess. `~/LLMs/` is an LM Studio/Windows download dir (`.exe`/`.dll`) but
  holds usable GGUFs — the one used for the Phase C run is
  `unsloth/Qwen2.5-VL-3B-Instruct-GGUF/Qwen2.5-VL-3B-Instruct-Q4_K_S.gguf` (1.8 GB, fits VRAM);
  `lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf` (5.6 GB) is the next candidate;
  `~/LLMs/Qwen3.6-35B-A3B-Q6_K.gguf` is 28 GB and will not fit in 16 GB.

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

## The chat thread (Phase C) — settled, do not re-litigate

`SentencePump` (`brain.py`) owns one turn at a time on a **daemon worker thread**: brain deltas →
`SentenceSplitter` → `tts.synthesize` → a `queue.Queue` of `(text, samples, rate)`. Two rules:

1. **The worker may run the brain and the voice; it must never touch `HollyFace` or the audio
   clock.** The UI thread drains `pump.sentences()` in its own poll loop and calls
   `session.say(text, samples=..., rate=...)` there. That is what keeps the settled rule — face
   `at` pinned to the audio append position — true for streamed replies too. Same reason there is
   no lock around the face queue.
2. **The splitter has one rule set, shared by the batch and the stream.** `SentenceSplitter`
   (`feed(delta)` / `end()`) and `split_sentences` (a thin wrapper over it, `junk="pass"`) must
   produce identical utterances for the same text however it is chunked — `tests/test_brain.py`
   asserts exactly that (whole / every-N-chars / word-wise / straddling-a-decimal chunkings).
   The subtle part: **a terminator at the buffer tail is *undecided*, not a boundary**, so
   `"12."` + `"5"` never splits `12.5`. EOS acts as whitespace, a newline forces an emit,
   whitespace collapses across chunk seams. Streaming uses `junk="hold"` (never emit a ≤1-char
   utterance); batch keeps `junk="pass"` so its documented contract is unchanged.

`--brain` defaults to **`none`** = "speak exactly what you typed", which is the pinned A1
behaviour; the brain is built *before* the window opens so a missing dependency is a message,
not a dead pygame window (rc 2).

## The barge-in (Phase E) — settled, do not re-litigate

Barge-in is the plan's three words — *stop playback, stop the queue, back to idle* — and each
part rides an existing seam:

1. **`LiveAudio.flush()` truncates at the *consumed* position, it does not empty the buffer.**
   The clock stays where the device is, `idle()` becomes true at once, and the next `append`
   starts at exactly that frozen position — so `say()`'s pinning rule (face `at` == the byte
   position the audio landed on) holds across a cut with no special case. The callback keeps
   padding silence; the clock stays frozen. That is why the cut cannot just zero the samples.
2. **The face half is `HollyFace.reset()`** — the component's own queue-clear that also clears
   the dirty check, so the next `frame_at(IDLE_ROW)` reports `changed=True` and redraws the
   mouth at rest exactly once. The host adds no return-to-rest logic of its own.
3. **`SentencePump.interrupt()` never blocks the UI thread** (unlike `stop()`'s join): cancel
   + drain, join ≤ 0.25 s best-effort. The race is closed twice — `_produce` drops any sentence
   that finishes synthesising after the cancel, and `sentences()` drains-and-discards while
   cancelled, so nothing from a cut turn can land on a clock that has just been truncated.
   `interrupted` (a property) is what stops the UI showing "thinking…".

## The idle overlay (Phase E) — settled, do not re-litigate

`--idle-motion` is an **opt-in host-side overlay** (`hollyhost/idle_motion.py`); with the flag
off the host is byte-identical to the pre-E host and the component's zero-draw idle stands.
The overlay's invariant: **whenever it is not mid-breath, the renderer's persistent frame shows
exactly the pose the component's dirty check believes it shows.** Hence the poll order in
`_interactive_ui`: `breathe.poll(active=face.pending > 0)` runs **before** `session.poll()`, and
if speech has just been queued the overlay first restores the pure idle row. Skip that order and
the component legitimately skips a draw (its row equals its `_last_row`) while the screen still
holds a breath pose. It runs on **wall time on purpose** — the audio clock freezes at the tail
while idle, and a breath is decoration, not something to sync. Throttled to ≤ 1 draw per step
(0.5 s default) with the overlay's own last-row check; the plan's ceiling is ~1-2 draws/s.
**The weight is measured, not guessed:** on the host bake, `aa` at 0.12 moves the mouth rect a
mean 0.88/255 — effectively invisible, and the first shipped default drew correctly while users
saw nothing. Default is now **0.25** (mean 1.8/255, ~15% of mouth pixels shift ≥ 4); speech
poses sit at 0.65 (mean 4.6); `--idle-weight` is the by-eye dial between.

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
- **A brain exception must salvage the buffered sentence.** If the stream dies mid-reply there is
  no trailing whitespace, so the last sentence is still an *undecided* boundary and would be
  silently dropped — the `except` branch calls `_finish(splitter)` so the user hears what the
  model actually managed to say.
- `SentenceSplitter._scan` is the **int cursor**, so the scanning method is `_scan_boundaries()`.
  Do not "clean up" the name back.
- `LlamaCppBrain.__init__` checks the GGUF path **before** `from llama_cpp import Llama`, so a
  missing file reports as a missing file instead of being masked by the missing module. It must
  also really store `self.max_tokens`.
- `_trim()` runs **after** the assistant message is appended, or the history keeps one turn too
  many (`history_turns=2` grows to 6 messages).
- **`Renderer.draw_row` normalises the top-2 picks by their sum**, so a bare `{aa: 0.12}` weight
  row renders the patch at **full alpha**. A subtle overlay row must put the complement on `sil`
  explicitly (`sil: 1-w, aa: w`) — pinned by `test_idle_motion.py`'s sum-to-1 assertion.
- **`LiveAudio.flush()` truncates, never clears.** Clearing to empty would make the next
  `append` return start 0 — *behind* the frozen clock — and silently break the pinning rule
  after a barge-in. `test_barge_in.py::test_next_append_pins_to_the_frozen_clock` pins this.
- **A clause cut must keep its comma with the head clause.** `--max-chars 120` against a live model
  produced an utterance starting with `", which is quite impressive…"` — the separator belonged to
  the tail. Fixed in `_clip_long`, pinned by both `test_tts.py` (no chunk starts with ``,;:``) and
  `test_brain.py` (the comma-cut path obeys the stream/batch equivalence under every chunking).

## Measured on this box (not assumed)

**Piper: in-process vs CLI** — this is why `make_tts("piper")` returns `PiperVoiceTTS`:

| path | cost |
|---|---|
| `piper` **CLI subprocess** per sentence (`PiperTTS`) | **1.42 s** (nearly all of it reloading the model) |
| in-process `PiperVoice.load()` | 0.85 s, **once**, at startup |
| in-process `voice.synthesize()` | **0.07–0.21 s** per 0.65–7.8 s of audio (~30× realtime) |

The CLI costs **more than a short sentence takes to play**, so a streamed reply starves the queue:
`brain_check.py` reported `STALL 0.51 s before sentence 2` with it. After the in-process switch the
same run reads `OK queue stays ahead`, first audio **74–158 ms** after a sentence is ready,
analysis 7–20 ms. Piper's Python API (piper-tts 1.8.0): `PiperVoice.load(path)`,
`voice.config.sample_rate`, `voice.synthesize(text, syn_config=SynthesisConfig(...))` yielding
chunks whose array attribute is **`chunk.audio_float_array`** (float32 — the exact
`synthesize(text)->(float32,rate)` contract, no int16 round-trip).

**Cross-voice classifier spot-check** (`tools/voice_check.py`, shipping config: headaudio +
auto `le` gate + gain, 12/s @ 30 fps) — this answers the animation HANDOFF's open
*"classifier on synthetic speech"* question for stock voices:

| voice | changes/s | idle | visemes/sentence | verdict |
|---|---|---|---|---|
| `en_GB-alan-medium` | 7.2 (7.0–8.2 over runs) | 41–47% | 9–12 | OK |
| `en_US-lessac-medium` | 7.6 (7.2–7.9 over runs) | 45–58% | 9–11 | OK |

Both match the component's real-speech baseline (~7 changes/s at 12/s). **Piper samples its own
prosody noise, so these move ±0.7 changes/s between runs** — one run is a spot-check, not a
benchmark. `mean top weight` is 1.000 (the settled one-frame fade is a hard cut → pure one-hot
rows). Analysis ~4.6 ms per second of speech.

**Real-device verification** (PipeWire default output):
- Batch UI: 119 polls / 12 draws (**10.1%**) / **0 underruns**, clock driven by the callback.
- `LiveAudio` stream: a 2.15 s Piper utterance tracked to within 2 ms, drained to idle,
  `last_status=None` (no PortAudio errors).
- `reference_player.py --audio /tmp/holly_alan.wav` on the real device: 295 polls / 44 draws
  (**14.9%**, higher than test_audio because the voice has more pose variety), return-to-rest.

**Phase C on real hardware** (2026-10-11, run by the user, by-eye verdict): `--brain llama` with
the 3B Q4_K_S GGUF above + `--tts piper` — the LLM answered, the voice was generated, and **the
mouth was in sync with the audio**. That is the plan's exit criterion met.

**The llama tier, measured** (`tools/brain_check.py --brain llama --tts piper`, the 3B Q4_K_S GGUF,
defaults `--ctx 2048 --gpu-layers -1 --max-tokens 200 --temp 0.7`):

| rep | 1st token | 1st sentence | **1st audio** | sents | 1st audio len | piper | analysis | chars | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 121 ms | 324 ms | **611 ms** | 1 | 10.51 s | 287 ms | 39 ms | 168 | 68.9 |
| 1 | 8 ms | 134 ms | **312 ms** | 1 | 6.83 s | 178 ms | 19 ms | 102 | 82.6 |

Same tier with the splitter's clause cap lowered (`--max-chars 120`):

| rep | 1st token | 1st sentence | **1st audio** | sents | 1st audio len | piper | analysis | chars | tok/s | tightest queue margin |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 127 ms | 330 ms | **530 ms** | 2 | 7.12 s | 200 ms | 21 ms | 168 | 67.6 | **+6.9 s ahead** |
| 1 | 8 ms | 134 ms | **285 ms** | 1 | 6.80 s | 151 ms | 25 ms | 102 | 89.6 | n/a (one sentence, 102 < 120) |

Model load 0.6 s, voice load 0.7–0.8 s (both paid before the window opens). Conclusions:

- **The GPU tier is settled: `--max-chars 120` gives a 6.9 s queue margin** on a 10.5 s reply, and
  first audio ~0.3–0.5 s. No latency-driven `--max-tokens` cap is needed — at 68–90 tok/s the brain
  writes a 100–170 char reply in under ~2.5 s while that reply is 6.8–10.5 s of audio, and Piper runs
  at 30–40× realtime. Set `--max-tokens` for conversational length, not for sync.
- **`--max-chars` is the first-audio lever, `--max-tokens` is not**: 611 → 530 ms came from cutting
  the model's run-on sentence at a clause boundary rather than waiting for a full stop.
- **That run found a real defect** (now fixed + pinned): the clause cut left the separator with the
  tail, so Holly was about to be handed `", which is quite impressive for a holographic entity."`
  as an utterance — a chunk starting with a comma opens the voice on a stray pause and the caption
  reads as a fragment. `_clip_long` now keeps the comma with the clause that owns it.
- A `sents 1` row measures **nothing** about overlap (nothing was ever queued behind anything), which
  is why `brain_check.py` now prints `n/a single sentence` there instead of a passing "OK", and now
  reports the **tightest queue margin** (positive = stalled, negative = that far ahead) so tiers are
  comparable.

**OmniVoice on the GPU, measured** (`--tts omnivoice`, cloning `voices/holly_base.flac` + its
`.txt` transcript; RTX 5060 Ti, fp16, default 32 diffusion steps):

| path | cost |
|---|---|
| `warmup()` — model + audio tokenizer + prompt encode, **once**, before the window opens | **9.2 s** cold page cache (the weight read itself is ~2.5 s cold, ~0.26 s once resident in the page cache) |
| steady-state `synthesize()` | **0.63 s** per 1.63 s of speech (**2.6× realtime**) |
| first sentence **before** `warmup()` existed | 10.56 s cold / 5.32 s warm — one-off load, none of it speech |

`tools/brain_check.py --brain script --tts omnivoice` after the warmup:

| rep | **1st audio** | sents | 1st audio len | omnivoice | analysis | chars | tok/s | tightest margin |
|---|---|---|---|---|---|---|---|---|
| 0 | **1604 ms** | 4 | 1.19 s | 1604 ms | 11 ms | 210 | 12.6 | **7.2 s ahead** |
| 1 | **733 ms** | 2 | 2.58 s | 733 ms | 5 ms | 137 | 19.0 | **1.5 s ahead** |

- **`warmup()` is what made the backend usable.** `OmniVoiceTTS` loads lazily so `make_tts` stays
  free, but that pushed 5–11 s of one-off cost into the *first sentence* — `1audio ms` of 5316 ms
  was pure model load. `app.py` and `brain_check.py` now call `warmup()` through
  `getattr(tts, "warmup", None)`, so backends without it are untouched. First audio is now 0.73–1.60 s.
- **The failure mode is a tiny first sentence, not the GPU.** Before the warmup the *same* script
  reply gave `STALL 0.28 s before sentence 2` warm and `OK` cold, because `Rude.` rendered as
  **0.69 s warm vs 1.17 s cold**: 0.69 s of audio + 0.25 s gap gave a 0.94 s budget and sentence 2
  landed 0.97 s later. Two rules follow. (1) **OmniVoice does not render the same text to the same
  length twice**, so a short first sentence makes the queue margin unstable run to run. (2) The
  fixed ~0.6 s per-call floor exceeds the audio a five-character sentence produces, so **do not set
  `--max-chars` aggressively low on this backend** — Piper at 30× realtime can afford short chunks,
  this cannot.
- **Piper is still ~10× quicker in absolute terms** (0.07–0.21 s/sentence vs ~0.63 s). OmniVoice
  clears the queue with 1.5–7.2 s to spare but has no headroom for very short sentences. `--tts-steps`
  (the `num_step` dial, default 32) is the lever if short replies ever stall; not needed on this box.
- **Weights and sidecars, all gitignored, all placed by hand:** `models/OmniVoice/` (copied, never
  downloaded by the backend — its `audio_tokenizer/` must be inside it or the tokenizer is fetched
  from HuggingFace at load), `voices/holly_base.flac` (the reference), `voices/holly_base.txt`
  (its **required** transcript — auto-transcription is refused because it would pull
  whisper-large-v3-turbo, ~1.5 GB), `voices/holly_base.pt` (the encoded prompt cache, ~12 kB,
  re-encoded only if the `.flac` is newer).

## File map

```
README.md                 user-facing: setup, run commands, Phase A status
hollyhost/
  app.py                  entry point. `--audio …` = batch (A0); no --audio = interactive text
                          line (A1). `--brain none|script|llama|http` attaches a brain (C).
                          `--dry` = hand-driven clock, no device/window.
  audio.py                the clock: SounddeviceOutput (batch), LiveAudio (appendable, freezes
                          at tail), FakeDevice + LiveAudio.advance (tests/--dry). `.flush()`
                          truncates at the consumed position for the Phase E barge-in.
  session.py              pure part: build_session (batch) + LiveSession (interactive) — face
                          queue and audio tail laid out on one clock so they cannot drift.
                          `.interrupt()` = the barge-in (flush + `HollyFace.reset()`).
  idle_motion.py          Phase E opt-in idle "breathing": throttled (≤ 2 draws/s) host-side
                          overlay; restores the pure idle pose whenever speech is queued; off
                          by default — the component's zero-draw idle is untouched when off.
  tts.py                  the seam: synthesize(text)->(float32,rate); resample-once/keep-both;
                          NullTTS / PiperVoiceTTS (default, in-process) / PiperTTS (subprocess,
                          batch only) / OmniVoiceTTS (opt-in GPU clone, lazy load + warmup()) /
                          ScriptTTS; make_tts factory; SentenceSplitter + split_sentences.
  brain.py                the other seam: reply_stream(messages) -> str deltas; ScriptBrain /
                          LlamaCppBrain / OpenAICompatBrain; make_brain; DEFAULT_PERSONA;
                          SentencePump (the chat thread, see above; `.interrupt()` abandons a
                          turn for barge-in — non-blocking, drain-on-cancel).
  normalize.py            Phase D TTS text preprocessing: preprocess(text) expands numbers/
                          decimals/unit-slashes/symbols/abbreviations and strips leaked markdown,
                          AFTER splitting and BEFORE synthesis, so phonemes (and visemes) are
                          unambiguous. Caption/history keep the original text.
  calibrate.py            the shared measurement (measure/report/starvation) + recommend() +
                          config load/write. Powers both tools/brain_check.py and app --calibrate;
                          model-free in CI with --brain script --tts null. hollyhost_config.json
                          (gitignored) records the chosen tier; _apply_config_defaults reads it.
tools/
  voice_check.py          cross-voice classifier spot-check (changes/s, idle %, visemes).
  brain_check.py          first-audio latency + per-sentence arrival + queue STALL verdict per
                          hardware tier. `--brain script|llama|http`, `--tts null|piper|omnivoice`.
                          Thin CLI over hollyhost/calibrate.py (shared with app --calibrate).
tests/
  test_clock.py           audio length == face clock invariant; dirty check; return-to-rest.
  test_playback_loop.py   the component's seam assertions, driven through the LIVE loop
                          (reachable frames, no sample past audio, gaps/tail idle,
                          prune-no-drift) + the two-rate multi-utterance seam test.
  test_tts.py             sample-rate contract + resampler quality + sentence splitting +
                          the in-process Piper backend (self-skips without piper-tts/a voice) +
                          the omnivoice backend (self-skips without the weights; the synthesis
                          test is gated on HOLLY_OMNIVOICE_DEVICE=cuda|cpu).
  test_brain.py           splitter equivalence under every chunking; the pump: first sentence
                          ready while the brain still generates, synthesis off the UI thread,
                          one turn at a time, errors surfaced not crashed, stop(), history trim,
                          and the pump preprocessing (spoken text to TTS, original caption).
  test_normalize.py       the Phase D preprocessing: decimals, unit slashes, symbols, markdown,
                          abbreviations; conservative (bare integers and unit-substring words are
                          left alone); idempotent; never re-splits an utterance.
  test_app.py             app.run() --dry dispatch smoke (batch + interactive + chat); the
                          --calibrate flow (writes/previews a config, rejects --brain none);
                          config-default precedence (explicit flag > config > built-in default);
                          the --barge-at rehearsal (returns 0, mouth at rest).
  test_barge_in.py        Phase E: flush freezes the clock at the cut; post-cut appends re-pin
                          to it; session interrupt = one rest redraw then zero-draw; the pump
                          drops queued AND in-flight sentences; new turns work after a barge-in.
  test_idle_motion.py     Phase E: ≤ 2 draws/s ceiling; subtle sum-to-1 rows; the idle-pose
                          restore before speech; no duplicate or wasted draws.
```

## Reproduce / verify everything

```sh
cd Holly-host
.venv/bin/python -m unittest discover -s tests -t .                 # 123 tests
.venv/bin/python tools/voice_check.py                               # cross-voice metrics
# the Phase C path with no model (script brain) — latency + STALL verdict:
.venv/bin/python tools/brain_check.py --brain script --tts piper
# the GPU clone instead (needs models/OmniVoice + voices/holly_base.flac/.txt, and the GPU):
.venv/bin/python tools/brain_check.py --brain script --tts omnivoice --tts-device cuda
# the product first-run calibration (same measurement, writes hollyhost_config.json):
.venv/bin/python -m hollyhost.app --calibrate --brain script --tts null            # model-free rehearsal
.venv/bin/python -m hollyhost.app --calibrate --brain llama --tts piper --model <gguf>  # this box's real tier
# the clone's own end-to-end synthesis test (prints the steady-state realtime factor):
HOLLY_OMNIVOICE_DEVICE=cuda .venv/bin/python -m unittest tests.test_tts.TestOmniVoice -v
# batch clock proof, no device:
.venv/bin/python -m hollyhost.app --audio holly-8bit-6000/test_audio.flac --audio holly-8bit-6000/test_audio.flac --gap 0.3 --dry
# chat path, no device, no model (a whole turn is ~18 s of speech, so give it the budget):
.venv/bin/python -m hollyhost.app --brain script --tts piper --dry --seconds 25
# the Phase E barge-in headless (cut at 2 s of clock; prints BARGE-IN, mouth must return to sil):
.venv/bin/python -m hollyhost.app --brain script --tts piper --dry --barge-at 2.0 --seconds 25
# real device (needs libportaudio2): type, Enter to speak; Tab (or Space on an empty line)
# stops Holly mid-sentence; add --idle-motion for the throttled idle breathing:
.venv/bin/python -m hollyhost.app --tts piper
```

## Running the brain on real hardware — the agent does not run inference

The agent must **not** run LLM inference, download GGUFs, or start heavy CUDA/pip builds: this
box is out of resources while the agent's own model is running, and the build is long. Hand the
user the exact commands instead and interpret their output. Piper ONNX inference is fine (cheap,
~0.2 s/sentence). **OmniVoice is in the same category as the LLM: hand it to the user.** Its fp16
weights are 2.4 GB on disk and ~3 GB of VRAM resident, which collides with the agent's own model
on a 16 GB card — including the OS page cache effect, a cold run and a warm run differ by ~5 s, so
the user should run it twice and read the second. The commands below are **verified working** (the
user ran them 2026-10-11).

```sh
# CPU-only bindings (slow but enough to prove the path):
.venv/bin/pip install -e ".[ui,dev,piper,llm]"
# GPU bindings (Blackwell sm_120 + CUDA 13; nvcc is NOT on PATH, so set it):
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120" \
  PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
  .venv/bin/pip install llama-cpp-python

.venv/bin/python tools/brain_check.py --brain llama --tts piper --max-chars 120 \
  --model /home/irreverend/LLMs/unsloth/Qwen2.5-VL-3B-Instruct-GGUF/Qwen2.5-VL-3B-Instruct-Q4_K_S.gguf
.venv/bin/python -m hollyhost.app --brain llama --tts piper --captions --max-chars 120 \
  --model /home/irreverend/LLMs/unsloth/Qwen2.5-VL-3B-Instruct-GGUF/Qwen2.5-VL-3B-Instruct-Q4_K_S.gguf
```

Zero-build alternative, already implemented: run `llama-server` / Ollama / LM Studio and use
`--brain http --base-url http://127.0.0.1:8080` (`OpenAICompatBrain` speaks SSE with stdlib
urllib — no extra dependency).

A `--dry` rehearsal needs a `--seconds` budget at least as long as the reply's audio; when it
expires mid-turn the harness prints `TRUNCATED …` and returns 0 rather than asserting about a
mouth that is legitimately mid-vowel (pinned by `test_app.py`). The full streamed turn above with
the real voice was measured: 1062 polls / 128 draws (**12.1%**) / mouth back at `sil`.

## Open items (in rough priority order)

1. **Re-measure only if the tier changes.** The 3B GPU tier is settled (6.9 s margin at
   `--max-chars 120`, first audio ~0.3–0.5 s). If the 9B GGUF or a CPU-only box is tried, re-run
   `tools/brain_check.py --brain llama --tts piper --max-chars 120 --model <gguf>` and read the
   **tightest margin** column, not the averages. If it STALLs, the levers are a smaller/faster
   model, `--gpu-layers`, or a cheaper voice — not the queue, and not the renderer.
2. **Production first-run calibration — DONE** (2026-10-13). `python -m hollyhost.app --calibrate`
   runs the same measurement as `tools/brain_check.py` (both now share `hollyhost/calibrate.py`:
   `measure`/`report`/`starvation`), prints the table, derives defaults via `recommend`, and writes
   them to `hollyhost_config.json` (gitignored; `--config` overrides the path, `--no-write` previews).
   A normal run reads that config through `_apply_config_defaults`: precedence is **explicit flag >
   config > built-in default**, and the built-ins match the pre-calibration values, so an uncalibrated
   box behaves exactly as before. Model-free in CI: `--calibrate --brain script --tts null` is what
   the tests run. `recommend` is honest about what the numbers decide — the queue margin sets
   `--max-chars`; `--max-tokens` is conversational length, not sync, so it is echoed not tuned;
   `--gpu-layers`/model are recorded as the tier measured, not invented. A follow-up could add a
   guided setup menu, but the supported entry point exists.
3. **The look — ACCEPTED** (2026-10-13). The user ran the live UI with `--tts piper` (plus
   `--idle-motion` and the `--key-hz 10` comparison) and judged the whole Phase E set — barge-in,
   idle breathing at the 0.25 default, the pose look — **nothing to change**. (The earlier
   omnivoice voice judgement was Phase D's "sounded like Holly".) The levers stay where they are
   for any future voice: the classifier's gate/gain knobs, `--fade-shape`, `--key-hz`,
   `--idle-weight` — **never the renderer.** New voices still get the `voice_check.py` +
   filmstrip spot-check before trust (open item 4).
4. **Phase B — voice abstraction.** Blocked until the voice corpus exists. Wire the fine-tuned
   Piper ONNX (CPU) and OmniVoice (GPU) behind `make_tts`/`synthesize` — a new backend only has to
   satisfy `synthesize(text)->(float32,rate)`; the rate may be anything. Spot-check each new voice
   with `tools/voice_check.py` + a filmstrip before trusting it.
5. **3b streaming-TTS refactor** — only if the host adopts chunked TTS that plays as it arrives
   (analysis currently always runs ahead of playback, which is what keeps the auto gate +
   `center` fade valid live). Note the pump already assumes whole-sentence synthesis; if TTS
   becomes chunked, `SentencePump` is where that changes.
6. **Phase D is accepted.** The TTS text preprocessing is built and tested
   (`hollyhost/normalize.py`), and the user judged the persona with a real voice
   (`--tts omnivoice`) — it **sounded like Holly**, so with the model not yet fine-tuned the
   persona is good enough for now. The later lever for richer character is the Red Dwarf LoRA
   fine-tune (its own side project), not the persona text or the pipeline.
7. **Phase E is built (2026-10-13), awaiting the human pass in item 3.** Barge-in, idle
   micro-motion and the look dials are implemented and tested (see the two settled sections);
   123 host tests pass. What is *not* done: **custom per-viseme gain curves** — the component
   already caps poses per viseme (`HA_VISEME_MAXS`, 0.65/0.75, on via `gain=True`); retuning
   those numbers is a component-side change, and the component is frozen — the host exposes
   `--no-gain` for the A/B instead. If the curves must change, that is a decision to take to
   the component repo, not to route around in the host.

## Kickoff prompt

> Continue the **Holly host** at `/home/irreverend/Projects/Holly-host`. Read this repo's
> `HANDOFF.md` first (authoritative state), then `README.md`, then `holly-8bit-6000/HANDOFF.md`
> (the finished animation component — consume it, never modify it). Phase A (skeleton, no LLM) is
> **done and verified on real hardware**; Phase C (the brain) is **done too** — the user ran
> `--brain llama` + a 3B GGUF + piper on this box and confirmed the answer was spoken with the
> mouth in sync; the latency table is in "Measured on this box". Phase D (persona + TTS text
> preprocessing) is **done and accepted** — `normalize.preprocess` is tested, and the user judged
> the persona with `--tts omnivoice` as sounding like Holly. Phase E (polish) is **done and
> accepted** (2026-10-13): barge-in (Tab / `--barge-at`), opt-in `--idle-motion` breathing
> (default peak weight 0.25, `--idle-weight` dial), and the look dials (`--no-gain`/`--key-hz`/
> `--fade-shape`) — the user ran them on the real device and asked for no changes. See the two
> settled sections. 135 host tests + 147 animation tests pass.
>
> Verify before anything else: `.venv/bin/python -m pytest -q` → 135 OK, and
> `cd holly-8bit-6000 && ../.venv/bin/python -m pytest -q` → 147 OK. Both suites run under this
> repo's single `.venv`. `libportaudio2` is installed, two stock Piper
> voices are in `voices/` (gitignored), and `llama-cpp-python` 0.3.36 (CUDA build, GPU offload
> works) is installed.
>
> Do not re-litigate the settled architecture. Clock: `t` is the device's consumed-frame position;
> `LiveAudio` freezes at the tail; `LiveSession` pins each utterance's face `at` to its audio
> buffer position; the component owns dirty-check / zero-draw idle / no-final-pose-latch. Chat
> thread: the pump may run brain + TTS off the UI thread but must never touch the face or the
> clock; `session.say` stays on the UI thread. The splitter and `split_sentences` must stay
> equivalent under any chunking. Barge-in: `flush()` truncates at the consumed position, the face
> half is `HollyFace.reset()`, `pump.interrupt()` never blocks the UI thread. Idle overlay: restore
> the idle pose **before** `session.poll()`, never after. Do not reimplement any of that in the host.
>
> **Hard constraint: do not run LLM inference, download GGUFs, or start CUDA/pip builds** — the
> agent's own model is using the machine. Hand the user the exact commands from "Running the brain
> on real hardware" and interpret their output. Piper ONNX inference is fine (cheap).
>
> Then pick up open item 1 only if you change model tier — the 3B GPU tier is measured and settled
> (`--max-chars 120`, 6.9 s queue margin, first audio ~0.3–0.5 s). There is **no open host work**
> otherwise: Phases A, C, D and E are done and accepted, and the production ask is **done**
> (`--calibrate` is the supported first-run step; shared `hollyhost/calibrate.py`, writes
> `hollyhost_config.json`). Phase B (the fine-tuned voice) stays blocked until the voice corpus
> exists; custom per-viseme gain curves are a decision for the component repo, not the host
> (open item 7).
