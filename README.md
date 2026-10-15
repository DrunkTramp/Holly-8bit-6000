# Holly host

The application shell around the [Holly-8bit-6000](../Holly) animation component: a
keyboard-driven Holly that speaks text with a lip-synced, retro pixel-art face in a Tkinter
window: chat log and input line on the left two thirds, the face (top) and a settings panel
(bottom) on the right third. You type, a voice synthesises it, Holly's mouth moves in time
with the audio *he himself is playing* — never a microphone.

The animation component is a finished library. This repo consumes it, it does not modify it.

## Integration model

The component is installed editable and imported as a package — no submodule, no `sys.path`
surgery. The host gets its **own venv** so it is not limited by whatever the animation
component's venv happens to have:

```sh
python3 -m venv .venv
.venv/bin/pip install -e ../Holly            # the animation component -> `import holly`
.venv/bin/pip install -e ".[ui,dev,piper]"   # host + pillow + pytest + piper-tts (+ sounddevice binding)
.venv/bin/pip install -e ".[ui,dev,piper,llm]"  # + llama.cpp bindings (Phase C, see "A brain")
# Tkinter ships with the system Python on most distros, but on Debian/Ubuntu it is a separate
# package: sudo apt install python3-tk
# the GPU clone backend is deliberately NOT a plain extra here: it needs a vendored clone and
# CUDA wheels installed first. See "OmniVoice GPU voice clone (opt-in)" below.
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
.venv/bin/python -m hollyhost.app --tts omnivoice                  # GPU clone of voices/holly_base.flac (see below)

# No audio device? Rehearse the identical loop with a hand-driven clock (no sounddevice needed):
.venv/bin/python -m hollyhost.app --audio ../Holly/test_audio.flac --dry
.venv/bin/python -m hollyhost.app --tts piper --dry
```

### The settings panel (bottom right) — nothing loads until you speak

The interactive window opens with **no model loaded at all**. The settings panel carries the
backend choices, and the first message triggers the load on a worker thread (the face and
chat stay responsive; that first line is kept and spoken once the voice is ready):

- **Voice tab** — TTS backend dropdown (`null` / `piper` / `omnivoice` / `script`) and the
  voice file (suggestions from `voices/`, or Browse).
- **Brain tab** — `none` (speak what you type), `llama` (a local GGUF loaded in-process;
  suggestions from `models/`, or Browse), or `http` (any OpenAI-compatible endpoint: host +
  port + optional API key, and a model name **propagated from the endpoint's own
  `/v1/models` list** via Fetch). `script` is the test brain.
- **Motion tab** — idle breathing overlay and its depth.

The CLI flags (`--tts`, `--brain`, `--model`, `--base-url`, `--http-model`, `--api-key`,
defaulting to `$OPENAI_API_KEY`) pre-fill the panel; once the first message has loaded the
backends, those fields lock — restart to change them.

**The panel remembers.** Its state is saved to `hollyhost_settings.json` (repo root,
gitignored; override with `--settings`) whenever the backends load and again when the window
closes, and restored at the next launch. Precedence is explicit flag > saved settings >
built-in default. Note the file can hold an API key in plain text — same per-machine,
gitignored logic as the calibration config.

`sounddevice` is **output only** — there is no input path anywhere.

By default the interactive line speaks **exactly what you typed** (`--brain none`). With a brain
attached, what you type becomes a *prompt* and Holly answers it (next section).

### Polish — Phase E

**Barge-in.** In the interactive window, **Tab** (or **Space** when the input line is empty)
stops Holly mid-sentence: playback is cut at the device position, the utterance queue and any
in-flight brain turn are dropped, and the mouth returns to rest. The clock survives the cut
intact — the buffer is truncated at what was actually consumed, so the next utterance pins to
that same position like any other. Rehearse it headless:

```sh
.venv/bin/python -m hollyhost.app --brain script --tts piper --dry --barge-at 2.0
```

**Idle breathing** (`--idle-motion`). An opt-in overlay that gives the idle mouth a slow,
gentle breath (`aa` at peak weight 0.25 — dial it with `--idle-weight`; on the bake, 0.12 is
effectively invisible and 0.65 is a speech pose — ≤ 2 draws/s). It never runs while speech is
queued — it restores the pure idle pose first — so the component's dirty check and zero-draw
idle stay exactly as shipped when the flag is off.

**Look dials** (for the human eye, not for sync): `--no-gain` A/Bs the classifier's per-viseme
display gain (HeadAudio `visemeMaxs` caps) against raw one-hot poses; `--key-hz 10` keeps the
2-frame fade from collapsing short holds to one 33 ms frame; `--fade-shape center|lag|lead`
places the fade around a pose change. Unset, each is the component's shipped default.

### A brain — Phase C

`--brain` puts a language model between the keyboard and the mouth. The reply is **streamed**:
as soon as the first sentence is complete it is synthesised and queued, so Holly starts talking
while the model is still writing the rest.

```sh
# no model at all — canned replies, but the real streaming/splitting/queue path:
.venv/bin/python -m hollyhost.app --brain script --tts piper --captions

# a GGUF through llama.cpp's Python bindings (`--max-chars 120` is the measured setting for
# this box — see "Measure before trusting a tier" below):
.venv/bin/python -m hollyhost.app --brain llama --model path/to/model.gguf --tts piper --max-chars 120 --captions

# anything that speaks OpenAI's HTTP API (llama-server, Ollama, LM Studio) — no build step:
.venv/bin/python -m hollyhost.app --brain http --base-url http://127.0.0.1:8080 --tts piper --captions

# a hosted endpoint: give no port and the scheme picks 443. `https://host:80` is the classic
# mistake — a TLS handshake aimed at a plain HTTP port (`SSL: WRONG_VERSION_NUMBER`):
.venv/bin/python -m hollyhost.app --brain http --base-url https://api.deepseek.com \
  --http-model deepseek-flash --api-key "$DEEPSEEK_API_KEY" --tts omnivoice --captions
```

`llama-cpp-python` is the only dependency that needs a compiler. On this box (RTX 5060 Ti,
CUDA 13 toolkit at `/usr/local/cuda`, Blackwell = `sm_120`) the GPU build is:

```sh
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120" \
  PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
  .venv/bin/pip install llama-cpp-python
```

(`nvcc` is not on `PATH` by default here — that `PATH=` is required, not cosmetic. Without CUDA
it builds CPU-only and still runs, just slower.) `--gpu-layers -1` offloads every layer; drop it
if the model does not fit in VRAM.

**Measure before trusting a tier.** `tools/brain_check.py` prints the numbers the exit criteria
are about — model load, ms to first token / first sentence / first **audio**, per-sentence
arrival time, and a `STALL` verdict telling you whether the queue can stay ahead of playback:

```sh
.venv/bin/python tools/brain_check.py --brain script --tts piper
.venv/bin/python tools/brain_check.py --brain llama --model path/to.gguf --tts piper
```

Measured on this box with the stock Piper voice: first audio **74–158 ms** after a sentence is
ready, analysis **7–20 ms**, and the queue stays ahead of a 12–18 s reply. A `STALL` almost
always means the *voice*, not the model — see the Piper note below.

With a real model on this box (3B Q4_K_S GGUF, all layers on the GPU): model load 0.6 s, first
audio **312–611 ms** after the prompt, and the brain writes 68–90 tokens/s against a voice running
at 30–40× realtime — so the queue is nowhere near the edge and **no reply-length cap is needed for
sync**. Choose `--max-tokens` for how long you want Holly's answers to be. Lowering the splitter's
clause cap to `--max-chars 120` is what actually improves the feel: first audio drops to
**285–530 ms** and a 10.5 s answer is delivered with a **6.9 s queue margin**, i.e. the next
utterance is ready 6.9 s before the voice could ever run dry. That is the setting this box uses.

**New machine? Calibrate it first.** The latency and margin table *is* the machine's configuration:
which model tier, how many `--gpu-layers`, and what `--max-chars`/`--max-tokens` should be all follow
from it. Run the supported first-run step, and every normal run afterwards reads the result as its
defaults:

```sh
.venv/bin/python -m hollyhost.app --calibrate --brain llama --tts piper --model path/to/model.gguf
```

It prints the table, derives the defaults, and writes them to `hollyhost_config.json` (gitignored;
`--config` chooses the path, `--no-write` just previews). Precedence on a normal run is **explicit
flag > config > built-in default**, so an uncalibrated box behaves exactly as before. `tools/brain_check.py`
is the same measurement as a developer tool — both share `hollyhost/calibrate.py`, and the CI rehearsal
is `--calibrate --brain script --tts null` (no model, no GPU).

### Persona and spoken text — Phase D

Holly's system prompt lives in **`holly_personality_prompt.md`** (repo root, tracked, editable
— that file is the character source; edit it to tune the voice). It is loaded at startup and
appended with the spoken-form rules (short spoken sentences, no markdown/symbols/lists — what
`normalize.preprocess` then enforces). Override the whole thing with `--persona text` or
`--persona @file`. Because the file persona is ~2k tokens, `--ctx` defaults to 4096.
On top of the prompt, every sentence is passed through `normalize.preprocess` **after** splitting
and **before** synthesis, so a reply that leaks `576x432`, `12/s`, `12.5`, `etc.` or `**bold**` is
rewritten into spoken words the voice can actually phonemise (`576 by 432`, `12 per second`, `12
point 5`, `et cetera`, plain text). Captions and history keep the original text. If your voice's own
normalizer already reads these correctly and you want the raw text spoken, pass `--no-normalize`.

### Stock Piper voice

`--tts piper` with no `--voice` uses the first `.onnx` in `voices/` (gitignored — it is a
63 MB download). Fetch one with:

```sh
.venv/bin/python -m piper.download_voices en_US-lessac-medium --download-dir voices
```

`--tts piper` runs the voice **in-process** (`PiperVoiceTTS`): the ONNX model loads once at
startup (~0.85 s) and each sentence costs 0.07–0.21 s (~30× realtime). That matters more than it
looks. The obvious implementation — shelling out to the `piper` CLI per sentence — costs **1.42 s
per call**, nearly all of it reloading the model, which is *longer than a short sentence takes to
play*: the queue starves and Holly goes silent mid-answer. `brain_check.py` reported exactly that
STALL before the backend was fixed. The subprocess form is still available as `PiperTTS` for batch
work, but it is never right for a streamed reply.

### OmniVoice GPU voice clone (opt-in)

`--tts omnivoice` runs the diffusion-LM cloner on the GPU, cloning `voices/holly_base.flac`
(gitignored, like everything in `voices/`) unless `--voice` names another reference. Setup on
this box (already done — recorded here for a fresh clone):

```sh
# 1. CUDA torch FIRST (the RTX 5060 Ti needs the cu128 wheels; PyPI's default torch is CPU-only):
uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0
# 2. the repo, vendored and editable (it is not on PyPI):
git clone --depth 1 https://github.com/k2-fsa/OmniVoice vendor/OmniVoice
uv pip install --python .venv/bin/python -e vendor/OmniVoice
# 3. weights by hand, never downloaded by the backend:
cp -r /path/to/OmniVoice models/OmniVoice    # includes audio_tokenizer/, or the tokenizer
                                             # would be fetched from HuggingFace at load
```

Cloning needs a **transcript of the reference**: write the exact words spoken in
`voices/holly_base.flac` to `voices/holly_base.txt`. The backend deliberately never
auto-transcribes — that path pulls whisper-large-v3-turbo (~1.5 GB) from HuggingFace, against
this repo's place-the-weights-by-hand rule. On first synthesis the reference is encoded once
into a voice prompt and cached as `voices/holly_base.pt`; later starts load the cache (delete
it, or replace the `.flac`, to re-encode). The model itself (~3 GB VRAM in fp16) loads lazily
on the first sentence, so `--tts omnivoice` costs nothing until Holly actually speaks.

Measured on this box (RTX 5060 Ti, fp16, default 32 diffusion steps), cloning
`voices/holly_base.flac`. Two calls per run, because the first is not comparable to the
second — it pays model load, CUDA init and the voice prompt:

| run | first sentence | steady state | weight loading |
| --- | --- | --- | --- |
| cold page cache | 10.74 s | **0.63 s** for 1.63 s of speech = **2.60× realtime** | 2.5 s (313 @ 214 it/s, 527 @ 505 it/s) |
| warm page cache | 5.70 s | **0.63 s** for 1.64 s of speech = **2.58× realtime** | 0.26 s (313 @ 1869 it/s, 527 @ 5600 it/s) |

Two things that are easy to misread. The 5 s gap between cold and warm is the **OS page
cache**, not VRAM — 2.4 GB of weights read off NVMe at ~214 it/s the first time, ~1869 it/s
once resident. And the ~5 s of first-sentence time is `import torch`/transformers, CUDA
context init and the prompt load; the per-sentence number is the steady-state one, and it is
comfortably faster than playback. Piper remains quicker in absolute terms (~30× realtime,
0.07–0.21 s/sentence), but the queue only needs *faster than playback*.

The 0.63 s has a **fixed floor** (prompt conditioning, audio-token decode) that does not
shrink with text length, so longer utterances amortise better and the realtime factor climbs
above 2.6×. That is a real trade against `--max-chars`: capping the first sentence short is
what makes `1audio ms` small, but each short sentence then pays that fixed cost again. With
the floor at ~0.6 s, a sentence shorter than ~0.6 s of speech costs more to make than to
play — so don't set `--max-chars` aggressively low on this backend.

Two follow-ups the streamed measurement forced.

**The model now warms up at startup.** `OmniVoiceTTS.warmup()` pays model load, CUDA init and
the prompt encode before the window opens; `app.py` and `brain_check.py` both call it through
`getattr(tts, "warmup", None)`, so backends without it are unaffected. Without it, `1audio ms`
for the first sentence was **10.56 s cold / 5.32 s warm** -- all one-off cost, none of it speech.

**A tiny first sentence can starve the queue, and this is the backend's real failure mode.**
`brain_check.py --brain script --tts omnivoice` on this box:

| rep | first sentence | its audio | next sentence arrived | budget (audio + gap) | margin |
| --- | --- | --- | --- | --- | --- |
| cold | `Rude.` | 1.17 s | +0.96 s | 1.42 s | **OK**, tightest 7.2 s ahead |
| warm | `Rude.` | **0.69 s** | +0.97 s | 0.94 s | **STALL 0.28 s** |

After the warmup landed, the same command runs clean: `tts load 9.2 s` in the header, first audio
**1604 ms** (4 sentences, tightest margin 7.2 s ahead) and **733 ms** (2 sentences, 1.5 s ahead),
no STALL.

Same text, same script brain, opposite verdicts -- because the fixed ~0.6 s per-call floor is
comparable to the audio a five-character sentence produces, and **OmniVoice does not render the
same text to the same length twice** (1.17 s vs 0.69 s for `Rude.`). So the margin on a short
first sentence is not a stable quantity on this backend; expect it to move run to run. Longer
sentences are safe (2.6× realtime and improving with length), which is why the tightest margin
in a whole reply lands on whichever sentence is shortest.

`--tts-steps` (the `num_step` dial, default 32) is the lever: fewer steps shrink the fixed floor.
Not needed at 2.6× realtime for normal sentences, but if short replies keep stalling, turn it
down and re-measure rather than assuming the GPU is the limit. The other option -- holding a
sentence shorter than some floor and merging it into the next -- would change
`SentenceSplitter` semantics for every backend and delay first audio, so it is deliberately
not done; `--max-chars` already shapes this from the other side.

Verified on `en_US-lessac-medium` (22.05 kHz): the shipping HeadAudio classifier drives the
mouth well on this synthetic speech (~7–8 pose-changes/s, 9–11 distinct visemes per sentence,
~5 ms analysis per second of speech) — the HANDOFF's open "classifier on synthetic speech"
question, answered positively for a stock voice. Judge each new voice by eye with
`../Holly/tools/reference_player.py` / `render_filmstrip.py`; if one gives poor visemes the fix is
the classifier's gate/gain knobs, never the renderer.

`tools/voice_check.py` runs the shipping classifier over a few sentences per voice and prints
the numbers that separate a good voice from a degenerate one (changes/s, idle %, distinct
visemes, analysis ms) — no device, no window. Measured across two voices:

| voice | changes/s | idle | visemes/sentence | verdict |
|---|---|---|---|---|
| `en_GB-alan-medium` (British male — apt for Holly) | 7–8 | 41–47% | 9–12 | OK |
| `en_US-lessac-medium` | 7.2–7.9 | 45–58% | 9–11 | OK |

Both match the component's real-speech baseline (~7 changes/s at 12/s). Piper samples its own
prosody noise, so re-running `voice_check.py` moves these by roughly ±0.7 changes/s — treat a
single run as a spot-check, not a benchmark. To eyeball the look
for a voice: synthesise a WAV (`piper -m voices/…onnx -f out.wav`) and run the animation
repo's `make_timeline.py --audio out.wav --out t.json` then
`render_filmstrip.py --timeline t.json --audio out.wav --video v.mp4`.

### Real audio output needs libportaudio (system package)

The `sounddevice` *binding* installs via pip, but opening a stream needs the native PortAudio
library, which is a system package (root):

```sh
sudo apt-get install -y libportaudio2
```

Without it, `import sounddevice` raises `OSError: PortAudio library not found` and the
interactive UI exits with a clear message. `--dry` runs the identical loop on a hand-driven
clock, so the whole pipeline is testable on a box with no audio hardware.

**Verified on real hardware** (PipeWire default device, this box): the batch UI ran
`sounddevice` output with the clock driven by the device callback — 119 polls / 12 draws
(10.1%) / **0 underruns** — and the interactive `LiveAudio` stream tracked a 2.15 s Piper
utterance to within 2 ms and drained to idle with no PortAudio status errors.

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
  app.py       entry point: A0 batch player + A1 interactive text line (Tkinter UI),
               + the --brain chat wiring
  tkui.py      the Tkinter shell: chat log + input (left 2/3), face canvas + settings panel
               (right 1/3, Voice/Brain/Motion tabs); backends load lazily at the first
               message, on a worker thread; mouth-rect-only redraws, same clock rule
  session.py   the pure part: build_session (batch) + LiveSession (interactive) — face queue
               and audio buffer laid out on one clock so they cannot drift; .interrupt() is
               the barge-in (cut the audio at the device position, reset the face queue)
  audio.py     the clock: SounddeviceOutput (batch) + LiveAudio (appendable, freezes at the
               tail while Holly waits for input; .flush() truncates at the consumed position
               for barge-in) + FakeDevice/LiveAudio.advance for tests
  idle_motion.py  Phase E opt-in idle "breathing": a throttled (≤ 2 draws/s) host-side overlay
                  that restores the pure idle pose whenever speech is queued
  tts.py       the seam: synthesize(text) -> (float32, rate); resample-once/keep-both contract;
               NullTTS / PiperVoiceTTS / PiperTTS / OmniVoiceTTS / ScriptTTS;
               SentenceSplitter + split_sentences
  brain.py     the other seam: reply_stream(messages) -> deltas; ScriptBrain / LlamaCppBrain /
               OpenAICompatBrain; SentencePump (chat thread -> queued ready-to-speak sentences;
               .interrupt() abandons a turn for barge-in)
tools/
  voice_check.py   cross-voice classifier spot-check (changes/s, idle %, visemes) — no device
  brain_check.py   first-audio latency + queue-STALL budget per hardware tier — no window
tests/
  test_clock.py           the audio length == face clock invariant, dirty check, return-to-rest
  test_playback_loop.py   the component's seam assertions, driven through the live loop
  test_tts.py             the sample-rate contract + resampler quality
  test_brain.py           the streaming splitter's equivalence property + the chat thread
  test_app.py             app.run() --dry dispatch smoke (batch, interactive, chat, barge-in)
  test_barge_in.py        Phase E: audio flush + face reset + pump abandon, clock pinning intact
  test_idle_motion.py     Phase E: throttle ceiling, subtlety, idle-pose restore
```

### The chat thread's one rule

`SentencePump` runs on a worker thread, and it may do the two slow things there — call the brain
and synthesise the voice. It must **never** touch `HollyFace` or the audio clock. The UI thread
drains `pump.sentences()` and calls `session.say(...)` itself, which is what keeps each
utterance's face position pinned to its audio position. Break that and the drift the whole
design exists to prevent comes back.

The splitter is one rule set shared by both paths: `SentenceSplitter` (incremental, for a token
stream) and `split_sentences` (batch) agree by construction, and a test feeds whole / every-N /
word-wise / straddling chunkings of a corpus through both and asserts they produce identical
utterances. That is what makes a terminator at a chunk seam (`"12."` + `"5"`) safe: it is
*undecided* until the next delta arrives, never a boundary.

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -t .
```

The tests run without a working audio device: `FakeDevice` and `LiveAudio.advance` drive the
clock by hand, and the pure queue logic needs no window. (`sounddevice`'s binding may be
installed but unused by the tests.) **No test runs an LLM** — the chat path is exercised through
`ScriptBrain`. The Piper tests do run real ONNX inference (it is cheap) and skip themselves if
`piper-tts` or a voice in `voices/` is missing.

## Status — Phases A, C, D and E

- **A0 (clock proof):** done. Batch queue drives the corner face from the device position;
  measured ~11% of polls cost a draw, idle is zero-draw, the mouth returns to rest on drain.
  Verified on a real output device (0 underruns, clock tracks the callback).
- **A1 (text to speech):** done. The interactive line synthesises typed text through a real
  stock Piper voice (`--tts piper`), folds it to 16k analysis + native playback, and speaks it
  on the same clock. Verified headless (`--dry`) and on a real device (stream drains to idle,
  clock matches audio length).
- **C (the brain):** done, and verified on this box by eye (2026-10-11): `--brain llama` with a
  3B Q4_K_S GGUF + `--tts piper` — the model answered, the voice played, the mouth was in sync.
  The reply is **streamed**: brain deltas → sentence splitter → synthesis on the chat thread → the
  utterance queue. Headless it measures first audio **74–158 ms** after a sentence is ready with
  the stock voice, analysis 7–20 ms, queue ahead of a 12–18 s reply. On the llama tier the same
  tool measures first audio **~0.3–0.5 s** and a **6.9 s queue margin** at `--max-chars 120`
  (3B Q4_K_S GGUF on an RTX 5060 Ti, 68–90 tok/s) — the brain outruns the voice, so that is the
  setting to use; `--max-tokens` is for conversational length, not for sync.
- **B (the fine-tuned voice):** blocked until a voice corpus exists.
- **D (persona + TTS text preprocessing):** done (2026-10-11). Before synthesis, `normalize.preprocess`
  rewrites a finished sentence into spoken form — `576x432` → "576 by 432", `12/s` → "12 per second",
  `12.5` → "12 point 5", `etc.` → "et cetera", `20°C` → "20 degrees Celsius", stray `**markdown**`
  stripped — so the phonemes, and therefore the visemes, are unambiguous. It runs **after** the
  splitter (never disturbs a decimal or a `Dr.` boundary) and **before** TTS; captions keep the
  original text. `--no-normalize` opts out. `DEFAULT_PERSONA` is rewritten for spoken output. The
  user ran it with `--tts omnivoice` and judged it **sounded like Holly** — with the model not yet
  fine-tuned, the persona is good enough for now (the real fine-tune is the later lever).
- **E (polish):** done and accepted (2026-10-13 — the user ran it on the real device with piper
  and asked for no changes). **Barge-in** (Tab / Space-on-empty-line; `--barge-at` headless):
  playback cut at the device position, queue and in-flight turn dropped, mouth back to rest —
  the pinning rule survives the cut by construction. **Idle micro-motion** (`--idle-motion`,
  off by default): throttled ≤ 2 draws/s breathing overlay, opt-in so the shipped zero-draw
  idle is untouched. **Look dials** `--no-gain` / `--key-hz` / `--fade-shape` expose the
  component's existing knobs (incl. its per-viseme `visemeMaxs` gain) for by-eye settling;
  *custom* per-viseme gain curves would be a component-side change and are not done here.
- **Still open:** the classifier-quality judgement on real TTS voices needs a person: run
  `../Holly/tools/reference_player.py` and `render_filmstrip.py` per voice; if a voice gives
  poor visemes the fix is the classifier's gate/gain knobs, never the renderer.
