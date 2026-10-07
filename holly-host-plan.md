# Holly-8bit-6000: whole-program plan (LLM + TTS + animated face)

Goal: a keyboard-driven conversational Holly. You type, a small local LLM answers in
Holly's voice, Holly's mouth lip-syncs the audio he generates. Runs on a Pi or a
low-end PC; a GPU box gets the nicer voice. No mic, no memory between sessions.

The animation component already exists and is done: `github.com/DrunkTramp/Holly-8bit-6000`
(PSD layers -> 15 Oculus visemes -> keyframed poses at 12/s @ 30 fps -> mouth-rect renders
at ~1.8 ms/draw; `speak(samples) -> timeline`, `row_at(audio_time)`). This plan builds the
host around it. The animation repo is consumed as a library, not modified.

## 1. Architecture

One Python process, four stages, three threads, one clock:

```
keyboard ──> chat thread: LLM (llama.cpp) ──> sentence splitter
                                                │  per-sentence
                                                v
                              TTS backend ──> utterance buffer (numpy)
                                                │
                              holly.speak(buffer) ──> pose timeline   (analysis, ~30 ms/utterance,
                                                │                       done BEFORE playback)
                                                v
                              audio out (sounddevice) ──> playback clock
                                                │
                              render thread: row_at(t), dirty-check, blit mouth rect
```

- **Utterance queue.** Each sentence is one utterance: TTS buffer + its pose timeline,
  queued back-to-back. Playback drains the queue; the animation clock is the audio
  output position — never `time.time()`. This is what keeps mouth and voice locked.
- **Zero-draw idle** is inherited: between utterances the base face sits there for free,
  no final-pose latch.
- **Pipeline overlap:** while sentence N plays, sentence N+1 is being synthesised and
  analysed, and the LLM is still generating. Perceived latency after the keypress is
  first-token latency + first-sentence TTS time only.

## 2. TTS: two backends behind one interface

Interface the host codes against: `synthesize(text) -> (np.float32 mono, sample_rate)`.

- **CPU path (Pi / low-end PC): Piper.** VITS -> ONNX via onnxruntime, explicitly
  optimised for Pi 4, runs arm64/armv7 binaries. Your pre-trained voice plugs in as a
  fine-tuned checkpoint exported to `.onnx` (+ matching `.onnx.json`); inference side is
  just `piper --model your-voice.onnx`. Fine-tune recipe (~1300 phrases from a
  HuggingFace checkpoint, `piper_train`, export ONNX) is well-documented in Piper's
  TRAINING.md.
- **GPU path: OmniVoice** (github.com/k2-fsa/OmniVoice, Apache 2.0). Zero-shot cloning
  from a 3–25 s reference clip — no training needed, just a good recording of the Holly
  voice. Claimed ~40x realtime on GPU. Reference clip becomes the asset; the model runs
  on the GPU box.
- **Sample-rate contract:** Piper outputs 16k/22.05k, OmniVoice 24k+. The viseme pipeline
  wants 16 kHz mono float32. Host resamples (or asks Piper for `--output-raw` 16k) for
  the *analysis* buffer; playback uses the native-rate buffer. Analysis and playback
  buffers must correspond 1:1 in time — resample once, keep both, never re-derive.
- Test both voices through the existing classifier before committing: the HeadAudio
  prototypes were trained on human speech; TTS output is cleaner, which should help, but
  eyeball a few utterances per voice with the filmstrip tool.

## 3. LLM: llama.cpp, quantized, local

- **Engine:** `llama-cpp-python` (or the llama.cpp server binary + HTTP; either fine —
  pick whichever the harness prefers). No cloud, no API keys.
- **Model tiers** (chat/completion, Q4_K_M quantization):
  - Pi 4 (4–8 GB): Qwen2.5-1.5B — usable, short replies.
  - Pi 5 (8 GB): Qwen2.5-3B comfortable; 7B possible with small context.
  - Old PC (16 GB): 7B comfortable, snappy enough.
- **Persona for now:** system prompt only — Holly's voice, lazy, 52x the intelligence of
  a human being, short spoken-style answers (TTS-friendly: no markdown, no lists, one or
  two sentences). Constrain generation length; long answers are latency and rambling.
- **Fine-tune later (separate track, not blocking):** Red Dwarf scripts as chat data —
  cast as user turns, Holly's lines as assistant turns. LoRA via llama.cpp's built-in
  training or axolotl on the GPU box; merge to GGUF and swap the model path. The host
  code doesn't change at all, which is why this can be deferred indefinitely.
- Context window stays small (2–4k). Fresh conversation each run: no persistence layer,
  history lives in RAM and dies with the process.

## 4. Host UI

pygame window (the animation component already renders into numpy surfaces; pygame is the
cheapest owner of one). The face is a **corner element**, not a full-window canvas.

- **Bake size settled: 576x432, retro pixel-art.** That is half of the 1152x864 source (the
  original line here said "384x288 = half" — 384x288 is a *third*), baked
  `--scale 0.5 --resample nearest`, and it is what `holly.DEFAULT_BUFFERS` now points at, so
  the host gets it without passing a path. Corner placement is the reason for the smaller
  canvas: the mouth rect is the only thing blitted, and at 576x432 it is 168x132 (22,176 px,
  0.72 ms/draw) instead of 335x265 (88,775 px, ~2.9 ms) — 1.52% vs 4.66% of one core at a
  60 Hz poll. Pixel-art scaling stays at build time, not runtime. For reference, nearest
  differs from a proper area downsample by only ~0.6/255 on this asset, because the source is
  already blocky.
- Text input line at the bottom; Holly's reply optionally shown as captions (useful while
  tuning TTS/visemes).
- Ctrl-C / Esc quits. No barge-in in v1 — Holly talks, you wait. (Barge-in is a Phase E
  option: stop playback, stop queue, back to idle.)

## 5. Phases

Each phase is harness-sized: independently testable, ends in something you can see/hear.

- **A. Skeleton (no LLM).** pygame window, keyboard input box, sounddevice output, utterance
  queue, playback clock driving `row_at()`. Type arbitrary text -> canned TTS (stock Piper
  voice is fine here) -> Holly mouths it in sync. Exit criteria: sync verified by eye,
  idle is zero-draw, no final-pose latch, mouth-rect-only blits.
- **B. Voice.** Wire the TTS abstraction: your fine-tuned Piper ONNX on CPU, OmniVoice on
  the GPU box, same interface. Resample contract in place. Exit criteria: both voices
  drive the mouth acceptably (filmstrip check per voice).
- **C. Brain.** llama.cpp integration + chat loop + sentence splitter feeding the queue.
  Exit criteria: type a question, hear a spoken answer with synced mouth; measure
  first-audio latency per hardware tier and cap reply length to fit it.
- **D. Persona.** Holly system prompt tuned for spoken output; TTS text preprocessing
  (expand numbers/abbreviations before synthesis — visemes follow phonemes, so TTS text
  quality directly drives mouth quality). Exit criteria: Holly sounds and answers like
  Holly.
- **E. Polish (optional, pull in any order).** Barge-in; idle micro-motion (old plan's
  Phase 4); the `2,3,3,2` hold-swing check — now measured: at 12/s the 2-frame fade collapses
  17% of poses to a single 33 ms frame, and `--key-hz 10` fixes it while keeping the fade, so
  settle it by eye on `build/debug/visemes_keys10.mp4` vs `build/debug/visemes.mp4` rather than
  by argument; Red Dwarf
  LoRA fine-tune as its own side project.

## 6. Risks and known unknowns

- **Classifier on synthetic speech.** HeadAudio prototypes expect human speech at 62.5 fps
  analysis. TTS is cleaner but prosody differs; if a voice produces poor visemes, the fix
  is the classifier's gate/gain knobs, not the renderer. Phase B's filmstrip check exists
  to catch this early.
- **Pi audio stack.** ALSA + sounddevice + pygame in one process can fight over devices;
  test audio output on the actual Pi in Phase A, not on the dev box only.
- **LLM latency on Pi 4.** 1.5B Q4 is roughly single-digit tokens/s on CPU; a two-sentence
  reply is ~2–4 s. Sentence streaming hides generation after the first sentence; if it's
  still too slow, drop to 1B or prefill-cached short prompts.
- **Repo boundary.** Settled: the host is a new repo that does `pip install -e
  /path/to/Holly-8bit-6000` and `import holly` — not a submodule with `sys.path` surgery.
  `holly.DEFAULT_BUFFERS`/`DEFAULT_MODEL` resolve against the animation repo's own root, not
  the process CWD, so the host can run from its own directory and still find the baked buffers
  and the vendored model. Keep the animation core host-agnostic as its HANDOFF demands; if the
  host needs a different handoff shape (it currently blits mouth-rect buffers), change the
  *call site*, not the core.

## 7. Budget sanity

Animation: **1.5% of one core during speech** at the 576x432 host bake (60 Hz poll, measured),
~4.7% at native 1152x864, **0% idle** — the dirty check skips ~79% of polls and idle costs
nothing. Piper: real-time or faster on a Pi 4. The remaining budget is all the LLM's — which is
why model size is the one dial that actually determines whether this lives on the Pi or on the
old PC.
