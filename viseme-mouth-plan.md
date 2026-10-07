# Real-time viseme mouth animation plan

Goal: live audio -> viseme poses -> cross-faded mouth layers from a PSD/GIMP file, on low-end CPU.

Previous version of this plan is preserved as `viseme-mouth-plan.v1.md`.

## Reality check (verified 2026-10-07)

- No public standalone OVRVTC repo exists. Meta's viseme engine = EOL Oculus Lipsync binary SDK (closed) + Quest-only Movement SDK.
- Engine choice: port HeadAudio (github.com/met4citizen/HeadAudio, MIT) — MFCC + Gaussian prototypes + Mahalanobis distance, outputs the 15 Oculus visemes in real time, no ML framework. Trained model `dist/model-en-mixed.bin` verified downloadable.
- 15 Oculus visemes: `sil PP FF TH DD kk CH SS nn RR aa E ih oh ou`
- **Timing model revised.** The original continuous ~100 viseme-frames/s model was built for frame-accurate A/V alignment. Watched output (`build/debug/visemes_continuous.mp4`) it reads as flickering, not talking: 10.3 pose changes/s and 16 runs lasting a single 16 ms frame. Replaced with cartoon keyframes — see section 2. Timing accuracy is an accepted trade.

## 1. Asset spec (PSD / GIMP)

- One layer per viseme; layer names exactly the viseme codes above.
- Identical canvas size; mouth registered at the same pixel position in every layer.
- Mouth-only content, transparent elsewhere; static base face layer at bottom.
- Author in GIMP, export as PSD (layer names survive). Read with psd-tools (v1.24.0 verified working).
- Build-time export script: validate layer names, composite each layer to RGBA numpy buffers at render resolution. Runtime never opens the PSD.
- Verified in `viseme.psd` (1152x864 RGB): `sil` is 100% opaque full canvas; the other 14 layers are mouth-only overlays sharing content bbox `(354,476)-(684,736)` — registration is exact. Union mouth crop `x=352 y=474, 335x265` = **8.9% of the canvas**, and patches are baked cropped to it. This is the single biggest lever on frame budget.

## 2. Timing model: cartoon keyframes

The mouth is a **pose**, held long enough to read, changed a few times a second, with a short cross-fade so a change is visible as a change rather than a cut.

Chosen rate (settled by eye on `test_audio.flac`): **12 pose slots/s at 30 fps**.

Render rate is **30 fps**, settled for two reasons: the low-CPU goal (60 fps doubles
draw cost for no gain in pose legibility — both hold a pure pose on 76.4% of frames),
and because the 2-frame fade at 30 fps is the *subtle* look being aimed for. A longer
fade at 60 fps is smoother but reads less like a deliberate pose change.

| Parameter | Value | Measured result on test audio |
|---|---|---|
| Pose slots | 12/s (83.3 ms each) | 93 slots over 7.68 s |
| Actual pose changes | — | **55 total, 7.1/s** |
| Mean hold | — | 138 ms (shortest 83 ms) |
| Cross-fade | **2 frames at 30 fps = 67 ms (chosen)** | one true 50/50 frame per change |
| Cross-fade | 3 frames at 60 fps = 50 ms (rejected: 2x draw cost) | exactly the intended 50 ms |
| Distinct drawn states | — | 54 of 233 frames @30 fps (23%) |

Rules that fall out of the measurements:

- **Achieved switch rate is ~57-59% of the slot rate**, because adjacent slots often pick the same pose. To target a *change* rate, divide by ~0.58: 12/s slots -> ~7 changes/s. 8/s slots -> ~4.5.
- **A fade expressed in seconds can miss its own midpoint.** With a 83-125 ms slot grid and a 33.3 ms frame grid, a 50 ms fade sampled at frame boundaries only ever produced blends of 0.074 and 0.259, never 0.5 — a fade in the numbers and a hard cut on screen. Fades are therefore quantised to **whole output frames**, and the ramp is driven by **frame index**, not time.
- Pooling the analysis frames per slot *is* the smoothing. ~5 analysis frames averaged per slot means one noisy frame cannot move the mouth.
- Sticky hysteresis (new mouth pose must beat the current by 0.05) suppresses flicker
  between similar shapes. It is **not** applied to or from `sil`: `sil` is often the
  argmax of a slot that straddles a speech onset, and a margin there delays mouth
  opening by a whole slot.
- **12/s does not divide 30 fps**, so pose slots alternate between 2 and 3 output
  frames in a symmetric `2,3,3,2` cycle. Measured pure-hold frames per slot: 31 slots
  hold 1 frame, 40 hold 2, 22 hold 3 — i.e. holds swing 67/100/100 ms on a repeating
  grid rather than being uniform. This is a grid artefact, not random jitter, and the
  cross-fade absorbs most of it; it may even read as less mechanical than a metronome.
  If a uniform hold is ever wanted at 30 fps, only 6/s (5 frames) or 10/s (3 frames)
  divide evenly — 10/s measures ~5.7 changes/s. `--key-hz 15` gives a uniform 2-frame
  slot but the 2-frame fade then consumes the whole slot, leaving no pure hold.

## 3. Pipeline

1. **Audio:** the utterance **the application itself plays** — file or TTS buffer (ffmpeg decode / numpy array) -> 16 kHz mono float32. **Mic capture is out of scope** (brief clarified 2026-10-07: the intent was never to animate microphone input; voice input/ASR would be a separate future subsystem of the host project).
2. **Framing:** 512-sample window, hop 256 -> 62.5 analysis fps. (The original "~100 frames/s from hop 256" was wrong: at 16 kHz, hop 256 is 62.5 fps; 100 fps needs hop 160.) Keyframing makes the analysis rate largely irrelevant — it only controls how many frames get averaged per pose.
3. **Features:** 13-15 MFCCs per frame (numpy FFT + mel filterbank), no scipy.
4. **Classify:** phoneme Gaussian prototypes (mean + inverse covariance), Mahalanobis distance, argmin phoneme, phoneme -> viseme lookup. HeadAudio ships the mapping.
5. **Keyframe:** pool weights per 83 ms slot -> argmax pose -> sticky hysteresis -> expand to render fps with a frame-quantised 2-3 frame cross-fade.
6. **Render:** static base layer + mouth-rect-only weighted overlays; at most two overlays blended per frame. **30 fps logical clock, drawn on demand — see section 5.**
7. **Sync:** the runtime is app-audio driven, so analysis runs **ahead of** the playback
    clock — the mouth can know every pose before playback reaches its slot. The ~115 ms
    pooling floor only applies to zero-lookahead streams (see section 5's rewrite).

CPU budget, measured: analysis **0.032 ms/frame** (~0.3% of a core at 62.5 fps). Rendering
was the bottleneck at 61.9 ms/draw and now costs **1.77 ms/draw** (section 4), so the whole
live pipeline sits at roughly **5% of one core** while speaking and **0% while idle**.

## 4. Rendering rule

Never hold-and-cut, never dissolve everything.

- Hold each pose for at least one full slot; change at most 12 times/s.
- Cross-fade over 2-3 output frames only. Longer and no pose is ever held pure.
- `sil` has no overlay patch. Fading toward `sil` means **lowering the active overlay's alpha** so the closed base mouth shows through — that is what makes closures ease instead of snap.
- The frame loop keeps a persistent premultiplied surface and rewrites only the mouth rect (~9% of pixels).

**Fixed.** The original `draw_row` unpremultiplied the whole 864x1152 canvas every frame:
3.3 ms of blend plus **58 ms of wasted conversion**, because the base layer is fully opaque
and 91% of the canvas never changes. Measured before/after:

| | original | mouth-rect, float64 | **mouth-rect, float32** |
|---|---|---|---|
| `draw_row` per call | **61.9 ms** | 4.09 ms | **1.77 ms** (35x faster) |
| resident buffers | ~104 MB | 13.6 MB | **10.8 MB** |
| sustained over the test clip | ~89% of a core | 8.5% | **5.0% of a core** |

Three separate wins, each measured:

1. **Mouth rect only, not the canvas** — 61.9 -> 4.09 ms. Bit-identical (max Δ 0 over 248
   frames), because outside the rect unpremultiplying was provably a no-op.
2. **float32 working precision** — 4.09 -> ~2.4 ms. Needs an explicit `float()` on the
   blend weight: `row[i]/total` is a numpy float64, and under NEP 50 a float32 array times
   a float64 *numpy* scalar silently promotes the whole expression back to float64.
3. **No boolean gather in `unpremultiply`** — 3.27 -> 1.28 ms on the rect. Clamping the
   alpha and using `np.where` avoids the gather/scatter of `rgb[mask] = ...`.

Output is **not** bit-identical any more, and that is accepted: over 308 frames and 307M
pixels, **0.0036% of pixels differ, every one by exactly 1/255, none by 2 or more**. That is
float32 blend rounding where `np.rint` lands on the far side of a .5 boundary.

Two traps found and handled while doing this:
- Multiplying by a float32 *reciprocal* of 255 instead of dividing by 255 introduced its own
  Δ1 error. Dividing directly is free and exact.
- Dropping `np.clip` before the uint8 cast would let an out-of-range value **wrap** (300 ->
  44, -5 -> 251) and appear as silent speckle. Clipping measured free, so it stays.

`test_mouth_rect_only_matches_the_full_canvas_algorithm` keeps the original whole-canvas
float64 computation as a **correctness oracle** with a tolerance of exactly 1, and asserts
**zero** difference outside the mouth rect where no float32 rounding can excuse it. Verified
it still bites: a deliberate 6% blend error fails both oracle tests.

`Renderer.mouth_rect` is now exposed as `(x, y, w, h)` so a host application can blit
88,775 pixels instead of 995,328.

**Aliasing contract:** `draw_row` returns the renderer's *persistent* buffer, not a fresh
array — copying 3 MB per draw is the cost this avoids. Callers retaining a frame across
draws must `.copy()` or use `Renderer.snapshot()`. This already bit once: the contact sheet
would have rendered 40 identical tiles. Pinned by
`test_collected_frames_must_be_copied_to_stay_distinct`.

## 5. Runtime model: on-demand, audio-driven

The target is **live display inside an application**, not video generation. So 30 fps is
the **logical animation clock** (the grid poses and fades are quantised to), *not* a
display refresh rate. Frames only advance while audio plays, and a frame only costs
anything if it differs from the one before it. The intended hardware is any low-end PC,
not specifically a Pi — the budget below is written per-core so it transfers.

Measured on the 12/s @ 30 fps timeline for `test_audio.flac` (233 frames, 7.73 s):

| | |
|---|---|
| frames needing a redraw | **111 (47.6%)** — the other 52.4% are identical to their predecessor |
| frames at pure `sil` (zero draw) | 75 (32.2%), in 8 idle periods averaging 312 ms |
| sustained draws/s **while speaking** | **19.6** (65.2% of speech frames change) |
| averaged over the whole clip | 14.3 draws/s |
| longest unbroken draw burst | 6 frames = 200 ms |

Consequences for the budget: at the fixed draw cost (**1.77 ms**, see section 4), sustained
drawing while speaking is `19.6 x 1.77` = **~3.5% of one core**, ~**5.0% measured over the
clip**, and **0% while idle**. Analysis adds 0.3%, so the whole live pipeline is roughly
**4-5% of one core during speech** — comfortable on a low-end PC, and it scales with core
speed rather than depending on a Pi being the target.

For comparison, the same draw loop on the *unfixed* renderer cost 61.9 ms/draw, which at
19.6 draws/s is **121% of a core** — more than one core, so frames would drop and the
mouth would visibly fall behind the voice.

The continuous model, by contrast, needs a redraw on **473 of 480 frames (98.5%)** — its
one-pole envelope changes every frame, so there is nothing to skip — and idles on only
1.5% of frames. **61.6 draws/s vs 14.3 draws/s: keyframing cuts actual draw work by 4.3x.**
That is a CPU win on top of the visual one, and it was not the reason the change was made.

Rules this imposes on Phase 3:

- **Clock from audio position, not wall time.** Frame index = `audio_time * 30`. If the
  app falls behind, skip to the newest pose rather than queueing stale draws.
- **Dirty-check before drawing.** Compare the new weight row to the last drawn one;
  skip the draw when identical. This halves the work for free.
- **Idle must be a true zero-draw state**, with the mouth settled on the base layer.
  Note this conflicts with Phase 4's idle micro-motion: breathing/blinking would
  reintroduce continuous draws, so if added it must be throttled to ~1-2 draws/s.
- **End of speech must settle to `sil`** rather than freezing on the last vowel. The VAD
  gate already produces this offline; the runtime must not latch the final pose.

### Latency floor — only for zero-lookahead streaming

Pooling is non-causal **relative to audio arrival**: a slot's pose can only be computed
when that slot's last analysis frame has arrived. For a stream that must be analysed
exactly as it plays, the mouth is at least one slot behind:
**~115 ms typical / ~149 ms worst at 12/s** (32 ms window + 83 ms slot + 33 ms frame
quantise); 157/190 ms at 8/s.

**But the audio source is the application's own output** (clarified 2026-10-07), so the
runtime can analyse ahead of the playback clock:

- Whole utterance in hand (the normal TTS case): every pose is known before playback
  reaches its slot — **the floor is zero and any fade shape is causal**.
- Chunked TTS: the floor is whatever the analysis buffer fails to cover. Keep it at least
  one slot (83 ms) ahead of playback and `center` remains valid.

**Consequence for `--fade-shape`:** the old "live loop must use `lag`" rule was premised
on microphone input and is **retired**. `center` — the look of the shipped configuration
(`visemes.mp4`) — is valid for the runtime as long as analysis leads playback. The choice
is now purely aesthetic (`lead` = cartoon anticipation). `test_lead_and_lag_position_the_fade`
still pins the mechanics. If a future zero-lookahead stream is ever required, `lag` is
still there.

## 6. Phases

- **Phase 0 — PSD audit/export script.** **Done.** `tools/export_visemes.py`: asserts expected layer names, validates the asset spec, bakes premultiplied uint8 RGBA buffers (`base` full canvas, `patches` cropped to the mouth rect) plus a manifest. Self-check vs PSD composite: mean diff 0.19, max 14.
- **Phase 1 — offline proof.** **Done.** One WAV/FLAC -> timeline JSON -> debug filmstrip + muxed video with waveform band and playhead, so A/V offset is visible.
- **Phase 1b — cartoon keyframe retiming.** **Done.** `holly/keyframes.py`. Replaces the attack/release envelope model for the default path; `--continuous` keeps the old one for A/B.
- **Phase 2 — port HeadAudio classifier.** **Done** (2026-10-07, + 2b look fixes). Front-end diff found 10/18 parameters mismatched; `features.py` rebuilt to mirror HeadAudio exactly. `HeadAudioClassifier` parses the model (phoneme->viseme map embedded in record headers), Mahalanobis argmin, silSensitivity, vote ring, log-energy VAD; verified against HeadAudio's own JS distance oracle. Shipping config chosen by the user = the zero-flag default (`visemes.mp4`).
- **Phase 3 — live animation runtime (app-audio driven).** **3a Done (2026-10-07).**
  `holly/runtime.py::HollyFace` (`speak`/`row_at`/`frame_at`) + `tools/reference_player.py`;
  32 contract tests in `tests/test_runtime.py`. Measured through the runtime on
  `test_audio.flac`: 13–18 ms analysis per 7.68 s utterance, 27.5% of 60 Hz polls cost a
  draw, 0 draws across 60 s of idle. Re-scoped 2026-10-07: the
  mouth animates to audio **the host plays** (TTS buffer/file), never to a microphone.
  Integration model confirmed: the main project generates whole TTS utterances, then
  plays them with the avatar in time — so **3a precompute is the path and 3b streaming
  is not needed** (revisit only if the host adopts chunked TTS). The host app itself
  **does not exist yet**: this repo is the animation component of a future project.
  **3a precompute (main path):** whole utterance -> existing pipeline in memory -> sample
  poses against the playback-position clock -> dirty-checked on-demand draw. Buffer-first
  means the auto le-gate works as-is, `center` fade is causal, and the latency floor is
  zero. Deliverable = host-agnostic component API (pure numpy in/out) + standalone pygame
  reference player. Handoff shape (callback / mouth-rect / full frame) deferred until the
  main project picks its stack; core must not pre-commit. ~~Fix the full-canvas
  unpremultiply~~ **done** — see section 4.
- **Phase 4 — polish.** `sil` idle micro-motion (throttled — see section 5), weight
  normalisation, per-viseme gain curves.

## 7. Alternatives

- **Oculus Lipsync SDK binary** (Windows C API, EOL): skips the port, closed binary.
- **Rhubarb Lip Sync:** precomputes 9-shape cues when audio is known in advance
  (TTS/dialogue). Zero runtime cost, and its discrete-cue output is a natural fit for the
  keyframe model — worth revisiting if any audio is pre-recorded.

## Deps (verified on this machine)

- python 3.12 venv (plan originally said 3.13; 3.12 works, no action needed)
- psd-tools 1.24.0, numpy 2.5.3, pillow 12.3.0 pinned in `requirements.txt`
- **no scipy** — the MFCC front end is pure numpy on purpose
- ffmpeg 6.1.1 + libx264 present; decode and video mux shell out to ffmpeg, so there is no
  decoder dependency
- pygame is pip-installable but **not installed yet** — optional, only as a standalone
  test host; the real host application may already own a surface to draw into.
  **sounddevice/libportaudio2 are NOT needed** — the runtime is retired from the
  mic-loop framing (2026-10-07): the audio source is the host's own output.

## Open questions

- ~~Render at 30 or 60 fps~~ **Settled: 30 fps** — low-CPU goal, and the 2-frame fade is the
  intended subtle look. See section 2 for the slot-grid consequence.
- ~~Fix the full-canvas unpremultiply~~ **Done, section 4.**
- ~~Drop the region surfaces from float64 to float32~~ **Done: 4.09 -> 1.77 ms/draw.**
- Is the `2,3,3,2` hold-length swing at 12/s @ 30 fps visible enough to matter? Compare
  against `--key-hz 10` (uniform 3-frame slots, ~5.7 changes/s) on the real classifier.
- ~~How does the host application want the frames~~ **Deferred: the host app doesn't
  exist yet** (2026-10-07) — this repo builds the animation *component*. Core stays
  host-agnostic (pure numpy); a pygame reference player proves the runtime. When the
  main project picks its stack, choose the handoff then — mouth-rect buffer is the
  standing recommendation (cheapest for both sides).
- ~~`--sticky` 0.05 and `--pool mean` were picked against the placeholder classifier~~
  **Re-tuned against the real one (Phase 2b sweep):** keep `mean` (`max` is degenerate for
  one-hot output); sticky 0.05 is inert-to-harmless and the shipped config keeps it.
- Bake at native 1152x864 or `--scale 0.5` for the target hardware. A retro pixel-art
  target would also change the scaling story (nearest-neighbour instead of the bilinear used
  today) — not yet specified.
- Analysis hop 256 (62.5 fps) or 160 (100 fps). Low priority now that keyframing owns the
  smoothing.
