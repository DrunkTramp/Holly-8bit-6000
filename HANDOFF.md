# Handoff: Phase 3a (component API + reference player) is done; the host application is next

Paste the **Kickoff prompt** at the bottom into a new session. Everything above it
is state that session needs and cannot infer from the code alone.

Plan of record: `viseme-mouth-plan.md`, already rewritten around the keyframe timing
model. The original continuous-100fps plan is preserved as `viseme-mouth-plan.v1.md`.

## Status

| Phase | State |
|---|---|
| 0 — PSD audit + buffer export | **Done.** `tools/export_visemes.py` |
| 1 — offline audio -> timeline -> debug filmstrip | **Done.** `tools/make_timeline.py`, `tools/render_filmstrip.py` |
| 1b — keyframed/cartoon retiming | **Done**, 12/s @ 30 fps settled. `holly/keyframes.py` |
| renderer cost fix | **Done.** 61.9 -> 1.77 ms/draw, float32. `holly/render.py` |
| 2 — port HeadAudio classifier to Python | **Done.** `holly/classify.py::HeadAudioClassifier`; front end re-matched first. |
| 3a — component API + reference player | **Done (2026-10-07).** `holly/runtime.py::HollyFace` — `speak()`/`row_at()`/`frame_at()` — plus `tools/reference_player.py`. See "Phase 3a". |
| host groundwork — `--resample nearest`, 576x432 settled, `pip install -e .` | **Done (2026-10-07).** `pyproject.toml`, `tests/test_bake.py`; a latent mouth-rect bug in every scaled bake was found and fixed. See "Scaled bakes". |
| 3b — streaming-TTS refactor | Not started, **conditional**: build it only if the host adopts chunked TTS that plays as it arrives. |
| 4 — polish (idle micro-motion, gain curves) | Not started |

147 tests pass: `.venv/bin/python -m unittest discover -s tests -t .`

## Change of brief: Phase 3 is not a mic loop (2026-10-07, later)

The user clarified the project's actual shape: **this repo is the animation part of a
larger application**, and Holly's mouth animates to **audio the application itself plays**
(TTS output, files). **At no point is the intent to animate microphone input.** Voice
input may exist in the larger project someday, but that is ASR — a separate subsystem;
this pipeline never needs a microphone.

**Integration model confirmed (2026-10-07):** the main project generates TTS audio, then
plays it in the app with the avatar speaking in time with it. Whole utterances, generated
before playback — so **3a (buffer-first) is the path; 3b (streaming) is not needed**
unless the host later adopts chunked TTS. Precompute cost is a rounding error: measured
~0.06 ms/frame -> **~30 ms to analyse a 7.7 s utterance**, done before playback starts;
every pose is known before its slot plays.

**The host app does not exist yet** (clarified 2026-10-07): this repo is the animation
*component* of a larger project to be built later. Therefore Phase 3a's deliverable is
(1) a **host-agnostic component API** — pure numpy in/out (`speak(samples) -> timeline`,
`row_at(audio_time) -> weight row`), no windowing or audio-output dependency in the core,
so whatever stack the main project eventually picks can consume it directly; and (2) a
**standalone pygame reference player** — plays any audio file with Holly speaking in
sync, proving playback-position clocking, dirty-check, zero-draw idle, and no final-pose
latching; it doubles as the eyeball harness for all future tuning (Phase 4 included).
The handoff-shape question (weight-row callback vs premultiplied mouth-rect buffer vs
full frame) is **deferred until the main project picks its stack** — the core must not
pre-commit. Standing recommendation for when the time comes: mouth-rect, cheapest for
both sides (88,775 px vs 995,328).

Consequences, all recorded where they matter:

- **The ~115 ms latency floor is retired for the main path.** It only applied because
  mic audio arrives non-causally. App audio can be analysed **before** the playback clock
  reaches each slot: whole-utterance buffer -> floor is zero; chunked TTS -> keep the
  analysis buffer >= one slot (83 ms) ahead of playback.
- **The "live loop must use `--fade-shape lag`" rule is retired with it.** `lag` existed
  only to keep the live path causal. `center` — the look of the shipped `visemes.mp4` —
  is valid for the runtime. The choice is now purely aesthetic (`lead` = anticipation).
  `test_lead_and_lag_position_the_fade` still pins the mechanics; nothing code-side changes.
- **No running noise-floor VAD work needed.** The auto `le` percentile gate is valid
  because the utterance buffer exists at call time; clean TTS audio is exactly what it
  was tuned on. `derive_vad_thresholds`/`derive_le_thresholds` whole-file passes stay fine.
- **`sounddevice`/`libportaudio2` are NOT needed.** pygame stays optional (test host only).
- The streaming refactor (per-frame stateful VAD counters, vote ring, pre-emphasis carry)
  is **deferred to a conditional 3b** — build it only if the host's TTS genuinely streams
  chunks that play as they arrive. 3a (buffer-first) is most of a normal TTS integration.

## Phase 3a: the component API and the reference player (2026-10-07)

`holly/runtime.py` is the whole deliverable: a host-agnostic class with no windowing,
audio-device or microphone dependency, holding the three things the runtime model needs.

```python
face = HollyFace(renderer=Renderer(DEFAULT_BUFFERS))   # 576x432 nearest bake; renderer optional
face.speak(tts_buffer, 16000)                 # analyse now, enqueue on the face clock
frame, rect, changed = face.frame_at(t)       # t = the host's audio output position
if changed:
    blit(frame, rect)                         # mouth rect only
```

- **`speak(samples, rate=16000, *, at=None) -> Utterance`** — buffer in, no file, no
  ffmpeg, no JSON round trip. Queues back-to-back by default; `at=` places an utterance
  explicitly (and rejects overlap — the renderer has one mouth).
- **`row_at(t) -> (15,)`** — never None: outside any utterance it returns `IDLE_ROW`,
  the base pose alone. **This is the no-final-pose-latch rule, enforced in the core** so
  no host has to remember it. `frame_index_at(t)` is `-1` in the same territory.
- **`frame_at(t) -> (frame, mouth_rect, changed)`** — the dirty check lives here: the row
  is compared against the last row drawn, and `changed=False` means blit nothing. Idle is
  therefore a true zero-draw state, with exactly one draw to restore the base mouth when
  an utterance ends.
- **Sample-rate contract is a hard error, not a silent resample.** The HeadAudio front end
  is pinned to 16 kHz mono float32, so `analyse()` raises for anything else and tells the
  host to keep the native-rate buffer for playback. Analysis and playback buffers must
  correspond 1:1 in time; the core refuses to be the place where that gets fudged.
- Utterance `audio_duration` (not timeline length) is what the queue advances by, because
  the queue must stay honest against the audio clock. `expand_keys` rounds the key count up
  to whole slots and then the timeline up to whole output frames, so a timeline can run past
  its audio by **up to one slot plus one frame** (116.7 ms worst case at 12/s @ 30 fps;
  measured 87 ms on `test_audio.flac`) — exposed as `Utterance.tail` and simply never reached
  by the clock, which is the right outcome: the tail is a held final pose, and latching one
  is forbidden.

**Measured through the runtime on `test_audio.flac` (7.68 s):** analysis **13–18 ms**
(~1.7–2.4 ms per second of speech, all of it before playback), 233 frames, 93 poses. At
the player's 60 Hz poll: **461 polls -> 127 draws, 334 skips** (27.5% of polls cost a
draw, 16.5 draws/s of speech). **60 s of idle polls: 0 draws.** These are the three
claims the phase existed to prove.

`tools/reference_player.py` is the only file in the repo that touches pygame, and it is
the eyeball harness for all future tuning: it mirrors `make_timeline.py`'s flags
(`--key-hz/--render-fps/--fade-s/--fade-shape/--pool/--sticky/--no-gain`), takes repeated
`--audio` to queue several utterances with `--gap` silence between them, and blits **only
the mouth rect** (`pygame.display.update(rect)`). The clock is `mixer.music.get_pos()`,
not `time.time()`; the loop refuses to jump the clock before the mixer has spun up, and
sets it past the end once the queue drains so the return-to-rest path is exercised.

### Sync geometry is now measured, not just eyeballed (2026-10-07)

Phase A's exit criteria were "sync verified by eye, idle is zero-draw, no final-pose latch,
mouth-rect-only blits". Three of the four were already automated; "by eye" was not, so
`tools/check_sync.py` measures it: a train of bursts at **known** times, then the offset
from each event to the **first** frame whose mouth leaves the base pose.

On a 12-burst harmonic-vowel train at 500 ms spacing, 12/s @ 30 fps:

| classifier | `lead` | `center` (shipping) | `lag` | spread of `center` |
|---|---|---|---|---|
| openness-ramp | −66.7 ms | −33.3 ms | 0.0 ms | 33.3 ms |
| headaudio | −66.7 ms | −33.3 ms | 0.0 ms | **0.0 ms** |

Three things worth carrying into the host:

- **The shipping `center` fade opens the mouth exactly one render frame early — 33 ms.**
  That is the fade straddling the slot boundary by design, and it is inside the half-slot
  (41.7 ms) tolerance the cartoon grid allows. It is also the kind of thing that is hard to
  see and easy to state, so if Holly ever reads as "slightly ahead", `--fade-shape lag` is
  the zero-offset option and the cost is that the old pose holds to the boundary.
- **Jitter is zero on the real classifier.** Every one of the 10 detected bursts reacted at
  the identical frame. Consistency is what makes a mouth read as attached to a voice; a
  33 ms lead that never varies looks better than a ±40 ms scatter that does.
- **Coverage is not a timing problem.** The real classifier ignored 2 of 12 vowel bursts. A
  synthetic harmonic tone is speech-like without being speech; whether a *TTS* voice gets
  good visemes is Phase B's filmstrip question, judged on the gate/gain knobs, not here.

`tests/test_sync.py` pins all of it: detection count, the one-frame `center` lead, the
strict `lead < center < lag` ordering one frame apart, and jitter bounds (≤ 1 slot for the
stub, ≤ 1 frame for the real classifier). It inherits its own test case onto the real
classifier rather than duplicating the assertions.

### Scaled bakes, nearest-neighbour, and a bug that only scaled buffers had (2026-10-07)

`tools/export_visemes.py` gained `--resample {bilinear,nearest}` (default `bilinear`, so the
existing behaviour is untouched), recorded as `manifest["resample"]`.

Measured on the base layer, nearest versus a proper area downsample (`BOX`): mean deviation
**0.61/255 at 1/2** and **0.53/255 at 1/3**, with 2.1% / 1.3% of pixels off by more than 8.
So for *this* asset the filter choice is nearly a no-op — the source is already blocky pixel
art, and nearest's real virtue is that it never invents an intensity the palette lacks
(`tests/test_bake.py` asserts exactly that subset property).

Baking at 0.5 then crashed `Renderer._over`: `(133,168,3)` against `(132,168,1)`. The cause
was in the manifest, not the renderer — `bake()` built the scaled mouth rect by rounding
**each edge independently**, while `resize()` rounds the **product**: `round(265 * 0.5) = 132`
but `round((y1+1)*0.5) - round(y0*0.5) = 133`. The rect now comes from the buffers that were
actually produced. This had never been caught because nothing in the offline pipeline ever
loaded a scaled bake — `build/visemes_half/` shipped with the same wrong rect. `tests/test_bake.py`
pins the invariant across scales (1.0, 0.5, 1/3, 0.7, 2.0) and constructs a real `Renderer`
from each bake, which is the host's actual entry condition.

Baked and verified through `HollyFace` (identical 233 frames / 107 draws at every size — the
timeline is resolution-independent, as it must be). Cost per `draw_row`, median of 233 draws:

| buffers | canvas | mouth rect | px/draw | ms/draw | core @ 12 draws/s |
|---|---|---|---|---|---|
| `build/visemes/` (native, offline tools) | 1152x864 | 335x265 | 88,775 | 2.84–3.06 | ~3.5% |
| **`build/visemes_pixel/` (the host asset)** | **576x432** | **168x132** | **22,176** | **0.71–0.73** | **~0.86%** |
| `build/visemes_third/` | 384x288 | 112x88 | 9,856 | 0.33–0.34 | ~0.40% |
| `build/visemes_half/` | 576x432 | 168x132 | 22,176 | — | bilinear, re-baked with the fix |

**Settled (2026-10-07): the host canvas is 576x432**, because the face sits in a *corner* of
the host UI rather than filling a window — and the mouth rect is the only thing a host ever
blits, so halving the linear size quarters the per-draw cost. `holly.runtime.DEFAULT_BUFFERS`
now points at `build/visemes_pixel/visemes.npz` (nearest, 0.5), with `NATIVE_BUFFERS` kept for
the debug filmstrip. End-to-end at the player's 60 Hz poll over `test_audio.flac`: **1.52% of
one core sustained** at 576x432 against 4.66% at native, same 107 draws. `tests/test_runtime.py
::TestHostBake` pins the size, the manifest's `nearest`/`0.5`, and the 4x rect ratio.

These ms figures are medians measured on this box across the whole `draw_row` call; the
earlier recorded **1.77 ms/draw** was the mouth-rect work at native under a different
harness. The decision-relevant number is the ratio, which is exactly the pixel count.

## Phase 2: HeadAudio classifier port (2026-10-07)

**The front end had to be rebuilt before the port could work.** The diff against
HeadAudio's `modules/mfcc.mjs` + `modules/parameters.mjs` found 10 of 18
parameters mismatched the old `features.py` choices. `features.py` is now a
line-by-line mirror of HeadAudio's front end; the shipped prototypes depend on
every one of these:

| Parameter | HeadAudio (now ours) | old Holly stub-era value |
|---|---|---|
| window | 512 @ 16 kHz, **Hamming** 0.54−0.46cos | 512, Hann |
| hop | 256 (62.5 fps) | 256 ✓ |
| pre-emphasis | 0.97, **streaming state** (taken from the overlap sample; frame 0 starts at 0) | 0.97 per-frame, zero initial |
| power bins | `|X|²/N`, k=0..255 (**Nyquist excluded**) | k=0..256 |
| mel bands | **40**, **30–7800 Hz**, plain triangles (NO area normalisation), bin=`floor(f·512/sr)`, f0-warp (identity at 150 Hz default) | 26, 0–8000, Slaney-normalised |
| coefficients | **12** (c1..c12 — **c0 dropped**; energy is separate `le=log10(ΣP)`) | 13 incl. c0 |
| DCT | DCT-II rows 1..12, scale √(2/40) — NOT orthonormal at c0 | orthonormal c0..c12 |
| cepstral liftering | **yes, L=22** on c1..c12 | none |
| tanh compression | **yes, `tanh(v)`** (r=1.0) — coefficients are bounded ±1 | none |
| deltas / CMN | disabled / none | none ✓ |

`FeatureSet` gained `log_energy` (HeadAudio's `le`, its VAD input). The
`OpennessRamp` stub still works off `centroid`/`rms`, but its outputs shifted
slightly because pre-emphasis/windowing changed — the archived stub artefacts
(`*_stub*` in `build/debug/`) are the old-feature ones.

**Model facts** (`model/model-en-mixed.bin`, 14,352 bytes, provenance + fetch
commands in `model/README.md`): 39 records × 368 bytes = big-endian packed
phoneme header (2 floats: codepoints at bytes 0-3, group byte 5, **viseme id
byte 7**), 12 float32 mu, 78 float32 inverse-covariance lower-triangle entries
(row-major `i, j<=i`). 38 IPA phonemes + synthetic `s1` silence. **The
phoneme→viseme map is embedded in the records** — parsing the model IS loading
the map (`GaussianModel.phoneme_viseme_map()`), and `HA_VISEME_TO_CANONICAL`
translates HeadAudio ids (`aa,E,I,O,U,PP,SS,TH,DD,FF,kk,nn,RR,CH,sil`) to our
canonical order (`sil,PP,FF,TH,DD,kk,CH,SS,nn,RR,aa,E,ih,oh,ou`). Note
I/O/U ↔ ih/oh/ou naming. `OPENNESS_ORDER` survives only inside the stub.

**Verified against the reference implementation, not just self-consistent:**
`tests/fixtures/headaudio_distances_oracle.csv` is HeadAudio's own JS output —
the 39×39 Mahalanobis matrix at the prototype means (silSensitivity 1.2 baked
in, `toFixed(1)`). Our float64 port matches every cell within 0.05 — i.e. to
the oracle's own rounding precision. Pinned by
`test_distances_match_the_js_classifier_output`, which also pins record ORDER
(phoneme header equality) and the lower-triangle packing orientation (a
transposed rebuild would pass every shape check and silently scramble accuracy —
`test_mahalanobis_matches_the_js_quadratic_form` cross-checks against a literal
transcription of the JS loop instead).

**Classifier behaviour** (`HeadAudioClassifier`, all defeatable flags, defaults
faithful to the JS): one-hot (n,15) weights; Mahalanobis argmin with
`--sil-sensitivity 1.2` (sil distances divided); 6-frame majority-vote ring
(`--no-vote` off-switch); vadgate.mjs log-energy hysteresis −40/−50 dBFS,
10 ms (`--no-vad`). VAD-inactive frames are `sil` and skip the vote ring, as
HeadAudio's `continue` does. Not ported (live-mic-only concerns): the `sc`
speaker-calibration prototype (group 255) and started/ended event bookkeeping.

**Measured on `test_audio.flac` with the real classifier** (12/s @ 30 fps,
center fade, sticky 0.05): analysis cost **0.06 ms/frame** (features ~0.023 +
classify ~0.036, budget was 1 ms); **7.8 pose changes/s** (65% of ceiling —
stub was 57-59%); mean hold 127 ms; redraws 121/233; idle `sil` only 6.5% in 2
periods. Histogram spread as expected: `CH`/`DD` 19.4% each, `E` 12.9%, `ih`
9.7%, `aa` down from 19.4%→4.3%, `sil` 35.6%→6.5%. 14 of 15 visemes used —
`oh` never fires on this clip. Spot-check against the clip's embedded transcript
(see Asset facts): "this is" → `TH TH TH SS ih ih ih`, "switching" → `ɹ→RR`,
`ʧ→CH`, `ŋ→nn`. Phonetics, not openness.

**Re-tuning `--sticky`/`--pool`/`--fade-shape` against the real classifier**
(`tools/sweep_phase2.py`, kept for Phase 3 reuse):

- One-hot pooling changed the game: a 12/s slot holds ~5 analysis frames, so
  vote fractions come in ~0.19 steps. **`--sticky 0.05` is now inert** (never
  fires); 0.2 → 7.4 changes/s (blocks 3-vs-2 slots, allows 4-vs-1); 0.35 → 7.2/s
  (blocks even 4-vs-2 — too sticky). Not a module default change (keyframes.py
  is off-limits per the seam rule); **0.2 is the recommended value if you want
  real hysteresis**, and `visemes_sticky02.mp4` exists to A/B it by eye — but that
  video is broken-gate-era and was superseded when the gate was fixed; the shipped
  configuration (chosen 2026-10-07) keeps the default `--sticky 0.05`.
- **`--pool max` is degenerate with one-hot output**: every viseme that appeared
  in the slot ties at 1.0 and canonical order makes `sil` win — it reads as
  "any sil frame closes the mouth". Keep `mean`.
- **`--sil-sensitivity` is inert on this clip**: 1.0/1.2/1.5 give identical
  results because the VAD gate, not the `s1` prototype, decides silence here.
  It may start to matter if the host's audio is noisier than clean TTS.
- The vote ring is worth keeping: `--no-vote` raises the change rate 7.8→8.3/s.
- `--fade-shape` does not change pose statistics (verified: center/lag/lead
  identical counts) — it only positions the cross-fade. `lag` output regenerated
  with the real classifier for the live-path eyeball check — a requirement later
  retired by the Phase 3 re-scope (see "Change of brief: Phase 3 is not a mic loop").

### Phase 2b: "the stub still looks best" — diagnosed and fixed (2026-10-07)

User watched the Phase 2 output and preferred the stub. Diagnosis
(`tools/diagnose_phase2.py`, kept): the classifier itself is **accurate** —
against the clip's embedded transcript, "this is" → `TH…SS ih ih ih`,
"unified" → `ih ih ih nn nn nn DD`, "speaking" → `PP…kk`, "switching" → `CH`,
"with" → `TH`; and the nearest-prototype distance p50 is 13.5 ≈ 12, which is
the expected χ² for in-distribution 12-dim samples (front end confirmed matched).
The bad look had two causes, neither of them classification:

1. **The shipped VAD gate never closes on this recording.** Quiet frames sit at
   `le` −4.9…−4.4; HeadAudio's fixed close threshold is −5.0 (−50 dBFS), so the
   gate stayed open **100%** of the clip: the mouth hung open through every
   pause (89/89 genuinely-quiet frames classified as speech). The stub's
   *percentile-derived* gate closed 35.6% of the time — that contrast was most
   of what "the stub looks better" was seeing.
   Fix: `derive_le_thresholds()` (same p20/p40 culture as `derive_vad_thresholds`,
   min gap 0.5 in log10) used when `--classifier headaudio --vad auto` (now the
   default). On this clip: le −4.52 → −3.28.
2. **Phoneme-true output is ~60% narrow-aperture consonants** (CH/DD/PP/SS/TH/kk),
   held at full opacity on the key grid. HeadAudio's own avatar driver hides this
   with 100 ms attack/decay alpha easing and `visemeMaxs` caps (0.65, 0.75 for
   PP/FF) — consonants never reach full strength there. Our hard keyframe holds
   showed every DD at 100%, which reads as teeth-clenching.
   Fix: `gain=True` on `HeadAudioClassifier` (off: `--no-gain`) applies
   `HA_VISEME_MAXS` at the weight level. **Honest caveat:** `expand_keys`
   rebuilds full-opacity pose rows, so the gain acts as a *pooling-time sil
   bias* (speech frames carry sil 0.25-0.35, so slots with mixed votes settle to
   sil more often) — not as a render-time alpha cap. A true alpha cap would need
   keyframes.py/timeline to carry sub-unity weights; deliberately not done.

Measured after the fixes (12/s @ 30 fps, center fade):

| | gate only (`--no-gain`) | gate + gain (**new default**) | gate + gain + `--smooth-before-keys` |
|---|---|---|---|
| pose changes/s | 6.5 | **6.9** | 4.9 |
| idle `sil` | 26.2% in 5 periods | **33.5% in 13 periods**, mean 200 ms | 33.0% in 7 periods |
| distinct poses | 12 | 12 | 10 |
| mean top weight | 0.893 | 0.886 | 0.918 |

Videos to compare against `visemes_stub.mp4`: `visemes.mp4` (gate+gain, the new
default), `visemes_ha_gate_only.mp4`, `visemes_ha_eased.mp4`. The earlier
sweep numbers in this file (sil 6.5%, 7.8 changes/s) were measured with the
broken gate and are superseded.

**User verdict (2026-10-07, watched):** gate+gain vs gate-only is "difficult to
tell" — measured: only 33/233 output frames differ, all of them sil-vs-pose
(gain adds ~1.1 s of micro-pause closures). The eased variant was rated **worst**
— pre-pooling attack/release blurs the consonant transitions; do not ship it.
`visemes.mp4` (gate+gain) stands as the default. The gain is kept not for its
visible effect but because it is HeadAudio's own display model and the base a
true alpha cap would build on (see next paragraph).

**Settled (2026-10-07):** the user re-watched the fixed set against
`visemes_stub.mp4` and chose **`visemes.mp4` (real classifier, auto gate + gain)
as the shipping driver** — the real classifier now beats the stub on looks, and
the stub reverts to being an A/B baseline only. This configuration is also the
zero-flag default of `tools/make_timeline.py`, so nothing extra needs pinning.

**What would actually make the gain visible:** a true `visemeMaxs` alpha cap —
carrying the pose weight through `expand_keys` so held consonants render at
0.65/0.75 over the base face instead of full opacity. The plumbing is nearly
there (`Keyframe.weight` already carries the pooled confidence but expansion
ignores it), and it stays inside every existing contract: rows still sum to 1,
still have at most two non-zero visemes, renderer and timeline format unchanged.
This touches `holly/keyframes.py`, which the Phase 2 seam rule fenced off — it is
now a user-approved polish decision, not a Phase 2 change. Not done.

## Change of brief (2026-10-07)

After watching `build/debug/visemes.mp4`, the continuous 100 fps model was
abandoned in favour of **cartoon timing**: pose changes a few times a second with a
very short (~0.05 s) cross-fade. The mouth is a *pose* held long enough to read, not a
signal tracked frame-by-frame. Timing accuracy against the audio is explicitly a
trade we accepted — legibility wins.

**Rate chosen by eye: `--key-hz 12` (~7 changes/s)**, which is now the module default
in `holly/keyframes.py::KEY_HZ`. The user compared 8/s and 12/s and preferred 12/s,
noting it also reads well against a real person, not just the retro pixel-art style
they are moving toward. `viseme-mouth-plan.md` has been rewritten around this model;
the original is preserved as `viseme-mouth-plan.v1.md`.

The old continuous path is still available as `--continuous` for A/B.

## Runtime model: on-demand drawing, not a video loop

**The end goal is live display inside an application, not video generation.** So 30 fps is
the **logical animation clock** — the grid poses and fades are quantised to — and *not* a
display refresh rate. Frames only advance while audio plays, and a frame only costs
anything if it differs from the one before it.

Measured on the chosen 12/s @ 30 fps timeline (`holly/keyframes.py::draw_stats`, printed
by `make_timeline.py`). *Stub-era: with the real classifier the same clip redraws
121/233 frames and idles only 6.4% in 2 periods — the mechanics below are unchanged,
the speech/idle balance shifted.*

| | |
|---|---|
| frames needing a redraw | **111 of 233 (47.6%)** — 122 are identical to their predecessor and free |
| frames at pure `sil` (zero draw) | 75 (32.2%) in **8 idle periods**, mean 312 ms |
| sustained draws/s while speaking | **19.6** (14.3/s averaged over the clip) |
| worst unbroken burst | **6 consecutive draws in 200 ms** |

At the mouth-rect-only blend cost (3.6 ms) that is ~22 ms of work inside 200 ms of wall
clock: **~11% of a core at peak, ~4% averaged, 0% while idle.** The plan's "rendering
dominates" concern is real but is only paid during speech bursts.

Phase 3 must therefore: clock from **audio position** (`frame = audio_time * 30`), not
wall time; **dirty-check** the weight row against the last drawn row and skip when equal;
treat idle as a true zero-draw state settled on the base layer; and **not latch the final
pose** when speech ends. Note the tension with Phase 4: idle micro-motion would
reintroduce continuous draws, so it must be throttled to ~1-2 draws/s.

For comparison, the continuous path needs a redraw on **473 of 480 frames (98.5%)** — its
one-pole envelope changes every frame, so there is nothing to skip — and idles on only
1.5% of frames. That is **61.6 draws/s vs 14.3 draws/s: keyframing cuts actual draw work
by 4.3x**, which is a CPU win on top of the visual one.

### Latency floor (mic-era; retired for the main path)

Pooling is non-causal **relative to audio arrival**: a slot's pose is only computable
when that slot's last analysis frame arrives. For a stream analysed exactly as it plays,
the mouth lags at least one slot: **~115 ms typical / ~149 ms worst at 12/s** (32 ms
window + 83 ms slot + 33 ms frame quantise); 157/190 ms at 8/s.

**The Phase 3 re-scope removes this for app-audio**: the runtime analyses the host's own
output ahead of the playback clock, so poses are known before their slots play. Whole
buffer -> zero floor; chunks -> floor = whatever lookahead the buffer doesn't cover
(keep it >= 83 ms).

**`--fade-shape` consequence, rewritten:** the old "live loop must use `lag`" rule existed
only because mic audio can't know the next pose early. It is **retired** — `center` (the
shipped `visemes.mp4` look) is valid for the runtime; `lead` is available if anticipation
is ever wanted; `lag` remains for a hypothetical zero-lookahead stream. Pinned mechanics:
`test_lead_and_lag_position_the_fade`. The "delay the audio by ~115 ms" advice was
mic-era and no longer applies.

**Render rate settled at 30 fps.** 60 fps was measured and rejected: both hold a pure
pose on 76.4% of frames, so 60 fps buys nothing for legibility and doubles draw cost —
the opposite of the low-CPU goal. The 2-frame (67 ms) fade at 30 fps is also the subtler
look. `build/debug/visemes_keys12_60fps.mp4` is kept only as the comparison.

## What keyframing does

`holly/keyframes.py` inserts one stage between smoothing and the timeline:

```
classify -> normalize -> [pool_keys -> expand_keys] -> build_timeline -> render
```

- `pool_keys` averages the analysis weight frames into one vector per **pose slot**
  (slot boundaries are exact multiples of `1/key_hz` in *time*, so the pose rate is
  exact regardless of analysis fps). Pooling is the smoothing now — an 8 Hz slot at
  62.5 analysis fps averages ~8 decisions, so one noisy frame cannot move the mouth.
- `pool_keys` picks the argmax pose, with optional **sticky hysteresis**: a new mouth
  pose must beat the current one by `--sticky` (default 0.05). The margin is **not**
  applied to or from `sil`, because `sil` is often the argmax of a slot that straddles
  a speech onset and a margin there would delay mouth opening by a whole slot.
- `expand_keys` emits a dense `(n_frames, 15)` matrix at the **render fps** where
  exactly two visemes are non-zero per frame. `holly/timeline.py` and `holly/render.py`
  are unchanged; keyframing is a re-timing transform, not a new output format.

### The fade trap this found

A cross-fade expressed in **seconds** can miss its midpoint entirely. With a 125 ms
key grid and a 33.3 ms frame grid (30 fps), a 50 ms fade sampled at frame boundaries
only ever produced blends of **0.074 and 0.259 — never 0.5**. It looked like a fade in
the numbers and was effectively a hard cut.

Fix: `fade_frames()` quantises the fade to **whole output frames** and `expand_keys`
drives the ramp from **frame index**, not time. Now a 2-frame fade always yields one
50/50 frame then the new pose. `--fade-s 0.05` at 30 fps becomes 2 frames = **67 ms**;
at 60 fps it is 3 frames = 50 ms exactly.

## Measured on `test_audio.flac` (7.68 s)

*(stub-era classifier; the real-classifier numbers are in the Phase 2 section above —
same ballpark for changes/s, but much less `sil`)*

| | continuous (old) | keyframed 8/s | **keyframed 12/s (chosen)** |
|---|---|---|---|
| timeline records | 480 @ 62.5 fps | 233 @ 30 fps | 233 @ 30 fps |
| pose changes | 79 (**10.3/s**) | 35 (**4.5/s**) | 55 (**7.1/s**) |
| mean hold | — | 215 ms | 138 ms |
| distinct weight rows | 472 of 480 (**98.3%**) | 39 of 233 (**16.7%**) | 54 of 233 (**23.2%**) |
| frames holding a pure pose | — | 85.0% | 76.4% (60 fps: 76.3%) |
| mean top weight | 0.416 | 0.925 | 0.882 |

The old output had **16 runs lasting a single 16 ms frame** — that is the flicker the
change was for. Keyframed output has no run shorter than one pose slot.

Achieved switch rate is consistently **~57-59% of `--key-hz`**, because adjacent slots
often pick the same pose. So `--key-hz 12` gives ~7 changes/s. To target a *change*
rate rather than a *slot* rate, divide by ~0.58.

Sweep (measured): key_hz 8 -> 4.5/s, 10 -> 5.7/s, **12 -> 7.1/s**, 14 -> 8.3/s, 16 -> 9.0/s.

### Slot-grid alignment caveat at 12/s @ 30 fps (measured)

12 does not divide 30, so slots alternate between 2 and 3 output frames in a symmetric
`2,3,3,2` cycle. Pure-hold frames per slot: **31 slots hold 1 frame, 40 hold 2, 22 hold 3**
— hold lengths swing 67/100/100 ms on a repeating grid rather than being uniform. It is a
grid artefact, not random jitter, and the cross-fade absorbs most of it.

Only `--key-hz 6` (5 frames) or `10` (3 frames) divide 30 evenly. **15 gives a uniform
2-frame slot but the 2-frame fade then consumes the whole slot, leaving no pure hold** —
measured min pure frames per slot = 1 at both 12 and 15, 2 at 10. If the swing ever reads
as a limp, `--key-hz 10` (~5.7 changes/s) is the uniform alternative; compare on the real
classifier before changing the default.

## CLI (flags on `tools/make_timeline.py`)

`--key-hz` (12) `--render-fps` (30) `--fade-s` (0.05) `--fade-shape center|lead|lag`
`--pool mean|max` `--sticky` (0.05) `--continuous` `--smooth-before-keys`.

Phase 2 adds: `--classifier headaudio|openness-ramp` (**headaudio is now the
default**), `--model` (`model/model-en-mixed.bin`), `--sil-sensitivity` (1.2),
`--no-vad`, `--no-vote`, `--no-gain` (skip the visemeMaxs display gain).
`--vad auto` now applies to both classifiers: rms percentiles for the stub,
log-energy percentiles for headaudio (its shipped −40/−50 dBFS gate is what
`--vad absolute` keeps; `--vad-floor/--vad-ceiling` only steer the stub).

`--fade-shape` positions the fade relative to the pose boundary: `center` straddles
it, `lead` reaches the new pose *before* it (cartoon anticipation, probably what you
want if the mouth feels late), `lag` holds the old pose to the boundary.

## Renderer cost: 61.9 ms -> 1.77 ms per draw

`Renderer.draw_row` used to unpremultiply the whole 864x1152 canvas every frame: 3.3 ms of
blend plus **58 ms of wasted conversion**. The waste was provable — the base layer is fully
opaque (alpha 254/255) and 91% of the canvas never changes, so outside the mouth rect
unpremultiplying changes RGB by mean 0.05/255. It was recomputing a constant.

| | original | mouth-rect f64 | **mouth-rect f32 (now)** |
|---|---|---|---|
| `draw_row` per call | **61.9 ms** | 4.09 ms | **1.77 ms** (35x) |
| resident buffers | ~104 MB | 13.6 MB | **10.8 MB** |
| sustained over the test clip | ~89% of a core | 8.5% | **5.0% of a core** |

Three wins, each measured separately:

1. **Mouth rect, not the canvas** — 61.9 -> 4.09 ms, bit-identical (max Δ 0 over 248 frames).
2. **float32 working precision** — 4.09 -> ~2.4 ms.
3. **No boolean gather in `unpremultiply`** — 3.27 -> 1.28 ms on the rect; clamp the alpha
   and use `np.where` instead of `rgb[mask] = ...`.

**Output is no longer bit-identical, and that is accepted per the user.** Over 308 frames and
307M pixels: **0.0036% of pixels differ, all by exactly 1/255, none by 2 or more.** That is
float32 blend rounding at a `np.rint` .5 boundary.

### Two traps worth remembering

- **NEP 50 scalar promotion.** `row[i] / total` is a *numpy* float64 scalar, and
  `float32_array * numpy_float64` promotes the whole expression to float64 — silently undoing
  the float32 work. The blend weight must be wrapped in `float()`. Verified with
  `test_region_math_stays_float32`.
- **Reciprocal vs divide.** Multiplying by a float32 `1/255` looked equivalent to dividing by
  255, but it introduced its own Δ1 error on 1.2% of pixels. Dividing directly is free and exact.
- **uint8 wraparound.** Dropping `np.clip` before the cast lets 300.0 -> 44 and -5.0 -> 251,
  i.e. silent speckle. Clipping measured free next to the other passes, so it stays.

### The oracle still bites

`test_mouth_rect_only_matches_the_full_canvas_algorithm` retains the original whole-canvas
float64 computation as the oracle, with tolerance **exactly 1**, and asserts **zero**
difference outside the mouth rect where no float32 rounding can excuse it.
`test_float32_rounding_never_exceeds_one_unit` sweeps 40 random weight rows and asserts the
same bound. A deliberate 6% blend error fails both.

`Renderer.mouth_rect` is exposed as `(x, y, w, h)` so a host can blit 88,775 px instead of
995,328.

### Aliasing contract — read this before writing a consumer

`draw_row` returns the renderer's **persistent** buffer, deliberately, because copying 3 MB
per draw is the cost being avoided. Anything retained across draws must be copied:
`renderer.snapshot()` (0.13 ms) or `.copy()`.

**This already caused one real bug.** `tools/render_filmstrip.py` collected frames and
converted them later, so every tile aliased one buffer holding the last pose — the contact
sheet would have been 40 identical images. Fixed with `.copy()` at both call sites and
pinned by `test_collected_frames_must_be_copied_to_stay_distinct`. Verified afterwards: 40
tiles, 20 distinct (correct, since the keyframed timeline holds poses).

Also note `test_only_the_mouth_rect_is_touched` would have gone **vacuous** under aliasing
(comparing two names for one object always yields zero). It now copies, and carries an
explicit guard that the mouth rect *does* differ so the test cannot silently stop testing.

### Still worth doing, not done

Nothing left in the render path. The remaining CPU is the numpy blend itself; further wins
would mean int16 fixed-point math or handing the blend to the host's blitter.

## The classifier contract (Phase 2 honoured it; Phase 3 must not break it)

Pipeline: `decode -> frame -> extract_features -> classify -> normalize -> keyframe -> timeline -> render`.

`holly/classify.py` — a classifier is any callable:

```python
classify(features: FeatureSet) -> np.ndarray   # shape (n_frames, 15), canonical viseme order
```

Everything downstream (keyframing, timeline JSON, renderer, filmstrip tools) is
classifier-agnostic and already tested. `HeadAudioClassifier` implements the
same `__call__` signature and drops in.

`holly/features.py` is now pinned to HeadAudio's front end (see the Phase 2
section above — **do not "tidy" it back**: every parameter there exists because
a shipped Gaussian prototype was trained on it). If a future phase wants
different features, it must retrain the prototypes too.

`OPENNESS_ORDER` is only used by the stub classifier now; the real map comes
from the model records.

## The stub classifier is now the A/B baseline

`OpennessRamp` stays selectable with `--classifier openness-ramp` and is kept
passing its tests, but the pipeline default is the real classifier. Its tell
remains the lopsided histogram (`aa` 19.4%, `sil` 35.6% on the test clip — now
archived as `visemes_stub.mp4`); the port spreads it (see Phase 2 measurements).

Phase 3 VAD note (rewritten after the re-scope): the runtime receives the host's own
audio output — clean, known, analysable before playback — so the whole-buffer percentile
gate (`derive_le_thresholds`) is valid as-is. HeadAudio's shipped absolute −40/−50 dBFS
gate (`--vad absolute`) stays available for a future streaming path where whole-file
statistics don't exist; a running noise-floor estimate is only needed if 3b ever meets
noisy chunked input.

## Environment (verified on this machine)

- Python **3.12.3** — the plan says 3.13; everything works on 3.12, no action needed.
- `psd-tools 1.24.0`, `numpy 2.5.3`, `pillow 12.3.0` in `.venv`. Pinned in `requirements.txt`.
- **No scipy.** The MFCC front end is pure numpy on purpose. Keep it that way.
- `ffmpeg 6.1.1` with `libx264`. Audio decode and debug-video muxing shell out to ffmpeg.
- `sounddevice` and `pygame` are **not installed** — commented out in `requirements.txt`.
  **Neither is needed as planned**: the mic loop was retired (see the Phase 3 re-scope),
  so no `sounddevice`/`libportaudio2`. pygame is optional, only as a standalone test host.
- Shell is **`sh`** — bash-isms like `${PIPESTATUS[0]}` fail.

## Asset facts (measured, not assumed)

`viseme.psd` is 1152x864, RGB, 15 top-level pixel layers named with a `.png` suffix
(`sil.png`, `PP.png`, ...). The exporter normalises the suffix.

- `sil` is 100% opaque over the full canvas — the static base face.
- The other 14 layers are mouth-only overlays sharing content bbox `(354,476)-(684,736)`, so registration is exact.
- Union mouth crop `x=352 y=474, 335x265` = **8.9% of the canvas**. Patches are baked cropped to this rect.

Baked buffers (`build/visemes/`):

- `visemes.npz` — key `base` `(864,1152,4)`, key `patches` `(14,265,335,4)`, **premultiplied uint8 RGBA**.
- `manifest.json` — canonical viseme order, `mouth_rect`, canvas sizes, `alpha: "premultiplied"`.
- `patches` is canonical order **minus `sil`**. Index it by viseme name via
  `Renderer.patch_index`, never by canonical index. This caused two separate
  off-by-one bugs; do not reintroduce a third.
- `build/visemes_half/` is a `--scale 0.5` variant. All of `build/` is regenerable.

`test_audio.flac` (7.68 s) carries its **transcript in the FLAC `prompt` metadata**
(ComfyUI TTS graph). The spoken text is the SRT from node 146: "Hello! This is
unified SRT TTS with character switching. [Alice] Hi there! I'm Alice speaking
with precise timing." — usable for spot-checking viseme accuracy without
guessing. `model/model-en-mixed.bin` + `tests/fixtures/headaudio_distances_oracle.csv`
are vendored from HeadAudio (MIT); fetch commands in `model/README.md`.

## Artefacts to watch

- `build/debug/visemes.mp4` — **12/s @ 30 fps, real classifier + auto gate + display gain: THE SHIPPED CONFIGURATION, chosen by the user (2026-10-07)**
- `build/debug/visemes_ha_gate_only.mp4` — same but `--no-gain`; visually a coin toss vs the default (33/233 frames differ), kept as reference
- `build/debug/visemes_ha_eased.mp4` — gate + gain + `--smooth-before-keys`; **rejected by the user as worst** — pre-pooling easing blurs consonant transitions, do not ship
- `build/debug/visemes_sticky02.mp4` — `--sticky 0.2` A/B (broken-gate era, superseded; kept for reference)
- `build/debug/visemes_keys12_lag.mp4` — shipped configuration with `--fade-shape lag`: now only a preview of the alternative fade shape (the live-path `lag` requirement was retired with the mic premise)
- `build/debug/visemes_stub.mp4`, `visemes_stub_lag.mp4` — the old stub output (old features), kept for A/B
- `build/debug/visemes_keys12_60fps.mp4` — 60 fps, kept only as the rejected comparison
- `build/debug/visemes_continuous.mp4` — the old continuous path (stub-era), kept for A/B
- `build/debug/strip.png` + `strip_ha_gate_only.png`, `strip_ha_eased.png`, `strip_keys12_sticky02.png`, `strip_keys12_lag.png`, `strip_stub*.png` — contact sheets

All have the source audio and a red playhead over a waveform band, so A/V offset is visible.
`timeline.json` is the real-classifier 12/s @ 30 fps timeline; `timeline_stub.json`
the archived stub one; `timeline_keys12_sticky02.json` / `timeline_keys12_lag.json`
the variants; `timeline_keys12_60.json` the 60 fps variant; `timeline_continuous.json`
the old path.

## File map

```
holly-host-plan.md          the LARGER project: LLM + TTS + this component, phased A-E
viseme-mouth-plan.md        plan of record (keyframe timing + on-demand runtime model)
viseme-mouth-plan.v1.md     original continuous-100fps plan, preserved
requirements.txt            build-time deps pinned; runtime deps now live in pyproject.toml
pyproject.toml              `pip install -e .` -> `import holly`; extras: build, player
model/model-en-mixed.bin    HeadAudio prototypes (MIT; see model/README.md)
model/README.md             provenance + fetch commands for the vendored model files
tools/export_visemes.py     Phase 0: audit PSD, validate spec, bake premultiplied buffers.
                            `--scale` + `--resample {bilinear,nearest}` for pixel-art bakes
tools/make_timeline.py      Phase 1+1b+2: audio -> timeline JSON (headaudio by default)
tools/render_filmstrip.py   Phase 1: timeline -> strip.png + muxed visemes.mp4
tools/sweep_phase2.py       Phase 2 tuning sweep: sticky/pool/fade-shape/sil-sensitivity A/B
tools/diagnose_phase2.py    Phase 2b: accuracy/distance/VAD-gate diagnostics vs the embedded transcript
tools/check_sync.py         When the mouth moves vs when the sound happens: burst train with
                            known event times -> signed ms offsets, per fade-shape
tools/reference_player.py   Phase 3a: standalone pygame player -- the sync proof and the tuning
                            harness. The ONLY file that imports pygame; the core never does.
holly/vise.py               canonical 15 visemes, index map
holly/audio.py              ffmpeg decode to 16 kHz mono f32, framing, rms
holly/features.py           HeadAudio-matched mel/MFCC front end, pure numpy (see Phase 2 section)
holly/classify.py           weight-matrix contract, HeadAudioClassifier + parser + viseme remap,
                            OpennessRamp stub (A/B baseline), top_two()
holly/keyframes.py          pose pooling, sticky hysteresis, frame-quantised cross-fade, draw_stats
holly/smooth.py             attack/release envelopes (continuous mode only), normalize_weights
holly/timeline.py           timeline JSON format + IO
holly/render.py             mouth-rect-only cross-fade renderer
holly/runtime.py            Phase 3a: HollyFace -- speak()/row_at()/frame_at(), the utterance
                            queue, the dirty check, IDLE_ROW. No windowing or audio deps.
tests/test_pipeline.py      86 tests pinning the inter-stage contracts
tests/test_runtime.py       32 tests pinning the component API's contracts
tests/test_bake.py          5 tests pinning manifest/buffer consistency at every bake scale
tests/test_sync.py          11 tests pinning the timing geometry (one-frame center lead, jitter)
tests/test_playback_loop.py 8 tests driving the queue from a fake output callback: seam
                            boundaries, no missed frames, gaps idle, prune does not drift
tests/fixtures/headaudio_distances_oracle.csv   JS-computed distance matrix (accuracy oracle)
build/                      all generated artefacts, regenerable
```

## Reproduce everything

```sh
.venv/bin/pip install -e .                 # the host does the same; then `import holly`
.venv/bin/python tools/export_visemes.py --preview build/preview/baked_sheet.jpg
.venv/bin/python tools/export_visemes.py --scale 0.5 --resample nearest --out build/visemes_pixel
.venv/bin/python tools/make_timeline.py --audio test_audio.flac
.venv/bin/python tools/render_filmstrip.py --audio test_audio.flac --video build/debug/visemes.mp4
.venv/bin/python tools/check_sync.py --compare
.venv/bin/python -m unittest discover -s tests -t .

# Phase 3a reference player (needs the optional pygame; the core does not):
.venv/bin/pip install pygame==2.6.1
.venv/bin/python tools/reference_player.py --audio test_audio.flac
.venv/bin/python tools/reference_player.py --audio a.wav --audio b.wav --gap 0.4 --scale 2
```

The model file is vendored in `model/` — if it ever goes missing, refetch per
`model/README.md` before running anything with the default classifier.

## Open decisions

- ~~`--key-hz`~~ **Settled: 12/s (~7 changes/s), chosen by eye.** Real classifier lands
  at 7.8/s at default sticky — same ballpark, no reason to revisit.
- ~~render 30 vs 60 fps~~ **Settled: 30 fps** — low-CPU goal plus the subtler 2-frame fade.
- ~~Pick `--sticky` by eye~~ **Settled by the shipped-config choice (2026-10-07): default 0.05.**
  Measured: 0.05 is inert with one-hot pooling (7.8 changes/s, broken-gate era), 0.2 = real
  hysteresis, 0.35 = sticky. With gate+gain the default lands at 6.9/s on target anyway;
  revisit only if the runtime shows chatter.
- ~~Which driver ships?~~ **Settled: the real classifier, gate+gain — the exact
  `visemes.mp4` configuration, chosen by the user over the stub (2026-10-07).** It is the
  zero-flag default of `make_timeline.py`. The pre-pooling eased variant was explicitly
  rejected by eye; gate-only vs gate+gain is visually a coin toss (33/233 frames) — keep
  the gain default, don't spend more eyeballs on it.
- **NEW: is the `2,3,3,2` hold-length swing at 12/s @ 30 fps visible?** See the slot-grid
  caveat above. `--key-hz 10` is the uniform-grid fallback if it reads as a limp.
- ~~`--fade-shape` — `lag` required for the live path~~ **Retired with the mic premise**
  (Phase 3 re-scope): analysis leads playback, so `center` (the shipped `visemes.mp4` look)
  is valid live. `visemes_keys12_lag.mp4` is now just a preview of an alternative shape.
- hop 256 (62.5 fps) or hop 160 (100 fps) for analysis. Keyframing makes this much less
  important; HeadAudio itself runs 62.5 fps and the prototypes expect it.
- ~~Bake at native 1152x864 or `--scale 0.5` for the target low-end hardware. The user is
  moving toward a **retro pixel-art style**, which likely means nearest-neighbour scaling
  rather than the bilinear used today — not yet specified or implemented.~~ **Settled
  (2026-10-07):** `--resample {bilinear,nearest}` exists, and the host bakes at **576x432
  nearest**. See "Scaled bakes" above for what nearest actually buys and for the rect bug
  it exposed.
- ~~`--sticky` 0.05 and `--pool mean` are untested against a real classifier~~ **Swept**
  (`tools/sweep_phase2.py`): keep `mean` (max is degenerate for one-hot); sticky see above.
- ~~**Bake size for the host (blocks host-plan Phase A)**~~ **Settled: 576x432, nearest.**
  With a correction to the plan: "384x288 = half of 1152x864" is wrong — half is **576x432**;
  384x288 is a **third** (`--scale 0.3333333333`). The face is a corner element of the host UI,
  so the smaller mouth rect wins: 22,176 px/draw against 88,775, measured 0.72 ms against
  ~2.9 ms, 1.52% against 4.66% of a core at a 60 Hz poll. `holly.runtime.DEFAULT_BUFFERS` is
  now `build/visemes_pixel/visemes.npz`; `NATIVE_BUFFERS` stays for the debug filmstrip, and
  `build/visemes_third/` remains if a tighter corner is ever wanted.
- ~~**Handoff shape is now a live question**~~ **Settled by the API, not by argument.**
  `HollyFace.frame_at()` returns exactly `(frame, mouth_rect, changed)` — the standing
  mouth-rect recommendation — and `row_at()` stays available for a host that wants to own
  compositing. The core is committed to neither.
- **NEW: repo boundary settled.** The host consumes this repo as `pip install -e
  Holly-8bit-6000` (`pyproject.toml`, package `holly`), not as a submodule with path
  surgery. `holly.DEFAULT_BUFFERS`/`DEFAULT_MODEL` resolve against the repo root rather than
  the process CWD, so `import holly` behaves identically from the host's own directory.
- ~~Phase 3 VAD levels: measure real mic levels~~ **Moot** — no mic. The auto le-gate is
  tuned on the host's own clean output; revisit only if 3b meets noisy streams.
- ~~Fix `Renderer.draw_row`'s 68 ms full-canvas unpremultiply~~ **Done: 1.77 ms/draw, 10.8 MB.**
- ~~Drop the region surfaces from float64 to float32~~ **Done: 4.09 -> 1.77 ms/draw.**
- Target hardware is **any low-end PC**, not specifically a Pi. Current budget: ~5% of one
  core during speech, 0% idle; analysis is now ~0.06 ms/frame (was ~0.03 — still trivial).
  Numbers are per-core, so they scale with clock speed.

## Kickoff prompt

> Start the **Holly host application** in a new repo alongside the animation component at
> `/home/irreverend/Projects/Holly`, consuming it as `pip install -e
> /home/irreverend/Projects/Holly` and then `import holly` (settled 2026-10-07 — not a
> submodule, no `sys.path` surgery). Read that repo's `HANDOFF.md` first: it is the
> authoritative state record and cannot be inferred from the code alone. `holly-host-plan.md`
> is the plan for the host; `viseme-mouth-plan.md` is the animation component's own plan.
>
> Verify before doing anything else, in the animation repo:
> `.venv/bin/python -m unittest discover -s tests -t .` should report **147 tests, OK**. Two
> generated assets must exist and **`build/` is gitignored, so a fresh clone has neither**:
> `build/visemes_pixel/visemes.npz` — the 576x432 nearest bake that `holly.DEFAULT_BUFFERS`
> points at, re-created with `tools/export_visemes.py --scale 0.5 --resample nearest --out
> build/visemes_pixel` — and `model/model-en-mixed.bin` (refetch per `model/README.md`).
>
> **The animation component is finished — do not rebuild or re-architect it.** Phase 3a is
> the whole host-facing surface:
>
> ```
> face = HollyFace(renderer=Renderer(DEFAULT_BUFFERS))   # or row_at(t) and composite yourself
> face.speak(tts_16k_float32)                            # analysis ~13-18 ms per 7.7 s, before playback
> frame, rect, changed = face.frame_at(t)                # t = audio output position, never time.time()
> if changed: blit(frame, rect)                          # 168x132 px at the host bake, 0.72 ms
> ```
>
> Its timing geometry is measured and pinned, not assumed: the shipping `center` fade leads the
> acoustic event by exactly one render frame (33 ms) with zero jitter on the real classifier
> (`tools/check_sync.py --compare`, `tests/test_sync.py`).
>
> The dirty check, the zero-draw idle and the no-final-pose-latch rule (`IDLE_ROW` outside
> speech) all live in the core — the host must not reimplement them, and must not latch a
> final pose when it thinks an utterance is over. `tools/reference_player.py` is the working
> example of a consumer plus the tuning harness; read it before writing the host's render loop.
>
> **Phase A (the task): the host skeleton, no LLM.** pygame window with the face in a corner,
> keyboard input line, `sounddevice` output, an utterance queue, and the playback clock driving
> `frame_at()`. Suggested order, because the first step needs no TTS at all:
>
> - **A0 — clock proof.** Queue `test_audio.flac` twice with a gap, play it through one
>   `sounddevice.OutputStream`, and derive `t` from the frames the output callback has actually
>   consumed (`t = frames_played / samplerate`). Exit: mouth is in sync by eye, idle is zero-draw,
>   only the 168x132 rect is ever blitted, and the mouth returns to rest when the queue drains.
>   `tests/test_playback_loop.py` is the worked model of exactly that loop, with the device
>   faked: a monotonic frame counter polled at 60 Hz, asserted to reach every reachable frame,
>   never sample past an utterance's audio, stay idle in the gaps, and not drift when played
>   utterances are pruned. Port its assertions into the host's own test rather than re-deriving them.
> - **A1 — text to speech.** Type arbitrary text -> stock Piper voice -> `speak()` -> same sync.
>
> Two things about that boundary that are easy to get wrong. **(1)** `analyse()` *raises* unless
> the buffer is 16 kHz mono float32 — by design, because the HeadAudio front end is pinned to it.
> Resample on the host side and keep the native-rate buffer for playback. The clock still lines up
> at two rates because the queue advances by **duration**, not sample count: 16k and 22.05k
> buffers of the same audio are the same number of seconds. **(2)** Keep analysis and playback
> buffers 1:1 in time — resample once, keep both, never re-derive one from the other.
>
> Constraints the host must not break, carried over from the Phase 3 re-scope: the
> microphone is out of scope entirely (the ~115 ms latency floor, the `--fade-shape lag`
> rule and the `sounddevice`-for-input framing are retired — `sounddevice` in the host is
> *output only*); analysis always runs ahead of playback, so the auto `le` percentile gate
> and `center` fade are valid live; the streaming refactor stays a conditional **3b** until
> the host adopts chunked TTS; and keep the batch classifier contract intact — the offline
> tools and the tests both depend on it.
>
> Do **not** re-tune `holly/features.py` toward "nicer" MFCC defaults: every parameter is
> pinned to HeadAudio's front end because the shipped prototypes were trained on it (the
> audit table in `HANDOFF.md` Phase 2 section lists all of them). Analysis cost is
> ~0.06 ms/frame total — well inside the 1 ms budget; keep it there.
>
> Settled, not to be re-litigated — all measured and recorded:
> **576x432 nearest** is the host bake (the face is a corner element; halving the linear size
> quarters the mouth-rect blit: 22,176 px / 0.72 ms against 88,775 px / 2.84-3.06 ms at native,
> 1.52% against 4.66% of one core at a 60 Hz poll). Pose rate **`--key-hz 12` @ 30 fps**.
> Analysis **~0.06 ms/frame**. The renderer's **1.77 ms/draw** figure in the sections above was
> the mouth-rect work at native under a different harness; the numbers in this paragraph are
> medians over the whole `draw_row` on this box — the ratio is the decision-relevant part.
> The shipping driver is the real classifier with the auto `le` gate + display gain — exactly
> `build/debug/visemes.mp4`, and the zero-flag default of both `make_timeline.py` and
> `HollyFace`. The pre-pooling eased variant was rejected by eye — do not resurrect it.
>
> Still open, in rough priority order — none of these block Phase A:
>
> - **Sync geometry is measured; the look still is not.** `tools/check_sync.py` says the
>   shipping `center` fade opens the mouth exactly one render frame (33 ms) early with **zero
>   jitter** on the real classifier, and `tests/test_sync.py` pins that. What only a human can
>   decide: whether a 33 ms lead *reads* right next to a voice, and whether `lag` (0 ms offset,
>   old pose holds to the boundary) looks stiffer. Run
>   `.venv/bin/python tools/reference_player.py --audio test_audio.flac` on a real device as
>   the first act of Phase A — it has only ever been run headless on SDL's dummy audio.
> - **`2,3,3,2` hold-length swing at 12/s @ 30 fps** — decide by eye in the live host; fall
>   back to `--key-hz 10` (uniform grid) if it reads as a limp.
> - **Classifier behaviour on synthetic speech** — TTS output is cleaner than the human
>   speech the HeadAudio prototypes were trained on, but prosody differs. Test each voice
>   with `tools/render_filmstrip.py` / the reference player; if a voice gives poor visemes the
>   fix is the classifier's gate/gain knobs, never the renderer.
> - **3b streaming-TTS refactor** — only if the host adopts chunked TTS that plays as it arrives.
> - **Phase 4 polish** — idle micro-motion, per-viseme gain curves.
> - **Red Dwarf LoRA** for the LLM — its own side project; the host code does not change for it.
