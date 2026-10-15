# Real-time viseme mouth animation plan
Goal: live audio -> viseme weights -> cross-faded mouth layers from a PSD/GIMP file, on low-end CPU.

## Reality check (verified 2026-10-07)
- No public standalone OVRVTC repo exists. Meta's viseme engine = EOL Oculus Lipsync binary SDK (closed) + Quest-only Movement SDK.
- Engine choice: port HeadAudio (github.com/met4citizen/HeadAudio, MIT) — MFCC + Gaussian prototypes + Mahalanobis distance, outputs the 15 Oculus visemes in real time, no ML framework. Trained model `dist/model-en-mixed.bin` verified downloadable.
- 15 Oculus visemes: sil PP FF TH DD kk CH SS nn RR aa E ih oh ou

## 1. Asset spec (PSD / GIMP)
- One layer per viseme; layer names exactly the viseme codes above.
- Identical canvas size; mouth registered at the same pixel position in every layer.
- Mouth-only content, transparent elsewhere; static base face layer at bottom.
- Author in GIMP, export as PSD (layer names survive). Read with psd-tools (v1.24.0 verified working).
- Build-time export script: validate layer names, composite each layer to RGBA numpy buffers at render resolution. Runtime never opens the PSD.

## 2. Pipeline
1. Audio: mic (sounddevice + libportaudio2) or file (ffmpeg decode) -> 16 kHz mono float32.
2. Framing: 512-sample window, 256 hop -> ~100 viseme frames/s.
3. Features: 13-15 MFCCs per frame (numpy FFT + mel filterbank).
4. Classify: for each phoneme Gaussian prototype (mean, inverse covariance) compute Mahalanobis distance; min-distance phoneme wins.
5. Map phoneme -> viseme via lookup table (HeadAudio ships the mapping).
6. Smooth: per-viseme attack (~30 ms) / release (~120 ms) envelopes.
7. Render: base layer + weighted mouth layers; or top-2 cross-fade (A at alpha 1, B at wB/(wA+wB)).
8. Sync: delay audio output ~50 ms so mouth and sound align.

CPU budget: analysis < 1 ms/frame in numpy; <10% of a Pi core at 100 fps. Rendering dominates, not analysis.

## 3. Phases
- Phase 0: PSD audit/export script (assert expected layer names, dump RGBA buffers).
- Phase 1: offline proof — one WAV -> viseme timeline JSON ({t, viseme, weight}); render debug filmstrip, watch against audio.
- Phase 2: port HeadAudio classifier (modules/headworklet.mjs + model-en-mixed.bin) to Python: MFCC, prototypes, Mahalanobis, phoneme->viseme map, attack/release.
- Phase 3: real-time loop — sounddevice ring buffer -> queue -> viseme frames at 100 Hz -> pygame draw at 30-60 fps.
- Phase 4: polish — sil idle micro-motion, weight normalization, per-viseme gain curves.

## 4. Rendering rule
Never hard-switch layers. Cross-fade the top-2 weighted visemes each frame with the attack/release envelope. This is what prevents the "popping between images" look.

## 5. Alternatives
- Oculus Lipsync SDK binary (Windows C API, EOL): skips the port, closed binary.
- Rhubarb Lip Sync: precompute 9-shape cues when audio is known in advance (TTS/dialogue); zero runtime cost.

## Deps (verified on this machine)
- python 3.13 venv; psd-tools 1.24.0 installed OK; numpy, pygame, sounddevice pip-installable; libportaudio2 installable via apt; ffmpeg present.
