#!/usr/bin/env python3
"""Phase 2 diagnostics: why does the real classifier read worse than the stub?

Checks, in order of suspicion:
  1. accuracy  - per-frame argmin viseme timeline against the clip's embedded
                 transcript ("Hello! This is unified SRT TTS with character
                 switching. / Hi there! I'm Alice speaking with precise timing.")
  2. distance  - how close do real frames land to the nearest prototype? (the
                 oracle CSV says prototype-to-prototype distances are ~0.3-5;
                 frames far outside that band mean off-distribution features)
  3. gate      - when does HeadAudio's log-energy VAD actually close, vs where
                 the audio is genuinely quiet (RMS percentiles)?
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from holly.audio import HOP_SIZE, TARGET_RATE, decode, frame, frame_times
from holly.classify import (
    HA_VISEME_TO_CANONICAL,
    HeadAudioClassifier,
    load_model,
    mahalanobis,
)
from holly.features import extract_features
from holly.vise import VISEMES

AUDIO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_audio.flac")
MODEL = Path("model/model-en-mixed.bin")

samples = decode(AUDIO)
frames = frame(samples, hop=HOP_SIZE)
fps = TARGET_RATE / HOP_SIZE
features = extract_features(frames, hop=HOP_SIZE)
times = frame_times(frames.shape[0], hop=HOP_SIZE)

model = load_model(MODEL)
classifier = HeadAudioClassifier(MODEL, vote_window=0, vad=False, analysis_fps=fps)

# --- 1+2: raw argmin per frame, distance stats ------------------------------
d = classifier.distances(features)
p = model.n_prototypes - 1 - np.argmin(d[:, ::-1], axis=1)
vis = np.array([HA_VISEME_TO_CANONICAL[int(v)] for v in model.visemes[p]])
min_d = d.min(axis=1)

print("nearest-prototype distance over the clip:")
for pct in (5, 25, 50, 75, 95):
    print(f"  p{pct:02d}: {np.percentile(min_d, pct):8.1f}")
print(f"  max: {min_d.max():8.1f}")
print("  (oracle reference: prototype-mean to OTHER prototype means runs 0.3-5.0;")
print("   the s1 sil column runs ~2400 because its covariance is tiny)")

# how far is each frame from the NON-sil prototypes only?
non_sil = model.visemes != 14
min_d_nonsil = d[:, non_sil].min(axis=1)
print(f"  nearest non-sil prototype: p50 {np.percentile(min_d_nonsil, 50):.1f} "
      f"p95 {np.percentile(min_d_nonsil, 95):.1f}")

# --- 3: VAD gate vs real quietness -------------------------------------------
le = np.asarray(features.log_energy, dtype=np.float64)
gate = classifier._vad_gate(le)
rms = np.asarray(features.rms, dtype=np.float64)
speech_rms = np.percentile(rms, 70)
quiet = rms < 0.10 * speech_rms  # frames at least 20 dB below the speech body

print("\nVAD gate (HeadAudio defaults -40/-50 dBFS):")
print(f"  le: p05 {np.percentile(le, 5):.2f}  p50 {np.percentile(le, 50):.2f}  "
      f"p95 {np.percentile(le, 95):.2f}   (gate opens > {classifier.active_le:.1f}, closes < {classifier.inactive_le:.1f})")
print(f"  gate open: {gate.mean():.1%} of frames")
print(f"  genuinely quiet frames (rms < 10% of speech level): {quiet.mean():.1%}")
print(f"  gate open WHILE quiet (mouth hangs open): {(gate & quiet).sum()} of {quiet.sum()} quiet frames")
print(f"  gate closed WHILE not quiet: {((~gate) & ~quiet).sum()} frames")

# --- 1: timestamped pose timeline for transcript alignment -------------------
print("\nper-frame argmin viseme (raw, no vote), 100 ms buckets:")
bucket = int(round(0.1 * fps))
n = len(vis)
for start in range(0, n, bucket):
    chunk = vis[start : start + bucket]
    seq = " ".join(VISEMES[int(i)] for i in chunk)
    t0 = times[start]
    bar = "#" * max(0, min(30, int(rms[start:start + bucket].mean() / speech_rms * 15)))
    print(f"  {t0:5.2f}s {seq:<60s} {bar}")
