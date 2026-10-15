#!/usr/bin/env python3
"""Phase 2 tuning sweep: re-check --sticky/--pool/--fade-shape against the real classifier.

The stub produced soft weights; the HeadAudio port emits one-hot frames, so the
pooling granularity and sticky-margin semantics changed under the defaults.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from holly.audio import HOP_SIZE, decode, frame
from holly.classify import HeadAudioClassifier
from holly.features import extract_features
from holly.keyframes import draw_stats, key_stats, keyframe_weights
from holly.smooth import normalize_weights

AUDIO = Path("test_audio.flac")
MODEL = Path("model/model-en-mixed.bin")

samples = decode(AUDIO)
frames = frame(samples, hop=HOP_SIZE)
fps = 16000 / HOP_SIZE
features = extract_features(frames, hop=HOP_SIZE)
duration = samples.size / 16000

CONFIGS = [
    ("baseline  sticky .05 mean vote", {}),
    ("sticky 0                      ", {"sticky": 0.0}),
    ("sticky 0.2                    ", {"sticky": 0.2}),
    ("sticky 0.35                   ", {"sticky": 0.35}),
    ("pool max                      ", {"pool": "max"}),
    ("no majority vote              ", {"vote_window": 0}),
    ("fade lag                      ", {"shape": "lag"}),
    ("fade lead                     ", {"shape": "lead"}),
    ("sil-sens 1.0                  ", {"sil_sensitivity": 1.0}),
    ("sil-sens 1.5                  ", {"sil_sensitivity": 1.5}),
]

for label, cfg in CONFIGS:
    classifier = HeadAudioClassifier(
        MODEL,
        sil_sensitivity=float(cfg.get("sil_sensitivity", 1.2)),
        vote_window=int(cfg.get("vote_window", 6)),
        vad=True,
        analysis_fps=fps,
    )
    weights = normalize_weights(classifier(features))
    keys, dense, out_times, out_fps = keyframe_weights(
        weights,
        analysis_fps=fps,
        key_hz=12.0,
        out_fps=30.0,
        fade_s=0.05,
        shape=cfg.get("shape", "center"),
        pool=cfg.get("pool", "mean"),
        sticky=cfg.get("sticky", 0.05),
    )
    stats = key_stats(keys, 12.0, duration)
    ds = draw_stats(dense, out_fps)
    top = [k.viseme for k in keys]
    sil_share = sum(1 for k in keys if k.viseme == "sil") * (1 / 12.0) / duration
    distinct = len({tuple(np.round(row, 3)) for row in dense})
    print(f"{label} | changes {stats['changes_per_s']:4.1f}/s | mean hold {stats['mean_hold_s']*1000:5.1f} ms "
          f"| min {stats['shortest_hold_s']*1000:4.0f} ms | sil {sil_share:5.1%} | distinct {stats['distinct']:3d} "
          f"| redraws {ds['redraws']:3d} | unused: {' '.join(sorted({'PP','FF','TH','DD','kk','CH','SS','nn','RR','aa','E','ih','oh','ou','sil'} - set(top)))}")
