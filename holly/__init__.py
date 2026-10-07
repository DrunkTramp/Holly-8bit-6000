"""Holly viseme pipeline.

Build time:  tools/export_visemes.py  -> premultiplied RGBA buffers
Phase 1:     audio -> frames -> features -> weights -> smoothed weights -> timeline
Phase 3:     the same weights arriving live, drawn by holly.render.Renderer
"""

from __future__ import annotations

from .audio import FRAME_SIZE, HOP_SIZE, TARGET_RATE, decode, frame, frame_times, rms
from .classify import HeadAudioClassifier, OpennessRamp, parse_model, top_two
from .features import FeatureSet, extract_features
from .smooth import attack_release, normalize_weights
from .timeline import build_timeline, read_timeline, sample_frames, timeline_weights, write_timeline
from .vise import BASE_VISEME, VISEMES, VISEME_INDEX

__all__ = [
    "BASE_VISEME",
    "FRAME_SIZE",
    "HOP_SIZE",
    "TARGET_RATE",
    "VISEMES",
    "VISEME_INDEX",
    "FeatureSet",
    "HeadAudioClassifier",
    "OpennessRamp",
    "attack_release",
    "build_timeline",
    "decode",
    "extract_features",
    "frame",
    "frame_times",
    "normalize_weights",
    "parse_model",
    "read_timeline",
    "rms",
    "sample_frames",
    "timeline_weights",
    "top_two",
    "write_timeline",
]
