"""Holly viseme pipeline.

Build time:  tools/export_visemes.py  -> premultiplied RGBA buffers
Phase 1:     audio -> frames -> features -> weights -> smoothed weights -> timeline
Phase 3a:    holly.runtime.HollyFace -- the host-facing component API
             (speak a buffer, sample poses against the host's playback clock)
"""

from __future__ import annotations

from .audio import FRAME_SIZE, HOP_SIZE, TARGET_RATE, decode, frame, frame_times, rms
from .classify import HeadAudioClassifier, OpennessRamp, parse_model, top_two
from .features import FeatureSet, extract_features
from .render import Renderer
from .runtime import (
    DEFAULT_BUFFERS,
    DEFAULT_MODEL,
    HollyFace,
    IDLE_ROW,
    NATIVE_BUFFERS,
    Utterance,
)
from .smooth import attack_release, normalize_weights
from .timeline import build_timeline, read_timeline, sample_frames, timeline_weights, write_timeline
from .vise import BASE_VISEME, VISEMES, VISEME_INDEX

__all__ = [
    "BASE_VISEME",
    "DEFAULT_BUFFERS",
    "DEFAULT_MODEL",
    "FRAME_SIZE",
    "HOP_SIZE",
    "IDLE_ROW",
    "NATIVE_BUFFERS",
    "TARGET_RATE",
    "VISEMES",
    "VISEME_INDEX",
    "FeatureSet",
    "HeadAudioClassifier",
    "HollyFace",
    "OpennessRamp",
    "Renderer",
    "Utterance",
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
