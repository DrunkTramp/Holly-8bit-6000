"""Audio decode and framing.

Everything downstream of the microphone works on 16 kHz mono float32 at a fixed
window/hop, so the decode step is the only place that has to know about codecs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

TARGET_RATE = 16000
FRAME_SIZE = 512
HOP_SIZE = 256


def decode(path: str | Path, target_rate: int = TARGET_RATE) -> np.ndarray:
    """Decode any file ffmpeg understands into mono float32 in [-1, 1].

    Decoding via an ffmpeg subprocess keeps the runtime free of a decoder
    dependency and handles wav/flac/mp3/ogg uniformly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"audio file not found: {path}")

    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(target_rate),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to decode {path}: {proc.stderr.decode('utf-8', 'replace').strip()}"
        )

    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    if samples.size == 0:
        raise RuntimeError(f"ffmpeg produced no samples for {path}")
    return samples


def frame(signal: np.ndarray, size: int = FRAME_SIZE, hop: int = HOP_SIZE) -> np.ndarray:
    """Slice the signal into (n_frames, size) overlapping windows.

    The tail is zero-padded by `size - hop` so the last partial window still
    gets a frame instead of silently dropping up to one hop of audio.
    """
    if hop <= 0 or size <= 0 or hop > size:
        raise ValueError(f"invalid framing: size={size} hop={hop}")

    signal = np.asarray(signal, dtype=np.float32)
    if signal.size < size:
        signal = np.pad(signal, (0, size))

    padded = np.pad(signal, (0, size - hop))
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    return np.ascontiguousarray(windows[::hop])


def frame_times(n_frames: int, hop: int = HOP_SIZE, rate: int = TARGET_RATE) -> np.ndarray:
    """Start time of each frame, in seconds."""
    return np.arange(n_frames, dtype=np.float64) * (hop / rate)


def rms(signal: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(signal), axis=-1))
