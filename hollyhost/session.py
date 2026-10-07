"""The pure part of the host: turn sources into a face queue and one playback buffer.

No audio device, no window. Everything that decides *when a pose happens* lives here so
it can be tested without portaudio or a display -- which is the whole point of Phase A0.

The invariant this module exists to protect:

    the number of seconds of audio written to the device  ==  the face clock

Both are derived from the same list of buffers, so they cannot drift. `holly.HollyFace`
advances its own queue by each utterance's *audio duration* (not its timeline length), and
we place utterances at exactly those times plus the gaps we also render as silence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from holly.audio import TARGET_RATE, decode
from holly.runtime import HollyFace

#: Playback rate. Holly's analysis buffer is 16 kHz mono float32 and the component refuses
#: anything else, so playing at the same rate keeps seconds == samples / rate exact and
#: removes a resampler from the clock path. TTS backends that emit 22.05k/24k must be
#: resampled to this for the *analysis* buffer; the playback buffer may stay native
#: because the queue advances by duration, not sample count.
SAMPLE_RATE = TARGET_RATE

#: The rate the viseme pipeline insists on. Same number as SAMPLE_RATE, named separately
#: because tts.py speaks about the *analysis* contract while session.py speaks about playback.
FACE_RATE = TARGET_RATE


@dataclass
class UtteranceSource:
    """One thing Holly will say: the audio, and where it lands on the face clock."""

    name: str
    samples: np.ndarray
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


def from_file(path: str | Path) -> np.ndarray:
    """Any file ffmpeg understands -> 16 kHz mono float32. Stands in for TTS in Phase A0."""
    return decode(str(path))


def build_session(
    buffers: list[np.ndarray],
    *,
    face: HollyFace | None = None,
    gap: float = 0.25,
    sample_rate: int = SAMPLE_RATE,
    names: list[str] | None = None,
) -> tuple[HollyFace, list[UtteranceSource], np.ndarray]:
    """Queue every buffer up front, and return the one continuous stream to play.

    Returns (face, sources, playback_buffer). `playback_buffer` contains the utterances
    separated by `gap` seconds of silence, so the device's frames-consumed position and
    the face clock are the same number by construction.

    Analysis happens here, before anything plays -- that is what lets the component use a
    whole-utterance percentile gate and a non-causal fade, and why there is no latency floor.
    """
    face = face or HollyFace()
    sources: list[UtteranceSource] = []
    chunks: list[np.ndarray] = []
    clock = 0.0

    for i, samples in enumerate(buffers):
        name = (names[i] if names else None) or getattr(samples, "name", None) or f"utterance-{i}"
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise ValueError(f"buffer {i} is empty")
        utterance = face.speak(samples, sample_rate, at=clock if clock > 0 else None)
        sources.append(
            UtteranceSource(
                name=name,
                samples=samples,
                start=utterance.start,
                duration=utterance.audio_duration,
            )
        )
        chunks.append(samples)
        clock = utterance.end + gap
        if i < len(buffers) - 1 and gap > 0:
            chunks.append(np.zeros(round(gap * sample_rate), dtype=np.float32))

    playback = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    return face, sources, playback


def played_duration(sources: list[UtteranceSource], playback: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """The two clocks, as one number to assert on."""
    return max(sources[-1].end if sources else 0.0, playback.size / sample_rate)


class LiveSession:
    """The interactive (A1) counterpart to `build_session`: utterances arrive over time.

    It owns the two things that must stay in lockstep and must not be reordered:

    * the **audio** tail (`LiveAudio`) -- what the device plays, and the clock;
    * the **face** queue (`HollyFace`) -- what the mouth shows, on the same clock.

    `say()` is the whole contract in one place: synthesise (or accept) audio, split it into
    a 16 kHz analysis buffer and a native-rate playback buffer (resample once, keep both),
    append the playback audio to the tail, and enqueue the face utterance at the exact
    clock position the audio landed on. Because the face `at` is derived from the same
    append that placed the bytes, mouth and voice cannot drift -- the property
    `build_session` proves for a batch holds incrementally here.

    The face is touched only from the UI thread (`say`/`poll`); the audio callback runs on
    the device thread and never sees the face. That is why no lock guards the queue.
    """

    def __init__(self, face: HollyFace, audio, *, gap: float = 0.25, tts=None):
        self.face = face
        self.audio = audio
        self.gap = float(gap)
        self.tts = tts

    def say(self, text: str, samples=None, rate: int | None = None):
        """Queue one utterance. `samples`/`rate` bypass TTS when the caller already has audio."""
        from .tts import prepare, resample

        if samples is None:
            if self.tts is None:
                raise ValueError("no audio given and no TTS backend on the session")
            samples, rate = self.tts.synthesize(text)
        if rate is None:
            rate = self.audio.rate

        prepared = prepare(text, samples, rate)  # analysis @ 16k, playback @ native rate
        playback = prepared.playback
        if prepared.rate != self.audio.rate:
            # One output stream, one rate: fold a foreign-rate voice into the device rate.
            playback = resample(playback, prepared.rate, self.audio.rate)

        start = self.audio.append(playback, gap=self.gap)
        return self.face.speak(prepared.analysis, FACE_RATE, at=start)

    def poll(self) -> tuple[np.ndarray, tuple[int, int, int, int], bool]:
        """Sample the face at the live clock, pruning what has already played.

        Returns the same `(frame, mouth_rect, changed)` as `HollyFace.frame_at`; the host
        blits the mouth rect only when `changed`.
        """
        t = self.audio.t
        self.face.prune(t)
        return self.face.frame_at(t)
