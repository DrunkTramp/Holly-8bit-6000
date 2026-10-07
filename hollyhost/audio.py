"""Audio output and the clock.

The clock is *frames the device has consumed*, never wall time. That single rule is what
keeps Holly's mouth locked to his voice: pygame may poll late, the LLM may stall the
thread, the scheduler may deschedule us -- none of it moves the audio position.

`sounddevice` is imported lazily so the pure logic and the tests work on a box without
libportaudio2. `FakeDevice` is the same interface driven by hand, which is how Phase A0
gets verified before there is audio hardware in the loop.
"""

from __future__ import annotations

import threading

import numpy as np

from .session import SAMPLE_RATE


class SounddeviceOutput:
    """One OutputStream playing one continuous buffer.

    The callback copies the next `frames` samples out of the buffer and advances the
    consumed counter. Underruns (a Pi under load) show up as the counter running ahead of
    what was actually filled, so `fill_pos` and `consumed` are tracked separately and the
    gap is observable rather than silent.
    """

    def __init__(self, buffer: np.ndarray, *, rate: int = SAMPLE_RATE, blocksize: int = 512,
                 device: int | str | None = None):
        import sounddevice as sd

        self._sd = sd
        self.rate = int(rate)
        self.blocksize = int(blocksize)
        self._buffer = np.clip(np.asarray(buffer, dtype=np.float32), -1.0, 1.0)
        self.consumed = 0
        self.fill_pos = 0
        self.device = device

    @property
    def t(self) -> float:
        """Seconds of audio the device has taken. This is the face clock."""
        return self.consumed / self.rate

    @property
    def finished(self) -> bool:
        return self.fill_pos >= self._buffer.size

    @property
    def underrun_frames(self) -> int:
        return max(0, self.consumed - self.fill_pos)

    def _callback(self, outdata, frames, _time, status) -> None:
        if status:
            # discarded / inputUnderflow / outputUnderflow -- keep the clock honest anyway,
            # but surface it: on ALSA a busy Pi produces these and the mouth must not lie.
            self.last_status = status
        take = min(frames, self._buffer.size - self.fill_pos)
        outdata[:take, 0] = self._buffer[self.fill_pos : self.fill_pos + take]
        if take < frames:
            outdata[take:, 0] = 0.0
        self.fill_pos += take
        self.consumed += frames

    def start(self):
        self.last_status = None
        stream = self._sd.OutputStream(
            samplerate=self.rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
            device=self.device,
        )
        self._stream = stream
        stream.start()
        return stream

    def stop(self) -> None:
        stream = getattr(self, "_stream", None)
        if stream is not None:
            stream.stop()
            stream.close()
            self._stream = None


class FakeDevice:
    """Same surface as SounddeviceOutput, advanced by hand. For tests and --dry-run."""

    def __init__(self, buffer: np.ndarray, *, rate: int = SAMPLE_RATE, blocksize: int = 512):
        self.rate = int(rate)
        self.blocksize = int(blocksize)
        self._size = int(np.asarray(buffer).size)
        self.consumed = 0

    @property
    def t(self) -> float:
        return self.consumed / self.rate

    @property
    def finished(self) -> bool:
        return self.consumed >= self._size

    def advance(self, frames: int) -> float:
        self.consumed += int(frames)
        return self.t

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


def list_devices() -> str:
    import sounddevice as sd

    return str(sd.query_devices())


class LiveAudio:
    """Appendable playback buffer + the face clock, for interactive (A1) playback.

    The batch device above plays one buffer built up front. Here utterances arrive over
    time (the user keeps typing), so the buffer grows while it plays. The one rule that
    keeps mouth and voice locked is unchanged: the clock is frames the callback has
    *actually emitted from real audio*, never wall time.

    When the buffer is exhausted the callback still fills the block (the device needs
    samples every cycle) but does **not** advance `fill`, so the clock freezes at the tail
    while Holly waits for the next utterance and never runs ahead of the audio. New audio
    is appended at the tail, and its face-clock start is exactly the buffer position it was
    appended at divided by the rate -- so the two clocks cannot drift, by construction.

    `start()` wires a real `sounddevice.OutputStream`; `advance()` drives the identical
    accounting by hand, which is how the loop is tested on a box with no audio hardware.
    """

    def __init__(self, rate: int = SAMPLE_RATE, *, blocksize: int = 512):
        self.rate = int(rate)
        self.blocksize = int(blocksize)
        self._lock = threading.Lock()
        self._samples = np.zeros(0, dtype=np.float32)
        self._fill = 0
        self._stream = None
        self.last_status = None

    # -- the clock ----------------------------------------------------------

    @property
    def t(self) -> float:
        """Seconds of real audio the callback has emitted. This is the face clock."""
        return self._fill / self.rate

    @property
    def queued_seconds(self) -> float:
        """End of everything appended so far, on the clock."""
        with self._lock:
            return self._samples.size / self.rate

    def idle(self) -> bool:
        """True when the clock has caught up with the tail (nothing left to play)."""
        with self._lock:
            return self._fill >= self._samples.size

    # -- queue --------------------------------------------------------------

    def append(self, chunk, *, gap: float = 0.0) -> float:
        """Append utterance audio (already at `self.rate`) to the tail.

        Returns its start in seconds on the clock. `gap` seconds of silence are inserted
        first when something is already queued, so back-to-back utterances keep the same
        face-clock spacing the queue was built with.
        """
        chunk = np.clip(np.asarray(chunk, dtype=np.float32).reshape(-1), -1.0, 1.0)
        if chunk.size == 0:
            raise ValueError("cannot append an empty utterance")
        with self._lock:
            if self._samples.size and gap > 0:
                pad = np.zeros(round(gap * self.rate), dtype=np.float32)
                self._samples = np.concatenate([self._samples, pad, chunk])
            elif self._samples.size:
                self._samples = np.concatenate([self._samples, chunk])
            else:
                self._samples = chunk
            return (self._samples.size - chunk.size) / self.rate

    # -- the callback -------------------------------------------------------

    def emit(self, frames: int) -> tuple[np.ndarray | None, int]:
        """Take up to `frames` samples out of the buffer, advancing the clock by what is real.

        Returns (block_or_None, n_real). The caller pads the rest of the device block with
        silence; `n_real` is what the clock moves, so exhausted buffer == frozen clock.
        """
        with self._lock:
            n = max(0, min(int(frames), self._samples.size - self._fill))
            block = self._samples[self._fill : self._fill + n].copy() if n else None
            self._fill += n
            return block, n

    def _callback(self, outdata, frames, _time, status) -> None:
        if status:
            self.last_status = status
        block, n = self.emit(frames)
        if n:
            outdata[:n, 0] = block
        if n < frames:
            outdata[n:, 0] = 0.0

    def start(self):
        import sounddevice as sd

        stream = sd.OutputStream(
            samplerate=self.rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self._stream = stream
        stream.start()
        return stream

    def stop(self) -> None:
        stream = getattr(self, "_stream", None)
        if stream is not None:
            stream.stop()
            stream.close()
            self._stream = None

    # -- manual drive (tests / --dry) --------------------------------------

    def advance(self, frames: int) -> float:
        """Drive the clock exactly as the callback would, without a device."""
        self.emit(frames)
        return self.t
