"""Audio output and the clock.

The clock is *frames the device has consumed*, never wall time. That single rule is what
keeps Holly's mouth locked to his voice: pygame may poll late, the LLM may stall the
thread, the scheduler may deschedule us -- none of it moves the audio position.

`sounddevice` is imported lazily so the pure logic and the tests work on a box without
libportaudio2. `FakeDevice` is the same interface driven by hand, which is how Phase A0
gets verified before there is audio hardware in the loop.
"""

from __future__ import annotations

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
