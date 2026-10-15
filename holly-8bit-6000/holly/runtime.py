"""Phase 3a: the host-facing component API.

Pure numpy in, numpy out. No windowing, no audio device, no microphone -- the host
owns the clock (which is its *audio output position*, never `time.time()`) and owns
whatever surface the pixels land on. This module owns three things:

    analysis   `speak(samples)` -> a pose timeline, computed before playback starts
    queue      utterances laid back-to-back on the face clock, gaps optional
    sampling   `row_at(t)` / `frame_at(t)` -> the pose for one clock value

The runtime model is on-demand drawing, so the two behaviours a host must not have to
implement itself are enforced here:

* **dirty-check** -- `frame_at` compares the weight row against the last row it drew
  and returns `changed=False` when they are equal, so the host blits nothing;
* **zero-draw idle, no final-pose latch** -- outside speech `row_at` returns
  `IDLE_ROW` (pure base pose) rather than the last pose of the utterance that just
  finished. The mouth returns to rest and then costs nothing.

Handoff shape is deliberately not committed to: `row_at` (weight row) and `frame_at`
(premultiplied mouth-rect buffer) are both thin views over the same timeline, and a
host can consume either.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import FRAME_SIZE, HOP_SIZE, TARGET_RATE, decode, frame
from .classify import (
    HeadAudioClassifier,
    OpennessRamp,
    derive_le_thresholds,
    derive_vad_thresholds,
    load_model,
)
from .features import extract_features
from .keyframes import (
    FADE_S,
    FADE_SHAPES,
    KEY_HZ,
    POOL_MODES,
    RENDER_FPS,
    STICKY_MARGIN,
    Keyframe,
    keyframe_weights,
)
from .render import Renderer
from .smooth import normalize_weights
from .vise import BASE_VISEME, VISEMES, VISEME_INDEX

#: Asset defaults resolve against this repo's root, not the process CWD. A host that
#: installs this package (`pip install -e Holly-8bit-6000`) and runs from its own
#: directory must still find the baked buffers and the vendored model.
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = REPO_ROOT / "model" / "model-en-mixed.bin"

#: The host's asset. 576x432 (half of the 1152x864 source) baked with `--resample nearest`:
#: the face is a corner element in the host UI, and the mouth rect is the only thing ever
#: blitted -- at this size it is 168x132 instead of 335x265, measured 0.73 ms/draw against
#: 2.84-3.06 ms at native. The offline tools keep their own native default
#: (`build/visemes/`); re-bake this one with
#:     tools/export_visemes.py --scale 0.5 --resample nearest --out build/visemes_pixel
DEFAULT_BUFFERS = REPO_ROOT / "build" / "visemes_pixel" / "visemes.npz"
#: Full-resolution bake, for the debug filmstrip/video and for eyeballing detail.
NATIVE_BUFFERS = REPO_ROOT / "build" / "visemes" / "visemes.npz"

CLASSIFIERS = ("headaudio", "openness-ramp")

#: HeadAudio's shipped gate, in dBFS. Overridden per utterance by `vad="auto"`,
#: which is the shipping default: with buffer-first analysis the whole-utterance
#: percentile gate is valid (the mic-era running noise-floor estimate is not needed).
VAD_ACTIVE_DB = -40.0
VAD_INACTIVE_DB = -50.0


def make_idle_row() -> np.ndarray:
    """The dense weight row for "Holly is not talking": the base pose, alone."""
    row = np.zeros(len(VISEMES), dtype=np.float64)
    row[VISEME_INDEX[BASE_VISEME]] = 1.0
    return row


#: Returned by `row_at` outside speech. Read-only so a host cannot scribble on the
#: state of every idle frame in the process by accident.
IDLE_ROW = make_idle_row()
IDLE_ROW.flags.writeable = False


@dataclass
class Utterance:
    """One analysed utterance sitting on the face clock.

    `start`/`audio_duration` are in seconds on the host's playback clock and are what
    the queue advances by, so the face stays locked to the audio output position even
    when utterances are queued back-to-back. `weights` can run past the audio by up to
    one pose slot plus one output frame (the key count is rounded up to whole slots, then
    the timeline up to whole frames); those tail frames are simply never reached, because
    the clock is the audio and the audio has ended.
    """

    index: int
    start: float
    audio_duration: float
    weights: np.ndarray
    keys: tuple[Keyframe, ...]
    render_fps: float
    analysis_ms: float

    @property
    def end(self) -> float:
        return self.start + self.audio_duration

    @property
    def frames(self) -> int:
        return int(self.weights.shape[0])

    @property
    def timeline_duration(self) -> float:
        return self.frames / self.render_fps

    @property
    def tail(self) -> float:
        """Seconds of held final slot the audio clock will not reach."""
        return max(0.0, self.timeline_duration - self.audio_duration)

    def covers(self, t: float) -> bool:
        return self.start <= t < self.end

    def row_covering(self, t: float) -> np.ndarray:
        """The row for a `t` this utterance covers. Assumes `covers(t)`."""
        k = min(int((t - self.start) * self.render_fps), self.frames - 1)
        return self.weights[max(k, 0)]

    def row_at(self, t: float) -> np.ndarray | None:
        """Weight row for absolute clock time `t`, or None if `t` is outside this one."""
        if not self.covers(t):
            return None
        return self.row_covering(t)

    def viseme_at(self, t: float) -> str:
        """Dominant viseme at `t`. Only meaningful while `covers(t)`."""
        row = self.row_at(t)
        if row is None:
            return BASE_VISEME
        return VISEMES[int(np.argmax(row))]


class HollyFace:
    """The animation component, as the host sees it.

    ```
    face = HollyFace(renderer=Renderer(DEFAULT_BUFFERS))
    face.speak(tts_samples)          # ~30 ms of analysis per 7.7 s utterance
    ...
    frame, rect, changed = face.frame_at(audio_output_position)
    if changed:
        blit(frame, rect)            # mouth rect only
    ```

    Every tuning knob defaults to the shipped `visemes.mp4` configuration -- the real
    HeadAudio classifier with the auto `le` gate + display gain, 12 poses/s at 30 fps,
    `center` fade. Because analysis runs ahead of playback, `center` is valid live: the
    mic-era `lag` requirement is retired.
    """

    def __init__(
        self,
        *,
        renderer: Renderer | None = None,
        model_path: str | Path | None = DEFAULT_MODEL,
        classifier: str = "headaudio",
        hop: int = HOP_SIZE,
        frame_size: int = FRAME_SIZE,
        key_hz: float = KEY_HZ,
        render_fps: float = RENDER_FPS,
        fade_s: float = FADE_S,
        fade_shape: str = "center",
        pool: str = POOL_MODES[0],
        sticky: float = STICKY_MARGIN,
        sil_sensitivity: float = 1.2,
        vote_window: int = 6,
        gain: bool = True,
        vad: str = "auto",
        vad_active_db: float = VAD_ACTIVE_DB,
        vad_inactive_db: float = VAD_INACTIVE_DB,
    ):
        if classifier not in CLASSIFIERS:
            raise ValueError(f"unknown classifier {classifier!r}, expected one of {CLASSIFIERS}")
        if fade_shape not in FADE_SHAPES:
            raise ValueError(f"unknown fade shape {fade_shape!r}, expected one of {FADE_SHAPES}")
        if pool not in POOL_MODES:
            raise ValueError(f"unknown pool mode {pool!r}, expected one of {POOL_MODES}")
        if key_hz <= 0.0 or render_fps <= 0.0:
            raise ValueError(f"invalid rates: key_hz={key_hz} render_fps={render_fps}")

        self.classifier = classifier
        self.hop = int(hop)
        self.frame_size = int(frame_size)
        self.key_hz = float(key_hz)
        self.render_fps = float(render_fps)
        self.fade_s = float(fade_s)
        self.fade_shape = fade_shape
        self.pool = pool
        self.sticky = float(sticky)
        self.sil_sensitivity = float(sil_sensitivity)
        self.vote_window = int(vote_window)
        self.gain = bool(gain)
        self.vad = vad
        self.vad_active_db = float(vad_active_db)
        self.vad_inactive_db = float(vad_inactive_db)

        self.model_path = Path(model_path) if model_path else None
        self._model = None  # parsed once, reused by every utterance

        self.utterances: list[Utterance] = []
        self._renderer = renderer
        self._last_row: np.ndarray | None = None
        self._n = 0
        self.stats = {"polls": 0, "draws": 0, "skips": 0, "analyses": 0, "analysis_ms": 0.0}

    # -- assets -------------------------------------------------------------

    @property
    def renderer(self) -> Renderer | None:
        return self._renderer

    def attach_renderer(self, renderer: Renderer | None) -> None:
        """Set (or clear) the renderer used by `frame_at`. Resets the dirty check."""
        self._renderer = renderer
        self._last_row = None

    def _gaussian_model(self):
        if self._model is None:
            if self.model_path is None:
                raise ValueError("no model_path given to HollyFace")
            self._model = load_model(self.model_path)
        return self._model

    def _build_classifier(self, analysis_fps: float, features) -> HeadAudioClassifier | OpennessRamp:
        """Per-utterance, because `vad="auto"` reads the utterance's own energy.

        The parsed Gaussian prototypes are shared; only the gate thresholds change.

        The auto gate is not cosmetic: HeadAudio's shipped -40/-50 dBFS thresholds assume
        a noise floor under -50 dBFS, which clean TTS output does not have -- measured on
        `test_audio.flac` its quiet frames sit at le -4.9..-4.4, so a fixed gate never
        closes and the mouth hangs open through every pause. This is the exact behaviour
        `tools/make_timeline.py` ships with, and the runtime must match it.
        """
        if self.classifier == "openness-ramp":
            floor, ceiling = derive_vad_thresholds(features.rms)
            return OpennessRamp(silence_floor=floor, speech_ceiling=ceiling)

        vad_active_db, vad_inactive_db = self.vad_active_db, self.vad_inactive_db
        if self.vad == "auto":
            floor_le, ceiling_le = derive_le_thresholds(features.log_energy)
            vad_inactive_db, vad_active_db = floor_le * 10.0, ceiling_le * 10.0
        elif self.vad not in ("absolute", "off"):
            raise ValueError(f"unknown vad mode {self.vad!r}, expected auto, absolute or off")

        return HeadAudioClassifier(
            model=self._gaussian_model(),
            sil_sensitivity=self.sil_sensitivity,
            vote_window=max(0, self.vote_window),
            vad=self.vad != "off",
            vad_active_db=vad_active_db,
            vad_inactive_db=vad_inactive_db,
            analysis_fps=analysis_fps,
            gain=self.gain,
        )

    # -- analysis -----------------------------------------------------------

    def analyse(
        self, samples: np.ndarray, sample_rate: int = TARGET_RATE
    ) -> tuple[np.ndarray, tuple[Keyframe, ...], float, float]:
        """samples (float32 mono at 16 kHz) -> (weights at render_fps, keys, analysis ms, render fps).

        No file, no JSON, no ffmpeg: the host's TTS buffer goes straight in.
        """
        samples = np.asarray(samples)
        if samples.ndim != 1:
            raise ValueError(f"expected a 1-D mono buffer, got shape {samples.shape}")
        if samples.size == 0:
            raise ValueError("empty utterance buffer")
        if int(sample_rate) != TARGET_RATE:
            raise ValueError(
                f"the viseme pipeline is pinned to {TARGET_RATE} Hz mono float32 (the HeadAudio "
                f"front end was trained at it); got {sample_rate} Hz. Resample on the host side "
                f"and keep the native-rate buffer for playback -- analysis and playback buffers "
                f"must correspond 1:1 in time."
            )

        t0 = time.perf_counter()
        analysis_fps = TARGET_RATE / self.hop
        frames = frame(samples, size=self.frame_size, hop=self.hop)
        features = extract_features(frames, sample_rate=TARGET_RATE, hop=self.hop)

        clf = self._build_classifier(analysis_fps, features)
        weights = normalize_weights(clf(features))

        keys, dense, _times, fps = keyframe_weights(
            weights,
            analysis_fps=analysis_fps,
            key_hz=self.key_hz,
            out_fps=self.render_fps,
            fade_s=self.fade_s,
            shape=self.fade_shape,
            pool=self.pool,
            sticky=self.sticky,
        )
        return dense, tuple(keys), (time.perf_counter() - t0) * 1000.0, fps

    def speak(
        self,
        samples: np.ndarray,
        sample_rate: int = TARGET_RATE,
        *,
        at: float | None = None,
    ) -> Utterance:
        """Analyse a buffer and enqueue it on the face clock.

        `at` places the utterance at an explicit clock time (the host's own schedule);
        by default it lands immediately after the last queued utterance, which is what
        back-to-back playback needs. Overlapping two utterances is an error -- the
        renderer has one mouth.
        """
        weights, keys, ms, fps = self.analyse(samples, sample_rate)
        audio_duration = int(np.asarray(samples).size) / float(sample_rate)

        start = self.utterances[-1].end if self.utterances else 0.0
        if at is not None:
            if at < 0.0:
                raise ValueError(f"utterance start cannot be negative: {at}")
            start = float(at)
        if self.utterances and start < self.utterances[-1].end - 1e-9:
            raise ValueError(
                f"utterance at {start:.3f}s overlaps {self.utterances[-1].end:.3f}s "
                f"(the last utterance's end)"
            )

        utterance = Utterance(
            index=self._n,
            start=start,
            audio_duration=audio_duration,
            weights=weights,
            keys=keys,
            render_fps=fps,
            analysis_ms=ms,
        )
        self._n += 1
        self.utterances.append(utterance)
        self.stats["analyses"] += 1
        self.stats["analysis_ms"] += ms
        return utterance

    def speak_file(self, path: str | Path, *, at: float | None = None) -> Utterance:
        """Convenience for the reference player: decode a file, then `speak` it."""
        samples = decode(path)
        return self.speak(samples, TARGET_RATE, at=at)

    # -- the clock ----------------------------------------------------------

    @property
    def queued_duration(self) -> float:
        """Seconds of face clock covered by the queue."""
        return self.utterances[-1].end if self.utterances else 0.0

    @property
    def pending(self) -> int:
        return len(self.utterances)

    def utterance_at(self, t: float) -> Utterance | None:
        for utterance in self.utterances:
            if utterance.covers(t):
                return utterance
        return None

    def frame_index_at(self, t: float) -> int:
        """Render-frame index for clock time `t`, or -1 when Holly is idle."""
        utterance = self.utterance_at(t)
        if utterance is None:
            return -1
        return min(int((t - utterance.start) * utterance.render_fps), utterance.frames - 1)

    def row_at(self, t: float) -> np.ndarray:
        """Dense (15,) weight row at clock time `t`.

        Returns `IDLE_ROW` outside any utterance -- including *after* the last one,
        which is the no-final-pose-latch rule. Never None, so a host can poll it at a
        fixed rate without thinking about boundaries.
        """
        utterance = self.utterance_at(t)
        if utterance is None:
            return IDLE_ROW
        return utterance.row_covering(t)

    def viseme_at(self, t: float) -> str:
        utterance = self.utterance_at(t)
        if utterance is None:
            return BASE_VISEME
        return utterance.viseme_at(t)

    def frame_at(self, t: float) -> tuple[np.ndarray, tuple[int, int, int, int], bool]:
        """Draw the pose for `t`. Returns (uint8 RGB frame, mouth rect, changed).

        `changed=False` means the mouth rect already holds this pose: draw nothing.
        The frame object is the renderer's persistent buffer (see the aliasing
        contract in `holly/render.py`) -- retain it only across a single poll.
        """
        if self._renderer is None:
            raise RuntimeError("frame_at needs a renderer; pass one to HollyFace")

        self.stats["polls"] += 1
        row = self.row_at(t)
        if self._last_row is not None and np.array_equal(row, self._last_row):
            self.stats["skips"] += 1
            return self._renderer.frame, self._renderer.mouth_rect, False

        frame = self._renderer.draw_row(row)
        self._last_row = np.array(row, copy=True)
        self.stats["draws"] += 1
        return frame, self._renderer.mouth_rect, True

    # -- queue housekeeping -------------------------------------------------

    def prune(self, t: float) -> int:
        """Drop utterances that ended at or before `t`. Returns how many went."""
        keep = [utterance for utterance in self.utterances if utterance.end > t]
        dropped = len(self.utterances) - len(keep)
        self.utterances = keep
        return dropped

    def reset(self) -> None:
        """Clear the queue and the dirty check, as if the process had just started.

        Does not touch the renderer's persistent frame: after a reset the next
        `frame_at` reports `changed=True` for whatever pose it lands on, including the
        idle one, which is how the mouth gets back to rest.
        """
        self.utterances.clear()
        self._last_row = None
        self._n = 0

    def draw_share(self) -> float:
        """Fraction of polls that actually cost a draw. The zero-draw-idle claim, measured."""
        polls = self.stats["polls"]
        return self.stats["draws"] / polls if polls else 0.0
