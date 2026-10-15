"""Frame features -> viseme weights.

The contract every classifier here satisfies:

    classify(features) -> (n_frames, 15) weights, one column per canonical viseme

Phase 3 and the renderer only ever see that matrix, so Phase 2 can replace
`OpennessRamp` with the HeadAudio Mahalanobis port without touching framing,
smoothing, the timeline format or the renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import N_COEFFS, FeatureSet
from .vise import BASE_VISEME, VISEMES, VISEME_INDEX

# Placeholder only. This is a mouth-openness ramp, not phonetics: it orders
# visemes by how open the mouth is and interpolates between neighbours. It
# exists to exercise the top-2 cross-fade, the smoothing and the timeline format
# end to end before a real classifier exists. Phase 2 replaces it.
OPENNESS_ORDER: tuple[str, ...] = (
    "PP", "FF", "TH", "DD", "kk", "SS", "CH", "nn", "RR", "E", "ih", "ou", "oh", "aa",
)

# Absolute fallbacks for the placeholder VAD. Real recordings vary wildly in
# level, so make_timeline.py derives thresholds from the file's own RMS
# distribution by default instead of trusting these.
SILENCE_FLOOR = 0.005
SPEECH_CEILING = 0.020

# Percentiles used when thresholds are derived automatically. Speech here sits an
# order of magnitude above the noise floor, so a gate anywhere in that gap works;
# 20/40 lands in it for typical close-mic recordings.
VAD_FLOOR_PERCENTILE = 20
VAD_CEILING_PERCENTILE = 40


def derive_vad_thresholds(rms: np.ndarray) -> tuple[float, float]:
    """Pick a silence gate from the recording's own frame-energy distribution."""
    floor = float(np.percentile(rms, VAD_FLOOR_PERCENTILE))
    ceiling = float(np.percentile(rms, VAD_CEILING_PERCENTILE))
    # Keep a usable gap even in a recording with no real silence at all.
    ceiling = max(ceiling, floor * 3.0, 1e-6)
    return floor, ceiling


def derive_le_thresholds(
    log_energy: np.ndarray,
    floor_percentile: float = VAD_FLOOR_PERCENTILE,
    ceiling_percentile: float = VAD_CEILING_PERCENTILE,
    min_gap: float = 0.5,
) -> tuple[float, float]:
    """HeadAudio-gate equivalent of derive_vad_thresholds, in `le` units.

    HeadAudio's shipped gate is absolute dBFS (-40/-50), which assumes the noise
    floor sits under -50 dBFS. Measured on test_audio.flac: quiet frames cluster
    at le -4.9..-4.4, so the fixed gate never closes and the mouth hangs open
    through every pause. Percentiles of the file's own log10-energy distribution
    adapt the way the rms-based stub gate already did. `min_gap` is 0.5 in log10
    energy (about 3x), mirroring the rms version's floor*3 rule.
    """
    floor = float(np.percentile(log_energy, floor_percentile))
    ceiling = float(np.percentile(log_energy, ceiling_percentile))
    ceiling = max(ceiling, floor + min_gap)
    return floor, ceiling


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _triangular_membership(openness: np.ndarray, n_knots: int) -> np.ndarray:
    """Hat-basis interpolation over `n_knots` evenly spaced openness knots.

    Each knot contributes max(0, 1 - |x - knot| / spacing). These hats sum to 1
    everywhere on [0, 1], which is what keeps the viseme weights normalised
    without a separate renormalisation pass.
    """
    positions = np.linspace(0.0, 1.0, n_knots)
    spacing = 1.0 / max(n_knots - 1, 1)
    return np.clip(1.0 - np.abs(openness[:, None] - positions[None, :]) / spacing, 0.0, 1.0)


class OpennessRamp:
    """Stand-in classifier: a VAD gate picks sil vs speech, spectral centroid picks openness."""

    name = "openness-ramp (placeholder)"

    def __init__(self, silence_floor: float = SILENCE_FLOOR, speech_ceiling: float = SPEECH_CEILING):
        self.silence_floor = float(silence_floor)
        self.speech_ceiling = float(speech_ceiling)

    def __call__(self, features: FeatureSet) -> np.ndarray:
        n = features.rms.shape[0]
        out = np.zeros((n, len(VISEMES)), dtype=np.float64)

        # Smoothstepped rather than linear: a linear ramp leaves every quiet-ish
        # frame with sil as the dominant viseme, which reads as a mouth that never
        # opens during normal speech.
        gap = max(self.speech_ceiling - self.silence_floor, 1e-9)
        speech = _smoothstep((features.rms - self.silence_floor) / gap)

        # Centroid is a crude openness proxy: brighter spectra -> wider aperture.
        openness = np.clip((features.centroid - 0.08) / 0.32, 0.0, 1.0)
        ramp = _triangular_membership(openness, len(OPENNESS_ORDER))

        base_idx = VISEME_INDEX[BASE_VISEME]
        out[:, base_idx] = 1.0 - speech
        for i, viseme in enumerate(OPENNESS_ORDER):
            out[:, VISEME_INDEX[viseme]] += speech * ramp[:, i]

        return out


# ---------------------------------------------------------------------------
# Phase 2: HeadAudio Gaussian-prototype classifier.
#
# Port of github.com/met4citizen/HeadAudio (MIT) modules/classifier.mjs +
# modules/training.mjs + modules/vadgate.mjs. The phoneme -> viseme map is not a
# table we carry: every record in model-en-mixed.bin embeds its phoneme (as 1-2
# UTF-8 codepoints in the header) and its HeadAudio viseme id, so parsing the
# model *is* loading the map. HeadAudio's viseme id ordering differs from our
# canonical order, hence HA_VISEME_TO_CANONICAL below.
# ---------------------------------------------------------------------------

# PARAMS.MODEL_VISEMES_N / MODEL_VISEME_SIL in parameters.mjs.
HA_VISEME_NAMES: tuple[str, ...] = (
    "aa", "E", "I", "O", "U", "PP", "SS", "TH", "DD", "FF", "kk", "nn", "RR", "CH", "sil",
)
HA_VISEME_SIL = 14

# HeadAudio viseme id -> index into holly.vise.VISEMES (sil, PP, FF, TH, DD, kk,
# CH, SS, nn, RR, aa, E, ih, oh, ou). I/O/U are the same shapes we name ih/oh/ou.
HA_VISEME_TO_CANONICAL: tuple[int, ...] = (
    VISEME_INDEX["aa"],   # 0  aa
    VISEME_INDEX["E"],    # 1  E
    VISEME_INDEX["ih"],   # 2  I
    VISEME_INDEX["oh"],   # 3  O
    VISEME_INDEX["ou"],   # 4  U
    VISEME_INDEX["PP"],   # 5  PP
    VISEME_INDEX["SS"],   # 6  SS
    VISEME_INDEX["TH"],   # 7  TH
    VISEME_INDEX["DD"],   # 8  DD
    VISEME_INDEX["FF"],   # 9  FF
    VISEME_INDEX["kk"],   # 10 kk
    VISEME_INDEX["nn"],   # 11 nn
    VISEME_INDEX["RR"],   # 12 RR
    VISEME_INDEX["CH"],   # 13 CH
    VISEME_INDEX["sil"],  # 14 sil
)

# headaudio.mjs visemeMaxs: the avatar driver never shows a viseme at full alpha.
# Mouth poses are capped (0.65, or 0.75 for the wide-open PP/FF) and ease back
# toward neutral. Phoneme-true output is ~60% narrow-aperture consonants on this
# clip; at full opacity that reads as teeth-clenching, which is why the raw
# one-hot port looked worse than the stub despite being accurate.
HA_VISEME_MAXS: tuple[float, ...] = (
    0.65, 0.65, 0.65, 0.65, 0.65,
    0.75, 0.65, 0.65, 0.65, 0.75,
    0.65, 0.65, 0.65, 0.65, 0.65,
)

# Binary record geometry (parameters.mjs): 2 header floats, then mu, then the
# lower triangle of the inverse covariance in row-major (i, j<=i) order.
RECORD_HEADER_FLOATS = 2
RECORD_MU_FLOATS = N_COEFFS
RECORD_SIGMA_FLOATS = N_COEFFS * (N_COEFFS + 1) // 2
RECORD_FLOATS = RECORD_HEADER_FLOATS + RECORD_MU_FLOATS + RECORD_SIGMA_FLOATS
RECORD_BYTES = RECORD_FLOATS * 4


def _tril_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Lower-triangle (row, col) pairs in the same order training.mjs packs them."""
    return np.tril_indices(n)


@dataclass
class GaussianModel:
    """Parsed model-en-mixed.bin: one Gaussian prototype per phoneme."""

    phonemes: tuple[str, ...]      # 1-2 codepoint IPA strings, "s1" for synthetic silence
    groups: np.ndarray             # (P,) uint8, from header byte 5
    visemes: np.ndarray            # (P,) uint8 HeadAudio viseme ids, from header byte 7
    mu: np.ndarray                 # (P, N_COEFFS) float64
    sigma_inv: np.ndarray          # (P, N_COEFFS, N_COEFFS) symmetric float64

    @property
    def n_prototypes(self) -> int:
        return self.mu.shape[0]

    def phoneme_viseme_map(self) -> dict[str, str]:
        """phoneme -> canonical viseme name, straight from the model records."""
        return {
            ph: VISEMES[HA_VISEME_TO_CANONICAL[int(vis)]]
            for ph, vis in zip(self.phonemes, self.visemes)
        }


def parse_model(data: bytes) -> GaussianModel:
    """Decode HeadAudio's binary prototype model.

    Header layout comes from training.mjs computePrototype/decodeBinaryRecord:
    the phoneme is a *big-endian* uint32 ``(cp0 << 16) | cp1`` at bytes 0-3
    (DataView.setUint32 defaults to BE), the group is byte 5 and the viseme id is
    byte 7. Everything after the header is little-endian float32.
    """
    if len(data) == 0 or len(data) % RECORD_BYTES:
        raise ValueError(
            f"model size {len(data)} is not a whole number of {RECORD_BYTES}-byte records"
        )

    records = np.frombuffer(data, dtype="<f4").reshape(-1, RECORD_FLOATS)
    raw = records.view(np.uint8).reshape(records.shape[0], RECORD_BYTES)

    phonemes: list[str] = []
    for row in raw:
        packed = int.from_bytes(bytes(row[0:4]), "big")
        cp1, cp2 = packed >> 16, packed & 0xFFFF
        if cp1 == 0:
            raise ValueError("model record has an empty phoneme header")
        phonemes.append(chr(cp1) + (chr(cp2) if cp2 else ""))

    groups = raw[:, 5].copy()
    visemes = raw[:, 7].copy()
    if not np.isin(visemes, np.arange(len(HA_VISEME_NAMES))).all():
        bad = sorted({int(v) for v in visemes if v >= len(HA_VISEME_NAMES)})
        raise ValueError(f"model records use unknown viseme ids {bad}")

    mu = records[:, RECORD_HEADER_FLOATS : RECORD_HEADER_FLOATS + RECORD_MU_FLOATS].astype(np.float64)
    lower = records[:, RECORD_HEADER_FLOATS + RECORD_MU_FLOATS :].astype(np.float64)

    rows, cols = _tril_indices(N_COEFFS)
    sigma_inv = np.zeros((records.shape[0], N_COEFFS, N_COEFFS), dtype=np.float64)
    sigma_inv[:, rows, cols] = lower
    sigma_inv[:, cols, rows] = lower  # classifier.mjs treats the storage as symmetric

    return GaussianModel(
        phonemes=tuple(phonemes),
        groups=groups,
        visemes=visemes,
        mu=mu,
        sigma_inv=sigma_inv,
    )


def load_model(path: str | Path) -> GaussianModel:
    return parse_model(Path(path).read_bytes())


def mahalanobis(vectors: np.ndarray, model: GaussianModel) -> np.ndarray:
    """(n_frames, P) squared Mahalanobis distances, exactly as classifier.mjs sums them.

    d = sum_i S_ii * diff_i^2 + sum_i sum_j<i 2 * S_ij * diff_i * diff_j, which is
    the quadratic form diff^T S diff with S rebuilt symmetric from the packed lower
    triangle. Vectorised: ~0.1 ms/frame for 39 prototypes at 12 coefficients.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != N_COEFFS:
        raise ValueError(f"expected (n_frames, {N_COEFFS}) mfcc, got {vectors.shape}")

    diff = vectors[:, None, :] - model.mu[None, :, :]
    return np.einsum("npi,pij,npj->np", diff, model.sigma_inv, diff, optimize=True)


class HeadAudioClassifier:
    """HeadAudio's Gaussian-prototype Mahalanobis classifier, ported.

    Satisfies the same contract as OpennessRamp: ``__call__(features) ->
    (n_frames, 15)`` weights in canonical viseme order. Each frame is one-hot --
    HeadAudio emits discrete visemes, and the keyframe pooler turns one-hots into
    per-slot distributions, which is its own smoothing.

    Faithful defaults, each defeatable for A/B tuning against the stub:

    * ``sil_sensitivity=1.2``  -- PARAMS.silSensitivity: sil prototype distances are
      divided by it, so silence must be beaten by a margin to close the mouth.
    * ``vote_window=6``        -- classifier.mjs majority vote over the last 6
      argmin visemes (ring initialised to sil).
    * ``vad=True``             -- vadgate.mjs log-energy hysteresis gate
      (-40 dBFS active / -50 dBFS inactive, 10 ms). Inactive frames skip the
      classifier entirely and stay sil, as HeadAudio's ``continue`` does.
    * ``gain=True``            -- headaudio.mjs visemeMaxs: mouth poses are capped
      at 0.65 (0.75 for PP/FF) with the remainder held by sil, the way HeadAudio's
      avatar driver never drives a shape to full alpha. ``gain=False`` gives the
      raw one-hot decisions.

    Not ported (live-mic concerns that offline pooling subsumes): the `sc`
    speaker-calibration prototype (group 255) and the started/ended event
    bookkeeping; the vote ring still initialises to sil.
    """

    name = "headaudio (mahalanobis gaussian prototypes)"

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        model: GaussianModel | None = None,
        sil_sensitivity: float = 1.2,
        vote_window: int = 6,
        vad: bool = True,
        vad_active_db: float = -40.0,
        vad_inactive_db: float = -50.0,
        vad_ms: float = 10.0,
        analysis_fps: float = 62.5,
        gain: bool = True,
    ):
        if model is None:
            if model_path is None:
                raise ValueError("HeadAudioClassifier needs model_path or a parsed model")
            model = load_model(model_path)
        if model.n_prototypes == 0:
            raise ValueError("parsed model has no prototypes")

        self.model = model
        self.sil_sensitivity = float(sil_sensitivity)
        self.vote_window = int(vote_window)
        self.vad = bool(vad)
        self.gain = bool(gain)
        self.active_le = vad_active_db / 10.0
        self.inactive_le = vad_inactive_db / 10.0
        # vadgate.mjs: frames = round(ms * (rate / hop) / 1000), at least 1.
        self.active_frames = max(1, round(vad_ms * analysis_fps / 1000.0))
        self.inactive_frames = max(1, round(vad_ms * analysis_fps / 1000.0))

    def distances(self, features: FeatureSet) -> np.ndarray:
        """Raw per-prototype distances with the sil-sensitivity adjustment applied."""
        d = mahalanobis(features.mfcc, self.model)
        if self.sil_sensitivity != 1.0:
            sil = self.model.visemes == HA_VISEME_SIL
            d[:, sil] /= self.sil_sensitivity
        return d

    def _vad_gate(self, log_energy: np.ndarray) -> np.ndarray:
        """bool per frame: True where vadgate.mjs would let a prediction through."""
        if not self.vad:
            return np.ones(log_energy.shape[0], dtype=bool)

        active = np.zeros(log_energy.shape[0], dtype=bool)
        is_active = 0  # the gate starts inactive, as in the JS constructor
        pre_active = 0
        pre_inactive = 0
        for i, le in enumerate(log_energy):
            if is_active:
                if le < self.inactive_le:
                    pre_inactive += 1
                    if pre_inactive >= self.inactive_frames:
                        is_active = 0
                        pre_active = 0
                elif pre_inactive > 0:
                    pre_inactive -= 1
            else:
                if le > self.active_le:
                    pre_active += 1
                    if pre_active >= self.active_frames:
                        is_active = 1
                        pre_inactive = 0
                elif pre_active > 0:
                    pre_active -= 1
            active[i] = bool(is_active)
        return active

    def __call__(self, features: FeatureSet) -> np.ndarray:
        n = features.mfcc.shape[0]
        out = np.zeros((n, len(VISEMES)), dtype=np.float64)
        sil_col = VISEME_INDEX[BASE_VISEME]
        out[:, sil_col] = 1.0

        gate = self._vad_gate(np.asarray(features.log_energy, dtype=np.float64))
        if not gate.any():
            return out

        d = self.distances(features)
        # JS picks the argmin with `d <= minD`, so the LAST prototype wins a tie.
        p = self.model.n_prototypes - 1 - np.argmin(d[:, ::-1], axis=1)
        argmin_visemes = self.model.visemes[p]  # HeadAudio ids

        # classifier.mjs majority vote: ring of `vote_window` visemes (init sil),
        # max count wins with the HIGHEST viseme id breaking ties (JS `>=`). Only
        # gated-through frames enter the ring; VAD-skipped frames leave it untouched.
        ring = np.full(max(self.vote_window, 1), HA_VISEME_SIL, dtype=np.int64) if self.vote_window > 0 else None
        counts = np.zeros(len(HA_VISEME_NAMES), dtype=np.int64)

        for i in np.nonzero(gate)[0]:
            if ring is not None:
                ring = np.roll(ring, -1)
                ring[-1] = argmin_visemes[i]
                counts[:] = np.bincount(ring, minlength=len(HA_VISEME_NAMES))
                vis = int(len(HA_VISEME_NAMES) - 1 - np.argmax(counts[::-1]))
            else:
                vis = int(argmin_visemes[i])
            row = out[i]
            row[:] = 0.0
            canonical = HA_VISEME_TO_CANONICAL[vis]
            if self.gain and vis != HA_VISEME_SIL:
                # HeadAudio's display model: cap the pose, hand the remainder to sil.
                alpha = HA_VISEME_MAXS[vis]
                row[sil_col] = 1.0 - alpha
                row[canonical] = alpha
            else:
                row[canonical] = 1.0

        return out


def top_two(weights: np.ndarray) -> tuple[int, int, float]:
    """(index_a, index_b, weight_b) for the two strongest visemes, renormalised.

    `weight_b` is the blend factor for the second viseme; the first carries
    `1 - weight_b`. Returns (-1, -1, 0.0) when the row is empty.
    """
    row = np.asarray(weights, dtype=np.float64)
    if row.ndim != 1:
        raise ValueError("top_two expects a single weight row")

    if row.size < 2 or row.sum() <= 0.0:
        return -1, -1, 0.0

    order = np.argsort(row)[::-1]
    a, b = int(order[0]), int(order[1])
    total = row[a] + row[b]
    if total <= 0.0:
        return a, b, 0.0
    return a, b, float(row[b] / total)
