"""Frame features: mel filterbank + MFCC, in plain numpy.

Every parameter here mirrors HeadAudio's front end
(github.com/met4citizen/HeadAudio, MIT: ``modules/mfcc.mjs`` +
``modules/parameters.mjs``) so the shipped Gaussian prototypes in
``model-en-mixed.bin`` describe the same feature space we feed them. The
parameter-by-parameter comparison is recorded in HANDOFF.md ("Phase 2 front-end
audit"); the short version is that HeadAudio uses a Hamming window, 40 un-normalised
mel bands spanning 30-7800 Hz, 12 coefficients with c0 dropped, cepstral lifting
(L=22), tanh compression, and log10 total power as a separate VAD feature --
none of which the earlier stand-in choices matched.

Deliberate deviations, both inert for the classifier:

* Pre-emphasis is applied per frame, but each frame's initial state is the real
  preceding sample (available because hop < window makes frames overlap), which
  reproduces HeadAudio's streaming ``y[n] = x[n] - a*x[n-1]`` exactly. HeadAudio
  pre-emphasises at the *source* rate before downsampling; we decode straight to
  16 kHz and pre-emphasise there, matching its 16 kHz-input path.
* HeadAudio's FFT loop covers bins 0..N/2-1 (Nyquist excluded); we slice the same
  bins off ``np.fft.rfft``.

No scipy: a 512-point FFT, a 40-band mel filterbank and a 12-coefficient DCT-II
are all cheap to express with numpy and keep the dependency surface small enough
to run on a low-end CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_MELS = 40            # PARAMS.MFCC_MEL_BANDS_N
N_COEFFS = 12          # PARAMS.MFCC_COEFF_N — c1..c12, c0 excluded (energy is `log_energy`)
PRE_EMPHASIS = 0.97    # PARAMS.AUDIO_PREEMPHASIS_ALPHA
LIFTER = 22            # PARAMS.MFCC_LIFTER
TANH_R = 1.0           # PARAMS.MFCC_COMPRESSION_TANH_R (compression is enabled)
MEL_LOW_HZ = 30.0      # mfcc.mjs _buildMelFilterbank lowFreq
MEL_HIGH_HZ = 7800.0   # mfcc.mjs _buildMelFilterbank highFreq
SPEAKER_MEAN_HZ = 150.0  # mfcc.mjs speakerMeanHz default; warp = clamp(f0/150, 0.6, 1.8)


@dataclass
class FeatureSet:
    mfcc: np.ndarray       # (n_frames, N_COEFFS) after liftering and tanh compression
    mel_log: np.ndarray    # (n_frames, N_MELS)
    centroid: np.ndarray   # (n_frames,) normalised to [0, 1] over the Nyquist range
    rms: np.ndarray        # (n_frames,)
    log_energy: np.ndarray # (n_frames,) HeadAudio `le`: log10 of frame power, VAD input


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def hamming_window(size: int) -> np.ndarray:
    """Symmetric Hamming window, matching mfcc.mjs _buildWindowHamming."""
    i = np.arange(size)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * i / (size - 1))


def mel_filterbank(
    n_filters: int,
    n_fft: int,
    sample_rate: int,
    low_hz: float = MEL_LOW_HZ,
    high_hz: float = MEL_HIGH_HZ,
    speaker_mean_hz: float = SPEAKER_MEAN_HZ,
) -> np.ndarray:
    """Triangular mel filterbank, (n_filters, n_fft // 2), plain (no area normalisation).

    Mirrors mfcc.mjs _buildMelFilterbank: bins are ``floor(f * n_fft / rate)`` over
    the bins 0..n_fft/2-1 that HeadAudio's power loop actually visits, the band
    edges span low_hz..high_hz on the mel scale, and the whole span is warped about
    ``low_hz`` by ``clamp(speaker_mean_hz / 150, 0.6, 1.8)`` (identity at the default).
    """
    n_bins = n_fft // 2
    warp = min(max(speaker_mean_hz / 150.0, 0.6), 1.8)
    mel_low = hz_to_mel(low_hz)
    mel_high = hz_to_mel(high_hz)
    step = (mel_high - mel_low) / (n_filters + 1)
    mel_points = mel_low + step * np.arange(n_filters + 2)
    hz_points = mel_to_hz(mel_low + (mel_points - mel_low) * warp)
    bin_points = np.floor(hz_points * n_fft / sample_rate).astype(int)

    weights = np.zeros((n_filters, n_bins), dtype=np.float64)
    for i in range(n_filters):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]

        # Rising edge covers bins [left, center), falling edge covers [center, right),
        # exactly as the two JS for-loops do. Empty ranges (degenerate filters) leave
        # the row zero, which the caller's log floor then handles.
        if center > left:
            hi = min(center, n_bins)
            if hi > left:
                bins = np.arange(left, hi)
                weights[i, left:hi] = (bins - left) / (center - left)

        if right > center:
            lo, hi = min(center, n_bins), min(right, n_bins)
            if hi > lo:
                bins = np.arange(lo, hi)
                weights[i, lo:hi] = (right - bins) / (right - center)

    return weights


def dct_matrix(n_coeffs: int, n_input: int) -> np.ndarray:
    """DCT-II rows 1..n_coeffs scaled by sqrt(2 / n_input), (n_coeffs, n_input).

    HeadAudio builds rows 0..n_coeffs but skips row 0 in compute(): the shipped
    model has no c0. Rows 1..M-1 of this basis are still orthonormal.
    """
    k = np.arange(1, n_coeffs + 1)[:, None]
    n = np.arange(n_input)[None, :]
    return np.sqrt(2.0 / n_input) * np.cos(np.pi * k * (n + 0.5) / n_input)


def cepstral_lifter(n_coeffs: int, lifter: int = LIFTER) -> np.ndarray:
    """mfcc.mjs _buildLifter: 1 + (L/2) sin(pi i / L) for i = 1..n_coeffs (c0 skipped)."""
    i = np.arange(1, n_coeffs + 1)
    return 1.0 + (lifter / 2.0) * np.sin(np.pi * i / lifter)


def pre_emphasise(
    frames: np.ndarray,
    coefficient: float = PRE_EMPHASIS,
    initial: np.ndarray | float = 0.0,
) -> np.ndarray:
    """y[n] = x[n] - a x[n-1] per frame, with `initial` supplying x[-1] per frame."""
    out = np.empty_like(frames)
    out[..., 0] = frames[..., 0] - coefficient * initial
    out[..., 1:] = frames[..., 1:] - coefficient * frames[..., :-1]
    return out


def extract_features(
    frames: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = N_MELS,
    n_coeffs: int = N_COEFFS,
    pre_emphasis: float = PRE_EMPHASIS,
    hop: int = 256,
    speaker_mean_hz: float = SPEAKER_MEAN_HZ,
) -> FeatureSet:
    frames = np.asarray(frames, dtype=np.float64)
    n_frames, size = frames.shape
    n_fft = size

    # Streaming pre-emphasis state: with hop < size, frames overlap, so the sample
    # immediately before frame i is a real element of frame i-1. Frame 0 starts from
    # silence, as HeadAudio's ring buffer and preemphasisPrevValue do.
    if 0 < hop < size:
        initial = np.zeros(n_frames, dtype=np.float64)
        initial[1:] = frames[:-1, size - hop - 1]
    else:
        initial = 0.0

    emphasized = pre_emphasise(frames, pre_emphasis, initial)
    windowed = emphasized * hamming_window(size)

    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2 / n_fft

    # HeadAudio's power loop covers bins 0..N/2-1; the Nyquist bin is never seen.
    power = spectrum[:, : n_fft // 2]
    log_energy = np.log10(power.sum(axis=1) + 1e-10)

    mel = power @ mel_filterbank(n_mels, n_fft, sample_rate, speaker_mean_hz=speaker_mean_hz).T
    mel_log = np.log(np.maximum(mel, 1e-10))

    mfcc = mel_log @ dct_matrix(n_coeffs, n_mels).T
    mfcc = mfcc * cepstral_lifter(n_coeffs)
    mfcc = TANH_R * np.tanh(mfcc / TANH_R)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    total = spectrum.sum(axis=1)
    centroid = np.where(total > 0, (spectrum * freqs).sum(axis=1) / np.maximum(total, 1e-12), 0.0)
    centroid = np.clip(centroid / (sample_rate / 2.0), 0.0, 1.0)

    return FeatureSet(
        mfcc=mfcc.astype(np.float32),
        mel_log=mel_log.astype(np.float32),
        centroid=centroid.astype(np.float32),
        rms=np.sqrt(np.mean(frames**2, axis=1)).astype(np.float32),
        log_energy=log_energy.astype(np.float32),
    )
