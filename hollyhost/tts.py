"""The TTS seam and the sample-rate contract.

One interface, so the LLM, the queue and the face never care which voice is running:

    synthesize(text) -> (np.float32 mono, sample_rate)

Two backends are planned (Piper on CPU, OmniVoice on GPU) and they emit different rates:
16k, 22.05k, 24k. The viseme pipeline accepts **only** 16 kHz mono float32 -- the HeadAudio
front end is pinned to it and `HollyFace.analyse()` raises otherwise. So this module owns
the one genuinely tricky rule in the host:

    resample once, keep both, never re-derive.

`prepare()` returns the 16k buffer for analysis AND the native-rate buffer for playback.
They must correspond 1:1 in time, which is why the resampler is band-limited rather than
linear: a sloppy resampler shifts formant energy, the classifier picks different visemes,
and the mouth disagrees with the voice in a way that looks like a lip-sync bug but isn't.

Playback may stay at the native rate, because the face queue advances by *duration*
(seconds), not sample count -- 16k and 24k buffers of the same audio are the same number
of seconds.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from holly.audio import TARGET_RATE

FACE_RATE = TARGET_RATE  # 16000


def _piper_executable(hint: str = "piper") -> str:
    """Resolve the piper CLI without depending on PATH.

    A venv's `bin/piper` is only on PATH when the venv is activated; when the host is run as
    `.venv/bin/python -m hollyhost.app` it is not. So: honour PATH, else look next to the
    running interpreter (the venv's own bin), else fall back to the bare name and let the
    subprocess surface the error.
    """
    found = shutil.which(hint)
    if found:
        return found
    sibling = Path(sys.executable).with_name(hint)
    return str(sibling) if sibling.exists() else hint


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Band-limited sinc resampling, pure numpy, no scipy.

    Windowed-sinc interpolation with the cutoff at the lower of the two Nyquist
    frequencies, so downsampling cannot alias and upsampling cannot image. Kaiser beta 8
    is a reasonable stopband/side-lobe trade for audio that is about to be analysed by a
    mel filterbank rather than listened to critically.
    """
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    src_rate, dst_rate = int(src_rate), int(dst_rate)
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32)
    if samples.size == 1:
        return samples.astype(np.float32)

    ratio = dst_rate / src_rate
    out_len = max(1, int(round(samples.size * ratio)))
    # Position of each output sample back in the input grid.
    pos = np.arange(out_len, dtype=np.float64) / ratio
    half = 48  # taps per side; 97 total. Narrow transition, still trivial for utterances.
    centers = np.floor(pos).astype(np.int64)
    offsets = pos - centers

    idx = np.arange(-half + 1, half + 1)
    # Cutoff in *input* sample-rate units: the lower of the two Nyquists. Downsampling
    # must filter at the output Nyquist or it aliases; upsampling must not filter below
    # the input Nyquist or it loses bandwidth.
    fc = min(0.5, 0.5 * ratio)
    window = np.kaiser(idx.size, 10.0)  # ~-100 dB stopband; beta 8 left 9 kHz in the skirt

    # Computed in blocks: the naive form materialises out_len x (2*half) float64s, which
    # for a 7 s utterance at 16 kHz is ~90 MB -- real memory for a thing that runs on a Pi.
    step = 4096
    out = np.empty(out_len, dtype=np.float64)
    for lo in range(0, out_len, step):
        hi = min(lo + step, out_len)
        x = offsets[lo:hi, None] - idx[None, :]
        taps = np.sinc(2.0 * fc * x) * window[None, :]
        # Normalise every output sample's taps to unit DC gain. Deriving the analytic gain
        # for a windowed kernel is where the algebra goes wrong; this is exact by
        # construction and keeps tone energy and DC level identical across the rate change.
        taps /= taps.sum(axis=1, keepdims=True)
        gather = np.clip(centers[lo:hi, None] + idx[None, :], 0, samples.size - 1)
        out[lo:hi] = np.sum(samples[gather] * taps, axis=1)
    return out.astype(np.float32)


@dataclass
class Prepared:
    """What the host hands to the face and to the speakers."""

    text: str
    analysis: np.ndarray  # 16 kHz mono float32 -> HollyFace.speak()
    playback: np.ndarray  # native rate -> the audio device
    rate: int             # playback rate

    @property
    def duration(self) -> float:
        return self.playback.size / self.rate


def prepare(text: str, samples: np.ndarray, rate: int) -> Prepared:
    """Enforce the contract: analysis is always 16k, playback stays native."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    rate = int(rate)
    if rate <= 0:
        raise ValueError(f"bad sample rate {rate}")
    analysis = samples if rate == FACE_RATE else resample(samples, rate, FACE_RATE)
    return Prepared(text=text, analysis=analysis, playback=samples, rate=rate)


class NullTTS:
    """Deterministic stand-in: a harmonic tone per sentence, no model, no downloads.

    Lets A1 wire text -> queue -> face -> audio with the whole pipeline exercised, before
    a voice exists. Not a stub to be embarrassed by -- it is what makes the seam testable.
    """

    name = "null"

    def __init__(self, rate: int = 22050, seconds_per_word: float = 0.18, f0: float = 130.0):
        self.rate = int(rate)
        self.seconds_per_word = seconds_per_word
        self.f0 = float(f0)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        words = max(1, len([w for w in str(text).split() if w]))
        n = max(self.rate // 20, round(words * self.seconds_per_word * self.rate))
        t = np.arange(n) / self.rate
        body = sum(np.sin(2 * np.pi * self.f0 * k * t) / k for k in (1, 2, 3, 4, 5))
        env = np.hanning(n)
        return (body / np.abs(body).max() * 0.4 * env).astype(np.float32), self.rate


class PiperTTS:
    """CPU path: piper's ONNX voice, run as a subprocess.

    `--output-raw` gives signed 16-bit PCM at the *voice's* native rate (16k or 22.05k
    depending on the checkpoint), so the rate is declared rather than sniffed.
    """

    name = "piper"

    def __init__(self, model: str | Path, *, rate: int = 22050, executable: str | None = None,
                 speaker: int | None = None, length_scale: float = 1.0):
        self.model = Path(model)
        self.rate = int(rate)
        self.executable = _piper_executable(executable or "piper")
        self.speaker = speaker
        self.length_scale = float(length_scale)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if not self.model.exists():
            raise FileNotFoundError(f"piper voice not found: {self.model}")
        cmd = [self.executable, "--model", str(self.model), "--output-raw"]
        if self.speaker is not None:
            cmd += ["--speaker", str(self.speaker)]
        if self.length_scale != 1.0:
            cmd += ["--length-scale", str(self.length_scale)]
        proc = subprocess.run(cmd, input=str(text).encode(), capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"piper failed: {proc.stderr.decode('utf-8', 'replace').strip()}")
        pcm = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0
        return pcm, self.rate


def _default_piper_voice():
    """A stock voice checked into `voices/` (gitignored), if one is present.

    Returns (path, sample_rate) read from the voice's own `.onnx.json`, so the rate is the
    checkpoint's real output rate rather than a guess. None when no voice has been fetched.
    """
    voices_dir = Path(__file__).resolve().parents[1] / "voices"
    if not voices_dir.is_dir():
        return None
    for onnx in sorted(voices_dir.glob("*.onnx")):
        cfg = onnx.with_suffix(".onnx.json")
        rate = 22050
        if cfg.exists():
            try:
                import json

                rate = int(json.loads(cfg.read_text())["audio"]["sample_rate"])
            except Exception:
                pass
        return onnx, rate
    return None


def make_tts(kind: str, *, voice=None, paths=None, rate: int | None = None):
    """Backend factory for the app: `null` runs anywhere, `piper` needs a voice + the CLI.

    `null` is the default so the Phase A skeleton runs with no downloads; it emits 22.05 kHz,
    which exercises the exact same resample-once/keep-both seam a real Piper voice will.
    With `--tts piper` and no `--voice`, a stock voice already in `voices/` is used.
    """
    kind = (kind or "null").lower()
    if kind == "null":
        return NullTTS(rate=rate or 22050)
    if kind == "piper":
        default = _default_piper_voice()
        if not voice:
            if not default:
                raise ValueError(
                    "--tts piper needs --voice, or a stock voice in voices/ "
                    "(fetch one: python -m piper.download_voices en_US-lessac-medium --download-dir voices)"
                )
            voice, vrate = default
            rate = rate or vrate
        return PiperTTS(voice, rate=rate or 22050)
    if kind == "script":
        if not paths:
            raise ValueError("--tts script needs at least one audio path")
        return ScriptTTS(paths, rate=rate or FACE_RATE)
    raise ValueError(f"unknown tts backend {kind!r} (null, piper, script)")


class ScriptTTS:
    """Play recorded files instead of synthesising. Useful until a voice is trained."""

    name = "script"

    def __init__(self, paths: list[str | Path], rate: int = FACE_RATE):
        from holly.audio import decode

        self._decode = decode
        self.paths = [Path(p) for p in paths]
        self.rate = int(rate)
        self._i = 0

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        path = self.paths[self._i % len(self.paths)]
        self._i += 1
        return self._decode(str(path), target_rate=self.rate), self.rate


_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "st", "prof", "inc", "ltd", "jr", "sr",
                  "vs", "etc", "approx", "dept", "est", "min", "max", "no"}


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    """Sentence chunks, one per utterance.

    This is the pipeline's overlap lever: the LLM streams tokens, and as soon as a
    sentence boundary appears it can be synthesised and analysed while the previous
    sentence is still playing. Perceived latency after the keypress is first-token latency
    plus first-sentence TTS time, not the whole reply.

    Splitting is deliberately conservative: a false boundary sends a fragment to TTS, which
    synthesises it badly *and* gives the viseme classifier nothing to chew on. A missed
    boundary only costs a little latency. So a period counts as a boundary only when it is
    followed by whitespace and the token before it is a real word end -- not a decimal
    (12.5), an initial (J. R.), or a known abbreviation (Dr.).
    """
    text = str(text).replace("\r\n", "\n")
    if not text.strip():
        return []

    out: list[str] = []
    buf = ""
    for line in text.split("\n"):
        line = " ".join(line.split())
        if not line:
            continue
        i = 0
        while i < len(line):
            ch = line[i]
            buf += ch
            i += 1
            if ch not in ".!?":
                continue
            if i < len(line) and line[i] not in " \t":
                continue  # 12.5, e-mail, J.R.
            token = buf.strip().split()[-1] if buf.strip() else ""
            stem = token.rstrip(".!?").lower()
            if stem.isdigit() or len(stem) <= 1 or stem in _ABBREVIATIONS:
                continue
            out.append(buf.strip())
            buf = ""
        if buf.strip():
            out.append(buf.strip())
            buf = ""

    # A long unpunctuated ramble is cut on a clause boundary so TTS gets a breath.
    final: list[str] = []
    for sentence in out:
        while len(sentence) > max_chars:
            cut = max(sentence.rfind(sep, 0, max_chars) for sep in ",;:")
            if cut <= 0:
                cut = max(sentence.rfind(" ", 0, max_chars), max_chars // 2)
            final.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            final.append(sentence)
    return [s for s in final if len(s) > 1] or final
