"""The TTS seam and the sample-rate contract.

One interface, so the LLM, the queue and the face never care which voice is running:

    synthesize(text) -> (np.float32 mono, sample_rate)

The backends (Piper on CPU, OmniVoice on GPU) emit different rates: 16k, 22.05k, 24k. The viseme pipeline accepts **only** 16 kHz mono float32 -- the HeadAudio
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
import threading
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


class PiperVoiceTTS:
    """CPU path, and the default: piper's ONNX voice loaded **once** and reused.

    Measured on this box (2026-10-11), and the reason the subprocess form is not the default:

        `piper` CLI, one call            1.42 s   (almost all of it reloading the model)
        in-process `PiperVoice.load()`   0.85 s   (paid once, at startup)
        in-process `synthesize()`        0.07-0.21 s per 0.65-7.8 s of audio (~30x realtime)

    A streamed answer arrives sentence by sentence, and the queue must have sentence *k+1*
    ready before sentence *k*'s audio finishes playing. A per-call cost larger than a short
    sentence's playback starves the queue and Holly goes silent mid-answer -- which is exactly
    what the CLI backend did (see `tools/brain_check.py`'s STALL verdict).

    Piper yields one chunk per phrase; they are concatenated and stay float32 at the voice's
    own rate (no int16 round-trip), so the analysis path sees the same numbers the speaker
    plays. The onnxruntime session is not documented as thread-safe; the app only ever calls
    this from the chat worker thread, but the lock makes a second caller harmless.
    """

    name = "piper"

    def __init__(self, model: str | Path, *, rate: int | None = None,
                 speaker: int | None = None, length_scale: float = 1.0):
        self.model = Path(model)
        if not self.model.exists():
            raise FileNotFoundError(f"piper voice not found: {self.model}")
        try:
            from piper import PiperVoice
            from piper.config import SynthesisConfig
        except ImportError as exc:  # the [piper] extra is optional
            raise ImportError(
                "the in-process piper backend needs piper-tts: pip install -e '.[piper]'"
            ) from exc
        # None = "whatever the checkpoint says", so a voice that ships its own tempo keeps it.
        self._config = SynthesisConfig(
            speaker_id=speaker,
            length_scale=None if float(length_scale) == 1.0 else float(length_scale),
        )
        self._voice = PiperVoice.load(str(self.model))
        self._lock = threading.Lock()
        self.rate = int(rate or self._voice.config.sample_rate)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        text = str(text)
        with self._lock:
            chunks = [np.asarray(c.audio_float_array, dtype=np.float32)
                      for c in self._voice.synthesize(text, syn_config=self._config)]
        if not chunks:
            raise RuntimeError(f"piper produced no audio for {text!r}")
        pcm = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        return pcm, self.rate


class PiperTTS:
    """CPU path, fallback: piper's ONNX voice run as a **subprocess**.

    Kept because it needs nothing importable in this process (useful if onnxruntime and the
    host's other native libs fight) -- but it pays ~1.4 s of model load per sentence, so it is
    only ever correct for batch work, never for a streamed reply. `make_tts("piper")` returns
    `PiperVoiceTTS` instead.

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


class OmniVoiceTTS:
    """GPU path, opt-in: OmniVoice zero-shot clone of a recorded reference voice.

    The model (``models/OmniVoice``, gitignored) and its audio tokenizer load **lazily**, on
    the first ``synthesize()`` -- construction stays free so the app can wire the seam before
    the GPU is touched. The reference voice is cloned once into a reusable
    :class:`VoiceClonePrompt` and cached as a ``.pt`` sidecar next to the reference audio
    (re-encoded only if the reference is newer than the cache).

    Cloning needs a transcript of the reference. We refuse to auto-transcribe: that would
    pull whisper-large-v3-turbo (~1.5 GB) from HuggingFace, and this repo's rule is that
    model weights are placed by hand, never downloaded. So the transcript lives in a ``.txt``
    sidecar next to the reference (or arrives as ``ref_text``).

    Output is float32 at the model's native 24 kHz; the seam's resample-once contract takes
    care of the 16 k analysis copy.
    """

    name = "omnivoice"

    def __init__(self, model: str | Path, ref_audio: str | Path, *, ref_text: str | None = None,
                 prompt: str | Path | None = None, language: str | None = "en",
                 device: str | None = None, dtype: str = "float16", rate: int | None = None,
                 num_step: int | None = None, guidance_scale: float | None = None):
        self.model_dir = Path(model)
        self.ref_audio = Path(ref_audio)
        if not self.ref_audio.exists():
            raise FileNotFoundError(f"omnivoice reference audio not found: {self.ref_audio}")
        self.ref_text = ref_text
        # None -> default sidecar next to the reference (voices/holly_base.pt for holly_base.flac).
        self.prompt_path = Path(prompt) if prompt else self.ref_audio.with_suffix(".pt")
        self.language = language
        self.device = device
        self.dtype = dtype
        self.rate = int(rate or 24000)  # reconciled against the model's own rate when it loads
        # Diffusion sampling knobs. None = the model's own defaults (32 steps, guidance 2.0).
        # Fewer steps trades quality for latency, which is the dial that decides whether a
        # sentence is ready before the previous one finishes playing.
        self.num_step = num_step
        self.guidance_scale = guidance_scale
        self._gen_config = None
        self._model = None
        self._voice_prompt = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from omnivoice import OmniVoice
        except ImportError as exc:
            raise ImportError(
                "the omnivoice backend needs the vendored repo and a CUDA torch; see README"
            ) from exc
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = OmniVoice.from_pretrained(
            str(self.model_dir), device_map=device,
            dtype=getattr(torch, self.dtype),
        )
        native = int(self._model.sampling_rate)
        if self.rate != native:
            print(f"[tts] omnivoice emits {native} Hz, not the declared {self.rate} Hz; "
                  "declaring " + str(native), file=sys.stderr)
            self.rate = native
        if self.num_step is not None or self.guidance_scale is not None:
            from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

            defaults = OmniVoiceGenerationConfig()
            self._gen_config = OmniVoiceGenerationConfig(
                num_step=int(self.num_step if self.num_step is not None else defaults.num_step),
                guidance_scale=float(self.guidance_scale
                                     if self.guidance_scale is not None
                                     else defaults.guidance_scale),
            )
        return self._model

    def _ensure_prompt(self):
        if self._voice_prompt is not None:
            return self._voice_prompt
        model = self._ensure_model()
        if self.prompt_path.exists() and \
                self.prompt_path.stat().st_mtime >= self.ref_audio.stat().st_mtime:
            from omnivoice.models.omnivoice import VoiceClonePrompt

            self._voice_prompt = VoiceClonePrompt.load(str(self.prompt_path))
            return self._voice_prompt
        ref_text = self.ref_text
        if ref_text is None:
            sidecar = self.ref_audio.with_suffix(".txt")
            if sidecar.exists():
                ref_text = sidecar.read_text().strip()
        if not ref_text:
            raise FileNotFoundError(
                f"no transcript for the omnivoice reference {self.ref_audio}. Write the exact "
                f"words spoken in it to {self.ref_audio.with_suffix('.txt')} (or pass ref_text)."
            )
        self._voice_prompt = model.create_voice_clone_prompt(
            ref_audio=str(self.ref_audio), ref_text=ref_text,
        )
        try:
            self._voice_prompt.save(str(self.prompt_path))
        except OSError:
            pass  # a read-only voices/ costs a re-encode next start, not a crash
        return self._voice_prompt

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        with self._lock:
            model = self._ensure_model()
            prompt = self._ensure_prompt()
            out = model.generate(text=str(text), language=self.language,
                                 voice_clone_prompt=prompt, generation_config=self._gen_config)
        if not out:
            raise RuntimeError(f"omnivoice produced no audio for {text!r}")
        return np.asarray(out[0], dtype=np.float32), self.rate

    def warmup(self) -> None:
        """Pay model load, CUDA init and the voice-prompt encode up front.

        Lazy loading keeps `make_tts` free, but it also means the *first sentence* absorbs
        ~5 s (warm page cache) to ~11 s (cold) of one-off cost -- `brain_check.py` reported
        `1audio ms` of 5316 ms purely because of that. The app calls this at startup so the
        user never waits on it mid-conversation. Idempotent, and it deliberately does not
        synthesise: the weights and prompt are the expensive part.
        """
        with self._lock:
            self._ensure_model()
            self._ensure_prompt()


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


def make_tts(kind: str, *, voice=None, paths=None, rate: int | None = None,
             device: str | None = None, num_step: int | None = None):
    """Backend factory for the app: `null` runs anywhere, `piper` needs a voice on disk.

    `null` is the default so the Phase A skeleton runs with no downloads; it emits 22.05 kHz,
    which exercises the exact same resample-once/keep-both seam a real Piper voice will.
    With `--tts piper` and no `--voice`, a stock voice already in `voices/` is used.
    `omnivoice` is the opt-in GPU clone: `--voice` is the reference audio, defaulting to
    `voices/holly_base.flac`. `device` is a torch device string ("cuda", "cuda:0", "cpu");
    by default it picks cuda when one is available.
    """
    kind = (kind or "null").lower()
    if kind == "null":
        return NullTTS(rate=rate or 22050)
    if kind == "omnivoice":
        root = Path(__file__).resolve().parents[1]
        model_dir = Path(paths[0]) if paths else root / "models" / "OmniVoice"
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"omnivoice model dir not found: {model_dir} (copy the weights there; see README)"
            )
        ref = Path(voice) if voice else root / "voices" / "holly_base.flac"
        return OmniVoiceTTS(model_dir, ref, rate=rate, device=device, num_step=num_step)
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
        return PiperVoiceTTS(voice, rate=rate)
    if kind == "script":
        if not paths:
            raise ValueError("--tts script needs at least one audio path")
        return ScriptTTS(paths, rate=rate or FACE_RATE)
    raise ValueError(f"unknown tts backend {kind!r} (null, piper, omnivoice, script)")


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

#: A sentence ends on one of these, and only under the conditions `_boundary` checks.
_TERMINATORS = ".!?"


def _boundary(buf: str, i: int) -> bool:
    """Is the terminator at `buf[i]` (already known to be followed by whitespace or EOS)
    a real sentence end?

    The token before it must be a real word end: not a decimal (12.5), not an initial
    (J. R.), not a known abbreviation (Dr.). One predicate, shared by the batch splitter
    and the streaming one, because the two disagreeing is exactly the kind of bug that
    only shows up live.
    """
    words = buf[: i + 1].split()
    if not words:
        return False
    stem = words[-1].rstrip(_TERMINATORS).lower()
    return not (stem.isdigit() or len(stem) <= 1 or stem in _ABBREVIATIONS)


def _clip_long(sentence: str, max_chars: int) -> list[str]:
    """Cut a long unpunctuated ramble on a clause boundary so TTS gets a breath."""
    out: list[str] = []
    while len(sentence) > max_chars:
        cut = max(sentence.rfind(sep, 0, max_chars) for sep in ",;:")
        if cut <= 0:
            cut = max(sentence.rfind(" ", 0, max_chars), max_chars // 2)
        head, tail = sentence[:cut], sentence[cut:]
        if tail[:1] in (",", ";", ":"):
            # The comma belongs to the clause that owns it. A chunk that *starts* with "," makes
            # the voice open on a stray pause and the caption read wrong -- `", which is quite
            # impressive for a holographic entity."` was real output from --max-chars 120.
            head, tail = head + tail[:1], tail[1:]
        out.append(head.strip())
        sentence = tail.strip()
    if sentence:
        out.append(sentence)
    return out


class SentenceSplitter:
    """Incremental sentence splitting for a token stream -- `feed()` deltas, `end()` at EOS.

    This is the pipeline's overlap lever: the LLM streams tokens, and as soon as a sentence
    boundary is *confirmed* it can be synthesised and analysed while the previous sentence
    is still playing. Perceived latency after the keypress is first-token latency plus
    first-sentence TTS time, not the whole reply.

    The one genuinely incremental subtlety: a terminator at the end of the buffer is
    **undecided**, because the rule needs the following character. `"12."` + `"5"` must not
    split. So a terminator found at the buffer tail is held until the next delta (or until
    `end()`, where end-of-stream counts as whitespace, exactly as it does for the end of a
    line in the batch form).

    Whitespace is collapsed to single spaces as text arrives, which has to be done across
    chunk seams: `"a "` + `" b"` is `"a b"`, and `"won"` + `"derful"` is `"wonderful"`.

    `junk` decides what happens to a one-character fragment ("!", "."). `"hold"` (the
    streaming default) never sends one to TTS -- it is merged into the next sentence and
    dropped at EOS if nothing follows, because a one-character utterance synthesises badly
    and gives the classifier nothing to chew on. `"pass"` emits it, which is what
    `split_sentences` needs to keep its documented batch contract.
    """

    def __init__(self, max_chars: int = 220, *, junk: str = "hold"):
        if junk not in ("hold", "pass"):
            raise ValueError(f"junk must be 'hold' or 'pass', not {junk!r}")
        self.max_chars = int(max_chars)
        self.junk = junk
        self._buf = ""      # normalized text since the last emitted sentence
        self._scan = 0      # everything before this index has been decided
        self._sep = False   # the last character seen was whitespace
        self._cr = False    # a trailing '\r' is held until we know if '\n' follows
        self._junk = ""
        self._spoken = False

    # -- public ------------------------------------------------------------

    def feed(self, piece: str) -> list[str]:
        """Absorb one streamed delta; return the sentences whose boundary is now confirmed."""
        if not piece:
            return []
        if self._cr:
            piece = "\r" + piece
            self._cr = False
        if piece.endswith("\r"):
            piece = piece[:-1]
            self._cr = True

        out: list[str] = []
        lines = piece.split("\n")
        for n, line in enumerate(lines):
            self._absorb(line)
            out.extend(self._scan_boundaries())
            if n < len(lines) - 1:  # a newline ends the sentence candidate, as in the batch form
                out.extend(self._flush())
        return out

    def end(self) -> list[str]:
        """Close the stream: end-of-stream is whitespace, so any held boundary resolves."""
        out = self._scan_boundaries()
        out.extend(self._flush())
        return out

    # -- internals ---------------------------------------------------------

    def _absorb(self, text: str) -> None:
        if not text:
            return
        seam_is_space = self._sep or text[:1].isspace()
        ends_in_space = text[-1:].isspace()
        body = " ".join(text.split())
        if not body:
            self._sep = True
            return
        if self._buf and not seam_is_space:
            self._buf += body          # a word split across the chunk seam
        elif self._buf:
            self._buf += " " + body
        else:
            self._buf = body
        self._sep = ends_in_space

    def _scan_boundaries(self) -> list[str]:
        out: list[str] = []
        i = self._scan
        while True:
            j = next((k for k in range(i, len(self._buf)) if self._buf[k] in _TERMINATORS), -1)
            if j < 0:
                break
            following = self._buf[j + 1 : j + 2]
            if following and following != " ":
                i = j + 1              # decided: 12.5, e-mail, J.R.
                continue
            if not following:
                self._scan = j         # undecided: the next delta tells us
                return out
            if not _boundary(self._buf, j):
                i = j + 1
                continue
            out.extend(self._emit(self._buf[: j + 1]))
            self._buf = self._buf[j + 1 :]
            i = 0
        self._scan = i
        return out

    def _flush(self) -> list[str]:
        out = self._emit(self._buf)
        self._buf = ""
        self._scan = 0
        self._sep = True
        if self._junk and not self._spoken:
            out.append(self._junk)     # nothing else was produced: say it rather than lose it
            self._spoken = True
        self._junk = ""
        return out

    def _emit(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        out: list[str] = []
        for piece in _clip_long(text, self.max_chars):
            if self.junk == "hold" and len(piece) <= 1:
                self._junk = (self._junk + " " + piece).strip()
                continue
            out.append(piece)
            self._spoken = True
        return out


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    """Sentence chunks, one per utterance -- the batch form of `SentenceSplitter`.

    Splitting is deliberately conservative: a false boundary sends a fragment to TTS, which
    synthesises it badly *and* gives the viseme classifier nothing to chew on. A missed
    boundary only costs a little latency. So a period counts as a boundary only when it is
    followed by whitespace and the token before it is a real word end -- not a decimal
    (12.5), an initial (J. R.), or a known abbreviation (Dr.).

    The batch form keeps one rule the streaming form does not: a one-character fragment is
    dropped *unless* it is all there is, so this never returns `[]` for non-empty input.
    """
    text = str(text).replace("\r\n", "\n")
    if not text.strip():
        return []
    splitter = SentenceSplitter(max_chars, junk="pass")
    out = splitter.feed(text) + splitter.end()
    return [s for s in out if len(s) > 1] or out
