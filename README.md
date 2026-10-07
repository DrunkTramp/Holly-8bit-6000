# Holly-8bit-6000
A lazy attempt at recreating Red Dwarf's Holly as an 8-bit 1980s era LLM powered AI

This repo is the **animation component**: audio in, lip-synced mouth-rect frames out.
The host application (keyboard, LLM, TTS) lives elsewhere and consumes this as a library —
see `holly-host-plan.md` for that, and `HANDOFF.md` for the state of this one.

## Use it

```sh
pip install -e /path/to/Holly-8bit-6000        # gives you `import holly`
pip install -e "/path/to/Holly-8bit-6000[build,player]"   # + PSD baking and the test window
```

```python
from holly import HollyFace, Renderer, DEFAULT_BUFFERS

face = HollyFace(renderer=Renderer(DEFAULT_BUFFERS))   # 576x432 nearest bake
face.speak(tts_samples)                                # 16 kHz mono float32, analysed up front

frame, rect, changed = face.frame_at(t)                # t = your audio output position
if changed:
    blit(frame, rect)                                  # 168x132 px, ~0.7 ms
```

`row_at(t)` is the same pose as a dense `(15,)` weight row, if you would rather composite it
yourself. Outside speech both return the base pose, so the mouth returns to rest and then
costs nothing — the dirty check, the zero-draw idle and the no-final-pose-latch rule are the
component's job, not the host's.

One number to know: the default `center` cross-fade opens the mouth **one render frame (33 ms)
before** the acoustic event, with zero jitter. That is the cartoon timing grid, and it is
measured and test-pinned — `tools/check_sync.py --compare` will show it, and `--fade-shape lag`
is the zero-offset alternative if a host ever wants it.

Nothing in the core opens a window or touches an audio device. `tools/reference_player.py`
(pygame) is the worked example and the tuning harness.

## Develop it

```sh
.venv/bin/pip install -e ".[build,player]"
.venv/bin/python tools/export_visemes.py --scale 0.5 --resample nearest --out build/visemes_pixel
.venv/bin/python tools/make_timeline.py --audio test_audio.flac
.venv/bin/python tools/reference_player.py --audio test_audio.flac
.venv/bin/python tools/check_sync.py --compare         # mouth-vs-sound offset, per fade shape
.venv/bin/python -m unittest discover -s tests -t .    # 139 tests
```

`build/` is generated and gitignored — a fresh clone needs the export step before anything
will render. Model provenance is in `model/README.md`.
