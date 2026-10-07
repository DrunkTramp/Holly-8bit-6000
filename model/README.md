# HeadAudio model files

Downloaded from github.com/met4citizen/HeadAudio (MIT License, Copyright (c)
2025 Mika Suominen) on 2026-10-07 for Phase 2 of the viseme pipeline.

- `model-en-mixed.bin` — from `dist/model-en-mixed.bin`, 14,352 bytes = 39
  Gaussian prototypes (38 IPA phonemes + synthetic `s1` silence), each 368 bytes:
  big-endian packed phoneme header, group byte, viseme byte, 12 float32 means,
  78 float32 inverse-covariance lower-triangle entries. This file embeds the
  phoneme -> viseme map; `holly/classify.py::parse_model` reads it.
- `tests/fixtures/headaudio_distances_oracle.csv` — from `tests/distances.csv`,
  the reference implementation's own 39x39 Mahalanobis distance matrix at the
  prototype means, used by `test_distances_match_the_js_classifier_output`.

Regenerate / re-fetch:

```sh
curl -sL -o model/model-en-mixed.bin \
  https://raw.githubusercontent.com/met4citizen/HeadAudio/main/dist/model-en-mixed.bin
curl -sL -o tests/fixtures/headaudio_distances_oracle.csv \
  https://raw.githubusercontent.com/met4citizen/HeadAudio/main/tests/distances.csv
```

The upstream MIT LICENSE text should accompany any redistribution of these
files. The full port attribution lives in `holly/classify.py`.
