"""Phase E: throttled idle micro-motion ("breathing") -- an opt-in host-side overlay.

The component owns the idle rule: outside speech `frame_at` returns `changed=False` and
the host draws nothing. That rule stands untouched -- with this feature off the host is
byte-identical to before. The plan's allowance (viseme-mouth-plan section 5) is a
*throttled* overlay of ~1-2 draws/s, so this module:

* runs only while no utterance is queued; the moment speech is queued it restores the
  pure idle pose (one draw, if it had drawn anything) and goes quiet until the turn is
  over -- which keeps the component's dirty check truthful: whenever the overlay is
  idle, the renderer's persistent frame shows exactly the pose the component believes
  it shows, so a speech frame whose row equals `IDLE_ROW` is still correctly skipped;
* draws at most one pose per `step` seconds, and only when the pose actually changed
  (the renderer has no dirty check of its own, so the overlay keeps its own last row);
* blends only `sil` with one low-weight viseme, and every cycle ends back on the pure
  idle row, so the mouth is never caught mid-breath.

Wall time drives it on purpose: the audio clock *freezes* at the tail while Holly
waits, and a breath is decoration, not something that must sync to anything. The
"never `time.time()`" rule governs the mouth-voice lock; nothing here is on that clock.
"""

from __future__ import annotations

import math
import time

from holly.runtime import make_idle_row
from holly.vise import BASE_VISEME, VISEME_INDEX


class IdleMotion:
    """A slow sin^2 breath on one viseme, drawn at most once per `step` seconds.

    `poll(active)` is called once per UI poll. `active` must mean "speech is queued or
    playing" (`face.pending > 0` is the right host-side signal: the queue keeps both
    utterances across the gap between them). Returns True when the overlay changed the
    renderer's frame, i.e. the host should blit the mouth rect.
    """

    def __init__(self, face, *, viseme: str = "aa", period: float = 4.0, step: float = 0.5,
                 max_weight: float = 0.25, clock=time.monotonic):
        if viseme not in VISEME_INDEX or viseme == BASE_VISEME:
            raise ValueError(f"unknown idle viseme {viseme!r}")
        if not 0.0 < max_weight < 0.5:
            raise ValueError(f"max_weight must stay subtle (speech caps at 0.65), got {max_weight}")
        if step <= 0 or period < 2 * step:
            raise ValueError(f"invalid rates: period={period} step={step}")
        self.face = face
        self.viseme = viseme
        self.period = float(period)
        self.step = float(step)
        self.max_weight = float(max_weight)
        self.draws = 0
        self._clock = clock
        self._index = VISEME_INDEX[viseme]
        self._t0 = self._clock()
        self._step_i: int | None = None
        self._last_w = 0.0
        self._active = False  # True while the last drawn pose is not the idle pose

    def poll(self, *, active: bool) -> bool:
        if active:
            drew = self._active
            if self._active:
                self._draw(0.0)  # restore the idle pose before the component samples speech
            self._t0 = self._clock()  # the breath restarts once the turn is over
            self._step_i = None
            return drew

        now = self._clock()
        i = int((now - self._t0) // self.step)
        if i == self._step_i:
            return False  # one draw per step at most: the throttle
        self._step_i = i
        phase = ((now - self._t0) % self.period) / self.period
        w = round(self.max_weight * math.sin(math.pi * phase) ** 2, 3)
        if w == self._last_w:
            return False  # the overlay's own dirty check
        self._draw(w)
        return True

    def _draw(self, weight: float) -> None:
        # `Renderer.draw_row` normalises the top-2 picks by their sum, so the complement
        # has to sit on `sil` explicitly: a bare {aa: w} row would render at full alpha.
        row = make_idle_row()
        if weight > 0.0:
            row[self._index] = weight
            row[VISEME_INDEX[BASE_VISEME]] = 1.0 - weight
        self._last_w = weight
        self._active = weight > 0.0
        self.face.renderer.draw_row(row)
        self.draws += 1
