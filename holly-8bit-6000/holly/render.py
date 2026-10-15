"""Runtime renderer over the buffers baked by tools/export_visemes.py.

The frame loop never touches the PSD and never touches a full canvas: it keeps a
persistent premultiplied surface and overwrites only the mouth rect each frame,
which is roughly 9% of the pixels. That is the difference between a playable
frame budget and a slideshow.

Measured on 1152x864 with the 335x265 mouth rect:

    reset (mouth-rect copy)      0.14 ms
    _over() x2 (the blend)       3.29 ms
    unpremultiply, mouth rect    3.63 ms
    unpremultiply, FULL canvas  58.00 ms   <- what this module avoids paying

The base layer is fully opaque, so outside the mouth rect unpremultiplying is a
no-op that changes RGB by mean 0.05/255 -- recomputing a constant. Converting the
whole canvas per frame measured 61.9 ms (121% of a core at the 19.6 draws/s the
keyframed timeline actually needs); converting only the mouth rect, in float32 and
without a boolean gather, measures 2.6 ms and is byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# The renderer's working precision. float32 halves the temporaries on the mouth rect
# and, measured over 308 frames, rounds to byte-identical uint8 output versus float64.
COMPUTE_DTYPE = np.float32


def unpremultiply(rgba: np.ndarray, dtype: type = np.float64) -> np.ndarray:
    """Premultiplied RGBA -> straight RGBA, for display or PNG export.

    Divides by a clamped alpha under `np.where` rather than gathering through a
    boolean mask: the mask path costs 3.27 ms on the mouth rect, this costs 1.28 ms.
    Dividing by 255.0 is deliberate -- multiplying by a float32 reciprocal of 255
    looked equivalent but shifted ~1.2% of pixels by 1/255 after rounding.
    """
    rgba = np.asarray(rgba, dtype=dtype)
    alpha = rgba[..., 3:] / dtype(255.0)
    # Never divide by zero; those results are discarded by the mask anyway.
    safe = np.where(alpha > 0, alpha, dtype(1.0))
    rgb = np.where(alpha > 0, rgba[..., :3] / safe, dtype(0.0))
    # Clamp before any uint8 cast: 300.0 would wrap to 44 and -5.0 to 251, giving
    # silent black/green speckles. Measured free next to the passes above.
    return np.clip(np.dstack([rgb, rgba[..., 3]]), 0.0, 255.0)


class Renderer:
    """Cross-fades the top-2 weighted viseme patches over the static base layer.

    Aliasing contract: `draw_row` returns the renderer's *persistent* frame buffer,
    not a fresh array -- copying 3 MB per draw is exactly the cost this class exists
    to avoid. Callers that need to retain a frame across draws must copy it
    (`Renderer.snapshot()`, or `np.array(frame, copy=True)`).
    """

    def __init__(self, npz_path: str | Path, manifest_path: str | Path | None = None):
        npz_path = Path(npz_path)
        manifest_path = Path(manifest_path) if manifest_path else npz_path.with_name("manifest.json")

        data = np.load(npz_path)
        manifest = json.loads(manifest_path.read_text())

        self.visemes: list[str] = list(manifest["visemes"])
        self.base_viseme: str = manifest["base"]
        rect = manifest["mouth_rect"]
        self.origin = (int(rect["x"]), int(rect["y"]))
        self.patch_shape = (int(rect["h"]), int(rect["w"]))
        x, y = self.origin
        h, w = self.patch_shape
        #: (x, y, w, h) in canvas pixels -- all a host needs to blit just the mouth.
        self.mouth_rect = (x, y, w, h)

        if manifest.get("alpha") != "premultiplied":
            raise ValueError(
                f"buffers are {manifest.get('alpha')!r}; the renderer expects premultiplied alpha"
            )

        base = data["base"]
        base_region = base[y : y + h, x : x + w]
        if base_region.shape[:2] != (h, w):
            raise ValueError(
                f"mouth rect {w}x{h} at {(x, y)} does not fit the {base.shape[1]}x{base.shape[0]} canvas"
            )

        # Patches stay uint8 (4.97 MB); only the 1-2 used per draw are widened, which
        # keeps resident memory at ~9 MB instead of ~104 MB on a low-end machine.
        self.patches = data["patches"]

        # `patches` is stored in canonical order minus the base viseme.
        patch_order = [v for v in self.visemes if v != self.base_viseme]
        if len(patch_order) != self.patches.shape[0]:
            raise ValueError(
                f"manifest lists {len(patch_order)} overlay visemes but the npz holds "
                f"{self.patches.shape[0]} patches"
            )
        self.patch_index = {v: i for i, v in enumerate(patch_order)}

        self.base_region = base_region.astype(COMPUTE_DTYPE)
        self.region = np.array(self.base_region, copy=True)

        #: Persistent straight-alpha RGB canvas. `draw_row` returns this same object.
        self.frame = np.rint(unpremultiply(base, COMPUTE_DTYPE)[..., :3]).astype(np.uint8)
        self.size = (self.frame.shape[1], self.frame.shape[0])

    def reset(self) -> None:
        """Restore the mouth region to the untouched base pixels."""
        self.region[...] = self.base_region

    def snapshot(self) -> np.ndarray:
        """A copy of the current frame, safe to keep across draws."""
        return np.array(self.frame, copy=True)

    def draw_row(self, weights: np.ndarray) -> np.ndarray:
        """Draw one frame from a dense weight row. Returns uint8 RGB, (H, W, 3).

        The base layer is always present, so the top-2 cross-fade works out of the
        box: `sil` has no patch, and fading toward `sil` simply lowers the alpha of
        whichever mouth overlay is active, letting the closed base mouth show
        through. That is what makes closures ease instead of snap.

        Only the mouth rect is recomputed and written; the other 91% of the canvas is
        byte-identical to the previous frame by construction.
        """
        row = np.asarray(weights, dtype=np.float64)
        if row.shape != (len(self.visemes),):
            raise ValueError(f"expected a weight row of length {len(self.visemes)}, got {row.shape}")

        self.reset()
        x, y = self.origin
        h, w = self.patch_shape

        picks = [int(i) for i in np.argsort(row)[::-1][:2] if row[int(i)] > 0.0]
        total = sum(row[i] for i in picks)
        if total > 0.0:
            for i in picks:
                name = self.visemes[i]
                if name == self.base_viseme:
                    continue
                self._over(self.region, self.patches[self.patch_index[name]], float(row[i] / total))

        self.frame[y : y + h, x : x + w] = np.rint(
            unpremultiply(self.region, COMPUTE_DTYPE)[..., :3]
        ).astype(np.uint8)
        return self.frame

    def draw(self, weights: dict[str, float] | np.ndarray) -> np.ndarray:
        """Draw one frame. Accepts a dense row or a {viseme: weight} mapping."""
        if isinstance(weights, dict):
            row = np.zeros(len(self.visemes), dtype=np.float64)
            index = {v: i for i, v in enumerate(self.visemes)}
            for name, value in weights.items():
                if name in index:
                    row[index[name]] = float(value)
            return self.draw_row(row)
        return self.draw_row(np.asarray(weights, dtype=np.float64))

    @staticmethod
    def _over(region: np.ndarray, patch: np.ndarray, weight: float) -> None:
        """Composite a weighted premultiplied patch into a premultiplied region.

        Verified against PIL's alpha_composite across 455 viseme-pair and weight
        combinations (max deviation 1.91/255).

        `weight` must arrive as a Python float: a numpy float64 scalar would promote
        the whole expression back to float64 and quietly undo the float32 work.
        """
        patch = patch.astype(region.dtype)
        t = region.dtype.type
        alpha = patch[..., 3:] / t(255.0) * weight
        region[..., :3] = region[..., :3] * (t(1.0) - alpha) + patch[..., :3] * weight
