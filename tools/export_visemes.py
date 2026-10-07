#!/usr/bin/env python3
"""Phase 0: audit a viseme PSD/GIMP export and bake runtime RGBA buffers.

Reads the authored PSD once, at build time, and writes flat numpy buffers that
the runtime can mmap-load without ever touching the PSD.

The audit enforces the asset spec from viseme-mouth-plan.md:
  * exactly the 15 Oculus viseme layers, named by viseme code
  * identical canvas size, mouth registered at the same pixels in every layer
  * `sil` is the opaque static base face; every other layer is a mouth-only
    overlay that lives inside one shared rect

That last check is not cosmetic. If every overlay is confined to a single
rect we can crop the runtime buffers to that rect and blit ~9% of the canvas
per frame instead of the whole face, which is the difference between a
playable frame budget and a slideshow on a low-end CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import Layer

# Canonical Oculus viseme set. Order is the storage order in the baked buffers.
VISEMES: tuple[str, ...] = (
    "sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "ih", "oh", "ou",
)
BASE_VISEME = "sil"

# An overlay that paints more than this fraction of the canvas is almost
# certainly a full-face drawing rather than a mouth patch.
FULL_FACE_ALARM = 0.25
# Overlays must agree on their content rect to within this many pixels.
REGISTRATION_TOLERANCE = 2


@dataclass
class LayerInfo:
    raw_name: str
    code: str
    src: Layer
    alpha: np.ndarray
    notes: list[str] = field(default_factory=list)

    @property
    def content_bbox(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) inclusive bbox of non-zero alpha, or None if empty."""
        ys, xs = np.nonzero(self.alpha)
        if len(xs) == 0:
            return (-1, -1, -1, -1)
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @property
    def coverage(self) -> float:
        return float((self.alpha > 0).mean())

    @property
    def opaque_fraction(self) -> float:
        return float((self.alpha >= 250).mean())


def canonicalize(raw_name: str, expected: set[str]) -> tuple[str | None, str | None]:
    """Map an authored layer name onto a canonical viseme code.

    GIMP/Photoshop users tend to leave file-style suffixes on layer names, so
    we tolerate surrounding whitespace and a trailing image extension, and fall
    back to a case-insensitive match with a warning.
    """
    name = raw_name.strip()
    for suffix in (".png", ".PNG", ".webp", ".WEBP", ".jpg", ".JPG"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    if name in expected:
        return name, None

    folded = {v.lower(): v for v in expected}
    if name.lower() in folded:
        return folded[name.lower()], f"case differs from canonical {folded[name.lower()]!r}"

    return None, f"not a recognised viseme code ({raw_name!r})"


def read_layers(psd: PSDImage) -> list[LayerInfo]:
    layers: list[LayerInfo] = []
    expected = set(VISEMES)

    for layer in psd:
        if layer.is_group():
            print(f"  skipping group {layer.name!r}", file=sys.stderr)
            continue

        code, note = canonicalize(layer.name, expected)
        if code is None:
            print(f"  WARN unknown layer: {note}", file=sys.stderr)
            continue

        image = layer.composite()
        if image is None:
            raise RuntimeError(f"layer {layer.name!r} produced no image; it may be empty or corrupt")
        rgba = np.array(image.convert("RGBA"), copy=True)
        info = LayerInfo(raw_name=layer.name, code=code, src=layer, alpha=rgba[..., 3])
        if note:
            info.notes.append(note)
        layers.append(info)

    return layers


def audit(layers: list[LayerInfo], canvas: tuple[int, int]) -> dict[str, str] | None:
    """Return an error mapping if the asset violates the spec, else None."""
    errors: dict[str, str] = {}
    codes = [layer.code for layer in layers]

    duplicates = {c for c in codes if codes.count(c) > 1}
    if duplicates:
        errors["duplicates"] = f"layers map to the same viseme: {sorted(duplicates)}"

    missing = sorted(set(VISEMES) - set(codes))
    if missing:
        errors["missing"] = f"expected viseme layers absent: {missing}"

    extra = sorted(set(codes) - set(VISEMES))
    if extra:
        errors["extra"] = f"unexpected viseme layers: {extra}"

    if errors:
        return errors

    by_code = {layer.code: layer for layer in layers}

    base = by_code[BASE_VISEME]
    if base.opaque_fraction < 0.95:
        errors["base_not_opaque"] = (
            f"{BASE_VISEME!r} is the static base layer and should be ~100% opaque "
            f"(got {base.opaque_fraction:.1%} opaque). Without an opaque base the "
            "cross-faded overlays composite onto nothing."
        )

    overlays = [layer for layer in layers if layer.code != BASE_VISEME]

    bboxes = {layer.content_bbox for layer in overlays}
    if len(bboxes) > 1:
        spread = {layer.code: layer.content_bbox for layer in overlays}
        errors["misregistered"] = (
            "mouth overlays do not share one content rect, so they are not registered "
            f"at the same pixel position: {spread}"
        )

    for layer in overlays:
        if layer.coverage > FULL_FACE_ALARM:
            errors[f"full_face_{layer.code}"] = (
                f"{layer.code!r} paints {layer.coverage:.1%} of the canvas; expected a "
                "mouth-only overlay. A full-face layer will ghost the eyes and hair "
                "when cross-faded."
            )

    return errors


def crop_rect(layers: list[LayerInfo], canvas: tuple[int, int], margin: int) -> tuple[int, int, int, int]:
    """Union bbox of every overlay's content, padded by `margin`, clamped to canvas."""
    width, height = canvas
    x0 = y0 = x1 = y1 = -1
    for layer in layers:
        if layer.code == BASE_VISEME:
            continue
        cx0, cy0, cx1, cy1 = layer.content_bbox
        if cx1 < 0:
            continue
        x0 = cx0 if x0 < 0 else min(x0, cx0)
        y0 = cy0 if y0 < 0 else min(y0, cy0)
        x1 = cx1 if x1 < 0 else max(x1, cx1)
        y1 = cy1 if y1 < 0 else max(y1, cy1)

    if x1 < 0:
        return 0, 0, width - 1, height - 1

    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(width - 1, x1 + margin)
    y1 = min(height - 1, y1 + margin)
    return x0, y0, x1, y1


def premultiply(rgba: np.ndarray) -> np.ndarray:
    """Straight alpha -> premultiplied alpha.

    Runtime blending becomes `out = base * (1 - a) + rgb_premul * weight`, one
    multiply per channel instead of two, and the weight scaling stays correct
    when two premultiplied patches are summed.
    """
    out = rgba.astype(np.float32)
    a = out[..., 3:] / 255.0
    out[..., :3] *= a
    return out


def resize_premultiplied(rgba: np.ndarray, scale: float) -> np.ndarray:
    """Resize premultiplied RGBA.

    Both the colour and alpha planes are resampled with the same filter, which
    is the correct operation on premultiplied data and avoids the dark fringes
    you get by resizing straight-alpha colour against an independent alpha.
    """
    height, width = rgba.shape[:2]
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    img = Image.fromarray(np.rint(rgba).astype(np.uint8), "RGBA").resize(new_size, Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.float32)


def rgba_of(layer: LayerInfo) -> np.ndarray:
    """Full-canvas straight-alpha RGBA for a layer, as uint8."""
    image = layer.src.composite()
    if image is None:
        raise RuntimeError(f"layer {layer.raw_name!r} produced no image; it may be empty or corrupt")
    return np.array(image.convert("RGBA"), dtype=np.uint8)


def bake(psd: PSDImage, layers: list[LayerInfo], scale: float, margin: int) -> dict:
    canvas = (psd.width, psd.height)
    rect = crop_rect(layers, canvas, margin)
    x0, y0, x1, y1 = rect

    by_code = {layer.code: layer for layer in layers}
    order = [v for v in VISEMES if v in by_code]

    base = premultiply(rgba_of(by_code[BASE_VISEME]))

    patches: list[np.ndarray] = []
    for code in order:
        if code == BASE_VISEME:
            continue
        rgba = rgba_of(by_code[code])
        cropped = rgba[y0 : y1 + 1, x0 : x1 + 1]
        patches.append(premultiply(cropped))

    if scale != 1.0:
        base = resize_premultiplied(base, scale)
        patches = [resize_premultiplied(p, scale) for p in patches]
        origin = (int(round(x0 * scale)), int(round(y0 * scale)))
        rect = (
            origin[0],
            origin[1],
            max(0, int(round((x1 + 1) * scale)) - 1),
            max(0, int(round((y1 + 1) * scale)) - 1),
        )
        canvas = (base.shape[1], base.shape[0])
    else:
        origin = (x0, y0)

    stacked = np.stack(patches).astype(np.uint8) if patches else np.zeros((0, 1, 1, 4), np.uint8)

    manifest = {
        "visemes": order,
        "base": BASE_VISEME,
        "source_canvas": {"width": psd.width, "height": psd.height},
        "render_canvas": {"width": canvas[0], "height": canvas[1]},
        "render_scale": scale,
        "mouth_rect": {"x": rect[0], "y": rect[1], "w": rect[2] - rect[0] + 1, "h": rect[3] - rect[1] + 1},
        "alpha": "premultiplied",
        "dtype": "uint8",
        "layout": {
            "base": "(H, W, 4) premultiplied RGBA, full canvas",
            "patches": "(N, h, w, 4) premultiplied RGBA, mouth crop, in `visemes` order minus the base",
        },
    }

    return {
        "base": base.astype(np.uint8),
        "patches": stacked,
        "manifest": manifest,
        "source_rect": (x0, y0, x1, y1),
        "rect_origin": origin,
    }


def verify_against_psd(baked: dict, psd: PSDImage, layers: list[LayerInfo]) -> tuple[float, float]:
    """Composite the baked buffers ourselves and compare to the PSD's own render.

    This is the registration self-check: if the crop offset or the layer order is
    wrong, the two images disagree and the numbers blow up. It catches the
    mistake without anyone having to eyeball a montage.

    Must be called with unscaled buffers: resampling shifts the bilinear sample
    grid relative to resizing the whole composite, which would read as a large
    difference at hard mouth edges and hide real registration errors.

    The comparison is done in premultiplied space. Comparing unpremultiplied RGB
    instead would divide by tiny alpha values along the feathered mouth edge and
    report huge differences where the buffers actually agree.
    """
    if baked["manifest"]["render_scale"] != 1.0:
        raise ValueError("verify_against_psd requires unscaled buffers")

    document = psd.composite()
    if document is None:
        raise RuntimeError("the PSD has no composite image")
    reference = np.array(document.convert("RGBA"), dtype=np.float32)
    reference[..., :3] *= reference[..., 3:] / 255.0

    base = baked["base"].astype(np.float32)

    # Recomposite in source stacking order (bottom -> top), not canonical order.
    order = [layer for layer in layers if layer.code != BASE_VISEME]
    # `patches` excludes the base, so its index space is the canonical order minus `sil`.
    patch_order = [v for v in baked["manifest"]["visemes"] if v != BASE_VISEME]
    index = {v: i for i, v in enumerate(patch_order)}
    out = base.copy()
    out[..., 3] = 255.0

    for layer in order:
        patch = baked["patches"][index[layer.code]].astype(np.float32)
        # Every patch was cropped from the same source rect, so they all land at
        # the same (scaled) origin.
        x0, y0 = baked["rect_origin"]
        h, w = patch.shape[:2]
        region = out[y0 : y0 + h, x0 : x0 + w]
        a = patch[..., 3:] / 255.0
        region[..., :3] = region[..., :3] * (1.0 - a) + patch[..., :3]

    diff = np.abs(out - reference)
    return float(diff.mean()), float(diff.max())


def write_preview(path: Path, baked: dict) -> None:
    manifest = baked["manifest"]
    base = baked["base"].astype(np.float32)
    names = [v for v in manifest["visemes"] if v != manifest["base"]]

    def unpre(img: np.ndarray) -> Image.Image:
        out = img.copy()
        a = out[..., 3:] / 255.0
        mask = a[..., 0] > 0
        rgb = np.zeros_like(out[..., :3])
        rgb[mask] = out[..., :3][mask] / a[mask]
        return Image.fromarray(np.rint(np.dstack([rgb, out[..., 3]])).astype(np.uint8), "RGBA")

    thumb_scale = max(0.25, min(1.0, 240.0 / max(base.shape[:2])))
    tiles = [unpre(resize_premultiplied(base, thumb_scale))]
    x0, y0 = manifest["mouth_rect"]["x"], manifest["mouth_rect"]["y"]
    for i in range(len(names)):
        frame = base.copy()
        frame[..., 3] = 255.0
        patch = baked["patches"][i].astype(np.float32)
        h, w = patch.shape[:2]
        region = frame[y0 : y0 + h, x0 : x0 + w]
        a = patch[..., 3:] / 255.0
        region[..., :3] = region[..., :3] * (1.0 - a) + patch[..., :3]
        tiles.append(unpre(resize_premultiplied(frame, thumb_scale)))

    tw, th = tiles[0].size
    sheet = Image.new("RGBA", (tw * len(tiles), th), (24, 24, 24, 255))
    for i, tile in enumerate(tiles):
        sheet.alpha_composite(tile, (i * tw, 0))
    sheet.convert("RGB").save(path, quality=90)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a viseme PSD and bake runtime RGBA buffers.")
    parser.add_argument("--psd", type=Path, default=Path("viseme.psd"))
    parser.add_argument("--out", type=Path, default=Path("build/visemes"))
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Render resolution multiplier. Use 0.5 to halve linear pixels on a weak CPU.",
    )
    parser.add_argument("--margin", type=int, default=2, help="Padding around the mouth crop, in source pixels.")
    parser.add_argument("--preview", type=Path, default=None, help="Write a contact sheet of the baked buffers here.")
    parser.add_argument("--allow-missing", action="store_true", help="Export the visemes that are present instead of failing.")
    args = parser.parse_args(argv)

    if not args.psd.exists():
        print(f"error: PSD not found: {args.psd}", file=sys.stderr)
        return 2

    psd = PSDImage.open(args.psd)
    print(f"auditing {args.psd}  canvas {psd.width}x{psd.height}")

    layers = read_layers(psd)
    print(f"  found {len(layers)} viseme layers of {len(VISEMES)} expected\n")

    header = f"{'viseme':<8} {'source layer':<16} {'opaque':>8} {'content':>9}  content bbox"
    print(header)
    print("-" * len(header))
    for layer in sorted(layers, key=lambda l: VISEMES.index(l.code)):
        for note in layer.notes:
            print(f"  WARN {layer.raw_name!r}: {note}")
        print(
            f"{layer.code:<8} {layer.raw_name:<16} "
            f"{layer.opaque_fraction:>7.1%} {layer.coverage:>8.1%}  {layer.content_bbox}"
        )
    print()

    errors = audit(layers, (psd.width, psd.height))
    if errors:
        print("AUDIT FAILED:", file=sys.stderr)
        for message in errors.values():
            print(f"  - {message}", file=sys.stderr)
        if not args.allow_missing:
            print("\nfix the layer names/registration, or pass --allow-missing to export a partial set.", file=sys.stderr)
            return 1
        layers = [l for l in layers if l.code in VISEMES]
        print("\ncontinuing with a partial set (--allow-missing)\n", file=sys.stderr)

    baked = bake(psd, layers, args.scale, args.margin)

    # Validate the asset at native resolution. Downscaling shifts the bilinear
    # sample grid by up to half a pixel relative to resizing the whole composite,
    # which would show up as a large diff at hard mouth edges and mask real
    # registration errors, so the strict gate only ever runs on unscaled buffers.
    check = baked if args.scale == 1.0 else bake(psd, layers, 1.0, args.margin)
    mean_diff, max_diff = verify_against_psd(check, psd, layers)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "visemes.npz",
        base=baked["base"],
        patches=baked["patches"],
    )
    (args.out / "manifest.json").write_text(json.dumps(baked["manifest"], indent=2) + "\n")

    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        write_preview(args.preview, baked)

    print("baked")
    rect = baked["manifest"]["mouth_rect"]
    canvas = baked["manifest"]["render_canvas"]
    print(f"  render canvas   {canvas['width']}x{canvas['height']} (scale {args.scale})")
    print(f"  mouth rect      x={rect['x']} y={rect['y']} {rect['w']}x{rect['h']}"
          f"  ({rect['w'] * rect['h'] / (canvas['width'] * canvas['height']):.1%} of canvas)")
    print(f"  base buffer     {baked['base'].shape}  {baked['base'].nbytes / 1e6:.2f} MB")
    print(f"  patch buffers   {baked['patches'].shape}  {baked['patches'].nbytes / 1e6:.2f} MB")
    print(f"  self-check      mean |diff| vs PSD composite = {mean_diff:.2f}, max = {max_diff:.0f}")
    print(f"  wrote           {args.out / 'visemes.npz'} and {args.out / 'manifest.json'}")
    if args.preview:
        print(f"  preview         {args.preview}")

    if max_diff > 16:
        print(
            "\nself-check failed: the baked buffers do not recomposite to the PSD render. "
            "The crop offset or stacking order is wrong.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
