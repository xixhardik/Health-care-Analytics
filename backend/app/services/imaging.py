"""Slice rendering for the 2D viewer.

The point of this module is that the volume does not cross the network *on this
path*. The frontend asks for one slice in one of three renderings and gets back a
small PNG; the arrays stay on the server, and a class hidden in the interface is
never transmitted because the filtering happens here.

**Scope note, Sprint 8.** That is no longer true of the application as a whole.
``services/volume_export.py`` and ``GET /api/analysis/{id}/volume`` do send the
full volume, because browser-side volumetric rendering cannot work without the
voxels. This module is unchanged and remains the 2D viewer's path; the
data-minimisation property described above applies to it, not to the volume
endpoint. The trade is recorded in
``outputs/reports/sprint7_final/architecture.md``.

Class colours live in one table here and are mirrored in the frontend theme, so
overlay colours are defined in a single place per layer rather than scattered
through component code.
"""

from __future__ import annotations

import io

import numpy as np

#: Semantic class -> RGB. Chosen to stay legible over greyscale MRI and to remain
#: distinguishable for the most common forms of colour vision deficiency.
CLASS_COLOURS: dict[int, tuple[int, int, int]] = {
    1: (56, 189, 248),    # vertebrae  - sky
    2: (251, 191, 36),    # discs      - amber
    3: (52, 211, 153),    # canal      - emerald
}

CLASS_NAMES = {1: "vertebra", 2: "intervertebral_disc", 3: "spinal_canal"}

RENDER_MODES = ("original", "mask", "overlay")


def _to_rgb(grey: np.ndarray) -> np.ndarray:
    return np.repeat(grey[:, :, None], 3, axis=2)


#: Colour of the selected-disc marker. Distinct from every class colour so the
#: highlight cannot be mistaken for a segmentation class.
HIGHLIGHT_COLOUR = (244, 244, 245)

#: Colour of the findings overlay. Red, and deliberately not one of the
#: segmentation class colours or the selected-disc white, so it cannot be read as
#: an anatomical class. Mirrored in frontend/lib/theme.ts as FINDING_OVERLAY.
FINDING_COLOUR = (239, 68, 68)

#: How a highlight request is interpreted. ``disc`` is the long-standing
#: selected-disc bracket; ``findings`` adds the red finding-associated regions.
HIGHLIGHT_MODES = ("none", "disc", "findings")

#: Instance labels encode disc N as 10 + N.
DISC_INSTANCE_OFFSET = 10


def render_slice(
    image_uint8: np.ndarray,
    semantic: np.ndarray,
    *,
    mode: str = "overlay",
    opacity: float = 0.45,
    classes: tuple[int, ...] = (1, 2, 3),
    instance: np.ndarray | None = None,
    highlight_disc: int | None = None,
    finding_discs: tuple[int, ...] = (),
    finding_opacity: float = 0.5,
) -> np.ndarray:
    """Compose one slice into an RGB array.

    ``original`` is the preprocessed greyscale slice, ``mask`` is the label map
    on black, ``overlay`` alpha-blends the mask over the image. Class visibility
    is applied here rather than client-side so a hidden class is never sent.

    ``highlight_disc`` draws a marker around one disc so the viewer stays in step
    with the disc selected in the results panel. It is drawn from the instance
    map, so the highlight always sits on the region the result describes.

    ``finding_discs`` tints those discs red to mark a region associated with a
    model-estimated finding. The tint is confined to the disc's own instance
    region, so it can never extend beyond the validated segmentation, and the
    caller decides which discs qualify - this function draws, it does not judge.
    Nothing here mutates the arrays it is given.
    """
    if mode not in RENDER_MODES:
        raise ValueError(f"Unknown render mode {mode!r}")

    grey = image_uint8.astype(np.uint8)
    selected = tuple(c for c in classes if c in CLASS_COLOURS)

    colour = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    for class_id in selected:
        colour[semantic == class_id] = CLASS_COLOURS[class_id]

    if mode == "original":
        composed = _to_rgb(grey)
    elif mode == "mask":
        composed = colour.copy()
    else:
        base = _to_rgb(grey).astype(np.float32)
        painted = (semantic > 0) & np.isin(semantic, selected)
        alpha = float(np.clip(opacity, 0.0, 1.0))
        blended = base.copy()
        blended[painted] = (
            base[painted] * (1.0 - alpha)
            + colour[painted].astype(np.float32) * alpha
        )
        composed = np.clip(blended, 0, 255).astype(np.uint8)

    # Findings first, so the selected-disc bracket stays readable on top of it.
    if finding_discs and instance is not None:
        composed = draw_finding_overlay(
            composed,
            instance,
            finding_discs,
            opacity=finding_opacity,
            emphasis_disc=highlight_disc,
        )
    if highlight_disc is not None and instance is not None:
        composed = draw_disc_highlight(composed, instance, highlight_disc)
    return composed


def disc_region(instance: np.ndarray, disc_index: int) -> np.ndarray:
    """Boolean mask of one disc on one slice, from the Sprint 5 instance map.

    The single place the disc-index-to-instance-label convention is applied, so
    the overlay and the bracket cannot disagree about which pixels are disc N.
    """
    return instance == (DISC_INSTANCE_OFFSET + disc_index)


def _boundary(region: np.ndarray) -> np.ndarray:
    """Pixels of ``region`` with a 4-neighbour outside it (array edge counts).

    A pixel is interior only when all four neighbours are also in the region; the
    boundary is what is left. Kept dependency-free rather than pulling in
    scipy.ndimage for one erosion.
    """
    interior = np.ones_like(region)
    interior[1:, :] &= region[:-1, :]
    interior[:-1, :] &= region[1:, :]
    interior[:, 1:] &= region[:, :-1]
    interior[:, :-1] &= region[:, 1:]
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False
    return region & ~interior


def draw_finding_overlay(
    rgb: np.ndarray,
    instance: np.ndarray,
    disc_indices: tuple[int, ...] | list[int],
    *,
    opacity: float = 0.5,
    emphasis_disc: int | None = None,
) -> np.ndarray:
    """Tint finding-associated disc regions red.

    Confined to each disc's instance region, so the overlay cannot mark anything
    the segmentation did not call a disc. A disc absent from this slice draws
    nothing rather than falling back to an approximation.

    ``emphasis_disc`` is rendered more strongly and outlined, so the disc selected
    in the results panel stands out among the marked ones. The emphasis changes
    opacity only - the region is the same mask either way.
    """
    out = rgb.astype(np.float32, copy=True)
    tint = np.asarray(FINDING_COLOUR, dtype=np.float32)
    base_alpha = float(np.clip(opacity, 0.0, 1.0))

    for disc_index in sorted(set(int(i) for i in disc_indices)):
        region = disc_region(instance, disc_index)
        if not region.any():
            continue
        selected = disc_index == emphasis_disc
        alpha = min(1.0, base_alpha + 0.22) if selected else base_alpha
        out[region] = out[region] * (1.0 - alpha) + tint * alpha
        if selected:
            # A solid edge, so the selected disc reads as chosen rather than just
            # brighter - which matters on a projector.
            out[_boundary(region)] = tint

    return np.clip(out, 0, 255).astype(np.uint8)


def draw_disc_highlight(
    rgb: np.ndarray, instance: np.ndarray, disc_index: int, *, pad: int = 4
) -> np.ndarray:
    """Draw a corner bracket around one disc.

    Brackets rather than a filled box or a full rectangle: the disc boundary stays
    visible, which matters when the point of looking is to judge the segmentation.
    """
    box = disc_outline_rows(instance, disc_index)
    if box is None:
        return rgb

    height, width = instance.shape
    top = max(0, box["row_min"] - pad)
    bottom = min(height - 1, box["row_max"] + pad)
    left = max(0, box["col_min"] - pad)
    right = min(width - 1, box["col_max"] + pad)

    out = rgb.copy()
    # Bracket length scales with the box so small discs do not get a full border.
    span_v = max(3, (bottom - top) // 4)
    span_h = max(3, (right - left) // 4)

    for row in (top, bottom):
        out[row, left : left + span_h] = HIGHLIGHT_COLOUR
        out[row, right - span_h + 1 : right + 1] = HIGHLIGHT_COLOUR
    for col in (left, right):
        out[top : top + span_v, col] = HIGHLIGHT_COLOUR
        out[bottom - span_v + 1 : bottom + 1, col] = HIGHLIGHT_COLOUR
    return out


def encode_png(rgb: np.ndarray, *, scale: int = 1) -> bytes:
    """Encode an RGB array as PNG bytes."""
    from PIL import Image

    image = Image.fromarray(rgb, mode="RGB")
    if scale and scale > 1:
        image = image.resize(
            (image.width * scale, image.height * scale), Image.NEAREST
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def disc_outline_rows(instance: np.ndarray, disc_index: int) -> dict | None:
    """Bounding box of one disc on one slice, for the viewer's focus indicator."""
    selector = disc_region(instance, disc_index)
    if not selector.any():
        return None
    rows, cols = np.where(selector)
    return {
        "row_min": int(rows.min()), "row_max": int(rows.max()),
        "col_min": int(cols.min()), "col_max": int(cols.max()),
        "area_px": int(selector.sum()),
    }
