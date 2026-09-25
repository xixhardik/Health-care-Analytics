"""Label semantics for the SPIDER lumbar spine masks.

The raw masks use a sparse, non-contiguous label vocabulary that was
confirmed empirically during dataset inspection (see
``outputs/preprocessing_reports/``) and matches the published convention:

======  ==================================================================
value   meaning
======  ==================================================================
0       background
1..9    vertebrae, numbered from the most inferior vertebra upwards
100     spinal canal
201..209 intervertebral discs, numbered from the most inferior IVD upwards
======  ==================================================================

Two derived label spaces are produced by the pipeline:

``semantic`` (4 classes)
    The target of the stated project goal - vertebrae vs intervertebral
    discs. The spinal canal is kept as its own class rather than merged
    into the background because it is annotated and discarding it would
    force the model to label canal voxels as background, which conflicts
    with the vertebra/disc boundaries around it.

        0 background, 1 vertebra, 2 IVD, 3 spinal canal

``instance`` (contiguous)
    A loss-less contiguous remap of the raw vocabulary so that the
    individual vertebra / disc identities survive preprocessing and can be
    used by a later sprint without re-reading the raw volumes.

        0 background, 1..9 vertebrae, 10 canal, 11..19 IVDs
"""

from __future__ import annotations

import numpy as np

# --- raw label vocabulary -------------------------------------------------
BACKGROUND_RAW: int = 0
VERTEBRA_RAW_RANGE: tuple[int, int] = (1, 9)  # inclusive
CANAL_RAW: int = 100
IVD_RAW_RANGE: tuple[int, int] = (201, 209)  # inclusive

# --- semantic label space -------------------------------------------------
SEM_BACKGROUND: int = 0
SEM_VERTEBRA: int = 1
SEM_IVD: int = 2
SEM_CANAL: int = 3

SEMANTIC_CLASSES: dict[int, str] = {
    SEM_BACKGROUND: "background",
    SEM_VERTEBRA: "vertebra",
    SEM_IVD: "intervertebral_disc",
    SEM_CANAL: "spinal_canal",
}

# --- instance label space -------------------------------------------------
INSTANCE_CANAL: int = 10
INSTANCE_IVD_OFFSET: int = 10  # raw 201 -> 11, raw 202 -> 12, ...


def describe_raw_label(value: int) -> str:
    """Return a human-readable description of a raw mask value."""
    if value == BACKGROUND_RAW:
        return "background"
    if VERTEBRA_RAW_RANGE[0] <= value <= VERTEBRA_RAW_RANGE[1]:
        return f"vertebra #{value} (counted from the most inferior)"
    if value == CANAL_RAW:
        return "spinal canal"
    if IVD_RAW_RANGE[0] <= value <= IVD_RAW_RANGE[1]:
        return f"IVD #{value - IVD_RAW_RANGE[0] + 1} (counted from the most inferior)"
    return f"UNKNOWN label {value}"


def is_known_raw_label(value: int) -> bool:
    """True if ``value`` belongs to the documented SPIDER label vocabulary."""
    return (
        value == BACKGROUND_RAW
        or VERTEBRA_RAW_RANGE[0] <= value <= VERTEBRA_RAW_RANGE[1]
        or value == CANAL_RAW
        or IVD_RAW_RANGE[0] <= value <= IVD_RAW_RANGE[1]
    )


def to_semantic(mask: np.ndarray) -> np.ndarray:
    """Collapse raw labels into the 4-class semantic label space.

    Uses boolean masks rather than a value-by-value loop so it stays fast on
    full volumes. Any unexpected value maps to background, which is the
    conservative choice; :func:`unknown_labels` is used separately to make
    sure that never silently happens.
    """
    mask = np.asarray(mask)
    out = np.zeros(mask.shape, dtype=np.uint8)

    vertebra = (mask >= VERTEBRA_RAW_RANGE[0]) & (mask <= VERTEBRA_RAW_RANGE[1])
    ivd = (mask >= IVD_RAW_RANGE[0]) & (mask <= IVD_RAW_RANGE[1])
    canal = mask == CANAL_RAW

    out[vertebra] = SEM_VERTEBRA
    out[ivd] = SEM_IVD
    out[canal] = SEM_CANAL
    return out


def to_instance(mask: np.ndarray) -> np.ndarray:
    """Remap raw labels onto a contiguous 0..19 range without losing identity."""
    mask = np.asarray(mask)
    out = np.zeros(mask.shape, dtype=np.uint8)

    vertebra = (mask >= VERTEBRA_RAW_RANGE[0]) & (mask <= VERTEBRA_RAW_RANGE[1])
    ivd = (mask >= IVD_RAW_RANGE[0]) & (mask <= IVD_RAW_RANGE[1])
    canal = mask == CANAL_RAW

    # Vertebra ids already sit in 1..9 and need no shifting.
    out[vertebra] = mask[vertebra].astype(np.uint8)
    out[canal] = INSTANCE_CANAL
    out[ivd] = (mask[ivd] - IVD_RAW_RANGE[0] + 1 + INSTANCE_IVD_OFFSET).astype(np.uint8)
    return out


def instance_label_name(value: int) -> str:
    """Human-readable name for a value in the contiguous instance space."""
    if value == 0:
        return "background"
    if 1 <= value <= 9:
        return f"vertebra_{value}"
    if value == INSTANCE_CANAL:
        return "spinal_canal"
    if 11 <= value <= 19:
        return f"ivd_{value - INSTANCE_IVD_OFFSET}"
    return f"unknown_{value}"


def unknown_labels(mask: np.ndarray) -> list[int]:
    """Return any values in ``mask`` outside the documented vocabulary."""
    return [int(v) for v in np.unique(mask) if not is_known_raw_label(int(v))]


def label_mapping_document() -> dict:
    """Serialisable description of both label spaces, saved next to the data.

    Written to ``data/processed/label_mapping.json`` so a later sprint can
    interpret the preprocessed masks without reading this source file.
    """
    return {
        "raw_vocabulary": {
            "background": BACKGROUND_RAW,
            "vertebrae": f"{VERTEBRA_RAW_RANGE[0]}-{VERTEBRA_RAW_RANGE[1]}"
            " (numbered from the most inferior vertebra upwards)",
            "spinal_canal": CANAL_RAW,
            "intervertebral_discs": f"{IVD_RAW_RANGE[0]}-{IVD_RAW_RANGE[1]}"
            " (numbered from the most inferior IVD upwards)",
        },
        "semantic_classes": {str(k): v for k, v in SEMANTIC_CLASSES.items()},
        "instance_classes": {
            str(v): instance_label_name(v) for v in [0, *range(1, 20)]
        },
        "notes": [
            "Semantic masks are the primary Sprint 1 target (vertebrae vs IVDs).",
            "Instance masks preserve individual vertebra/disc identity.",
            "Masks are only ever resized with nearest-neighbour interpolation.",
        ],
    }
