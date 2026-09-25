"""Sprint 5 - disc indexing post-processing on existing predictions.

No retraining, and the segmentation output is never altered. Every method here
reads the frozen ``semantic`` prediction and produces a better ``instance``
label map, so the pixels assigned to each class stay byte-identical and only the
*identity* attached to each disc changes.

Why the baseline fails
----------------------
Production indexing (``disc_features.instances_from_semantic``) numbers discs by
ordering connected components within a single slice, bottom-up. The index of a
disc is therefore a function of *how many components that one slice happens to
contain*. One spurious component below a disc, or one missed disc, renumbers
everything above it. The Sprint 3 audit measured the consequence: discs are
found ~96.7% of the time but indexed correctly only ~84.1% of the time, and
most of the failures are +/-1 shifts.

The methods
-----------
``5A`` series-level ordering
    Associate disc candidates across the slices of a series into row-aligned
    *tracks*, order the tracks once at series level, and propagate that identity
    to every slice. A disc missed on one slice then no longer renumbers its
    neighbours, because the index comes from the disc's position in the series
    rather than from the component count of that slice.

``5B`` spurious-component rejection
    Drop candidates that cannot be discs before ordering: too small, or lying
    outside the column band the series' discs occupy.

``5C`` informative-slice restriction
    Lateral slices carry only disc fragments. A prediction-derived
    informativeness score decides which slices are trustworthy enough to
    contribute to track estimation, and (reported separately) which slices to
    measure on.

``5D`` vertebral-body identification
    Vertebrae are *not* numbered by counting connected components - the audit
    established that a vertebra occupies roughly two components in a sagittal
    slice (body plus posterior elements), so ordered components are structurally
    invalid for them. Instead the vertebral body is identified as the component
    anterior to the spinal canal, and each body is numbered from the disc tracks
    it sits above.

Geometric conventions, all verified against the ground truth on the train split
before use (see ``scripts/_probe_geometry.py`` output recorded in the report):

* row 0 is superior, so a larger mean row is more inferior
* vertebra ``N`` is the body immediately *superior* to disc ``N`` (97.7% of
  ground-truth discs), which is the convention
  ``disc_features._neighbouring_vertebrae`` already relies on
* vertebral bodies sit ~23 px anterior of the canal centroid column; posterior
  elements sit ~11 px behind it
* adjacent discs are 33.8 px apart (median), 5th percentile 25.7 px, so a
  track-clustering threshold must stay well under 25 px

Level naming
------------
Integer disc indices only. The dataset does not state which vertebra is L5, so
no anatomical level name (L1/L2/.../S1) is produced anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from src.preprocessing.labels import SEM_CANAL, SEM_IVD, SEM_VERTEBRA

#: Instance label space (Sprint 1): 0 background, 1-9 vertebrae, 10 canal,
#: 11-19 discs.
INSTANCE_CANAL = 10
IVD_OFFSET = 10
MAX_DISCS = 9

#: The taxonomy's minimum component area, unchanged from the Sprint 3 audit so
#: the error categories stay comparable.
MIN_AREA_PX = 20


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PostProcessParams:
    """Every tunable of the post-processing pipeline.

    Defaults reproduce the *baseline* behaviour as closely as the structure
    allows, so a run with ``use_series_tracks=False`` and no filtering is the
    control.
    """

    # 5B - candidate rejection
    min_area_px: int = MIN_AREA_PX
    col_tolerance_px: float | None = None      # None disables the column band

    # 5A - series-level tracks
    use_series_tracks: bool = True
    cluster_px: float = 12.0                   # single-linkage row threshold
    min_track_support: float = 0.25            # fraction of informative slices
    assign_max_dist_px: float = 18.0           # component -> track gate

    # 5C - informative slices (prediction-derived only)
    informative_min_ivd_px: int = 0            # 0 = every slice is informative
    informative_min_components: int = 0

    # 5D - vertebral bodies
    use_vertebral_bodies: bool = True
    body_canal_margin_px: float = 0.0

    max_discs: int = MAX_DISCS

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Component extraction
# ---------------------------------------------------------------------------


def extract_components(binary: np.ndarray, min_area: int) -> list[dict]:
    """Connected components of a boolean mask, most inferior first.

    Mirrors ``indexing_diagnostics._components`` so candidate extraction is
    identical to what the audit measured.
    """
    from scipy import ndimage

    labelled, n = ndimage.label(binary)
    found: list[dict] = []
    for component in range(1, n + 1):
        selector = labelled == component
        area = int(selector.sum())
        if area < min_area:
            continue
        rows, cols = np.where(selector)
        found.append(
            {
                "area_px": area,
                "mean_row": float(rows.mean()),
                "mean_col": float(cols.mean()),
                "row_min": int(rows.min()),
                "row_max": int(rows.max()),
                "selector": selector,
            }
        )
    found.sort(key=lambda item: -item["mean_row"])
    return found


def canal_column(semantic: np.ndarray) -> float | None:
    """Mean column of the predicted spinal canal, or None if absent."""
    canal = semantic == SEM_CANAL
    if not canal.any():
        return None
    return float(np.where(canal)[1].mean())


# ---------------------------------------------------------------------------
# Baseline (reproduces production indexing exactly)
# ---------------------------------------------------------------------------


def baseline_disc_regions(
    semantic: np.ndarray, *, min_area: int = MIN_AREA_PX,
    max_discs: int = MAX_DISCS,
) -> dict[int, np.ndarray]:
    """Per-slice ordered-component indexing, i.e. the current behaviour.

    Reproduces ``disc_features.instances_from_semantic`` including its
    ``max_instances`` truncation, so it can serve as the control.
    """
    components = extract_components(semantic == SEM_IVD, min_area)[:max_discs]
    return {i + 1: comp["selector"] for i, comp in enumerate(components)}


# ---------------------------------------------------------------------------
# 5C - informativeness (prediction-derived; never uses ground truth)
# ---------------------------------------------------------------------------


def slice_informativeness(semantic: np.ndarray, *, min_area: int) -> dict:
    """Score how much disc evidence a slice carries.

    Derived only from the prediction, so the criterion is available at test time
    without any ground-truth annotation.
    """
    ivd = semantic == SEM_IVD
    components = extract_components(ivd, min_area)
    return {
        "ivd_area_px": int(ivd.sum()),
        "n_components": len(components),
        "foreground_area_px": int((semantic > 0).sum()),
    }


def is_informative(score: dict, params: PostProcessParams) -> bool:
    return (
        score["ivd_area_px"] >= params.informative_min_ivd_px
        and score["n_components"] >= params.informative_min_components
    )


# ---------------------------------------------------------------------------
# 5B - candidate rejection
# ---------------------------------------------------------------------------


def filter_components(
    components: list[dict], *, params: PostProcessParams,
    reference_col: float | None,
) -> tuple[list[dict], list[dict]]:
    """Split candidates into kept and rejected.

    Two conservative rules:

    * **area** - already applied at extraction, repeated here so a larger
      ``min_area_px`` than the taxonomy's 20 px can be tested.
    * **column band** - discs of one series occupy a narrow column range. A
      candidate far from the series' reference column is anatomically
      implausible and is usually a fragment of something else.
    """
    kept: list[dict] = []
    rejected: list[dict] = []
    for component in components:
        if component["area_px"] < params.min_area_px:
            rejected.append({**component, "reason": "area"})
            continue
        if (
            params.col_tolerance_px is not None
            and reference_col is not None
            and abs(component["mean_col"] - reference_col) > params.col_tolerance_px
        ):
            rejected.append({**component, "reason": "column_band"})
            continue
        kept.append(component)
    return kept, rejected


# ---------------------------------------------------------------------------
# 5A - series-level tracks
# ---------------------------------------------------------------------------


def _single_linkage_1d(values: list[float], threshold: float) -> list[list[int]]:
    """Group 1-D values into clusters separated by more than ``threshold``."""
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    clusters: list[list[int]] = [[order[0]]]
    for position in order[1:]:
        if values[position] - values[clusters[-1][-1]] <= threshold:
            clusters[-1].append(position)
        else:
            clusters.append([position])
    return clusters


def build_series_tracks(
    per_slice: dict[str, list[dict]],
    informative: dict[str, bool],
    *,
    params: PostProcessParams,
) -> list[dict]:
    """Associate disc candidates across slices into ordered series-level tracks.

    Only informative slices vote, so lateral slices holding disc fragments
    cannot invent a track. Tracks with too little support across the series are
    discarded as spurious. The surviving tracks are ordered once, most inferior
    first, and that ordering becomes the disc identity for the whole series.
    """
    rows: list[float] = []
    owners: list[str] = []
    for slice_id, components in per_slice.items():
        if not informative.get(slice_id, True):
            continue
        for component in components:
            rows.append(component["mean_row"])
            owners.append(slice_id)

    if not rows:
        return []

    n_voting = max(1, sum(1 for v in informative.values() if v))
    clusters = _single_linkage_1d(rows, params.cluster_px)

    tracks: list[dict] = []
    for cluster in clusters:
        member_rows = [rows[i] for i in cluster]
        supporters = {owners[i] for i in cluster}
        tracks.append(
            {
                "centre_row": float(np.median(member_rows)),
                "n_votes": len(cluster),
                "n_slices": len(supporters),
                "support_frac": len(supporters) / n_voting,
            }
        )

    tracks = [t for t in tracks if t["support_frac"] >= params.min_track_support]
    # Most inferior first, matching the dataset's bottom-up numbering.
    tracks.sort(key=lambda t: -t["centre_row"])
    tracks = tracks[: params.max_discs]
    for position, track in enumerate(tracks, start=1):
        track["index"] = position
    return tracks


def assign_to_tracks(
    components: list[dict], tracks: list[dict], *, params: PostProcessParams,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    """Give every component the identity of its nearest series-level track.

    Within one slice a track can only be claimed once; if two components compete
    the larger one wins and the loser is dropped, because a single disc cannot
    legitimately appear twice at the same row in the same slice.
    """
    if not tracks:
        return {}, [{**c, "reason": "no_tracks"} for c in components]

    candidates: list[tuple[float, int, dict]] = []
    unassigned: list[dict] = []
    for component in components:
        distances = [
            (abs(component["mean_row"] - track["centre_row"]), track["index"])
            for track in tracks
        ]
        distance, index = min(distances)
        if distance > params.assign_max_dist_px:
            unassigned.append({**component, "reason": "no_track_within_gate"})
            continue
        candidates.append((distance, index, component))

    # Larger area wins a contested track; ties break on the closer centre.
    candidates.sort(key=lambda item: (-item[2]["area_px"], item[0]))
    regions: dict[int, np.ndarray] = {}
    for distance, index, component in candidates:
        if index in regions:
            unassigned.append({**component, "reason": "track_already_claimed"})
            continue
        regions[index] = component["selector"]
    return regions, unassigned


# ---------------------------------------------------------------------------
# 5D - vertebral bodies
# ---------------------------------------------------------------------------


def vertebral_bodies(
    semantic: np.ndarray, *, params: PostProcessParams,
    reference_canal_col: float | None,
) -> tuple[list[dict], list[dict]]:
    """Identify vertebral *bodies*, not vertebra connected components.

    A vertebra occupies roughly two components in a sagittal slice: the body,
    which is anterior, and the posterior elements. Counting components therefore
    cannot number vertebrae. Here the canal centroid column splits the two:
    components anterior to the canal are bodies, the rest are posterior
    elements and are excluded from numbering (they stay in the vertebra semantic
    class, they simply do not receive an identity).
    """
    components = extract_components(semantic == SEM_VERTEBRA, params.min_area_px)
    column = canal_column(semantic)
    if column is None:
        column = reference_canal_col
    if column is None:
        # Without a canal reference the split is not defensible, so fall back to
        # treating every component as a body candidate and let the track/disc
        # geometry filter them.
        return components, []

    bodies: list[dict] = []
    posterior: list[dict] = []
    for component in components:
        if component["mean_col"] < column - params.body_canal_margin_px:
            bodies.append(component)
        else:
            posterior.append(component)
    return bodies, posterior


def number_bodies_from_discs(
    bodies: list[dict], disc_rows: dict[int, float],
) -> dict[int, np.ndarray]:
    """Number vertebral bodies so that body ``N`` sits directly above disc ``N``.

    Verified convention: vertebra ``N`` is superior to disc ``N``. Rather than
    ordering bodies independently - which would reintroduce the same
    count-dependent fragility - each body's index is the number of disc tracks
    strictly inferior to it. That ties vertebra identity to disc identity, which
    is what the measurement code's label arithmetic assumes.
    """
    if not bodies or not disc_rows:
        return {}
    regions: dict[int, np.ndarray] = {}
    for body in bodies:
        below = sum(1 for row in disc_rows.values() if row > body["mean_row"])
        if below < 1 or below > MAX_DISCS:
            continue
        if below in regions:
            # Two bodies map to the same slot; keep the larger.
            if int(regions[below].sum()) >= body["area_px"]:
                continue
        regions[below] = body["selector"]
    return regions


# ---------------------------------------------------------------------------
# Instance map assembly
# ---------------------------------------------------------------------------


def assemble_instance_map(
    semantic: np.ndarray,
    disc_regions: dict[int, np.ndarray],
    vertebra_regions: dict[int, np.ndarray] | None,
) -> np.ndarray:
    """Build the Sprint 1 instance label space from identified regions.

    Layer order matches ``disc_features.instance_mask_from_semantic``: vertebrae
    first, then the canal, then discs on top.
    """
    out = np.zeros(semantic.shape, dtype=np.uint8)
    if vertebra_regions:
        for index, selector in vertebra_regions.items():
            out[selector] = index
    out[semantic == SEM_CANAL] = INSTANCE_CANAL
    for index, selector in disc_regions.items():
        out[selector] = IVD_OFFSET + index
    return out


# ---------------------------------------------------------------------------
# Series pipeline
# ---------------------------------------------------------------------------


def process_series(
    semantic_by_slice: dict[str, np.ndarray], *, params: PostProcessParams,
) -> dict:
    """Run the full post-processing pipeline over one series.

    Returns the new instance map per slice plus per-series diagnostics.
    """
    scores = {
        slice_id: slice_informativeness(semantic, min_area=params.min_area_px)
        for slice_id, semantic in semantic_by_slice.items()
    }
    informative = {
        slice_id: is_informative(score, params) for slice_id, score in scores.items()
    }
    # If the criterion rejects everything, fall back to using all slices rather
    # than returning nothing.
    if not any(informative.values()):
        informative = {slice_id: True for slice_id in semantic_by_slice}

    # Series reference columns, from informative slices only.
    disc_cols: list[float] = []
    canal_cols: list[float] = []
    raw_components: dict[str, list[dict]] = {}
    for slice_id, semantic in semantic_by_slice.items():
        components = extract_components(semantic == SEM_IVD, params.min_area_px)
        raw_components[slice_id] = components
        if informative[slice_id]:
            disc_cols.extend(c["mean_col"] for c in components)
        column = canal_column(semantic)
        if column is not None:
            canal_cols.append(column)
    reference_col = float(np.median(disc_cols)) if disc_cols else None
    reference_canal_col = float(np.median(canal_cols)) if canal_cols else None

    # 5B
    filtered: dict[str, list[dict]] = {}
    n_rejected = 0
    reject_reasons: dict[str, int] = {}
    for slice_id, components in raw_components.items():
        kept, rejected = filter_components(
            components, params=params, reference_col=reference_col
        )
        filtered[slice_id] = kept
        n_rejected += len(rejected)
        for item in rejected:
            reject_reasons[item["reason"]] = reject_reasons.get(item["reason"], 0) + 1

    # 5A
    tracks = (
        build_series_tracks(filtered, informative, params=params)
        if params.use_series_tracks else []
    )

    instance_maps: dict[str, np.ndarray] = {}
    n_unassigned = 0
    unassigned_reasons: dict[str, int] = {}
    for slice_id, semantic in semantic_by_slice.items():
        components = filtered[slice_id]
        if params.use_series_tracks and tracks:
            disc_regions, unassigned = assign_to_tracks(
                components, tracks, params=params
            )
            n_unassigned += len(unassigned)
            for item in unassigned:
                unassigned_reasons[item["reason"]] = (
                    unassigned_reasons.get(item["reason"], 0) + 1
                )
        else:
            # Per-slice ordering, on the filtered candidates.
            ordered = components[: params.max_discs]
            disc_regions = {i + 1: c["selector"] for i, c in enumerate(ordered)}

        vertebra_regions: dict[int, np.ndarray] | None = None
        if params.use_vertebral_bodies:
            bodies, _posterior = vertebral_bodies(
                semantic, params=params,
                reference_canal_col=reference_canal_col,
            )
            disc_rows = {
                index: float(np.where(selector)[0].mean())
                for index, selector in disc_regions.items()
            }
            vertebra_regions = number_bodies_from_discs(bodies, disc_rows)
        else:
            vertebrae, _ = _legacy_vertebra_regions(semantic, params)
            vertebra_regions = vertebrae

        instance_maps[slice_id] = assemble_instance_map(
            semantic, disc_regions, vertebra_regions
        )

    return {
        "instance_maps": instance_maps,
        "n_tracks": len(tracks),
        "tracks": [
            {k: v for k, v in track.items() if k != "selector"} for track in tracks
        ],
        "n_slices": len(semantic_by_slice),
        "n_informative_slices": int(sum(informative.values())),
        "informative": informative,
        "reference_col": reference_col,
        "reference_canal_col": reference_canal_col,
        "n_components_rejected": n_rejected,
        "reject_reasons": reject_reasons,
        "n_components_unassigned": n_unassigned,
        "unassigned_reasons": unassigned_reasons,
    }


def _legacy_vertebra_regions(
    semantic: np.ndarray, params: PostProcessParams,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    """Ordered-component vertebra numbering, i.e. the current (invalid) scheme.

    Retained only so method 5D can be compared against what it replaces.
    """
    components = extract_components(semantic == SEM_VERTEBRA, params.min_area_px)
    components = components[:MAX_DISCS]
    return {i + 1: c["selector"] for i, c in enumerate(components)}, []


# ---------------------------------------------------------------------------
# Scoring - the corrected taxonomy, applied to an INSTANCE map
# ---------------------------------------------------------------------------


def classify_discs(
    truth_instance: np.ndarray,
    predicted_regions: dict[int, np.ndarray],
    *,
    min_area: int = MIN_AREA_PX,
) -> dict:
    """Classify every ground-truth disc into the Sprint 3 failure taxonomy.

    The rules are copied unchanged from
    ``indexing_diagnostics.analyse_slice`` - best-overlap match, ``merged`` when
    the matched component also covers another ground-truth disc, ``split`` when
    more than one predicted component overlaps substantially, ``correct`` when
    the indices agree, ``shifted`` otherwise. The only difference is that
    candidates come from a supplied *index -> region* mapping instead of being
    re-derived from the semantic map, which is what lets a post-processed
    identity be scored on exactly the same definition.
    """
    truth_regions = {
        value - IVD_OFFSET: (truth_instance == value)
        for value in range(IVD_OFFSET + 1, IVD_OFFSET + MAX_DISCS + 1)
        if int((truth_instance == value).sum()) >= min_area
    }

    records: list[dict] = []
    for truth_index in sorted(truth_regions):
        region = truth_regions[truth_index]
        truth_area = int(region.sum())

        overlaps = {
            index: int((region & selector).sum())
            for index, selector in predicted_regions.items()
        }
        overlapping = {k: v for k, v in overlaps.items() if v > 0}

        if not overlapping:
            records.append(
                {
                    "truth_index": truth_index,
                    "category": "missed",
                    "matched_index": None,
                    "offset": None,
                    "truth_area_px": truth_area,
                    "overlap_px": 0,
                    "iou": 0.0,
                    "n_overlapping_components": 0,
                    "merged_with": None,
                }
            )
            continue

        best_index = max(overlapping, key=overlapping.get)
        best_region = predicted_regions[best_index]
        overlap = overlapping[best_index]
        union = int((region | best_region).sum())

        also_covers = [
            other for other, other_region in truth_regions.items()
            if other != truth_index
            and int((other_region & best_region).sum()) >= min_area
        ]
        substantial = [k for k, v in overlapping.items() if v >= min_area]

        if also_covers:
            category = "merged"
        elif len(substantial) > 1:
            category = "split"
        elif best_index == truth_index:
            category = "correct"
        else:
            category = "shifted"

        records.append(
            {
                "truth_index": truth_index,
                "category": category,
                "matched_index": best_index,
                "offset": best_index - truth_index,
                "truth_area_px": truth_area,
                "overlap_px": overlap,
                "iou": round(overlap / union, 4) if union else 0.0,
                "n_overlapping_components": len(substantial),
                "merged_with": also_covers or None,
            }
        )

    spurious = []
    for index, selector in predicted_regions.items():
        total = sum(
            int((selector & region).sum()) for region in truth_regions.values()
        )
        if total < min_area:
            spurious.append({"predicted_index": index,
                             "area_px": int(selector.sum())})

    return {
        "discs": records,
        "n_truth_discs": len(truth_regions),
        "n_predicted_components": len(predicted_regions),
        "count_matches": len(truth_regions) == len(predicted_regions),
        "count_delta": len(predicted_regions) - len(truth_regions),
        "n_spurious_components": len(spurious),
        "n_correct": sum(1 for r in records if r["category"] == "correct"),
    }


def regions_from_instance(
    instance: np.ndarray, *, min_area: int = MIN_AREA_PX
) -> dict[int, np.ndarray]:
    """Recover the index -> region mapping from an instance label map."""
    regions: dict[int, np.ndarray] = {}
    for value in range(IVD_OFFSET + 1, IVD_OFFSET + MAX_DISCS + 1):
        selector = instance == value
        if int(selector.sum()) >= min_area:
            regions[value - IVD_OFFSET] = selector
    return regions


def summarise_records(
    disc_records: list[dict], slice_records: list[dict]
) -> dict:
    """Aggregate taxonomy records, matching the Sprint 3 summary fields."""
    import pandas as pd

    if not disc_records:
        return {}
    discs = pd.DataFrame(disc_records)
    slices = pd.DataFrame(slice_records)
    total = len(discs)
    counts = discs["category"].value_counts().to_dict()
    shifted = discs[discs["category"] == "shifted"]
    offsets = (
        shifted["offset"].value_counts().sort_index().to_dict()
        if len(shifted) else {}
    )
    slices = slices.copy()
    slices["all_correct"] = slices["n_correct"] == slices["n_truth_discs"]

    by_index = (
        discs.groupby("truth_index")
        .agg(
            n=("category", "size"),
            correct=("category", lambda s: int((s == "correct").sum())),
            shifted=("category", lambda s: int((s == "shifted").sum())),
            merged=("category", lambda s: int((s == "merged").sum())),
            split=("category", lambda s: int((s == "split").sum())),
            missed=("category", lambda s: int((s == "missed").sum())),
        )
        .reset_index()
    )
    by_index["pct_correct"] = (100 * by_index["correct"] / by_index["n"]).round(2)

    return {
        "n_discs_analysed": int(total),
        "n_slices_analysed": int(len(slices)),
        "categories": {
            category: {
                "n": int(counts.get(category, 0)),
                "pct": round(100 * counts.get(category, 0) / total, 2),
            }
            for category in ("correct", "shifted", "merged", "split", "missed")
        },
        "pct_index_correct": round(100 * counts.get("correct", 0) / total, 2),
        "pct_region_found": round(
            100 * (total - counts.get("missed", 0)) / total, 2
        ),
        "shifted_offsets": {str(k): int(v) for k, v in offsets.items()},
        "pct_shifted_by_one": round(
            100 * sum(v for k, v in offsets.items() if abs(k) == 1) / total, 2
        ) if offsets else 0.0,
        "pct_slices_count_correct": round(
            100 * float(slices["count_matches"].mean()), 2
        ),
        "pct_slices_all_discs_correct": round(
            100 * float(slices["all_correct"].mean()), 2
        ),
        "total_spurious_components": int(slices["n_spurious_components"].sum()),
        "pct_slices_with_spurious": round(
            100 * float((slices["n_spurious_components"] > 0).mean()), 2
        ),
        "by_disc_index": by_index.to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Compact cache, so parameter sweeps do not re-read the image data
# ---------------------------------------------------------------------------
#
# The taxonomy outcome depends on three things: the pixel overlap between each
# ground-truth disc and each predicted candidate, the geometry of each candidate,
# and the index assigned to each candidate. Only the last of those changes when a
# parameter changes. Precomputing the first two collapses a sweep from minutes of
# array I/O per configuration to milliseconds of arithmetic, and keeps memory
# flat - storing the boolean regions for the validation split would need well
# over a gigabyte.


def build_slice_cache(
    truth_instance: np.ndarray,
    semantic: np.ndarray,
    *,
    min_area: int = MIN_AREA_PX,
) -> dict:
    """Summarise one slice into everything the taxonomy needs, without pixels."""
    truth_areas: dict[int, int] = {}
    truth_regions: dict[int, np.ndarray] = {}
    for value in range(IVD_OFFSET + 1, IVD_OFFSET + MAX_DISCS + 1):
        selector = truth_instance == value
        area = int(selector.sum())
        if area >= min_area:
            truth_areas[value - IVD_OFFSET] = area
            truth_regions[value - IVD_OFFSET] = selector

    components = extract_components(semantic == SEM_IVD, min_area)
    compact: list[dict] = []
    for component in components:
        overlaps = {
            index: int((component["selector"] & region).sum())
            for index, region in truth_regions.items()
        }
        compact.append(
            {
                "area_px": component["area_px"],
                "mean_row": component["mean_row"],
                "mean_col": component["mean_col"],
                "overlaps": {k: v for k, v in overlaps.items() if v > 0},
            }
        )

    return {
        "truth_areas": truth_areas,
        "components": compact,
        "informativeness": slice_informativeness(semantic, min_area=min_area),
        "canal_col": canal_column(semantic),
    }


def classify_from_cache(
    entry: dict,
    assigned: dict[int, int],
    *,
    min_area: int = MIN_AREA_PX,
) -> dict:
    """Run the taxonomy from a cached slice and a candidate -> index assignment.

    ``assigned`` maps the *position* of a component in ``entry["components"]`` to
    the disc index it was given. Components absent from the mapping were rejected
    and take no part, exactly as if they had never been predicted.

    Produces the same fields as :func:`classify_discs`; the two are cross-checked
    against each other in the driver script.
    """
    truth_areas = entry["truth_areas"]
    components = entry["components"]

    # index -> (position, area, overlaps)
    by_index: dict[int, dict] = {}
    for position, index in assigned.items():
        component = components[position]
        by_index[index] = {
            "position": position,
            "area_px": component["area_px"],
            "overlaps": component["overlaps"],
        }

    records: list[dict] = []
    for truth_index in sorted(truth_areas):
        truth_area = truth_areas[truth_index]
        overlapping = {
            index: data["overlaps"].get(truth_index, 0)
            for index, data in by_index.items()
        }
        overlapping = {k: v for k, v in overlapping.items() if v > 0}

        if not overlapping:
            records.append(
                {
                    "truth_index": truth_index,
                    "category": "missed",
                    "matched_index": None,
                    "offset": None,
                    "truth_area_px": truth_area,
                    "overlap_px": 0,
                    "iou": 0.0,
                    "n_overlapping_components": 0,
                }
            )
            continue

        best_index = max(overlapping, key=overlapping.get)
        best = by_index[best_index]
        overlap = overlapping[best_index]
        union = truth_area + best["area_px"] - overlap

        also_covers = [
            other for other in truth_areas
            if other != truth_index
            and best["overlaps"].get(other, 0) >= min_area
        ]
        substantial = [k for k, v in overlapping.items() if v >= min_area]

        if also_covers:
            category = "merged"
        elif len(substantial) > 1:
            category = "split"
        elif best_index == truth_index:
            category = "correct"
        else:
            category = "shifted"

        records.append(
            {
                "truth_index": truth_index,
                "category": category,
                "matched_index": best_index,
                "offset": best_index - truth_index,
                "truth_area_px": truth_area,
                "overlap_px": overlap,
                "iou": round(overlap / union, 4) if union else 0.0,
                "n_overlapping_components": len(substantial),
            }
        )

    n_spurious = sum(
        1 for data in by_index.values()
        if sum(data["overlaps"].values()) < min_area
    )

    return {
        "discs": records,
        "n_truth_discs": len(truth_areas),
        "n_predicted_components": len(by_index),
        "count_matches": len(truth_areas) == len(by_index),
        "count_delta": len(by_index) - len(truth_areas),
        "n_spurious_components": n_spurious,
        "n_correct": sum(1 for r in records if r["category"] == "correct"),
    }


# ---------------------------------------------------------------------------
# Assignment strategies operating on the cache
# ---------------------------------------------------------------------------


def assign_baseline(entry: dict, params: PostProcessParams) -> dict[int, int]:
    """Per-slice ordered components - the current production behaviour."""
    positions = list(range(len(entry["components"])))
    # entry["components"] is already most-inferior-first.
    kept = [
        p for p in positions
        if entry["components"][p]["area_px"] >= params.min_area_px
    ][: params.max_discs]
    return {position: i + 1 for i, position in enumerate(kept)}


def assign_series(
    entries: dict[str, dict], params: PostProcessParams
) -> tuple[dict[str, dict[int, int]], dict]:
    """Series-level assignment (5A + 5B + 5C) over cached slices of one series."""
    informative = {
        slice_id: is_informative(entry["informativeness"], params)
        for slice_id, entry in entries.items()
    }
    if not any(informative.values()):
        informative = {slice_id: True for slice_id in entries}

    # Series reference column from informative slices only (5B).
    columns = [
        component["mean_col"]
        for slice_id, entry in entries.items() if informative[slice_id]
        for component in entry["components"]
    ]
    reference_col = float(np.median(columns)) if columns else None

    kept: dict[str, list[int]] = {}
    reject_reasons: dict[str, int] = {}
    for slice_id, entry in entries.items():
        positions = []
        for position, component in enumerate(entry["components"]):
            if component["area_px"] < params.min_area_px:
                reject_reasons["area"] = reject_reasons.get("area", 0) + 1
                continue
            if (
                params.col_tolerance_px is not None
                and reference_col is not None
                and abs(component["mean_col"] - reference_col)
                > params.col_tolerance_px
            ):
                reject_reasons["column_band"] = (
                    reject_reasons.get("column_band", 0) + 1
                )
                continue
            positions.append(position)
        kept[slice_id] = positions

    if not params.use_series_tracks:
        assignment = {
            slice_id: {
                position: i + 1
                for i, position in enumerate(positions[: params.max_discs])
            }
            for slice_id, positions in kept.items()
        }
        return assignment, {
            "n_tracks": 0, "reject_reasons": reject_reasons,
            "n_informative_slices": int(sum(informative.values())),
            "reference_col": reference_col, "unassigned_reasons": {},
        }

    # 5A - build tracks from informative slices only.
    rows: list[float] = []
    owners: list[str] = []
    for slice_id, positions in kept.items():
        if not informative[slice_id]:
            continue
        for position in positions:
            rows.append(entries[slice_id]["components"][position]["mean_row"])
            owners.append(slice_id)

    n_voting = max(1, sum(1 for v in informative.values() if v))
    tracks: list[dict] = []
    for cluster in _single_linkage_1d(rows, params.cluster_px):
        supporters = {owners[i] for i in cluster}
        tracks.append(
            {
                "centre_row": float(np.median([rows[i] for i in cluster])),
                "n_votes": len(cluster),
                "n_slices": len(supporters),
                "support_frac": len(supporters) / n_voting,
            }
        )
    tracks = [t for t in tracks if t["support_frac"] >= params.min_track_support]
    tracks.sort(key=lambda t: -t["centre_row"])
    tracks = tracks[: params.max_discs]
    for position, track in enumerate(tracks, start=1):
        track["index"] = position

    assignment: dict[str, dict[int, int]] = {}
    unassigned_reasons: dict[str, int] = {}
    for slice_id, positions in kept.items():
        if not tracks:
            assignment[slice_id] = {
                position: i + 1
                for i, position in enumerate(positions[: params.max_discs])
            }
            continue
        candidates: list[tuple[float, int, int, int]] = []
        for position in positions:
            component = entries[slice_id]["components"][position]
            distance, index = min(
                (abs(component["mean_row"] - t["centre_row"]), t["index"])
                for t in tracks
            )
            if distance > params.assign_max_dist_px:
                unassigned_reasons["no_track_within_gate"] = (
                    unassigned_reasons.get("no_track_within_gate", 0) + 1
                )
                continue
            candidates.append((distance, index, position, component["area_px"]))

        candidates.sort(key=lambda item: (-item[3], item[0]))
        taken: dict[int, int] = {}
        for distance, index, position, _area in candidates:
            if index in taken:
                unassigned_reasons["track_already_claimed"] = (
                    unassigned_reasons.get("track_already_claimed", 0) + 1
                )
                continue
            taken[index] = position
        assignment[slice_id] = {position: index for index, position in taken.items()}

    return assignment, {
        "n_tracks": len(tracks),
        "tracks": tracks,
        "reject_reasons": reject_reasons,
        "unassigned_reasons": unassigned_reasons,
        "n_informative_slices": int(sum(informative.values())),
        "reference_col": reference_col,
        "informative": informative,
    }


# ---------------------------------------------------------------------------
# Generic structure classifier (used to score method 5D on vertebrae)
# ---------------------------------------------------------------------------


def classify_generic(
    truth_regions: dict[int, np.ndarray],
    predicted_regions: dict[int, np.ndarray],
    *,
    min_area: int = MIN_AREA_PX,
) -> list[dict]:
    """Apply the taxonomy rules to any indexed structure, not just discs.

    Used for vertebrae so method 5D is judged on the same definition of
    ``correct`` / ``shifted`` / ``merged`` / ``split`` / ``missed`` as the disc
    indexing, rather than on a bespoke metric.
    """
    records: list[dict] = []
    for truth_index in sorted(truth_regions):
        region = truth_regions[truth_index]
        overlaps = {
            index: int((region & selector).sum())
            for index, selector in predicted_regions.items()
        }
        overlapping = {k: v for k, v in overlaps.items() if v > 0}

        if not overlapping:
            records.append({"truth_index": truth_index, "category": "missed",
                            "matched_index": None, "offset": None})
            continue

        best_index = max(overlapping, key=overlapping.get)
        best_region = predicted_regions[best_index]
        also_covers = [
            other for other, other_region in truth_regions.items()
            if other != truth_index
            and int((other_region & best_region).sum()) >= min_area
        ]
        substantial = [k for k, v in overlapping.items() if v >= min_area]

        if also_covers:
            category = "merged"
        elif len(substantial) > 1:
            category = "split"
        elif best_index == truth_index:
            category = "correct"
        else:
            category = "shifted"

        records.append({"truth_index": truth_index, "category": category,
                        "matched_index": best_index,
                        "offset": best_index - truth_index})
    return records


def vertebra_truth_regions(
    truth_instance: np.ndarray, *, min_area: int = MIN_AREA_PX
) -> dict[int, np.ndarray]:
    """Ground-truth vertebra instances, keyed by their own label 1..9."""
    regions: dict[int, np.ndarray] = {}
    for value in range(1, MAX_DISCS + 1):
        selector = truth_instance == value
        if int(selector.sum()) >= min_area:
            regions[value] = selector
    return regions
