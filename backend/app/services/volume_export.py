"""Serialise a completed analysis's volume for browser-side 3D rendering.

Architectural note - this is a deliberate reversal
--------------------------------------------------
Until Sprint 8 this backend held one line firmly: the volume never crossed the
network. ``imaging.py`` rendered slices server-side and sent PNGs, which meant a
class toggled off in the interface was never transmitted at all.

A real volumetric renderer cannot work that way. WebGL volume rendering needs the
voxels. So this module exists, and with it the data-minimisation property above
no longer holds for clients that call the volume endpoint: the image volume and
both label maps are sent in full, including labels for classes the user has
hidden.

That trade was made consciously for a localhost single-user research tool. It is
recorded in ``outputs/reports/sprint7_final/architecture.md`` and it is the reason
the endpoint is restricted to completed analyses and never serves raw
dataset files.

Wire format
-----------
One request, one response, no base64 inflation::

    [0:4]            uint32 little-endian  header_length
    [4:4+H]          UTF-8 JSON header
    [4+H:]           channel payloads, concatenated in header["channels"] order

The header is self-describing, so a client parses it with a ``DataView`` and a
``TextDecoder`` and needs no out-of-band knowledge. Channel payloads are plain
``uint8``, which is what the arrays already are on disk, so nothing is re-encoded
or quantised on the way out. gzip does the compression: measured on the sample
study, 6.2 MB of raw voxels leaves as roughly 1 MB.
"""

from __future__ import annotations

import json
import struct

import numpy as np

from backend.app.services.imaging import CLASS_NAMES, DISC_INSTANCE_OFFSET

#: Bumped if the binary layout changes, so a cached client payload can never be
#: misread by a newer or older reader.
VOLUME_FORMAT_VERSION = 1

#: Channel order in the payload. ``image`` is the display greyscale; ``semantic``
#: is the four-class map; ``instance`` carries per-disc identity (disc N is
#: ``DISC_INSTANCE_OFFSET + N``), which is what lets the 3D view highlight the
#: same disc the 2D view and the results panel are describing.
CHANNELS = ("image", "semantic", "instance")


def build_volume_payload(
    arrays: dict,
    *,
    analysis_id: str,
    study: dict | None = None,
    pipeline_version: str | None = None,
    finding_discs: tuple[int, ...] = (),
) -> tuple[bytes, dict]:
    """Return ``(payload_bytes, header)`` for one analysis.

    ``arrays`` is what ``AnalysisStore.read_arrays`` returns. Nothing here mutates
    it: each channel is copied into C-contiguous uint8 before serialisation.
    """
    image = np.ascontiguousarray(arrays["images"], dtype=np.uint8)
    semantic = np.ascontiguousarray(arrays["semantic"], dtype=np.uint8)
    instance = np.ascontiguousarray(arrays["instance"], dtype=np.uint8)

    if not (image.shape == semantic.shape == instance.shape):
        raise ValueError(
            f"Volume channels disagree on shape: image={image.shape}, "
            f"semantic={semantic.shape}, instance={instance.shape}"
        )
    if image.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got {image.ndim}-D {image.shape}")

    blobs = {"image": image.tobytes(), "semantic": semantic.tobytes(),
             "instance": instance.tobytes()}

    channels = []
    offset = 0
    for name in CHANNELS:
        blob = blobs[name]
        channels.append({"name": name, "offset": offset, "length": len(blob),
                         "dtype": "uint8"})
        offset += len(blob)

    slices, rows, cols = (int(n) for n in image.shape)
    study = study or {}
    in_plane = study.get("in_plane_spacing_mm")
    slice_spacing = study.get("slice_spacing_mm")

    header = {
        "format_version": VOLUME_FORMAT_VERSION,
        "analysis_id": analysis_id,
        "pipeline_version": pipeline_version,
        # Voxel index order in the payload, stated explicitly so a renderer does
        # not have to guess which axis is which.
        "axis_order": ["slice", "row", "col"],
        "dimensions": {"slices": slices, "rows": rows, "cols": cols},
        # Preprocessing resampled in-plane to 1.0 mm/px, which is what makes every
        # measurement expressible in millimetres. Through-plane spacing comes from
        # the source study when it is known, and is null rather than guessed when
        # it is not.
        "spacing_mm": {
            "row": 1.0,
            "col": 1.0,
            "slice": float(slice_spacing) if slice_spacing else None,
        },
        "native_in_plane_spacing_mm": in_plane,
        "channels": channels,
        "semantic_labels": {str(k): v for k, v in CLASS_NAMES.items()},
        "disc_instance_offset": DISC_INSTANCE_OFFSET,
        # The server's finding-overlay decision travels with the volume, so the 3D
        # view colours exactly the discs the 2D view tints. The browser does not
        # re-derive which discs qualify.
        "finding_discs": sorted(int(d) for d in finding_discs),
        "slice_ids": [str(s) for s in arrays.get("slice_ids", [])],
        "total_bytes": offset,
        "notice": (
            "Voxel data for browser-side rendering. Research prototype; not a "
            "diagnostic image. Do not redistribute: derived from licensed "
            "research data."
        ),
    }

    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload = bytearray(struct.pack("<I", len(encoded)))
    payload.extend(encoded)
    for name in CHANNELS:
        payload.extend(blobs[name])

    return bytes(payload), header


def parse_volume_payload(payload: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    """Inverse of :func:`build_volume_payload`. Used by the tests.

    Kept here rather than in the test file so the format has exactly one
    definition and a round-trip is checked against the real writer.
    """
    if len(payload) < 4:
        raise ValueError("Payload is too short to contain a header length.")
    (header_length,) = struct.unpack("<I", payload[:4])
    header = json.loads(payload[4:4 + header_length].decode("utf-8"))

    body = payload[4 + header_length:]
    dims = header["dimensions"]
    shape = (dims["slices"], dims["rows"], dims["cols"])

    channels: dict[str, np.ndarray] = {}
    for spec in header["channels"]:
        start = spec["offset"]
        chunk = body[start:start + spec["length"]]
        if len(chunk) != spec["length"]:
            raise ValueError(
                f"Channel {spec['name']!r} is truncated: expected "
                f"{spec['length']} bytes, got {len(chunk)}."
            )
        channels[spec["name"]] = np.frombuffer(
            chunk, dtype=np.uint8
        ).reshape(shape)
    return header, channels
