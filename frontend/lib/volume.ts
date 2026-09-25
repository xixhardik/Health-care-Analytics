/**
 * Volume client for the 3D viewer.
 *
 * Parses the one-request binary payload served by
 * `GET /api/analysis/{id}/volume`. The format is defined once on the server in
 * `backend/app/services/volume_export.py`; this is its only reader.
 *
 *   [0:4]    uint32 little-endian  header_length
 *   [4:4+H]  UTF-8 JSON header
 *   [4+H:]   channel payloads, concatenated in header.channels order
 *
 * Nothing here interprets anatomy. Which discs carry findings and what the disc
 * instance offset is both arrive in the header, decided server-side, so the 3D
 * view marks exactly what the 2D view marks rather than re-deriving it.
 */

import { API_BASE } from "./api";

export interface VolumeChannelSpec {
  name: "image" | "semantic" | "instance";
  offset: number;
  length: number;
  dtype: "uint8";
}

export interface VolumeHeader {
  format_version: number;
  analysis_id: string;
  pipeline_version: string | null;
  axis_order: ["slice", "row", "col"];
  dimensions: { slices: number; rows: number; cols: number };
  spacing_mm: { row: number; col: number; slice: number | null };
  native_in_plane_spacing_mm: number[] | null;
  channels: VolumeChannelSpec[];
  semantic_labels: Record<string, string>;
  disc_instance_offset: number;
  finding_discs: number[];
  slice_ids: string[];
  total_bytes: number;
  notice: string;
}

export interface LoadedVolume {
  header: VolumeHeader;
  /** Display greyscale, one byte per voxel. */
  image: Uint8Array;
  /** Four-class semantic map: 0 background, 1 vertebra, 2 disc, 3 canal. */
  semantic: Uint8Array;
  /** Per-instance map. Disc N is `header.disc_instance_offset + N`. */
  instance: Uint8Array;
  /** slices * rows * cols, precomputed because every consumer needs it. */
  voxelCount: number;
}

/** The format this client understands. A mismatch is refused, not guessed at. */
export const SUPPORTED_VOLUME_FORMAT = 1;

export class VolumeError extends Error {
  constructor(
    message: string,
    readonly code:
      | "TOO_SHORT"
      | "BAD_HEADER"
      | "UNSUPPORTED_FORMAT"
      | "MISSING_CHANNEL"
      | "TRUNCATED"
      | "BAD_DIMENSIONS",
  ) {
    super(message);
    this.name = "VolumeError";
  }
}

export function volumeUrl(analysisId: string): string {
  return `${API_BASE}/api/analysis/${analysisId}/volume`;
}

/**
 * Parse a volume payload.
 *
 * Validates rather than trusts: a truncated body, an absent channel or a
 * dimension product that disagrees with the declared channel length all raise,
 * because a silently-wrong volume would render as plausible-looking anatomy that
 * is not the patient's.
 */
export function parseVolume(buffer: ArrayBuffer): LoadedVolume {
  if (buffer.byteLength < 4) {
    throw new VolumeError("Volume payload is too short to contain a header.", "TOO_SHORT");
  }

  const view = new DataView(buffer);
  const headerLength = view.getUint32(0, true);
  if (headerLength <= 0 || 4 + headerLength > buffer.byteLength) {
    throw new VolumeError(
      `Declared header length ${headerLength} does not fit in a ${buffer.byteLength}-byte payload.`,
      "BAD_HEADER",
    );
  }

  let header: VolumeHeader;
  try {
    header = JSON.parse(
      new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)),
    ) as VolumeHeader;
  } catch (cause) {
    throw new VolumeError(`Volume header is not valid JSON: ${cause}`, "BAD_HEADER");
  }

  if (header.format_version !== SUPPORTED_VOLUME_FORMAT) {
    throw new VolumeError(
      `Volume format version ${header.format_version} is not supported by this client (expects ${SUPPORTED_VOLUME_FORMAT}).`,
      "UNSUPPORTED_FORMAT",
    );
  }

  const { slices, rows, cols } = header.dimensions ?? ({} as VolumeHeader["dimensions"]);
  if (!(slices > 0 && rows > 0 && cols > 0)) {
    throw new VolumeError(
      `Volume dimensions are not usable: ${JSON.stringify(header.dimensions)}.`,
      "BAD_DIMENSIONS",
    );
  }
  const voxelCount = slices * rows * cols;

  const bodyStart = 4 + headerLength;
  const channel = (name: VolumeChannelSpec["name"]): Uint8Array => {
    const spec = header.channels?.find((c) => c.name === name);
    if (!spec) {
      throw new VolumeError(`Volume payload has no '${name}' channel.`, "MISSING_CHANNEL");
    }
    if (spec.length !== voxelCount) {
      throw new VolumeError(
        `Channel '${name}' declares ${spec.length} bytes but the dimensions imply ${voxelCount}.`,
        "BAD_DIMENSIONS",
      );
    }
    const start = bodyStart + spec.offset;
    if (start + spec.length > buffer.byteLength) {
      throw new VolumeError(
        `Channel '${name}' is truncated: needs bytes ${start}..${start + spec.length} of ${buffer.byteLength}.`,
        "TRUNCATED",
      );
    }
    return new Uint8Array(buffer, start, spec.length);
  };

  return {
    header,
    image: channel("image"),
    semantic: channel("semantic"),
    instance: channel("instance"),
    voxelCount,
  };
}

/** Fetch and parse one analysis's volume. One request, server-cached immutable. */
export async function fetchVolume(
  analysisId: string,
  signal?: AbortSignal,
): Promise<LoadedVolume> {
  const response = await fetch(volumeUrl(analysisId), { signal });
  if (!response.ok) {
    let code = `HTTP_${response.status}`;
    let message = `The volume could not be loaded (HTTP ${response.status}).`;
    try {
      const body = await response.json();
      if (body?.error?.code) {
        code = body.error.code;
        message = body.error.message ?? message;
      }
    } catch {
      // A non-JSON error body is not worth a second failure mode.
    }
    throw new VolumeError(message, code as VolumeError["code"]);
  }
  return parseVolume(await response.arrayBuffer());
}

/**
 * Build the scalar array the 3D renderer colours by.
 *
 * One byte per voxel encoding what to draw, so the renderer needs a single
 * transfer function rather than three overlapping volumes:
 *
 *   0                      hidden
 *   1..3                   semantic class (vertebra / disc / canal)
 *   FINDING_CODE           a disc the server marked finding-associated
 *   SELECTED_CODE          the disc currently selected
 *
 * The finding and selected codes come from the server's decision and the UI's
 * selection respectively; nothing about anatomy is inferred here.
 */
export const FINDING_CODE = 8;
export const SELECTED_CODE = 9;

export function buildLabelScalars(
  volume: LoadedVolume,
  options: {
    showSegmentation: boolean;
    showFindings: boolean;
    visibleClasses: number[];
    selectedDisc: number | null;
  },
): Uint8Array {
  const { semantic, instance, header, voxelCount } = volume;
  const { showSegmentation, showFindings, visibleClasses, selectedDisc } = options;
  const out = new Uint8Array(voxelCount);

  const classVisible = new Uint8Array(4);
  for (const id of visibleClasses) {
    if (id >= 1 && id <= 3) classVisible[id] = 1;
  }

  const offset = header.disc_instance_offset;
  const findingInstances = new Set(
    showFindings ? header.finding_discs.map((d) => offset + d) : [],
  );
  const selectedInstance = selectedDisc == null ? -1 : offset + selectedDisc;

  for (let i = 0; i < voxelCount; i += 1) {
    const label = semantic[i]!;
    const inst = instance[i]!;

    // Selection wins over finding, which wins over plain class, so the disc the
    // user is looking at is never hidden behind another cue.
    if (selectedInstance >= 0 && inst === selectedInstance) {
      out[i] = SELECTED_CODE;
      continue;
    }
    if (findingInstances.size > 0 && findingInstances.has(inst)) {
      out[i] = FINDING_CODE;
      continue;
    }
    if (showSegmentation && label >= 1 && label <= 3 && classVisible[label]) {
      out[i] = label;
    }
  }
  return out;
}
