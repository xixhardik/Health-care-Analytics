/**
 * Sprint 8 - the volume client.
 *
 * Parsing is tested against payloads built in the backend's exact wire layout
 * (see `buildVolumePayload` in fixtures), so a format drift on either side shows
 * up here rather than as silently wrong anatomy in the 3D view.
 */

import { describe, expect, it } from "vitest";

import {
  FINDING_CODE,
  SELECTED_CODE,
  SUPPORTED_VOLUME_FORMAT,
  VolumeError,
  buildLabelScalars,
  parseVolume,
  volumeUrl,
} from "@/lib/volume";
import { buildVolumePayload } from "./fixtures";

describe("volume parsing", () => {
  it("parses the header the backend emits", () => {
    const volume = parseVolume(buildVolumePayload());

    expect(volume.header.format_version).toBe(SUPPORTED_VOLUME_FORMAT);
    expect(volume.header.axis_order).toEqual(["slice", "row", "col"]);
    expect(volume.header.pipeline_version).toBe("SPIDER-Lumbar-v1");
  });

  it("reports the correct dimensions and voxel count", () => {
    const volume = parseVolume(
      buildVolumePayload({ slices: 6, rows: 8, cols: 10 }),
    );

    expect(volume.header.dimensions).toEqual({ slices: 6, rows: 8, cols: 10 });
    expect(volume.voxelCount).toBe(6 * 8 * 10);
  });

  it("exposes all three channels at full length", () => {
    const volume = parseVolume(buildVolumePayload({ slices: 3, rows: 4, cols: 5 }));
    const expected = 3 * 4 * 5;

    expect(volume.image).toHaveLength(expected);
    expect(volume.semantic).toHaveLength(expected);
    expect(volume.instance).toHaveLength(expected);
  });

  it("keeps the semantic and instance arrays distinct", () => {
    // A parser that aliased one channel onto another would still "work" until the
    // 3D view coloured vertebrae as discs.
    const volume = parseVolume(buildVolumePayload());

    expect(Array.from(volume.semantic)).not.toEqual(Array.from(volume.image));
    expect(Array.from(volume.instance)).not.toEqual(Array.from(volume.semantic));
    expect(new Set(volume.semantic)).toEqual(new Set([0, 1, 2]));
  });

  it("carries the server's disc-instance convention rather than assuming it", () => {
    const volume = parseVolume(buildVolumePayload({ discInstanceOffset: 20 }));

    expect(volume.header.disc_instance_offset).toBe(20);
    expect(Array.from(volume.instance)).toContain(22);
  });

  it("carries the server's finding decision", () => {
    const volume = parseVolume(buildVolumePayload({ findingDiscs: [2, 5] }));
    expect(volume.header.finding_discs).toEqual([2, 5]);
  });

  it("builds a stable URL for an analysis", () => {
    expect(volumeUrl("abc123abc123abcd")).toContain(
      "/api/analysis/abc123abc123abcd/volume",
    );
  });
});

describe("volume parsing failures", () => {
  it("refuses a payload too short to hold a header", () => {
    expect(() => parseVolume(new ArrayBuffer(2))).toThrowError(
      /too short/i,
    );
  });

  it("refuses a header length that does not fit the payload", () => {
    const buffer = new ArrayBuffer(16);
    new DataView(buffer).setUint32(0, 9999, true);
    expect(() => parseVolume(buffer)).toThrowError(/does not fit/i);
  });

  it("refuses an unsupported format version instead of guessing", () => {
    let thrown: unknown;
    try {
      parseVolume(buildVolumePayload({ formatVersion: 99 }));
    } catch (cause) {
      thrown = cause;
    }
    expect(thrown).toBeInstanceOf(VolumeError);
    expect((thrown as VolumeError).code).toBe("UNSUPPORTED_FORMAT");
  });

  it("refuses a truncated body rather than rendering partial anatomy", () => {
    let thrown: unknown;
    try {
      parseVolume(buildVolumePayload({ truncate: 120 }));
    } catch (cause) {
      thrown = cause;
    }
    expect(thrown).toBeInstanceOf(VolumeError);
    expect(["TRUNCATED", "BAD_HEADER"]).toContain((thrown as VolumeError).code);
  });
});

describe("label scalars for the 3D transfer function", () => {
  const volume = parseVolume(buildVolumePayload({ findingDiscs: [2] }));

  it("paints segmentation classes when segmentation is shown", () => {
    const scalars = buildLabelScalars(volume, {
      showSegmentation: true,
      showFindings: false,
      visibleClasses: [1, 2, 3],
      selectedDisc: null,
    });
    expect(new Set(scalars)).toEqual(new Set([0, 1, 2]));
  });

  it("hides everything when segmentation and findings are both off", () => {
    const scalars = buildLabelScalars(volume, {
      showSegmentation: false,
      showFindings: false,
      visibleClasses: [1, 2, 3],
      selectedDisc: null,
    });
    expect(scalars.every((v) => v === 0)).toBe(true);
  });

  it("respects per-class visibility", () => {
    const scalars = buildLabelScalars(volume, {
      showSegmentation: true,
      showFindings: false,
      visibleClasses: [2],
      selectedDisc: null,
    });
    // Vertebra (1) must be gone, disc (2) must remain.
    expect(Array.from(scalars)).not.toContain(1);
    expect(Array.from(scalars)).toContain(2);
  });

  it("marks finding-associated discs with the finding code", () => {
    const scalars = buildLabelScalars(volume, {
      showSegmentation: true,
      showFindings: true,
      visibleClasses: [1, 2, 3],
      selectedDisc: null,
    });
    expect(Array.from(scalars)).toContain(FINDING_CODE);
  });

  it("does not mark findings when the layer is off", () => {
    const scalars = buildLabelScalars(volume, {
      showSegmentation: true,
      showFindings: false,
      visibleClasses: [1, 2, 3],
      selectedDisc: null,
    });
    expect(Array.from(scalars)).not.toContain(FINDING_CODE);
  });

  it("marks only the discs the server listed", () => {
    // Disc 2 exists in the fixture volume; disc 5 does not.
    const none = parseVolume(buildVolumePayload({ findingDiscs: [5] }));
    const scalars = buildLabelScalars(none, {
      showSegmentation: true,
      showFindings: true,
      visibleClasses: [1, 2, 3],
      selectedDisc: null,
    });
    expect(Array.from(scalars)).not.toContain(FINDING_CODE);
  });

  it("gives the selected disc precedence over the finding cue", () => {
    const scalars = buildLabelScalars(volume, {
      showSegmentation: true,
      showFindings: true,
      visibleClasses: [1, 2, 3],
      selectedDisc: 2,
    });
    expect(Array.from(scalars)).toContain(SELECTED_CODE);
    // Disc 2 was the only finding disc, and selection now owns those voxels.
    expect(Array.from(scalars)).not.toContain(FINDING_CODE);
  });

  it("selects a disc even when segmentation is hidden", () => {
    const scalars = buildLabelScalars(volume, {
      showSegmentation: false,
      showFindings: false,
      visibleClasses: [],
      selectedDisc: 2,
    });
    expect(Array.from(scalars)).toContain(SELECTED_CODE);
  });

  it("returns one byte per voxel", () => {
    const scalars = buildLabelScalars(volume, {
      showSegmentation: true,
      showFindings: true,
      visibleClasses: [1, 2, 3],
      selectedDisc: null,
    });
    expect(scalars).toHaveLength(volume.voxelCount);
  });
});
