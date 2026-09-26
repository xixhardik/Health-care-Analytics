/**
 * Sprint 8 commit 3 - Recovery Tracker.
 *
 * The functional coverage (timeline, selection, comparison, summary) matters, but
 * the assertions that matter most are the ones about language. This feature is one
 * careless noun away from claiming that SPIDER contains postoperative follow-up,
 * so several tests read the rendered DOM and assert that specific phrasings are
 * absent.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  COMPARISON_CAVEAT,
  COMPARISON_FIELDS,
  CROSS_STUDY_DISCLAIMER,
  IMAGING_STAGE_IDS,
  LONGITUDINAL_UNAVAILABLE,
  SURGICAL_CONSIDERATION_CAVEAT,
  TIMELINE,
  computeStageMetrics,
  formatMetric,
  metricDelta,
  surgicalConsideration,
  workflowSummary,
} from "@/lib/recovery";
import { demoCase, result as resultFixture } from "./fixtures";

/* ------------------------------------------------ mocks */

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), back: vi.fn() }),
  usePathname: () => "/recovery",
}));

// The 3D viewer is exercised by viewer3d.test.tsx; here it is a stub so these
// tests are about the tracker, not about WebGL.
vi.mock("@/components/Mri3DViewer", () => ({
  Mri3DViewer: ({
    analysisId,
    highlightDisc,
  }: {
    analysisId: string;
    highlightDisc?: number | null;
  }) => (
    <div data-testid="viewer-3d" data-analysis={analysisId}>
      {highlightDisc != null ? `3D disc ${highlightDisc}` : "3D no selection"}
    </div>
  ),
}));

const { RecoveryTracker, LongitudinalUnavailable } = await import(
  "@/components/RecoveryTracker"
);

/* ------------------------------------------------ stage fixtures */

/** Four stage results with the real measured arc from the four demo studies. */
function stageResult(options: {
  analysisId: string;
  discCount: number;
  meanHeight: number;
  findingDiscs: number[];
  grades: (number | null)[];
}) {
  const discs = options.grades.map((grade, i) => ({
    ...resultFixture.discs[0]!,
    index: i + 1,
    pfirrmann_grade: grade,
    narrowing: i === 0,
    bulging: i < 2,
    herniation: false,
    modic_any: i === 0,
    measurements: {
      ...resultFixture.discs[0]!.measurements,
      canal_width_at_disc_mm: 12 + i * 0.5,
      area_mm2: 200 + i * 10,
    },
  }));
  return {
    ...resultFixture,
    analysis_id: options.analysisId,
    discs,
    summary: {
      ...resultFixture.summary,
      disc_count: options.discCount,
      mean_disc_height_mm: options.meanHeight,
      discs_with_findings_count: options.findingDiscs.length,
    },
    finding_overlay: {
      ...resultFixture.finding_overlay,
      disc_indices: options.findingDiscs,
      discs: options.findingDiscs.map((index) => ({ index, findings: ["Disc narrowing"] })),
    },
  };
}

const READY_STAGES = {
  pre_surgery: {
    status: "ready" as const,
    analysisId: "a".repeat(16),
    result: stageResult({
      analysisId: "a".repeat(16), discCount: 9, meanHeight: 4.64,
      findingDiscs: [1, 2, 3, 4, 5, 6, 7, 8], grades: [5, 5, 5, 5, 5, 5, 3, 5, 5],
    }),
  },
  post_surgery: {
    status: "ready" as const,
    analysisId: "b".repeat(16),
    result: stageResult({
      analysisId: "b".repeat(16), discCount: 7, meanHeight: 5.8,
      findingDiscs: [1, 2, 3, 4, 5, 6, 7], grades: [5, 4, 4, 5, 4, 5, 4],
    }),
  },
  month_3: {
    status: "ready" as const,
    analysisId: "c".repeat(16),
    result: stageResult({
      analysisId: "c".repeat(16), discCount: 7, meanHeight: 7.79,
      findingDiscs: [1, 2], grades: [5, 3, 1, 1, 1, 1, 1],
    }),
  },
  month_6: {
    status: "ready" as const,
    analysisId: "d".repeat(16),
    result: stageResult({
      analysisId: "d".repeat(16), discCount: 6, meanHeight: 8.49,
      findingDiscs: [1], grades: [3, 1, 2, 1, 1, 1],
    }),
  },
};

function renderTracker(
  overrides: Partial<React.ComponentProps<typeof RecoveryTracker>> = {},
) {
  const onSelectNode = vi.fn();
  const utils = render(
    <RecoveryTracker
      demoCase={demoCase}
      stages={READY_STAGES}
      activeNodeId="diagnosis"
      onSelectNode={onSelectNode}
      {...overrides}
    />,
  );
  return { ...utils, onSelectNode };
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 404, json: async () => ({}) })) as unknown as typeof fetch);
});

/* -------------------------------------------------------------------------- */
/* Pure logic                                                                  */
/* -------------------------------------------------------------------------- */

describe("recovery metrics", () => {
  it("reads every metric from the pipeline result", () => {
    const metrics = computeStageMetrics(READY_STAGES.pre_surgery.result);

    expect(metrics.discCount).toBe(9);
    expect(metrics.meanDiscHeightMm).toBeCloseTo(4.64);
    expect(metrics.findingDiscCount).toBe(8);
    expect(metrics.maxPfirrmannGrade).toBe(5);
    expect(metrics.narrowingCount).toBe(1);
    expect(metrics.meanCanalWidthMm).not.toBeNull();
  });

  it("returns null rather than zero for an unavailable measurement", () => {
    const stripped = {
      ...READY_STAGES.month_6.result,
      summary: { ...READY_STAGES.month_6.result.summary, mean_disc_height_mm: null },
      discs: READY_STAGES.month_6.result.discs.map((d) => ({
        ...d,
        pfirrmann_grade: null,
        measurements: { ...d.measurements, canal_width_at_disc_mm: null },
      })),
    };
    const metrics = computeStageMetrics(stripped);

    expect(metrics.meanDiscHeightMm).toBeNull();
    expect(metrics.maxPfirrmannGrade).toBeNull();
    expect(metrics.meanCanalWidthMm).toBeNull();
    // A missing measurement is not a measurement of zero.
    expect(metrics.meanDiscHeightMm).not.toBe(0);
  });

  it("counts only findings reported true, never nulls", () => {
    const nulls = {
      ...READY_STAGES.month_6.result,
      discs: READY_STAGES.month_6.result.discs.map((d) => ({
        ...d, narrowing: null, bulging: null, herniation: null, modic_any: null,
      })),
    };
    const metrics = computeStageMetrics(nulls);
    expect(metrics.narrowingCount).toBe(0);
    expect(metrics.bulgingCount).toBe(0);
  });

  it("formats an unavailable value as text, not as a number", () => {
    const field = COMPARISON_FIELDS.find((f) => f.key === "meanDiscHeightMm")!;
    expect(formatMetric(null, field)).toBe("unavailable");
    expect(formatMetric(4.64, field)).toBe("4.64 mm");
  });

  it("describes a difference without calling it a change", () => {
    const field = COMPARISON_FIELDS.find((f) => f.key === "meanDiscHeightMm")!;
    expect(metricDelta(4.64, 5.8, field).text).toBe("+1.16 mm");
    expect(metricDelta(5.8, 4.64, field).text).toBe("−1.16 mm");
    expect(metricDelta(4.64, 4.64, field).text).toBe("no difference");
    expect(metricDelta(null, 4.64, field).text).toBe("not comparable");
    expect(metricDelta(4.64, null, field).direction).toBe("unknown");
  });

  it("produces no composite recovery score of any kind", () => {
    const metrics = computeStageMetrics(READY_STAGES.pre_surgery.result);
    for (const key of Object.keys(metrics)) {
      expect(key.toLowerCase()).not.toContain("recovery");
      expect(key.toLowerCase()).not.toContain("percent");
      expect(key.toLowerCase()).not.toContain("score");
    }
  });
});

describe("timeline definition", () => {
  it("has the five requested positions in order", () => {
    expect(TIMELINE.map((n) => n.label)).toEqual([
      "Diagnosis",
      "Surgical Evaluation",
      "Post-Surgery",
      "3-Month Recovery",
      "6-Month Recovery",
    ]);
  });

  it("marks Surgical Evaluation as having no imaging of its own", () => {
    const node = TIMELINE.find((n) => n.id === "surgical_evaluation")!;
    expect(node.hasOwnImaging).toBe(false);
    // It reuses the baseline study rather than inventing a scan.
    expect(node.stageId).toBe("pre_surgery");
  });

  it("lists four imaging stages, without duplicates", () => {
    expect(IMAGING_STAGE_IDS).toEqual([
      "pre_surgery", "post_surgery", "month_3", "month_6",
    ]);
  });
});

describe("surgical evaluation consideration", () => {
  it("prompts evaluation and lists the findings behind it", () => {
    const consideration = surgicalConsideration(READY_STAGES.pre_surgery.result);
    expect(consideration.warranted).toBe(true);
    expect(consideration.wording).toBe("Imaging findings may warrant surgical evaluation.");
    expect(consideration.supportingFindings.length).toBeGreaterThan(0);
  });

  it("makes no recommendation and invents no indication", () => {
    const { wording, caveat } = surgicalConsideration(READY_STAGES.pre_surgery.result);
    const text = `${wording} ${caveat}`.toLowerCase();
    for (const forbidden of [
      "we recommend", "you should", "requires surgery", "surgery is needed",
      "must undergo", "is indicated",
    ]) {
      expect(text).not.toContain(forbidden);
    }
    expect(caveat).toContain("not a clinical recommendation");
  });

  it("does not prompt when nothing positive was reported", () => {
    const quiet = {
      ...READY_STAGES.month_6.result,
      summary: { ...READY_STAGES.month_6.result.summary, findings: [] },
    };
    expect(surgicalConsideration(quiet).warranted).toBe(false);
  });
});

describe("workflow summary", () => {
  it("states that true longitudinal follow-up is absent", () => {
    const rows = workflowSummary({
      stageCount: 5, distinctStudyCount: 4, sourceDataset: "SPIDER",
    });
    const followUp = rows.find((r) => r.label === "True longitudinal follow-up")!;
    expect(followUp.state).toBe("no");
    expect(followUp.value).toMatch(/not available/i);
    expect(rows.find((r) => r.label === "Workflow type")!.value).toMatch(/simulated/i);
  });
});

/* -------------------------------------------------------------------------- */
/* Rendering                                                                   */
/* -------------------------------------------------------------------------- */

describe("recovery tracker rendering", () => {
  it("renders all five timeline stages as selectable steps", () => {
    renderTracker();
    // Scoped to the timeline region: the disc list and summary are lists too.
    const timeline = within(
      screen.getByRole("region", { name: /Demonstration timeline/i }),
    );

    for (const node of TIMELINE) {
      expect(
        timeline.getByRole("button", { name: new RegExp(node.label, "i") }),
      ).toBeInTheDocument();
    }
  });

  it("marks the active stage and reports each stage's real counts", () => {
    renderTracker();

    const active = screen.getByRole("button", { name: /Diagnosis/i });
    expect(active).toHaveAttribute("aria-current", "step");
    // Pre-surgery: 9 discs, 8 finding-associated, 4.64 mm.
    expect(within(active).getByText("9")).toBeInTheDocument();
    expect(within(active).getByText("8")).toBeInTheDocument();
    expect(within(active).getByText("4.64 mm")).toBeInTheDocument();
  });

  it("shows a non-identifying study reference per stage", () => {
    renderTracker();
    expect(screen.getAllByText("Demonstration study A").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Demonstration study D").length).toBeGreaterThan(0);
  });

  it("reports selecting a stage to its owner", async () => {
    const { onSelectNode } = renderTracker();
    await userEvent.click(screen.getByRole("button", { name: /6-Month Recovery/i }));
    expect(onSelectNode).toHaveBeenCalledWith("month_6");
  });

  it("labels Surgical Evaluation as an assessment of baseline imaging", () => {
    renderTracker({ activeNodeId: "surgical_evaluation" });
    expect(screen.getByText(/Assessment of baseline imaging/i)).toBeInTheDocument();
    expect(
      screen.getAllByText(/not a separate scan/i).length,
    ).toBeGreaterThan(0);
  });

  it("keeps the disclaimer permanently visible on every stage", () => {
    for (const nodeId of TIMELINE.map((n) => n.id)) {
      const { unmount } = renderTracker({ activeNodeId: nodeId });
      expect(screen.getByText(CROSS_STUDY_DISCLAIMER)).toBeInTheDocument();
      expect(screen.getByText(/Simulated Longitudinal Demo/)).toBeInTheDocument();
      unmount();
    }
  });
});

describe("comparison panel", () => {
  it("captions itself as a cross-study trend, not recovery", () => {
    renderTracker();
    expect(screen.getAllByText(COMPARISON_CAVEAT).length).toBeGreaterThan(0);
  });

  it("shows one column per imaging stage", () => {
    renderTracker();
    const table = screen.getByRole("table");
    for (const label of [
      "Pre-Surgery", "Post-Surgery", "3-Month Recovery", "6-Month Recovery",
    ]) {
      expect(
        within(table).getByRole("columnheader", { name: new RegExp(label, "i") }),
      ).toBeInTheDocument();
    }
  });

  it("renders the real measured disc-height arc with differences", () => {
    renderTracker();
    const table = screen.getByRole("table");

    for (const value of ["4.64 mm", "5.80 mm", "7.79 mm", "8.49 mm"]) {
      expect(within(table).getByText(value)).toBeInTheDocument();
    }
    // Differences, labelled as such; the first column is the baseline.
    expect(within(table).getByText("+1.16 mm")).toBeInTheDocument();
    expect(within(table).getAllByText("baseline").length).toBeGreaterThan(0);
  });

  it("shows unavailable metrics as unavailable rather than zero", () => {
    const stages = {
      ...READY_STAGES,
      month_6: {
        ...READY_STAGES.month_6,
        result: {
          ...READY_STAGES.month_6.result,
          summary: {
            ...READY_STAGES.month_6.result.summary,
            mean_disc_height_mm: null,
          },
        },
      },
    };
    renderTracker({ stages });
    expect(within(screen.getByRole("table")).getAllByText("unavailable").length)
      .toBeGreaterThan(0);
  });

  it("marks a stage that has not been analysed", () => {
    renderTracker({
      stages: { ...READY_STAGES, month_6: { status: "pending" } },
    });
    expect(screen.getAllByText(/not analysed/i).length).toBeGreaterThan(0);
  });

  it("never presents a recovery percentage as a value", () => {
    renderTracker();

    // No percentage figure anywhere in the comparison table: every field is a
    // count or a millimetre measurement.
    const table = screen.getByRole("table");
    expect(table.textContent ?? "").not.toMatch(/\d\s*%/);

    // And nothing anywhere pairs a number with recovery language.
    const text = document.body.textContent ?? "";
    expect(text).not.toMatch(/\d+\s*%\s*(recovered|recovery|improvement)/i);
    expect(text).not.toMatch(/(recovery|improvement)\s*(score|index)\s*[:=]?\s*\d/i);
  });

  it("explicitly disavows deriving a recovery percentage", () => {
    renderTracker();
    expect(
      screen.getByText(/no recovery percentage is derived from it/i),
    ).toBeInTheDocument();
  });
});

describe("scientific language", () => {
  it("never describes the stages with forbidden clinical wording", () => {
    for (const nodeId of TIMELINE.map((n) => n.id)) {
      const { unmount } = renderTracker({ activeNodeId: nodeId });
      const text = (document.body.textContent ?? "").toLowerCase();
      for (const forbidden of [
        "actual postoperative scan",
        "actual recovery scan",
        "longitudinal follow-up scan",
        "same-patient follow-up",
        "confirmed recovery",
        "successful surgery",
      ]) {
        expect(text, `"${forbidden}" appeared on ${nodeId}`).not.toContain(forbidden);
      }
      unmount();
    }
  });

  it("uses the approved vocabulary instead", () => {
    renderTracker();
    const text = document.body.textContent ?? "";
    expect(text).toMatch(/Simulated Longitudinal Demo/);
    expect(text).toMatch(/Demonstration study/);
    expect(text).toMatch(/Research prototype/i);
  });

  it("carries the surgical-evaluation caveat wherever the prompt appears", () => {
    renderTracker();
    expect(screen.getByText(SURGICAL_CONSIDERATION_CAVEAT)).toBeInTheDocument();
  });
});

describe("final workflow summary", () => {
  it("shows stage counts, capability rows and the follow-up limitation", () => {
    renderTracker();

    expect(screen.getByText("Demonstration stages")).toBeInTheDocument();
    expect(screen.getByText("4 distinct SPIDER studies")).toBeInTheDocument();
    expect(screen.getByText("2D MRI")).toBeInTheDocument();
    expect(screen.getByText("3D MRI")).toBeInTheDocument();
    expect(screen.getByText("Segmentation")).toBeInTheDocument();
    expect(screen.getByText("Finding analysis")).toBeInTheDocument();
    expect(screen.getByText("True longitudinal follow-up")).toBeInTheDocument();
    expect(screen.getByText(/Not available in this dataset/i)).toBeInTheDocument();
  });
});

describe("2D / 3D stage synchronisation", () => {
  it("drives the 3D viewer with the active stage's analysis", async () => {
    renderTracker();
    const viewer = await screen.findByTestId("viewer-3d");
    expect(viewer).toHaveAttribute("data-analysis", "a".repeat(16));
  });

  it("loads a different analysis when the stage changes", async () => {
    const { rerender } = renderTracker();
    expect(await screen.findByTestId("viewer-3d")).toHaveAttribute(
      "data-analysis", "a".repeat(16),
    );

    rerender(
      <RecoveryTracker
        demoCase={demoCase}
        stages={READY_STAGES}
        activeNodeId="month_6"
        onSelectNode={vi.fn()}
      />,
    );

    expect(await screen.findByTestId("viewer-3d")).toHaveAttribute(
      "data-analysis", "d".repeat(16),
    );
  });

  it("propagates a selected disc into the 3D viewer", async () => {
    renderTracker();
    await userEvent.click(screen.getAllByRole("button", { name: /^Disc/ })[0]!);

    expect(await screen.findByText(/3D disc 1/)).toBeInTheDocument();
    // Announced in both the viewer toolbar and the disc panel header.
    expect(screen.getAllByText(/Disc 1 selected/i).length).toBeGreaterThan(0);
  });

  it("clears a selected disc that does not exist in the new stage", async () => {
    // Pre-surgery has 9 discs; month_6 has 6. Selecting disc 9 then switching
    // must not leave a dangling selection.
    const { rerender } = renderTracker();
    const discButtons = screen.getAllByRole("button", { name: /^Disc/ });
    await userEvent.click(discButtons[discButtons.length - 1]!);
    expect((await screen.findAllByText(/Disc 9 selected/i)).length)
      .toBeGreaterThan(0);

    rerender(
      <RecoveryTracker
        demoCase={demoCase}
        stages={READY_STAGES}
        activeNodeId="month_6"
        onSelectNode={vi.fn()}
      />,
    );

    await waitFor(() =>
      expect(screen.queryByText(/Disc 9 selected/i)).not.toBeInTheDocument(),
    );
    expect(await screen.findByText(/3D no selection/)).toBeInTheDocument();
  });

  it("offers both 2D and 3D views of the stage", async () => {
    renderTracker();
    const group = screen.getByRole("group", { name: /Viewer dimension/i });
    expect(within(group).getByRole("button", { name: /2D View/i })).toBeInTheDocument();

    await userEvent.click(within(group).getByRole("button", { name: /2D View/i }));
    expect(screen.getByAltText(/Lumbar MRI slice/i)).toBeInTheDocument();
  });

  it("reports a stage that failed to analyse", () => {
    renderTracker({
      stages: {
        ...READY_STAGES,
        pre_surgery: { status: "failed", error: "Source study missing." },
      },
    });
    expect(screen.getByText(/Stage unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/Source study missing\./)).toBeInTheDocument();
  });
});

describe("real single-study state", () => {
  it("states that longitudinal follow-up is unavailable, inventing no stages", () => {
    render(<LongitudinalUnavailable analysisId="abc123abc123abcd" />);

    expect(screen.getByText(LONGITUDINAL_UNAVAILABLE)).toBeInTheDocument();
    expect(screen.getByText("Single-study analysis")).toBeInTheDocument();
    expect(screen.getByText(/no change over time is measured/i)).toBeInTheDocument();
    // No fabricated timeline.
    expect(screen.queryByText(/3-Month/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/6-Month/i)).not.toBeInTheDocument();
  });
});
