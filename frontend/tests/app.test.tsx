/**
 * Frontend behaviour tests.
 *
 * Covers the flows the brief calls out: upload, progress state, result
 * rendering, disc selection, and error state. The API module is mocked so these
 * stay fast and deterministic; the real wiring is proven by the backend suite and
 * the end-to-end smoke test.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api";
import {
  health as healthFixture,
  result as resultFixture,
  statusFailed,
  statusProcessing,
} from "./fixtures";

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      health: vi.fn(),
      upload: vi.fn(),
      run: vi.fn(),
      status: vi.fn(),
      result: vi.fn(),
      history: vi.fn(),
      remove: vi.fn(),
      loadSample: vi.fn(),
      // The real URL builder, not a stub: what the viewer asks the server for is
      // part of the behaviour under test, and a stub would hide it.
      sliceUrl: actual.api.sliceUrl,
      downloadUrl: (id: string) => `/download/${id}`,
    },
  };
});

const { api } = await import("@/lib/api");
const { ProcessingView } = await import("@/components/ProcessingView");
const { ResultsView } = await import("@/components/ResultsView");
const { DiscList } = await import("@/components/DiscPanel");
const { MriViewer } = await import("@/components/MriViewer");
const NewAnalysisPage = (await import("@/app/new/page")).default;
const DashboardPage = (await import("@/app/page")).default;

/*
 * Every page that mounts reads `GET /api/health` first: the dashboard needs the
 * pipeline identity and the served metric table, the upload page needs
 * `sample_study_available`. Both call it as a promise, so the mock has to resolve
 * a real payload rather than `undefined`. The fixture mirrors the backend
 * response shape, which keeps these tests offline while still failing if the
 * `HealthResponse` contract drifts.
 */
beforeEach(() => {
  vi.mocked(api.health).mockResolvedValue(healthFixture);
});

/* -------------------------------------------------------------------------- */
/* Upload flow                                                                 */
/* -------------------------------------------------------------------------- */

describe("upload flow", () => {
  it("rejects an unsupported extension before contacting the API", async () => {
    render(<NewAnalysisPage />);
    const input = document.getElementById("mri-file") as HTMLInputElement;

    await userEvent.upload(
      input,
      new File(["notes"], "notes.txt", { type: "text/plain" }),
      // The input carries an `accept` filter; userEvent honours it and would
      // drop the file before the handler runs, which is not what is under test.
      { applyAccept: false },
    );

    expect(await screen.findByText(/not a supported MRI volume/i)).toBeInTheDocument();
    // No pointless round trip for a file we already know is wrong.
    expect(api.upload).not.toHaveBeenCalled();
  });

  it("uploads a volume and shows the server-reported study details", async () => {
    vi.mocked(api.upload).mockResolvedValue({
      analysis_id: "abc123abc123abcd",
      status: "queued",
      study: resultFixture.study,
      created_at: "2026-09-24T20:00:00Z",
      message: "Upload validated. Start the analysis to process this study.",
      is_sample: false,
    });

    render(<NewAnalysisPage />);
    const input = document.getElementById("mri-file") as HTMLInputElement;
    await userEvent.upload(
      input,
      new File(["volume-bytes"], "33_t2.mha", { type: "application/octet-stream" }),
    );

    expect(await screen.findByText("Validated")).toBeInTheDocument();
    expect(screen.getByText(/Ready to analyse/i)).toBeInTheDocument();
    // Volume facts must come from the response, not be guessed client-side.
    expect(screen.getByText("24")).toBeInTheDocument();
    expect(screen.getByText("242 × 305 px")).toBeInTheDocument();
    expect(screen.getByText(/filename heuristic/i)).toBeInTheDocument();
    // The demo entry point is gated on the server confirming the volume exists,
    // so it is only offered because health reported sample_study_available.
    expect(
      screen.getByRole("button", { name: /Load Sample Study/i }),
    ).toBeInTheDocument();
  });

  it("surfaces a backend rejection with its message and code", async () => {
    vi.mocked(api.upload).mockRejectedValue(
      new ApiError(
        "UNREADABLE_VOLUME",
        "Unable to read the MRI volume. The file may be corrupt.",
        400,
      ),
    );

    render(<NewAnalysisPage />);
    await userEvent.upload(
      document.getElementById("mri-file") as HTMLInputElement,
      new File(["junk"], "corrupt.mha", { type: "application/octet-stream" }),
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /Unable to read the MRI volume/i,
    );
    expect(screen.getByText("UNREADABLE_VOLUME")).toBeInTheDocument();
    expect(screen.getByText("Rejected")).toBeInTheDocument();
  });
});

/* -------------------------------------------------------------------------- */
/* Progress state                                                              */
/* -------------------------------------------------------------------------- */

describe("progress state", () => {
  it("shows the backend's exact progress, not a synthetic value", () => {
    render(<ProcessingView status={statusProcessing} />);

    expect(screen.getByText("60%")).toBeInTheDocument();
    const bar = screen.getByRole("progressbar", { name: /analysis progress/i });
    expect(bar).toHaveAttribute("aria-valuenow", "60");
    expect(screen.getByText("Disc indexing")).toBeInTheDocument();
  });

  it("marks completed stages and the active stage distinctly", () => {
    render(<ProcessingView status={statusProcessing} />);

    // Preprocessing is in stages_completed; Disc Indexing is current.
    const preprocessing = screen.getByText("Preprocessing").closest("li")!;
    expect(within(preprocessing).getByText("done")).toBeInTheDocument();

    const indexing = screen.getByText("Disc Indexing").closest("li")!;
    expect(within(indexing).getByText("in progress")).toBeInTheDocument();
  });

  it("renders a failure with the backend's error and a retry", () => {
    const onRetry = vi.fn();
    render(<ProcessingView status={statusFailed} onRetry={onRetry} />);

    expect(screen.getByText("Analysis failed")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(
      /Analysis failed during segmentation/i,
    );
    expect(screen.getByText("ANALYSIS_FAILED")).toBeInTheDocument();
  });
});

/* -------------------------------------------------------------------------- */
/* Result rendering                                                            */
/* -------------------------------------------------------------------------- */

describe("result rendering", () => {
  it("renders the study, discs and template findings", () => {
    render(<ResultsView result={resultFixture} />);

    const heading = screen.getByRole("heading", { level: 1 });
    expect(heading).toHaveTextContent("33_t2.mha");

    // The study meta line is a list of discrete facts, not one interpolated
    // string, so each is asserted on its own node within the header block. The
    // pipeline version sits beside the study facts: a result always carries the
    // identity of the pipeline that produced it.
    const header = within(heading.parentElement!);
    expect(header.getByText("2 discs")).toBeInTheDocument();
    expect(header.getByText("24 slices")).toBeInTheDocument();
    expect(header.getByText(resultFixture.pipeline.version)).toBeInTheDocument();
    expect(header.getByText("SPIDER-Lumbar-v1")).toBeInTheDocument();

    // Disc rows, with the Pfirrmann grade shown per disc.
    expect(screen.getByTitle("Model-estimated Pfirrmann grade 5")).toBeInTheDocument();
    expect(screen.getByTitle("Model-estimated Pfirrmann grade 2")).toBeInTheDocument();

    // Deterministic summary text from the backend.
    expect(
      screen.getByText(/No anatomical level name is asserted/i),
    ).toBeInTheDocument();
    expect(screen.getAllByText(/No postoperative assessment is provided/i).length).toBeGreaterThan(0);
  });

  it("states why unsupported findings are absent instead of implying a negative", () => {
    render(<ResultsView result={resultFixture} />);

    expect(screen.getAllByText(/Not reported by design/i).length).toBeGreaterThan(0);
    expect(
      screen.getByText(/nominal Modic type was never modelled/i),
    ).toBeInTheDocument();
    expect(screen.getAllByText(/Too few positive cases to validate/i).length).toBeGreaterThan(0);
  });

  it("always shows the research disclaimer", () => {
    render(<ResultsView result={resultFixture} />);
    expect(
      screen.getByText(/require expert radiological\s+review/i),
    ).toBeInTheDocument();
  });
});

/* -------------------------------------------------------------------------- */
/* Disc selection                                                              */
/* -------------------------------------------------------------------------- */

describe("disc selection", () => {
  it("calls back with the chosen disc", async () => {
    const onSelect = vi.fn();
    render(
      <DiscList
        discs={resultFixture.discs}
        selectedIndex={null}
        onSelect={onSelect}
      />,
    );

    await userEvent.click(screen.getAllByRole("button")[0]!);
    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({ index: 1 }),
    );
  });

  /**
   * Sprint 7 renders each disc value as a structured row of label, value and a
   * provenance chip, and the backend decides the formatting. Locating a row by
   * *both* its label and its provenance is deliberate: it makes the assertion
   * fail if a measurement is ever relabelled as an estimate, or vice versa.
   *
   * The detail panel exists twice in jsdom (desktop aside and mobile sheet), so
   * the first matching row is used.
   */
  const structuredRow = (label: string, provenance: string): HTMLElement => {
    const row = screen
      .getAllByText(label)
      .map((node) => node.parentElement)
      .find(
        (candidate): candidate is HTMLElement =>
          !!candidate && !!within(candidate).queryByText(provenance),
      );
    expect(row, `no "${label}" row tagged "${provenance}"`).toBeTruthy();
    return row!;
  };

  it("opens a detail panel with measurements and validated provenance", async () => {
    render(<ResultsView result={resultFixture} />);

    await userEvent.click(screen.getAllByRole("button", { name: /Disc/ })[0]!);

    expect((await screen.findAllByText("Disc 1")).length).toBeGreaterThan(0);

    // Measurements: the fixture's central height is 5.9 mm and its area 231 mm²,
    // rendered as the backend formatted them and tagged as measured off the
    // segmentation rather than predicted.
    const height = structuredRow("Disc height (central)", "Segmentation-derived");
    expect(within(height).getByText("5.90 mm")).toBeInTheDocument();
    expect(resultFixture.discs[0]!.measurements.height_mm_central).toBe(5.9);

    const area = structuredRow("Disc area", "Segmentation-derived");
    expect(within(area).getByText("231.0 mm²")).toBeInTheDocument();

    // A model output is labelled as an estimate, with qualitative evidence
    // strength and never a numeric "confidence".
    const grade = structuredRow("Pfirrmann grade", "Model-estimated");
    expect(within(grade).getByText("5")).toBeInTheDocument();
    expect(within(grade).getByText("Moderate evidence")).toBeInTheDocument();

    // An unmodelled target reads as unavailable, not as a negative finding.
    const modic = structuredRow("Modic type", "Not modelled");
    expect(within(modic).getByText("unavailable")).toBeInTheDocument();
    expect(
      screen.getAllByText(/nominal Modic type was never modelled/i).length,
    ).toBeGreaterThan(0);

    // The estimate must carry how well it validated.
    expect(screen.getAllByText(/Validated test pr auc 0.8779/i).length).toBeGreaterThan(0);
    // Identity support must be described as tracking, not clinical confidence.
    expect(screen.getAllByText(/not a clinical confidence/i).length).toBeGreaterThan(0);
  });

  it("shows an empty state when no disc was identified", () => {
    render(<DiscList discs={[]} selectedIndex={null} onSelect={vi.fn()} />);
    expect(screen.getByText("No disc was identified")).toBeInTheDocument();
  });
});

/* -------------------------------------------------------------------------- */
/* Viewer                                                                      */
/* -------------------------------------------------------------------------- */

describe("MRI viewer", () => {
  it("navigates slices and reports the position", async () => {
    render(<MriViewer analysisId="abc123abc123abcd" sliceCount={24} />);

    expect(screen.getByText("Slice 13 / 24")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /next slice/i }));
    expect(screen.getByText("Slice 14 / 24")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /previous slice/i }));
    expect(screen.getByText("Slice 13 / 24")).toBeInTheDocument();
  });

  it("supports keyboard navigation", async () => {
    render(<MriViewer analysisId="abc123abc123abcd" sliceCount={24} />);
    const canvas = screen.getByRole("group", { name: /MRI slice viewer/i });

    canvas.focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(screen.getByText("Slice 14 / 24")).toBeInTheDocument();

    await userEvent.keyboard("{Home}");
    expect(screen.getByText("Slice 1 / 24")).toBeInTheDocument();
  });

  it("exposes class visibility toggles", async () => {
    render(<MriViewer analysisId="abc123abc123abcd" sliceCount={24} />);

    const discs = screen.getByRole("checkbox", { name: /Intervertebral discs/i });
    expect(discs).toBeChecked();
    await userEvent.click(discs);
    expect(discs).not.toBeChecked();
  });

  it("does not offer a findings overlay when none was supplied", () => {
    render(<MriViewer analysisId="abc123abc123abcd" sliceCount={24} />);
    expect(screen.queryByText("Findings Overlay")).not.toBeInTheDocument();
  });
});

/* -------------------------------------------------------------------------- */
/* Findings overlay                                                            */
/* -------------------------------------------------------------------------- */

describe("findings overlay", () => {
  const overlay = resultFixture.finding_overlay;
  const viewer = (props: Partial<React.ComponentProps<typeof MriViewer>> = {}) =>
    render(
      <MriViewer
        analysisId="abc123abc123abcd"
        sliceCount={24}
        findingOverlay={overlay}
        {...props}
      />,
    );

  const sliceImage = () =>
    screen.getByAltText(/Lumbar MRI slice/i) as HTMLImageElement;

  it("is not offered when the server marked no disc", () => {
    // A control that could only ever produce an unchanged image would mislead.
    viewer({ findingOverlay: { ...overlay, disc_indices: [], discs: [] } });
    expect(screen.queryByText("Findings Overlay")).not.toBeInTheDocument();
  });

  it("is offered with the count of discs the server marked", () => {
    viewer();
    expect(screen.getByText("Findings Overlay")).toBeInTheDocument();
    expect(screen.getByText("1 disc marked")).toBeInTheDocument();
  });

  it("asks the server for the overlay only once switched on", async () => {
    viewer();

    // Off: the request is exactly the pre-existing one.
    expect(sliceImage().src).not.toContain("highlight=findings");

    await userEvent.click(screen.getByRole("checkbox", { name: /^Off$/ }));

    // On: the rendering is requested from the backend, which owns the rule.
    expect(sliceImage().src).toContain("highlight=findings");
    expect(sliceImage().src).toContain("finding_opacity=");
  });

  it("returns to the original rendering when switched off again", async () => {
    viewer();
    const before = sliceImage().src;

    const toggle = screen.getByRole("checkbox", { name: /^Off$/ });
    await userEvent.click(toggle);
    expect(sliceImage().src).not.toEqual(before);

    await userEvent.click(screen.getByRole("checkbox", { name: /^On$/ }));
    expect(sliceImage().src).toEqual(before);
  });

  it("keeps the selected disc in the request alongside the overlay", async () => {
    viewer({ highlightDisc: 1 });
    await userEvent.click(screen.getByRole("checkbox", { name: /^Off$/ }));

    const { src } = sliceImage();
    expect(src).toContain("highlight_disc=1");
    expect(src).toContain("highlight=findings");
  });

  it("states that the red is a finding-associated region, not damaged tissue", async () => {
    viewer();
    await userEvent.click(screen.getByRole("checkbox", { name: /^Off$/ }));

    // The wording is served by the backend and rendered verbatim.
    expect(
      screen.getByText(/not a diagnosis of damaged tissue/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/not from a pixel-level pathology model/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Marked from the validated disc segmentation/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/without a supported positive finding are left unmarked/i),
    ).toBeInTheDocument();
  });

  it("shows a legend distinguishing the finding layer from the anatomy", async () => {
    viewer();
    await userEvent.click(screen.getByRole("checkbox", { name: /^Off$/ }));

    const legend = within(screen.getByRole("list", { name: /Overlay legend/i }));
    expect(legend.getByText("Model-estimated finding")).toBeInTheDocument();
    expect(legend.getByText("Anatomical segmentation")).toBeInTheDocument();
    expect(legend.getByText("MRI")).toBeInTheDocument();
  });

  it("exposes an opacity control for the finding layer", async () => {
    viewer();
    await userEvent.click(screen.getByRole("checkbox", { name: /^Off$/ }));

    const slider = screen.getByLabelText("Finding") as HTMLInputElement;
    expect(slider).toHaveValue("0.5");
    // The segmentation opacity control is untouched by the new one.
    expect(screen.getByLabelText("Opacity")).toHaveValue("0.45");
  });

  it("marks only the discs the server listed, in the disc list", () => {
    render(
      <DiscList
        discs={resultFixture.discs}
        selectedIndex={null}
        onSelect={vi.fn()}
        markedIndices={overlay.disc_indices}
      />,
    );
    // Disc 1 carries a positive narrowing finding; disc 2's is false.
    expect(
      screen.getAllByTitle(/marked by the findings overlay/i),
    ).toHaveLength(overlay.disc_indices.length);
  });

  it("marks nothing when the server marked nothing", () => {
    render(
      <DiscList
        discs={resultFixture.discs}
        selectedIndex={null}
        onSelect={vi.fn()}
        markedIndices={[]}
      />,
    );
    expect(
      screen.queryByTitle(/marked by the findings overlay/i),
    ).not.toBeInTheDocument();
  });

  it("never marks a disc when the result carries no overlay decision", () => {
    render(
      <DiscList
        discs={resultFixture.discs}
        selectedIndex={null}
        onSelect={vi.fn()}
      />,
    );
    expect(
      screen.queryByTitle(/marked by the findings overlay/i),
    ).not.toBeInTheDocument();
  });
});

/* -------------------------------------------------------------------------- */
/* Error and empty states                                                      */
/* -------------------------------------------------------------------------- */

describe("dashboard states", () => {
  it("shows an empty state when there are no analyses", async () => {
    vi.mocked(api.history).mockResolvedValue({ items: [], total: 0, counts: {} });
    render(<DashboardPage />);

    expect(await screen.findByText("No MRI analyses yet")).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /Start New Analysis/i }),
    ).toBeInTheDocument();

    // An empty history must not empty the pipeline panel: identity and the
    // research figures come from health, and every figure is served rather than
    // hard-coded in the UI.
    expect(
      await screen.findByText(healthFixture.pipeline.version),
    ).toBeInTheDocument();
    const metric = healthFixture.metric_table[0]!;
    expect(screen.getByText(metric.label)).toBeInTheDocument();
    expect(screen.getByText(metric.value)).toBeInTheDocument();
    // Those figures describe the pipeline on a held-out split, not this user's
    // data, and the UI has to say so.
    expect(
      screen.getByText(/They describe the pipeline, not this study/i),
    ).toBeInTheDocument();
  });

  it("shows an actionable error when the API is unreachable", async () => {
    vi.mocked(api.history).mockRejectedValue(
      new ApiError("NETWORK_ERROR", "Cannot reach the analysis API.", 0),
    );
    render(<DashboardPage />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /Cannot reach the analysis API/i,
    );
    expect(screen.getByText("NETWORK_ERROR")).toBeInTheDocument();
  });

  it("never fabricates history rows", async () => {
    vi.mocked(api.history).mockResolvedValue({ items: [], total: 0, counts: {} });
    render(<DashboardPage />);

    await waitFor(() => screen.getByText("No MRI analyses yet"));
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });
});
