/**
 * Sprint 8 - the 3D viewer component, and 2D/3D synchronisation.
 *
 * jsdom has no WebGL, so VTK.js is mocked module-by-module. What these tests can
 * prove is the wiring: that the component fetches the volume once, builds label
 * scalars from the *server's* finding decision and the shared selected disc, and
 * releases every VTK object on unmount. What they cannot prove is that pixels
 * appear on a GPU - that needs a browser, and is verified separately.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { buildVolumePayload, demoCase, demoStageResult, result as resultFixture } from "./fixtures";

/* -------------------------------------------------------------------------- */
/* VTK.js mock                                                                */
/* -------------------------------------------------------------------------- */

/** Counts construction and deletion so a leak on unmount is detectable. */
const vtkLedger = {
  created: [] as string[],
  deleted: [] as string[],
  renders: 0,
  resets: 0,
  scalarData: [] as Uint8Array[],
  reset() {
    this.created = [];
    this.deleted = [];
    this.renders = 0;
    this.resets = 0;
    this.scalarData = [];
  },
};

function fakeObject(kind: string, extra: Record<string, unknown> = {}) {
  vtkLedger.created.push(kind);
  return {
    delete: vi.fn(() => vtkLedger.deleted.push(kind)),
    ...extra,
  };
}

const newInstance = (kind: string, extra: Record<string, unknown> = {}) =>
  vi.fn(() => fakeObject(kind, extra));

vi.mock("@kitware/vtk.js/Rendering/Core/Volume", () => ({
  default: {
    newInstance: vi.fn(() =>
      fakeObject("Volume", {
        setMapper: vi.fn(),
        setVisibility: vi.fn(),
        getProperty: vi.fn(() => ({
          setRGBTransferFunction: vi.fn(),
          setScalarOpacity: vi.fn(),
          setInterpolationTypeToLinear: vi.fn(),
          setInterpolationTypeToNearest: vi.fn(),
          setScalarOpacityUnitDistance: vi.fn(),
        })),
      }),
    ),
  },
}));

vi.mock("@kitware/vtk.js/Rendering/Core/VolumeMapper", () => ({
  default: {
    newInstance: vi.fn(() =>
      fakeObject("VolumeMapper", {
        setInputData: vi.fn(),
        setSampleDistance: vi.fn(),
      }),
    ),
  },
}));

vi.mock("@kitware/vtk.js/Common/DataModel/ImageData", () => ({
  default: {
    newInstance: vi.fn(() => {
      const scalars = {
        setData: vi.fn((data: Uint8Array) => vtkLedger.scalarData.push(data)),
      };
      return fakeObject("ImageData", {
        setDimensions: vi.fn(),
        setSpacing: vi.fn(),
        modified: vi.fn(),
        getPointData: vi.fn(() => ({
          setScalars: vi.fn(),
          getScalars: vi.fn(() => scalars),
        })),
      });
    }),
  },
}));

vi.mock("@kitware/vtk.js/Common/Core/DataArray", () => ({
  default: {
    newInstance: vi.fn((options: { values?: Uint8Array }) => {
      if (options?.values) vtkLedger.scalarData.push(options.values);
      return fakeObject("DataArray");
    }),
  },
}));

vi.mock("@kitware/vtk.js/Rendering/Core/ColorTransferFunction", () => ({
  default: { newInstance: vi.fn(() => fakeObject("ColorTransferFunction", { addRGBPoint: vi.fn() })) },
}));

vi.mock("@kitware/vtk.js/Common/DataModel/PiecewiseFunction", () => ({
  default: { newInstance: vi.fn(() => fakeObject("PiecewiseFunction", { addPoint: vi.fn() })) },
}));

vi.mock("@kitware/vtk.js/Rendering/Core/RenderWindow", () => ({
  default: {
    newInstance: vi.fn(() =>
      fakeObject("RenderWindow", {
        addRenderer: vi.fn(),
        addView: vi.fn(),
        render: vi.fn(() => {
          vtkLedger.renders += 1;
        }),
      }),
    ),
  },
}));

vi.mock("@kitware/vtk.js/Rendering/Core/Renderer", () => ({
  default: {
    newInstance: vi.fn(() =>
      fakeObject("Renderer", {
        addVolume: vi.fn(),
        resetCamera: vi.fn(() => {
          vtkLedger.resets += 1;
        }),
        resetCameraClippingRange: vi.fn(),
        getActiveCamera: vi.fn(() => ({ elevation: vi.fn() })),
      }),
    ),
  },
}));

vi.mock("@kitware/vtk.js/Rendering/OpenGL/RenderWindow", () => ({
  default: {
    newInstance: vi.fn(() =>
      fakeObject("OpenGLRenderWindow", {
        setContainer: vi.fn(),
        setSize: vi.fn(),
      }),
    ),
  },
}));

vi.mock("@kitware/vtk.js/Rendering/Core/RenderWindowInteractor", () => ({
  default: {
    newInstance: vi.fn(() =>
      fakeObject("RenderWindowInteractor", {
        setView: vi.fn(),
        initialize: vi.fn(),
        setContainer: vi.fn(),
        setInteractorStyle: vi.fn(),
      }),
    ),
  },
}));

vi.mock(
  "@kitware/vtk.js/Interaction/Style/InteractorStyleTrackballCamera",
  () => ({ default: { newInstance: newInstance("TrackballCamera") } }),
);

// Side-effect-only module in production; a no-op here so the real WebGL factories
// are not registered inside jsdom.
vi.mock("@kitware/vtk.js/Rendering/Profiles/Volume", () => ({}));

/* -------------------------------------------------------------------------- */

const { Mri3DViewer } = await import("@/components/Mri3DViewer");

const ANALYSIS_ID = "abc123abc123abcd";

function mockVolumeFetch(payload: ArrayBuffer = buildVolumePayload({ findingDiscs: [1] })) {
  const fetchMock = vi.fn(async () => ({
    ok: true,
    arrayBuffer: async () => payload,
  })) as unknown as typeof fetch;
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock as unknown as ReturnType<typeof vi.fn>;
}

beforeEach(() => {
  vtkLedger.reset();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("3D viewer", () => {
  it("fetches the volume once and builds a scene", async () => {
    const fetchMock = mockVolumeFetch();
    render(<Mri3DViewer analysisId={ANALYSIS_ID} />);

    await waitFor(() =>
      expect(screen.queryByText(/Loading volume/i)).not.toBeInTheDocument(),
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String((fetchMock as never as { mock: { calls: string[][] } }).mock.calls[0]![0])).toContain(
      `/api/analysis/${ANALYSIS_ID}/volume`,
    );
    // Two volumes: the greyscale MRI and the label overlay.
    expect(vtkLedger.created.filter((k) => k === "Volume")).toHaveLength(2);
    expect(vtkLedger.created).toContain("RenderWindow");
    expect(vtkLedger.renders).toBeGreaterThan(0);
  });

  it("renders a real VTK container rather than a placeholder image", async () => {
    mockVolumeFetch();
    render(<Mri3DViewer analysisId={ANALYSIS_ID} />);
    await waitFor(() =>
      expect(screen.queryByText(/Loading volume/i)).not.toBeInTheDocument(),
    );

    expect(screen.getByTestId("vtk-container")).toBeInTheDocument();
    // No <img> anywhere: this is not a pre-rendered picture.
    expect(document.querySelector("img")).toBeNull();
  });

  it("exposes camera, layer and opacity controls", async () => {
    mockVolumeFetch();
    render(<Mri3DViewer analysisId={ANALYSIS_ID} />);
    await waitFor(() =>
      expect(screen.queryByText(/Loading volume/i)).not.toBeInTheDocument(),
    );

    expect(screen.getByRole("button", { name: /Reset camera/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /Fullscreen/i })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /^MRI$/ })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: /Segmentation/i })).toBeChecked();
    expect(screen.getByLabelText(/MRI opacity/i)).toHaveValue("0.35");
    expect(screen.getByText(/drag rotate/i)).toBeInTheDocument();
  });

  it("resets the camera on request", async () => {
    mockVolumeFetch();
    render(<Mri3DViewer analysisId={ANALYSIS_ID} />);
    await waitFor(() =>
      expect(screen.queryByText(/Loading volume/i)).not.toBeInTheDocument(),
    );
    const before = vtkLedger.resets;

    await userEvent.click(screen.getByRole("button", { name: /Reset camera/i }));
    expect(vtkLedger.resets).toBeGreaterThan(before);
  });

  it("re-uploads label scalars when the selected disc changes", async () => {
    mockVolumeFetch();
    const { rerender } = render(
      <Mri3DViewer analysisId={ANALYSIS_ID} highlightDisc={null} />,
    );
    await waitFor(() =>
      expect(screen.queryByText(/Loading volume/i)).not.toBeInTheDocument(),
    );
    const before = vtkLedger.scalarData.length;

    rerender(<Mri3DViewer analysisId={ANALYSIS_ID} highlightDisc={2} />);

    await waitFor(() =>
      expect(vtkLedger.scalarData.length).toBeGreaterThan(before),
    );
    // SELECTED_CODE is 9; the newest scalars must contain it.
    const latest = vtkLedger.scalarData.at(-1)!;
    expect(Array.from(latest)).toContain(9);
  });

  it("reports the highlighted disc in the overlay caption", async () => {
    mockVolumeFetch();
    render(<Mri3DViewer analysisId={ANALYSIS_ID} highlightDisc={3} />);
    await waitFor(() =>
      expect(screen.queryByText(/Loading volume/i)).not.toBeInTheDocument(),
    );
    expect(screen.getByText(/disc 3 highlighted/i)).toBeInTheDocument();
  });

  it("offers the server's finding discs and hands selection back", async () => {
    mockVolumeFetch();
    const onSelectDisc = vi.fn();
    render(
      <Mri3DViewer
        analysisId={ANALYSIS_ID}
        findingOverlay={resultFixture.finding_overlay}
        onSelectDisc={onSelectDisc}
      />,
    );
    await waitFor(() =>
      expect(screen.queryByText(/Loading volume/i)).not.toBeInTheDocument(),
    );

    expect(screen.getByText("1 finding-associated")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "1" }));
    expect(onSelectDisc).toHaveBeenCalledWith(1);
  });

  it("states that red is not a pixel-level diagnosis", async () => {
    mockVolumeFetch();
    render(<Mri3DViewer analysisId={ANALYSIS_ID} />);
    await waitFor(() =>
      expect(screen.queryByText(/Loading volume/i)).not.toBeInTheDocument(),
    );
    expect(
      screen.getByText(/not a pixel-level diagnosis of damaged tissue/i),
    ).toBeInTheDocument();
  });

  it("disables the findings layer when the server marked nothing", async () => {
    mockVolumeFetch();
    render(
      <Mri3DViewer
        analysisId={ANALYSIS_ID}
        findingOverlay={{
          ...resultFixture.finding_overlay,
          disc_indices: [],
          discs: [],
        }}
      />,
    );
    await waitFor(() =>
      expect(screen.queryByText(/Loading volume/i)).not.toBeInTheDocument(),
    );
    expect(screen.getByRole("checkbox", { name: /Findings/i })).toBeDisabled();
  });

  it("releases every VTK object on unmount", async () => {
    mockVolumeFetch();
    const { unmount } = render(<Mri3DViewer analysisId={ANALYSIS_ID} />);
    await waitFor(() =>
      expect(screen.queryByText(/Loading volume/i)).not.toBeInTheDocument(),
    );
    expect(vtkLedger.deleted).toHaveLength(0);

    unmount();

    // Renderers, actors, mappers, the interactor and the GL window all go.
    for (const kind of [
      "RenderWindow", "Renderer", "OpenGLRenderWindow",
      "RenderWindowInteractor", "Volume", "VolumeMapper",
    ]) {
      expect(vtkLedger.deleted, `${kind} was not deleted`).toContain(kind);
    }
    expect(vtkLedger.deleted.filter((k) => k === "Volume")).toHaveLength(2);
  });

  it("shows an actionable message when the volume is unavailable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 409,
        json: async () => ({
          error: { code: "VOLUME_NOT_READY", message: "Not ready yet." },
        }),
      })) as unknown as typeof fetch,
    );

    render(<Mri3DViewer analysisId={ANALYSIS_ID} />);

    expect(await screen.findByText(/3D view unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/Not ready yet\./i)).toBeInTheDocument();
    // The 2D path must be advertised as unaffected.
    expect(screen.getByText(/2D viewer is unaffected/i)).toBeInTheDocument();
  });

  it("survives a malformed volume payload", async () => {
    mockVolumeFetch(new ArrayBuffer(2));
    render(<Mri3DViewer analysisId={ANALYSIS_ID} />);
    expect(await screen.findByText(/3D view unavailable/i)).toBeInTheDocument();
  });
});

/* -------------------------------------------------------------------------- */
/* 2D / 3D synchronisation and the demo-stage banner                          */
/* -------------------------------------------------------------------------- */

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), back: vi.fn() }),
  usePathname: () => "/analysis/abc123abc123abcd",
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      ...actual.api,
      longitudinalCase: vi.fn(async () => demoCase),
      loadDemoStage: vi.fn(async () => ({ analysis_id: "f".repeat(16) })),
      sliceUrl: actual.api.sliceUrl,
      downloadUrl: (id: string) => `/download/${id}`,
    },
  };
});

const { ResultsView } = await import("@/components/ResultsView");

describe("2D / 3D synchronisation", () => {
  beforeEach(() => {
    vtkLedger.reset();
    mockVolumeFetch();
  });

  it("offers both views and starts in 2D", () => {
    render(<ResultsView result={resultFixture} />);

    const group = screen.getByRole("group", { name: /Viewer dimension/i });
    const twoD = within(group).getByRole("button", { name: /2D View/i });
    expect(twoD).toHaveAttribute("aria-pressed", "true");
    expect(within(group).getByRole("button", { name: /3D View/i })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    // The existing 2D viewer is present and untouched.
    expect(screen.getByAltText(/Lumbar MRI slice/i)).toBeInTheDocument();
  });

  it("switches to the 3D view on request and keeps the 2D viewer mounted", async () => {
    render(<ResultsView result={resultFixture} />);

    await userEvent.click(screen.getByRole("button", { name: /3D View/i }));

    expect(await screen.findByTestId("vtk-container")).toBeInTheDocument();
    // Still mounted, so slice position and prefetch cache survive a tab flip.
    expect(screen.getByAltText(/Lumbar MRI slice/i)).toBeInTheDocument();
  });

  it("propagates the disc selected in the panel into the 3D view", async () => {
    render(<ResultsView result={resultFixture} />);
    await userEvent.click(screen.getByRole("button", { name: /3D View/i }));
    await screen.findByTestId("vtk-container");

    await userEvent.click(screen.getAllByRole("button", { name: /^Disc/ })[0]!);

    expect(await screen.findByText(/disc 1 highlighted/i)).toBeInTheDocument();
  });

  it("passes the same server finding decision to the 3D view", async () => {
    render(<ResultsView result={resultFixture} />);
    await userEvent.click(screen.getByRole("button", { name: /3D View/i }));
    await screen.findByTestId("vtk-container");

    // resultFixture.finding_overlay marks exactly disc 1.
    expect(screen.getByText("1 finding-associated")).toBeInTheDocument();
  });
});

describe("simulated longitudinal demo stage", () => {
  beforeEach(() => {
    vtkLedger.reset();
    mockVolumeFetch();
  });

  /** The banner element, located by its badge. */
  const banner = () =>
    screen.getByText("Simulated Longitudinal Demo").closest("div")!
      .parentElement!;

  it("shows the SIMULATED LONGITUDINAL DEMO banner in the actual UI", () => {
    render(<ResultsView result={demoStageResult} />);
    // Exact match: the disclaimer also contains "Simulated longitudinal
    // demonstration", so a loose regex would match two nodes.
    expect(screen.getByText("Simulated Longitudinal Demo")).toBeInTheDocument();
  });

  it("states that the stages are different studies, not follow-up scans", () => {
    render(<ResultsView result={demoStageResult} />);

    expect(
      screen.getByText(/different SPIDER studies to demonstrate the longitudinal workflow/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/not postoperative follow-up scans from the same patient/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/does not contain true postoperative longitudinal follow-up/i),
    ).toBeInTheDocument();
  });

  it("identifies the current stage without exposing a patient identifier", () => {
    render(<ResultsView result={demoStageResult} />);

    // Scoped to the banner: "Baseline" also appears as the analysis timepoint.
    const inBanner = within(banner());
    expect(inBanner.getByText(/Demo stage: Pre-Surgery/i)).toBeInTheDocument();
    expect(inBanner.getByText("Baseline")).toBeInTheDocument();
    expect(inBanner.getByText("Demonstration study A")).toBeInTheDocument();
    // The source patient id is provenance, not something the banner shows.
    expect(inBanner.queryByText(/177/)).not.toBeInTheDocument();
  });

  it("never calls the stages postoperative or follow-up scans", () => {
    render(<ResultsView result={demoStageResult} />);
    const text = document.body.textContent ?? "";

    for (const forbidden of [
      "actual postoperative scan",
      "recovery scan",
      "longitudinal follow-up scan",
      "same-patient follow-up",
    ]) {
      expect(text.toLowerCase()).not.toContain(forbidden);
    }
  });

  it("does not display any recovery percentage", () => {
    render(<ResultsView result={demoStageResult} />);
    const text = document.body.textContent ?? "";
    expect(text).not.toMatch(/recovery\s*(percentage|score)/i);
    expect(text).not.toMatch(/\d+%\s*recover/i);
  });

  it("offers all four stages and marks the current one", async () => {
    render(<ResultsView result={demoStageResult} />);

    for (const label of [
      "Pre-Surgery", "Post-Surgery", "3-Month Recovery", "6-Month Recovery",
    ]) {
      expect(await screen.findByRole("button", { name: new RegExp(label, "i") }))
        .toBeInTheDocument();
    }
    const current = await screen.findByRole("button", { name: /Pre-Surgery/i });
    expect(current).toHaveAttribute("aria-current", "step");
    expect(current).toBeDisabled();
  });

  it("loads a different stage through the real pipeline endpoint", async () => {
    const { api } = await import("@/lib/api");
    render(<ResultsView result={demoStageResult} />);

    const target = await screen.findByRole("button", { name: /6-Month Recovery/i });
    await userEvent.click(target);

    expect(api.loadDemoStage).toHaveBeenCalledWith("LS-DEMO-001", "month_6");
  });

  it("says the stages are different studies processed by the same pipeline", async () => {
    render(<ResultsView result={demoStageResult} />);
    expect(
      await screen.findByText(/each stage is a different SPIDER study/i),
    ).toBeInTheDocument();
  });

  it("shows no demo banner for an ordinary study", () => {
    render(<ResultsView result={resultFixture} />);
    expect(screen.queryByText(/Simulated Longitudinal Demo/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Demo stages/i)).not.toBeInTheDocument();
  });
});
