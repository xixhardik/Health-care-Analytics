/**
 * Frontend UX polish: workspace layout, and idle auto-rotation in the 3D viewer.
 *
 * jsdom has no layout engine and no WebGL, so these tests do not pretend to
 * measure pixels or validate rendering. What they pin is what actually broke and
 * what could silently regress:
 *
 *  - the layout *contract* - which element is height-bounded and which one owns
 *    the scrollbar. That contract is the entire fix for the viewer sitting a
 *    screen and a half down the page.
 *  - the auto-rotation *lifecycle* - when it starts, what stops it, when it
 *    resumes, and that nothing survives unmount.
 *  - that the findings overlay cannot resize the MRI viewport. Since pixel
 *    heights are unmeasurable here, what is asserted is that the canvas box and
 *    the controls box are not functions of the overlay toggle at all.
 */

import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { buildVolumePayload, result as resultFixture } from "./fixtures";

/* -------------------------------------------------------------------------- */
/* VTK mock - tracks camera azimuth so rotation is observable                  */
/* -------------------------------------------------------------------------- */

const vtk = {
  azimuthCalls: [] as number[],
  renders: 0,
  deleted: [] as string[],
  reset() {
    this.azimuthCalls = [];
    this.renders = 0;
    this.deleted = [];
  },
};

const camera = {
  azimuth: vi.fn((deg: number) => vtk.azimuthCalls.push(deg)),
  elevation: vi.fn(),
};

function obj(kind: string, extra: Record<string, unknown> = {}) {
  return { delete: vi.fn(() => vtk.deleted.push(kind)), ...extra };
}

vi.mock("@kitware/vtk.js/Rendering/Profiles/Volume", () => ({}));

vi.mock("@kitware/vtk.js/Rendering/Core/Volume", () => ({
  default: {
    newInstance: vi.fn(() =>
      obj("Volume", {
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
      obj("VolumeMapper", { setInputData: vi.fn(), setSampleDistance: vi.fn() }),
    ),
  },
}));
vi.mock("@kitware/vtk.js/Common/DataModel/ImageData", () => ({
  default: {
    newInstance: vi.fn(() => {
      const scalars = { setData: vi.fn() };
      return obj("ImageData", {
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
  default: { newInstance: vi.fn(() => obj("DataArray")) },
}));
vi.mock("@kitware/vtk.js/Rendering/Core/ColorTransferFunction", () => ({
  default: { newInstance: vi.fn(() => obj("ColorTransferFunction", { addRGBPoint: vi.fn() })) },
}));
vi.mock("@kitware/vtk.js/Common/DataModel/PiecewiseFunction", () => ({
  default: { newInstance: vi.fn(() => obj("PiecewiseFunction", { addPoint: vi.fn() })) },
}));
vi.mock("@kitware/vtk.js/Rendering/Core/RenderWindow", () => ({
  default: {
    newInstance: vi.fn(() =>
      obj("RenderWindow", {
        addRenderer: vi.fn(),
        addView: vi.fn(),
        render: vi.fn(() => {
          vtk.renders += 1;
        }),
      }),
    ),
  },
}));
vi.mock("@kitware/vtk.js/Rendering/Core/Renderer", () => ({
  default: {
    newInstance: vi.fn(() =>
      obj("Renderer", {
        addVolume: vi.fn(),
        resetCamera: vi.fn(),
        resetCameraClippingRange: vi.fn(),
        getActiveCamera: vi.fn(() => camera),
      }),
    ),
  },
}));
vi.mock("@kitware/vtk.js/Rendering/OpenGL/RenderWindow", () => ({
  default: {
    newInstance: vi.fn(() =>
      obj("OpenGLRenderWindow", { setContainer: vi.fn(), setSize: vi.fn() }),
    ),
  },
}));
vi.mock("@kitware/vtk.js/Rendering/Core/RenderWindowInteractor", () => ({
  default: {
    newInstance: vi.fn(() =>
      obj("RenderWindowInteractor", {
        setView: vi.fn(),
        initialize: vi.fn(),
        setContainer: vi.fn(),
        setInteractorStyle: vi.fn(),
      }),
    ),
  },
}));
vi.mock("@kitware/vtk.js/Interaction/Style/InteractorStyleTrackballCamera", () => ({
  default: { newInstance: vi.fn(() => obj("TrackballCamera")) },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => "/analysis/abc123abc123abcd",
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      ...actual.api,
      longitudinalCase: vi.fn(),
      loadDemoStage: vi.fn(),
      sliceUrl: actual.api.sliceUrl,
      downloadUrl: (id: string) => `/download/${id}`,
    },
  };
});

const { Mri3DViewer } = await import("@/components/Mri3DViewer");
const { ResultsView } = await import("@/components/ResultsView");

const ANALYSIS_ID = "abc123abc123abcd";

/** Drives requestAnimationFrame manually so rotation is deterministic. */
let rafQueue: FrameRequestCallback[] = [];
let rafId = 0;
let now = 0;

function flushFrames(count = 1, msPerFrame = 16) {
  for (let i = 0; i < count; i += 1) {
    const queued = rafQueue;
    rafQueue = [];
    now += msPerFrame;
    for (const cb of queued) cb(now);
  }
}

beforeEach(() => {
  vtk.reset();
  camera.azimuth.mockClear();
  rafQueue = [];
  rafId = 0;
  now = 0;
  vi.useFakeTimers({ shouldAdvanceTime: true });

  vi.stubGlobal(
    "requestAnimationFrame",
    (cb: FrameRequestCallback) => {
      rafQueue.push(cb);
      rafId += 1;
      return rafId;
    },
  );
  vi.stubGlobal("cancelAnimationFrame", () => {
    rafQueue = [];
  });
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      arrayBuffer: async () => buildVolumePayload({ findingDiscs: [1] }),
    })) as unknown as typeof fetch,
  );
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function mountViewer(
  props: Partial<React.ComponentProps<typeof Mri3DViewer>> = {},
) {
  const utils = render(<Mri3DViewer analysisId={ANALYSIS_ID} {...props} />);
  await waitFor(() =>
    expect(screen.queryByText(/Loading volume/i)).not.toBeInTheDocument(),
  );
  return utils;
}

/**
 * Let the idle timer elapse and the resulting effect run.
 *
 * Wrapped in `act` because the timer callback calls setState from outside React;
 * without it the state update and the effect that starts the animation would not
 * have flushed by the time the frames are driven.
 */
async function goIdle() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2000);
  });
}

/* -------------------------------------------------------------------------- */
/* Problem 1: layout contract                                                  */
/* -------------------------------------------------------------------------- */

describe("analysis workspace layout", () => {
  beforeEach(() => {
    vi.useRealTimers();
  });

  /** The workspace root, which owns the viewport height bound. */
  const workspace = () =>
    screen.getByRole("heading", { level: 1 }).closest("div.flex.min-h-0.flex-col")!
      .parentElement ?? document.body;

  it("bounds the workspace to the viewport below the header", () => {
    const { container } = render(<ResultsView result={resultFixture} />);
    const root = container.firstElementChild as HTMLElement;

    // Without a definite height here, the sidebar's overflow-y-auto has nothing
    // to scroll inside and the whole page grows instead.
    expect(root.className).toContain("lg:h-[calc(100vh-3.5rem)]");
    expect(root.className).toContain("lg:overflow-hidden");
    // Below lg the layout stays a scrolling document.
    expect(root.className).not.toMatch(/(^|\s)h-\[calc/);
  });

  it("makes the findings sidebar the scrolling region, not the page", () => {
    const { container } = render(<ResultsView result={resultFixture} />);
    const aside = container.querySelector("aside")!;

    expect(aside.className).toContain("overflow-y-auto");
    // min-h-0 is what lets a flex child actually shrink and scroll.
    expect(aside.className).toContain("min-h-0");
  });

  it("keeps min-h-0 on every ancestor between the root and the viewer", () => {
    // A single missing min-h-0 in a nested flex/grid chain silently defeats
    // internal scrolling, which is why this is asserted rather than assumed.
    const { container } = render(<ResultsView result={resultFixture} />);
    const grid = container.querySelector("div.grid")!;
    expect(grid.className).toContain("min-h-0");
    expect(grid.className).toContain("flex-1");

    const viewerColumn = grid.firstElementChild as HTMLElement;
    expect(viewerColumn.className).toContain("min-h-0");
  });

  it("lets the viewer shrink inside the bounded workspace", () => {
    const { container } = render(<ResultsView result={resultFixture} />);
    const viewerBox = container.querySelector("div.min-h-\\[380px\\]")!;
    // Keeps a floor on stacked layouts, but yields on the bounded one.
    expect(viewerBox.className).toContain("lg:min-h-0");
  });

  it("does not introduce horizontal scrolling", () => {
    const { container } = render(<ResultsView result={resultFixture} />);
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).not.toContain("overflow-x-auto");
    expect(root.className).not.toContain("overflow-x-scroll");
  });

  it("still renders every viewer control after the layout change", async () => {
    render(<ResultsView result={resultFixture} />);

    expect(screen.getByRole("group", { name: /Rendering mode/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Zoom in/i })).toBeInTheDocument();
    expect(screen.getByLabelText("Opacity")).toBeInTheDocument();
    expect(screen.getByLabelText("Slice position")).toBeInTheDocument();
    expect(
      screen.getByRole("checkbox", { name: /Intervertebral discs/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Next slice/i })).toBeInTheDocument();
  });

  it("fills the viewer with the MRI at 1x instead of a fixed small box", () => {
    render(<ResultsView result={resultFixture} />);
    const image = screen.getByAltText(/Lumbar MRI slice/i) as HTMLImageElement;

    // object-contain fills the available space without cropping or stretching.
    expect(image.style.width).toBe("100%");
    expect(image.style.height).toBe("100%");
    expect(image.style.objectFit).toBe("contain");
  });

  it("switches to explicit pixel sizing when zoomed, so panning works", async () => {
    render(<ResultsView result={resultFixture} />);
    await userEvent.click(screen.getByRole("button", { name: /Zoom in/i }));

    const image = screen.getByAltText(/Lumbar MRI slice/i) as HTMLImageElement;
    expect(image.style.width).toBe(`${352 * 1.5}px`);
    expect(image.style.height).toBe(`${256 * 1.5}px`);
    expect(image.style.objectFit).toBe("contain");
  });
});

/* -------------------------------------------------------------------------- */
/* Problem 3: the findings overlay must not resize the MRI viewport            */
/* -------------------------------------------------------------------------- */

describe("findings overlay and MRI viewport size", () => {
  beforeEach(() => {
    vi.useRealTimers();
  });

  const canvas = () => screen.getByRole("group", { name: /MRI slice viewer/i });
  const controls = () => screen.getByTestId("viewer-controls");
  const sliceImage = () =>
    screen.getByAltText(/Lumbar MRI slice/i) as HTMLImageElement;
  const findingsToggle = (state: "On" | "Off") =>
    screen.getByRole("checkbox", { name: new RegExp(`^${state}$`) });

  it("gives the overlay controls a constant desktop height and their own scrollbar", () => {
    render(<ResultsView result={resultFixture} />);

    // A fixed box with its own scrollbar is what stops this strip taking height
    // out of the canvas when the overlay expands.
    expect(controls().className).toContain("lg:h-[7.5rem]");
    expect(controls().className).toContain("lg:overflow-y-auto");
    // Gutter reserved up front, so the scrollbar appearing cannot reflow the rows.
    expect(controls().className).toContain("lg:[scrollbar-gutter:stable]");
  });

  it("leaves the overlay controls unbounded below lg, where the page scrolls", () => {
    render(<ResultsView result={resultFixture} />);

    // The cap is lg-prefixed only; a scroll region inside a scrolling document
    // would be worse than a taller page.
    expect(controls().className).not.toMatch(/(^|\s)h-\[/);
    expect(controls().className).not.toMatch(/(^|\s)overflow-y-auto/);
  });

  it("does not change the canvas box when the overlay is switched on", async () => {
    render(<ResultsView result={resultFixture} />);
    const before = canvas().className;

    await userEvent.click(findingsToggle("Off"));

    // jsdom has no layout engine, so the *contract* is asserted rather than a
    // pixel height: the canvas box is not a function of the overlay being on.
    expect(sliceImage().src).toContain("highlight=findings");
    expect(canvas().className).toBe(before);
  });

  it("does not change the controls box when the overlay is switched on", async () => {
    render(<ResultsView result={resultFixture} />);
    const before = controls().className;

    await userEvent.click(findingsToggle("Off"));

    // The height is keyed on server-decided availability, never on the toggle.
    expect(controls().className).toBe(before);
  });

  it("stays stable across repeated toggling", async () => {
    render(<ResultsView result={resultFixture} />);
    const canvasBox = canvas().className;
    const controlsBox = controls().className;

    for (let round = 0; round < 2; round += 1) {
      await userEvent.click(findingsToggle("Off"));
      expect(canvas().className).toBe(canvasBox);
      expect(controls().className).toBe(controlsBox);

      await userEvent.click(findingsToggle("On"));
      expect(canvas().className).toBe(canvasBox);
      expect(controls().className).toBe(controlsBox);
    }
  });

  it("keeps the overlay explanation rendered instead of dropping it to save height", async () => {
    render(<ResultsView result={resultFixture} />);
    await userEvent.click(findingsToggle("Off"));

    // Scrollable, not hidden. The sentence that stops the red being read as
    // tissue-level pathology has to stay on screen.
    expect(
      screen.getByText(/not a diagnosis of damaged tissue/i),
    ).toBeInTheDocument();

    const legend = screen.getByLabelText("Overlay legend");
    expect(controls().contains(legend)).toBe(true);
  });

  it("keeps the finding opacity control on the primary overlay row", async () => {
    render(<ResultsView result={resultFixture} />);
    await userEvent.click(findingsToggle("Off"));

    const row = screen.getByText("Findings Overlay").parentElement!;

    // Same row as the toggle and the marked-disc count, so the control most
    // likely to be reached for is not below the fold of the scrolling strip.
    expect(row.contains(findingsToggle("On"))).toBe(true);
    expect(row.contains(screen.getByText("1 disc marked"))).toBe(true);
    expect(row.contains(screen.getByLabelText("Finding"))).toBe(true);
    // And it drops onto its own line when the row runs out of width.
    expect(row.className).toContain("flex-wrap");
  });

  it("floors the canvas so nothing below it can flatten the image", () => {
    render(<ResultsView result={resultFixture} />);

    expect(canvas().className).toContain("flex-1");
    // A zero-basis flex child is the first thing space is taken from, so it
    // needs a floor of its own on the bounded desktop layout.
    expect(canvas().className).toContain("lg:min-h-[200px]");
    // The stacked layout keeps its larger floor.
    expect(canvas().className).toContain("min-h-[320px]");
  });

  it("puts mode, slice and zoom in one row instead of two", () => {
    render(<ResultsView result={resultFixture} />);
    const bar = screen.getByRole("group", { name: /Rendering mode/i })
      .parentElement!;

    expect(bar.contains(screen.getByLabelText("Slice position"))).toBe(true);
    expect(bar.contains(screen.getByRole("button", { name: /Zoom in/i }))).toBe(
      true,
    );
    // Wraps back to separate lines when the column is too narrow.
    expect(bar.className).toContain("flex-wrap");
  });

  it("puts the image first in the viewer column, with the controls under it", () => {
    render(<ResultsView result={resultFixture} />);
    const column = canvas().parentElement!;

    expect(column.firstElementChild).toBe(canvas());
    expect(column.lastElementChild).toBe(controls());
  });

  it("does not introduce horizontal scrolling in the viewer", () => {
    render(<ResultsView result={resultFixture} />);

    expect(canvas().className).not.toContain("overflow-x");
    expect(controls().className).not.toContain("overflow-x");
  });
});

/* -------------------------------------------------------------------------- */
/* Problem 2: idle auto-rotation                                               */
/* -------------------------------------------------------------------------- */

describe("3D idle auto-rotation", () => {
  it("does not rotate before the idle delay has elapsed", async () => {
    await mountViewer();
    flushFrames(3);
    expect(camera.azimuth).not.toHaveBeenCalled();
  });

  it("starts rotating once the viewer has been idle", async () => {
    await mountViewer();
    await goIdle();
    flushFrames(3);

    expect(camera.azimuth).toHaveBeenCalled();
    // By test id, not text: the toggle label also reads "Auto-rotate".
    expect(await screen.findByTestId("auto-rotate-indicator")).toBeInTheDocument();
  });

  it("rotates slowly, on one axis, in small time-based increments", async () => {
    await mountViewer();
    // Scene construction tilts the camera once; only rotation is under test here.
    camera.elevation.mockClear();
    await goIdle();
    flushFrames(4, 16);

    expect(vtk.azimuthCalls.length).toBeGreaterThan(0);
    // 15 deg/sec over a 16 ms frame is well under a degree.
    for (const degrees of vtk.azimuthCalls) {
      expect(degrees).toBeGreaterThan(0);
      expect(degrees).toBeLessThan(1);
    }
    // Azimuth only: no elevation churn, so the volume does not tumble.
    expect(camera.elevation).not.toHaveBeenCalled();
  });

  it("does nothing when the renderer never became ready", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 409,
        json: async () => ({ error: { code: "VOLUME_NOT_READY", message: "Not ready." } }),
      })) as unknown as typeof fetch,
    );
    render(<Mri3DViewer analysisId={ANALYSIS_ID} />);
    expect(await screen.findByText(/3D view unavailable/i)).toBeInTheDocument();

    await goIdle();
    flushFrames(3);
    expect(camera.azimuth).not.toHaveBeenCalled();
  });

  it("pauses immediately when the user starts dragging", async () => {
    await mountViewer();
    await goIdle();
    flushFrames(2);
    expect(camera.azimuth).toHaveBeenCalled();

    const before = camera.azimuth.mock.calls.length;
    screen.getByTestId("vtk-container").dispatchEvent(
      new MouseEvent("pointerdown", { bubbles: true }),
    );
    flushFrames(3);

    expect(camera.azimuth.mock.calls.length).toBe(before);
    await waitFor(() =>
      expect(screen.queryByTestId("auto-rotate-indicator")).not.toBeInTheDocument(),
    );
  });

  it("pauses on wheel and on touch", async () => {
    await mountViewer();
    const container = screen.getByTestId("vtk-container");

    for (const event of [
      new WheelEvent("wheel", { bubbles: true }),
      new Event("touchstart", { bubbles: true }),
    ]) {
      await goIdle();
      flushFrames(1);
      const before = camera.azimuth.mock.calls.length;
      container.dispatchEvent(event);
      flushFrames(3);
      expect(camera.azimuth.mock.calls.length).toBe(before);
    }
  });

  it("ignores pointer movement with no button held", async () => {
    await mountViewer();
    await goIdle();
    flushFrames(1);
    const before = camera.azimuth.mock.calls.length;

    // Hovering is not camera interaction.
    screen.getByTestId("vtk-container").dispatchEvent(
      new MouseEvent("pointermove", { bubbles: true, buttons: 0 }),
    );
    flushFrames(2);

    expect(camera.azimuth.mock.calls.length).toBeGreaterThan(before);
  });

  it("resumes after the user stops interacting", async () => {
    await mountViewer();
    screen.getByTestId("vtk-container").dispatchEvent(
      new MouseEvent("pointerdown", { bubbles: true }),
    );
    flushFrames(2);
    const during = camera.azimuth.mock.calls.length;

    await goIdle();
    flushFrames(3);

    expect(camera.azimuth.mock.calls.length).toBeGreaterThan(during);
  });

  it("pauses when the camera is reset", async () => {
    await mountViewer();
    await goIdle();
    flushFrames(1);
    const before = camera.azimuth.mock.calls.length;

    await userEvent.click(screen.getByRole("button", { name: /Reset camera/i }));
    flushFrames(3);

    expect(camera.azimuth.mock.calls.length).toBe(before);
  });

  it("pauses when a visualisation control changes", async () => {
    await mountViewer();
    await goIdle();
    flushFrames(1);
    const before = camera.azimuth.mock.calls.length;

    await userEvent.click(screen.getByRole("checkbox", { name: /Segmentation/i }));
    flushFrames(3);

    expect(camera.azimuth.mock.calls.length).toBe(before);
  });

  it("stays off when the user switches it off, and does not resume on idle", async () => {
    await mountViewer();
    await userEvent.click(screen.getByRole("checkbox", { name: /Auto-rotate/i }));

    await goIdle();
    flushFrames(5);

    expect(camera.azimuth).not.toHaveBeenCalled();
    expect(screen.queryByTestId("auto-rotate-indicator")).not.toBeInTheDocument();
  });

  it("rotates again once the user switches it back on", async () => {
    await mountViewer();
    const toggle = screen.getByRole("checkbox", { name: /Auto-rotate/i });

    await userEvent.click(toggle);
    await goIdle();
    flushFrames(3);
    expect(camera.azimuth).not.toHaveBeenCalled();

    await userEvent.click(toggle);
    await goIdle();
    flushFrames(3);
    expect(camera.azimuth).toHaveBeenCalled();
  });

  it("defaults to off when the system asks for reduced motion", async () => {
    vi.stubGlobal("matchMedia", (query: string) => ({
      matches: query.includes("prefers-reduced-motion"),
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }));

    await mountViewer();
    expect(screen.getByRole("checkbox", { name: /Auto-rotate/i })).not.toBeChecked();

    await goIdle();
    flushFrames(3);
    expect(camera.azimuth).not.toHaveBeenCalled();
  });

  it("stops rotating and releases VTK on unmount", async () => {
    const { unmount } = await mountViewer();
    await goIdle();
    flushFrames(2);
    expect(camera.azimuth).toHaveBeenCalled();

    unmount();
    const afterUnmount = camera.azimuth.mock.calls.length;
    flushFrames(5);
    await vi.advanceTimersByTimeAsync(4000);
    flushFrames(5);

    // No frames and no late idle timer firing after teardown.
    expect(camera.azimuth.mock.calls.length).toBe(afterUnmount);
    expect(vtk.deleted).toContain("RenderWindow");
    expect(vtk.deleted).toContain("Renderer");
  });

  it("does not leave a stale animation running when the study changes", async () => {
    const { rerender } = await mountViewer();
    await goIdle();
    flushFrames(2);
    expect(camera.azimuth).toHaveBeenCalled();

    // A stage change swaps the analysis id, which rebuilds the whole scene.
    rerender(<Mri3DViewer analysisId={"f".repeat(16)} />);
    await waitFor(() => expect(vtk.deleted).toContain("RenderWindow"));

    camera.azimuth.mockClear();
    flushFrames(4);
    // The previous scene's loop must not still be turning the camera.
    expect(camera.azimuth).not.toHaveBeenCalled();
  });

  it("keeps disc selection and findings working while rotating", async () => {
    const onSelectDisc = vi.fn();
    await mountViewer({
      highlightDisc: 1,
      findingOverlay: resultFixture.finding_overlay,
      onSelectDisc,
    });
    await goIdle();
    flushFrames(2);
    expect(camera.azimuth).toHaveBeenCalled();

    expect(screen.getByText(/disc 1 highlighted/i)).toBeInTheDocument();
    expect(screen.getByText("1 finding-associated")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "1" }));
    expect(onSelectDisc).toHaveBeenCalledWith(1);
  });
});
