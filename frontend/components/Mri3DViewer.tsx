/**
 * Volumetric 3D viewer, rendered by VTK.js over WebGL.
 *
 * This renders the actual MRI volume served by `GET /api/analysis/{id}/volume` -
 * the same voxels the 2D viewer's PNGs are composed from. There is no
 * pre-rendered imagery and no faked depth.
 *
 * Two volumes share one camera: the greyscale MRI, and a label volume built from
 * the semantic and instance maps. Keeping them separate is what lets MRI opacity,
 * segmentation visibility, findings visibility and disc selection be independent
 * controls without re-fetching anything.
 *
 * Synchronisation: which discs are finding-associated arrives in the volume
 * header, decided by the server, and the selected disc is passed in from
 * `ResultsView` - the same state the 2D viewer and the disc panel use. This
 * component derives neither, so the two views cannot disagree.
 *
 * Loaded through `next/dynamic` with `ssr: false`: VTK.js touches `window` and
 * WebGL at import time, so it must never run during a server render.
 */

"use client";

import {
  AlertTriangle,
  Box,
  Loader2,
  Maximize2,
  Minimize2,
  RotateCcw,
} from "lucide-react";
import * as React from "react";

import { FINDING_OVERLAY, SEG_CLASSES } from "@/lib/theme";
import type { FindingOverlayInfo } from "@/lib/types";
import {
  FINDING_CODE,
  SELECTED_CODE,
  VolumeError,
  buildLabelScalars,
  fetchVolume,
  type LoadedVolume,
} from "@/lib/volume";
import { Badge, Button, Skeleton, Toggle, cn } from "./ui";

export interface Mri3DViewerProps {
  analysisId: string;
  /** Disc selected in the results panel. Highlighted here too. */
  highlightDisc?: number | null;
  /** The server's finding-overlay decision, for the legend and the count. */
  findingOverlay?: FindingOverlayInfo | null;
  /** Lets the 3D view hand selection back to the 2D workflow. */
  onSelectDisc?: (discIndex: number) => void;
}

/** Everything VTK owns, so unmount can release all of it. */
interface Scene {
  renderWindow: { delete: () => void; render: () => void };
  renderer: { delete: () => void; resetCamera: () => void };
  openGL: { delete: () => void; setContainer: (c: HTMLElement | null) => void };
  interactor: { delete: () => void; setContainer: (c: HTMLElement | null) => void };
  mriActor: { delete: () => void; setVisibility: (v: boolean) => void };
  labelActor: { delete: () => void; setVisibility: (v: boolean) => void };
  mriProperty: unknown;
  labelData: { getPointData: () => { getScalars: () => { setData: (d: Uint8Array) => void } } };
  labelMapper: { delete: () => void };
  mriMapper: { delete: () => void };
  setMriOpacity: (value: number) => void;
  resetCamera: () => void;
  render: () => void;
}

export function Mri3DViewer({
  analysisId,
  highlightDisc,
  findingOverlay,
  onSelectDisc,
}: Mri3DViewerProps) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const sceneRef = React.useRef<Scene | null>(null);
  const volumeRef = React.useRef<LoadedVolume | null>(null);

  const [status, setStatus] = React.useState<"loading" | "ready" | "failed">(
    "loading",
  );
  const [error, setError] = React.useState<string | null>(null);
  const [showMri, setShowMri] = React.useState(true);
  const [mriOpacity, setMriOpacity] = React.useState(0.35);
  const [showSegmentation, setShowSegmentation] = React.useState(true);
  const [showFindings, setShowFindings] = React.useState(true);
  const [visibleClasses, setVisibleClasses] = React.useState<number[]>([1, 2, 3]);
  const [fullscreen, setFullscreen] = React.useState(false);

  const markedDiscs = findingOverlay?.disc_indices ?? [];

  /* ------------------------------------------------ build the scene once */
  React.useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    async function build() {
      setStatus("loading");
      setError(null);
      try {
        // Imported here rather than at module scope so the WebGL-touching code
        // is only pulled in when the 3D tab is actually opened.
        // The rendering profile registers the OpenGL view-node factories for
        // volume rendering. Without it vtk.js raises "No vtkOpenGLViewNodeFactory
        // implementation found for vtkRenderer" and nothing draws - in a real
        // browser as well as here. It must be imported before any renderable is
        // constructed, and the Volume profile is imported rather than All so the
        // geometry, glyph and molecule pipelines stay out of the bundle.
        await import("@kitware/vtk.js/Rendering/Profiles/Volume");

        const [
          { default: vtkVolume },
          { default: vtkVolumeMapper },
          { default: vtkImageData },
          { default: vtkDataArray },
          { default: vtkColorTransferFunction },
          { default: vtkPiecewiseFunction },
          { default: vtkRenderWindow },
          { default: vtkRenderer },
          { default: vtkOpenGLRenderWindow },
          { default: vtkRenderWindowInteractor },
          { default: vtkInteractorStyleTrackballCamera },
        ] = await Promise.all([
          import("@kitware/vtk.js/Rendering/Core/Volume"),
          import("@kitware/vtk.js/Rendering/Core/VolumeMapper"),
          import("@kitware/vtk.js/Common/DataModel/ImageData"),
          import("@kitware/vtk.js/Common/Core/DataArray"),
          import("@kitware/vtk.js/Rendering/Core/ColorTransferFunction"),
          import("@kitware/vtk.js/Common/DataModel/PiecewiseFunction"),
          import("@kitware/vtk.js/Rendering/Core/RenderWindow"),
          import("@kitware/vtk.js/Rendering/Core/Renderer"),
          import("@kitware/vtk.js/Rendering/OpenGL/RenderWindow"),
          import("@kitware/vtk.js/Rendering/Core/RenderWindowInteractor"),
          import("@kitware/vtk.js/Interaction/Style/InteractorStyleTrackballCamera"),
        ]);

        const volume = await fetchVolume(analysisId, controller.signal);
        if (cancelled) return;
        volumeRef.current = volume;

        const container = containerRef.current;
        if (!container) return;

        const { slices, rows, cols } = volume.header.dimensions;
        // Through-plane spacing is null when the source study did not report it;
        // 1.0 keeps the volume renderable without inventing a thickness claim.
        const zSpacing = volume.header.spacing_mm.slice ?? 1.0;
        const spacing: [number, number, number] = [
          volume.header.spacing_mm.col,
          volume.header.spacing_mm.row,
          zSpacing,
        ];

        const renderWindow = vtkRenderWindow.newInstance();
        const renderer = vtkRenderer.newInstance({ background: [0, 0, 0] });
        renderWindow.addRenderer(renderer);

        const openGL = vtkOpenGLRenderWindow.newInstance();
        openGL.setContainer(container);
        renderWindow.addView(openGL);
        const { width, height } = container.getBoundingClientRect();
        openGL.setSize(Math.max(1, Math.floor(width)), Math.max(1, Math.floor(height)));

        const interactor = vtkRenderWindowInteractor.newInstance();
        interactor.setView(openGL);
        interactor.initialize();
        interactor.setContainer(container);
        // Trackball camera gives rotate (drag), zoom (wheel / right-drag) and
        // pan (middle-drag) without hand-rolling any of them.
        interactor.setInteractorStyle(
          vtkInteractorStyleTrackballCamera.newInstance(),
        );

        /* -------------------------------------------------- the MRI volume */
        const mriData = vtkImageData.newInstance();
        mriData.setDimensions(cols, rows, slices);
        // setSpacing takes an array, not three arguments.
        mriData.setSpacing(spacing);
        mriData.getPointData().setScalars(
          vtkDataArray.newInstance({
            name: "mri",
            numberOfComponents: 1,
            values: volume.image,
          }),
        );

        const mriColour = vtkColorTransferFunction.newInstance();
        mriColour.addRGBPoint(0, 0, 0, 0);
        mriColour.addRGBPoint(255, 1, 1, 1);
        const mriAlpha = vtkPiecewiseFunction.newInstance();
        // Air and the darkest tissue stay transparent, so the spine is not buried
        // inside a solid grey block.
        mriAlpha.addPoint(0, 0);
        mriAlpha.addPoint(40, 0);
        mriAlpha.addPoint(255, 1);

        const mriMapper = vtkVolumeMapper.newInstance();
        mriMapper.setInputData(mriData);
        mriMapper.setSampleDistance(0.7);
        const mriActor = vtkVolume.newInstance();
        mriActor.setMapper(mriMapper);
        const mriProperty = mriActor.getProperty();
        mriProperty.setRGBTransferFunction(0, mriColour);
        mriProperty.setScalarOpacity(0, mriAlpha);
        mriProperty.setInterpolationTypeToLinear();
        mriProperty.setScalarOpacityUnitDistance(0, 2.0);
        renderer.addVolume(mriActor);

        /* ------------------------------------------ the label volume */
        const labelData = vtkImageData.newInstance();
        labelData.setDimensions(cols, rows, slices);
        labelData.setSpacing(spacing);
        labelData.getPointData().setScalars(
          vtkDataArray.newInstance({
            name: "labels",
            numberOfComponents: 1,
            values: buildLabelScalars(volume, {
              showSegmentation: true,
              showFindings: true,
              visibleClasses: [1, 2, 3],
              selectedDisc: null,
            }),
          }),
        );

        const labelColour = vtkColorTransferFunction.newInstance();
        const rgb = (hex: string): [number, number, number] => [
          parseInt(hex.slice(1, 3), 16) / 255,
          parseInt(hex.slice(3, 5), 16) / 255,
          parseInt(hex.slice(5, 7), 16) / 255,
        ];
        labelColour.addRGBPoint(0, 0, 0, 0);
        for (const item of SEG_CLASSES) {
          labelColour.addRGBPoint(item.id, ...rgb(item.hex));
        }
        // Same red the 2D findings overlay uses, from the same theme constant.
        labelColour.addRGBPoint(FINDING_CODE, ...rgb(FINDING_OVERLAY.hex));
        labelColour.addRGBPoint(SELECTED_CODE, 1, 1, 1);

        const labelAlpha = vtkPiecewiseFunction.newInstance();
        labelAlpha.addPoint(0, 0);
        labelAlpha.addPoint(0.9, 0);
        labelAlpha.addPoint(1, 0.45);
        labelAlpha.addPoint(3, 0.45);
        labelAlpha.addPoint(FINDING_CODE, 0.7);
        labelAlpha.addPoint(SELECTED_CODE, 0.95);

        const labelMapper = vtkVolumeMapper.newInstance();
        labelMapper.setInputData(labelData);
        labelMapper.setSampleDistance(0.7);
        const labelActor = vtkVolume.newInstance();
        labelActor.setMapper(labelMapper);
        const labelProperty = labelActor.getProperty();
        labelProperty.setRGBTransferFunction(0, labelColour);
        labelProperty.setScalarOpacity(0, labelAlpha);
        // Nearest-neighbour: interpolating label ids would invent classes that
        // are not in the segmentation.
        labelProperty.setInterpolationTypeToNearest();
        labelProperty.setScalarOpacityUnitDistance(0, 1.2);
        renderer.addVolume(labelActor);

        renderer.resetCamera();
        renderer.getActiveCamera().elevation(-20);
        renderer.resetCameraClippingRange();
        renderWindow.render();

        sceneRef.current = {
          renderWindow, renderer, openGL, interactor,
          mriActor, labelActor, mriProperty, labelData, labelMapper, mriMapper,
          setMriOpacity: (value: number) => {
            const fn = vtkPiecewiseFunction.newInstance();
            fn.addPoint(0, 0);
            fn.addPoint(40, 0);
            fn.addPoint(255, value);
            (mriProperty as { setScalarOpacity: (i: number, f: unknown) => void })
              .setScalarOpacity(0, fn);
          },
          resetCamera: () => {
            renderer.resetCamera();
            renderer.getActiveCamera().elevation(-20);
            renderer.resetCameraClippingRange();
            renderWindow.render();
          },
          render: () => renderWindow.render(),
        } as unknown as Scene;

        setStatus("ready");
      } catch (cause) {
        if (cancelled || (cause as Error)?.name === "AbortError") return;
        const message =
          cause instanceof VolumeError
            ? cause.message
            : `The 3D viewer could not be initialised. ${(cause as Error)?.message ?? cause}`;
        setError(message);
        setStatus("failed");
      }
    }

    build();

    return () => {
      cancelled = true;
      controller.abort();
      // VTK objects hold GPU resources and DOM listeners; React unmounting the
      // container is not enough to release either.
      const scene = sceneRef.current;
      if (scene) {
        try {
          scene.interactor.setContainer(null);
          scene.openGL.setContainer(null);
          scene.mriActor.delete();
          scene.labelActor.delete();
          scene.mriMapper.delete();
          scene.labelMapper.delete();
          scene.interactor.delete();
          scene.openGL.delete();
          scene.renderer.delete();
          scene.renderWindow.delete();
        } catch {
          // A partially-built scene must still not block unmount.
        }
        sceneRef.current = null;
      }
      volumeRef.current = null;
    };
  }, [analysisId]);

  /* ---------------------------------- keep the size right on resize */
  React.useEffect(() => {
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      const scene = sceneRef.current;
      if (!scene) return;
      const { width, height } = container.getBoundingClientRect();
      (scene.openGL as unknown as { setSize: (w: number, h: number) => void })
        .setSize(Math.max(1, Math.floor(width)), Math.max(1, Math.floor(height)));
      scene.render();
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [status]);

  /* ------------- rebuild the label scalars when any cue changes */
  React.useEffect(() => {
    const scene = sceneRef.current;
    const volume = volumeRef.current;
    if (!scene || !volume || status !== "ready") return;
    scene.labelData
      .getPointData()
      .getScalars()
      .setData(
        buildLabelScalars(volume, {
          showSegmentation,
          showFindings,
          visibleClasses,
          selectedDisc: highlightDisc ?? null,
        }),
      );
    (scene.labelData as unknown as { modified: () => void }).modified();
    scene.render();
  }, [status, showSegmentation, showFindings, visibleClasses, highlightDisc]);

  /* --------------------------------- MRI visibility and opacity */
  React.useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || status !== "ready") return;
    scene.mriActor.setVisibility(showMri);
    scene.setMriOpacity(mriOpacity);
    scene.render();
  }, [status, showMri, mriOpacity]);

  const toggleClass = (id: number, next: boolean) =>
    setVisibleClasses((current) =>
      next ? [...current, id].sort() : current.filter((v) => v !== id),
    );

  const controlsDisabled = status !== "ready";

  return (
    <div
      className={cn(
        "flex min-h-0 flex-col",
        fullscreen && "fixed inset-0 z-50 bg-surface-0",
      )}
    >
      {/* ------------------------------------------------- toolbar */}
      <div className="flex flex-wrap items-center gap-2 border-b border-line-subtle px-3 py-2">
        <span className="flex items-center gap-1.5 text-xs text-ink-muted">
          <Box className="h-3.5 w-3.5" aria-hidden />
          Volume rendering
        </span>
        {markedDiscs.length > 0 ? (
          <Badge tone={showFindings ? "danger" : "muted"}>
            {markedDiscs.length} finding-associated
          </Badge>
        ) : null}
        <div className="ml-auto flex items-center gap-1">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => sceneRef.current?.resetCamera()}
            disabled={controlsDisabled}
            aria-label="Reset camera"
          >
            <RotateCcw className="h-3.5 w-3.5" aria-hidden />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setFullscreen((v) => !v)}
            aria-label={fullscreen ? "Exit fullscreen" : "Fullscreen"}
          >
            {fullscreen ? (
              <Minimize2 className="h-3.5 w-3.5" aria-hidden />
            ) : (
              <Maximize2 className="h-3.5 w-3.5" aria-hidden />
            )}
          </Button>
        </div>
      </div>

      {/* ------------------------------------------------- canvas */}
      <div className="relative min-h-[320px] flex-1 bg-black">
        <div
          ref={containerRef}
          data-testid="vtk-container"
          className="absolute inset-0"
          role="img"
          aria-label="Three-dimensional volume rendering of the lumbar MRI study"
        />
        {status === "loading" ? (
          <div className="absolute inset-0 grid place-items-center">
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="h-5 w-5 animate-spin text-accent" aria-hidden />
              <p className="text-xs text-ink-muted">Loading volume…</p>
              <Skeleton className="h-1 w-32 rounded-full" />
            </div>
          </div>
        ) : null}
        {status === "failed" ? (
          <div className="absolute inset-0 grid place-items-center px-6">
            <div className="max-w-sm text-center">
              <AlertTriangle
                className="mx-auto h-5 w-5 text-severity-high"
                aria-hidden
              />
              <p className="mt-2 text-sm text-ink">3D view unavailable</p>
              <p className="mt-1 text-xs leading-relaxed text-ink-faint">
                {error}
              </p>
              <p className="mt-2 text-2xs leading-relaxed text-ink-faint">
                The 2D viewer is unaffected and still shows this study.
              </p>
            </div>
          </div>
        ) : null}
        {status === "ready" ? (
          <p className="pointer-events-none absolute bottom-2 left-2 rounded bg-surface-0/85 px-2 py-1 font-mono text-2xs text-ink-faint backdrop-blur">
            drag rotate · wheel zoom · middle-drag pan
            {highlightDisc != null ? ` · disc ${highlightDisc} highlighted` : ""}
          </p>
        ) : null}
      </div>

      {/* ------------------------------------------------- controls */}
      <div className="space-y-2.5 border-t border-line-subtle px-3 py-2.5">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="label-caps">Layers</span>
          <Toggle
            checked={showMri}
            onChange={setShowMri}
            label="MRI"
            disabled={controlsDisabled}
          />
          <Toggle
            checked={showSegmentation}
            onChange={setShowSegmentation}
            label="Segmentation"
            swatch={SEG_CLASSES[1]?.hex}
            disabled={controlsDisabled}
          />
          <Toggle
            checked={showFindings}
            onChange={setShowFindings}
            label="Findings"
            swatch={FINDING_OVERLAY.hex}
            disabled={controlsDisabled || markedDiscs.length === 0}
          />
        </div>

        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="label-caps">Classes</span>
          {SEG_CLASSES.map((item) => (
            <Toggle
              key={item.id}
              checked={visibleClasses.includes(item.id)}
              onChange={(next) => toggleClass(item.id, next)}
              label={item.label}
              swatch={item.hex}
              disabled={controlsDisabled || !showSegmentation}
            />
          ))}
        </div>

        <div className="flex items-center gap-3">
          <label htmlFor="mri-3d-opacity" className="label-caps shrink-0">
            MRI opacity
          </label>
          <input
            id="mri-3d-opacity"
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={mriOpacity}
            disabled={controlsDisabled || !showMri}
            onChange={(event) => setMriOpacity(Number(event.target.value))}
            className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-surface-3 accent-accent disabled:opacity-40"
          />
          <span className="w-9 text-right font-mono text-2xs text-ink-muted">
            {Math.round(mriOpacity * 100)}%
          </span>
        </div>

        {markedDiscs.length > 0 ? (
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="label-caps">Finding discs</span>
            {markedDiscs.map((index) => (
              <button
                key={index}
                type="button"
                onClick={() => onSelectDisc?.(index)}
                aria-pressed={highlightDisc === index}
                className={cn(
                  "rounded border px-1.5 py-0.5 font-mono text-2xs transition-colors",
                  highlightDisc === index
                    ? "border-accent bg-accent/15 text-accent"
                    : "border-line text-ink-muted hover:text-ink",
                )}
              >
                {index}
              </button>
            ))}
            <span className="text-2xs text-ink-faint">
              selecting here also selects it in the 2D view and the disc panel
            </span>
          </div>
        ) : null}

        <p className="text-2xs leading-relaxed text-ink-faint">
          Red marks disc regions associated with a model-estimated finding, drawn
          from the validated disc segmentation. It is not a pixel-level diagnosis
          of damaged tissue.
        </p>
      </div>
    </div>
  );
}

export default Mri3DViewer;
