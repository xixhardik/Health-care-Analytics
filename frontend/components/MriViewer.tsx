/**
 * Slice viewer.
 *
 * Slices are fetched one at a time as PNGs from the backend, so the volume never
 * crosses the network. Neighbouring slices are prefetched into the browser cache
 * so stepping through feels immediate; the backend marks the responses immutable,
 * which makes a revisit free.
 */

"use client";

import {
  ChevronLeft,
  ChevronRight,
  Contrast,
  Eye,
  EyeOff,
  Layers,
  RotateCcw,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import * as React from "react";

import { api } from "@/lib/api";
import { FINDING_OVERLAY, SEG_CLASSES } from "@/lib/theme";
import type { FindingOverlayInfo, SliceMode } from "@/lib/types";
import { Badge, Button, Skeleton, Toggle, cn } from "./ui";

const MODES: { id: SliceMode; label: string; icon: React.ReactNode }[] = [
  { id: "original", label: "MRI", icon: <Contrast className="h-3.5 w-3.5" /> },
  { id: "overlay", label: "Overlay", icon: <Layers className="h-3.5 w-3.5" /> },
  { id: "mask", label: "Mask", icon: <Eye className="h-3.5 w-3.5" /> },
];

const ZOOM_STEPS = [1, 1.5, 2, 3, 4] as const;

export interface MriViewerProps {
  analysisId: string;
  sliceCount: number;
  /** Jump target, e.g. a disc's representative slice. */
  focusSlice?: number | null;
  /**
   * Disc index to mark in the image. Kept in step with the disc selected in the
   * results panel, so the two views always describe the same structure.
   */
  highlightDisc?: number | null;
  /**
   * The server's decision about which disc regions the findings overlay marks.
   * Absent or empty means the control has nothing to show and is not offered.
   */
  findingOverlay?: FindingOverlayInfo | null;
}

export function MriViewer({
  analysisId,
  sliceCount,
  focusSlice,
  highlightDisc,
  findingOverlay,
}: MriViewerProps) {
  const [index, setIndex] = React.useState(() => Math.floor(sliceCount / 2));
  const [mode, setMode] = React.useState<SliceMode>("overlay");
  const [opacity, setOpacity] = React.useState(0.45);
  const [visible, setVisible] = React.useState<number[]>([1, 2, 3]);
  const [zoomStep, setZoomStep] = React.useState(0);
  const [loaded, setLoaded] = React.useState(false);
  const [failed, setFailed] = React.useState(false);
  const [showFindings, setShowFindings] = React.useState(false);
  const [findingOpacity, setFindingOpacity] = React.useState(0.5);

  // Offered only when the server actually marked something. A control that could
  // only ever produce an unchanged image would be misleading.
  const markedDiscs = findingOverlay?.disc_indices ?? [];
  const findingsAvailable = markedDiscs.length > 0;
  const findingsOn = findingsAvailable && showFindings;

  const containerRef = React.useRef<HTMLDivElement>(null);
  const zoom = ZOOM_STEPS[zoomStep] ?? 1;

  // Follow an external focus request (clicking a disc in the results panel).
  React.useEffect(() => {
    if (focusSlice == null) return;
    if (focusSlice < 0 || focusSlice >= sliceCount) return;
    setIndex(focusSlice);
  }, [focusSlice, sliceCount]);

  // One place builds the request, so the preview and the prefetch can never ask
  // for different renderings of the same slice.
  const sliceRequest = React.useCallback(
    (at: number) =>
      api.sliceUrl(analysisId, at, {
        mode,
        opacity,
        classes: visible,
        highlightDisc,
        highlight: findingsOn ? "findings" : "disc",
        findingOpacity,
      }),
    [analysisId, mode, opacity, visible, highlightDisc, findingsOn, findingOpacity],
  );

  const url = React.useMemo(() => sliceRequest(index), [sliceRequest, index]);

  // Prefetch the neighbours so arrow-key stepping does not flash.
  React.useEffect(() => {
    if (mode === "original" && !findingsOn) return;
    for (const offset of [-1, 1, -2, 2]) {
      const neighbour = index + offset;
      if (neighbour < 0 || neighbour >= sliceCount) continue;
      const image = new Image();
      image.src = sliceRequest(neighbour);
    }
  }, [sliceRequest, index, mode, sliceCount, findingsOn]);

  React.useEffect(() => {
    setLoaded(false);
    setFailed(false);
  }, [url]);

  const step = React.useCallback(
    (delta: number) =>
      setIndex((current) =>
        Math.max(0, Math.min(sliceCount - 1, current + delta)),
      ),
    [sliceCount],
  );

  // Arrow keys move through slices whenever the viewer has focus.
  const onKeyDown = (event: React.KeyboardEvent) => {
    switch (event.key) {
      case "ArrowLeft":
      case "ArrowDown":
        event.preventDefault();
        step(-1);
        break;
      case "ArrowRight":
      case "ArrowUp":
        event.preventDefault();
        step(1);
        break;
      case "Home":
        event.preventDefault();
        setIndex(0);
        break;
      case "End":
        event.preventDefault();
        setIndex(sliceCount - 1);
        break;
      case "+":
      case "=":
        event.preventDefault();
        setZoomStep((s) => Math.min(ZOOM_STEPS.length - 1, s + 1));
        break;
      case "-":
        event.preventDefault();
        setZoomStep((s) => Math.max(0, s - 1));
        break;
      case "0":
        event.preventDefault();
        setZoomStep(0);
        break;
      default:
        break;
    }
  };

  const toggleClass = (id: number, next: boolean) =>
    setVisible((current) =>
      next
        ? [...current, id].sort()
        : current.filter((value) => value !== id),
    );

  const overlayControlsDisabled = mode === "original";

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/*
        ---------------------------------------------------- canvas

        First child, and the only one that grows. Everything else below is a
        fixed-height strip, so the image gets whatever the workspace has left.

        `lg:min-h-[200px]` is a floor, not a size. The canvas is `flex-1` with a
        zero basis, which means it is the first thing flexbox takes space away
        from; without a floor, anything that grows underneath it can drive the
        image down to nothing. That is exactly how expanding the findings overlay
        used to flatten the MRI into a strip.
      */}
      <div
        ref={containerRef}
        tabIndex={0}
        onKeyDown={onKeyDown}
        role="group"
        aria-label={`MRI slice viewer, slice ${index + 1} of ${sliceCount}. Use arrow keys to change slice.`}
        className="relative flex min-h-[320px] flex-1 items-center justify-center overflow-auto bg-black focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent lg:min-h-[200px]"
      >
        {!loaded && !failed ? (
          <Skeleton className="absolute inset-6 rounded" />
        ) : null}

        {failed ? (
          <div className="px-6 text-center">
            <EyeOff className="mx-auto h-5 w-5 text-ink-faint" aria-hidden />
            <p className="mt-2 text-xs text-ink-muted">
              This slice could not be rendered.
            </p>
            <Button
              size="sm"
              variant="secondary"
              className="mt-3"
              onClick={() => {
                setFailed(false);
                setLoaded(false);
              }}
            >
              Retry
            </Button>
          </div>
        ) : (
          /* eslint-disable-next-line @next/next/no-img-element */
          <img
            key={url}
            src={url}
            alt={`Lumbar MRI slice ${index + 1} of ${sliceCount}, ${mode} view`}
            onLoad={() => setLoaded(true)}
            onError={() => setFailed(true)}
            draggable={false}
            className={cn(
              "select-none transition-opacity duration-150",
              loaded ? "opacity-100" : "opacity-0",
            )}
            /*
             * At 1x the image fills the available space and `object-contain`
             * keeps the aspect ratio - it letterboxes rather than cropping or
             * stretching. Previously this was a fixed 352x256 box, which left the
             * MRI small in the middle of a large black canvas.
             *
             * Above 1x the size is explicit in pixels so the container's
             * `overflow-auto` gives real pan-and-zoom.
             */
            style={
              zoom === 1
                ? { width: "100%", height: "100%", objectFit: "contain" }
                : {
                    width: `${352 * zoom}px`,
                    height: `${256 * zoom}px`,
                    maxWidth: "none",
                    objectFit: "contain",
                    imageRendering: zoom >= 3 ? "pixelated" : "auto",
                  }
            }
          />
        )}

        <div className="pointer-events-none absolute bottom-2 left-2 flex items-center gap-2 rounded bg-surface-0/85 px-2 py-1 font-mono text-2xs text-ink backdrop-blur">
          <span>
            Slice {index + 1} / {sliceCount}
          </span>
          {highlightDisc != null ? (
            <span className="text-accent">· disc {highlightDisc} marked</span>
          ) : null}
          {findingsOn ? (
            <span style={{ color: FINDING_OVERLAY.hex }}>
              · findings overlay on
            </span>
          ) : null}
        </div>
      </div>

      {/*
        ---------------------------------------------------- control bar

        Rendering mode, slice position and zoom, on one row. These were two rows -
        one above the image and one below - which cost the canvas a whole row of
        height for no functional gain. `flex-wrap` splits them back onto separate
        lines when the column is too narrow, so nothing is ever pushed out of
        reach on a small screen.
      */}
      <div className="flex flex-wrap items-center gap-2 border-t border-line-subtle px-3 py-2">
        <div
          className="flex rounded-md border border-line bg-surface-2 p-0.5"
          role="group"
          aria-label="Rendering mode"
        >
          {MODES.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setMode(item.id)}
              aria-pressed={mode === item.id}
              className={cn(
                "inline-flex items-center gap-1.5 rounded px-2 py-1 text-2xs font-medium transition-colors",
                mode === item.id
                  ? "bg-accent/15 text-accent"
                  : "text-ink-faint hover:text-ink",
              )}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </div>

        <div className="flex min-w-[9rem] flex-1 items-center gap-2">
          <Button
            size="sm"
            variant="secondary"
            onClick={() => step(-1)}
            disabled={index === 0}
            aria-label="Previous slice"
          >
            <ChevronLeft className="h-3.5 w-3.5" aria-hidden />
          </Button>
          <input
            type="range"
            min={0}
            max={Math.max(0, sliceCount - 1)}
            value={index}
            onChange={(event) => setIndex(Number(event.target.value))}
            aria-label="Slice position"
            className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-surface-3 accent-accent"
          />
          <Button
            size="sm"
            variant="secondary"
            onClick={() => step(1)}
            disabled={index >= sliceCount - 1}
            aria-label="Next slice"
          >
            <ChevronRight className="h-3.5 w-3.5" aria-hidden />
          </Button>
        </div>

        <div className="flex items-center gap-1">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setZoomStep((s) => Math.max(0, s - 1))}
            disabled={zoomStep === 0}
            aria-label="Zoom out"
          >
            <ZoomOut className="h-3.5 w-3.5" aria-hidden />
          </Button>
          <span className="w-10 text-center font-mono text-2xs text-ink-muted">
            {zoom.toFixed(1)}x
          </span>
          <Button
            size="sm"
            variant="ghost"
            onClick={() =>
              setZoomStep((s) => Math.min(ZOOM_STEPS.length - 1, s + 1))
            }
            disabled={zoomStep === ZOOM_STEPS.length - 1}
            aria-label="Zoom in"
          >
            <ZoomIn className="h-3.5 w-3.5" aria-hidden />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setZoomStep(0)}
            disabled={zoomStep === 0}
            aria-label="Reset zoom"
          >
            <RotateCcw className="h-3.5 w-3.5" aria-hidden />
          </Button>
        </div>
      </div>

      {/*
        ---------------------------------------------------- overlay controls

        A fixed-height strip on desktop that scrolls its own contents.

        This is the whole fix. Switching the findings overlay on adds roughly
        130px of slider, legend and explanation, and while this block was free to
        grow it took that height straight out of the canvas - so turning the
        overlay on flattened the MRI. A constant height means the canvas is the
        same size with the overlay on and off. Nothing is deleted and nothing is
        permanently hidden: the explanation is still rendered, in full, and
        reachable by scrolling this strip.

        The height is keyed on `findingsAvailable`, which the server decides once
        per study, and deliberately not on `showFindings`. The user's toggle
        therefore cannot change this element's box at all. A study with no
        findings gets no cap, so it does not show a half-empty panel.

        `scrollbar-gutter: stable` reserves the scrollbar's width up front. Without
        it the scrollbar appears only once the overlay expands, which narrows the
        rows and reflows them - a visible twitch for no reason.

        Below `lg` there is no cap: the page is a scrolling document there, and a
        scroll region inside a scrolling page is worse than a taller page.
      */}
      <div
        data-testid="viewer-controls"
        className={cn(
          "space-y-2.5 border-t border-line-subtle px-3 py-2.5 lg:space-y-2 lg:py-2",
          findingsAvailable &&
            "lg:h-[7.5rem] lg:overflow-y-auto lg:[scrollbar-gutter:stable]",
        )}
      >
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="label-caps">Classes</span>
          {SEG_CLASSES.map((item) => (
            <Toggle
              key={item.id}
              checked={visible.includes(item.id)}
              onChange={(next) => toggleClass(item.id, next)}
              label={item.label}
              swatch={item.hex}
              disabled={overlayControlsDisabled}
            />
          ))}
          {overlayControlsDisabled ? (
            <Badge tone="muted">Switch to Overlay or Mask to filter</Badge>
          ) : null}
        </div>

        <div className="flex items-center gap-3">
          <label
            htmlFor="overlay-opacity"
            className="label-caps shrink-0"
          >
            Opacity
          </label>
          <input
            id="overlay-opacity"
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={opacity}
            disabled={mode !== "overlay"}
            onChange={(event) => setOpacity(Number(event.target.value))}
            className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-surface-3 accent-accent disabled:opacity-40"
          />
          <span className="w-9 text-right font-mono text-2xs text-ink-muted">
            {Math.round(opacity * 100)}%
          </span>
        </div>

        {/* ------------------------------------------ findings overlay */}
        {findingsAvailable ? (
          <div className="space-y-2 rounded-md border border-line-subtle bg-surface-2/40 px-2.5 py-2">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
              <span className="label-caps">Findings Overlay</span>
              <Toggle
                checked={showFindings}
                onChange={setShowFindings}
                label={showFindings ? "On" : "Off"}
                swatch={FINDING_OVERLAY.hex}
              />
              <Badge tone={findingsOn ? "danger" : "muted"}>
                {markedDiscs.length} disc
                {markedDiscs.length === 1 ? "" : "s"} marked
              </Badge>
            </div>

            {findingsOn ? (
              <>
                <div className="flex items-center gap-3">
                  <label
                    htmlFor="finding-opacity"
                    className="label-caps shrink-0"
                  >
                    Finding
                  </label>
                  <input
                    id="finding-opacity"
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={findingOpacity}
                    onChange={(event) =>
                      setFindingOpacity(Number(event.target.value))
                    }
                    className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-surface-3 disabled:opacity-40"
                    style={{ accentColor: FINDING_OVERLAY.hex }}
                  />
                  <span className="w-9 text-right font-mono text-2xs text-ink-muted">
                    {Math.round(findingOpacity * 100)}%
                  </span>
                </div>

                {/* Legend: red / green / white, in the app's existing idiom. */}
                <ul
                  className="flex flex-wrap items-center gap-x-3 gap-y-1"
                  aria-label="Overlay legend"
                >
                  <li className="flex items-center gap-1.5 text-2xs text-ink-muted">
                    <span
                      aria-hidden
                      className="h-2.5 w-2.5 shrink-0 rounded-sm"
                      style={{ backgroundColor: FINDING_OVERLAY.hex }}
                    />
                    {FINDING_OVERLAY.label}
                  </li>
                  <li className="flex items-center gap-1.5 text-2xs text-ink-muted">
                    <span
                      aria-hidden
                      className="h-2.5 w-2.5 shrink-0 rounded-sm"
                      style={{ backgroundColor: "#34d399" }}
                    />
                    Anatomical segmentation
                  </li>
                  <li className="flex items-center gap-1.5 text-2xs text-ink-muted">
                    <span
                      aria-hidden
                      className="h-2.5 w-2.5 shrink-0 rounded-sm border border-line bg-white"
                    />
                    MRI
                  </li>
                </ul>

                {/*
                  The single most important sentence in this component: it stops
                  the red being read as tissue-level pathology, which the system
                  has no model for.
                */}
                <p className="text-2xs leading-relaxed text-ink-faint">
                  {findingOverlay?.note ??
                    "Red marks disc regions associated with a model-estimated finding. It is not a pixel-level diagnosis of damaged tissue."}
                </p>
                <p className="text-2xs leading-relaxed text-ink-faint">
                  Marked from the validated disc segmentation for disc{" "}
                  <span className="font-mono text-ink-muted">
                    {markedDiscs.join(", ")}
                  </span>
                  . Discs without a supported positive finding are left unmarked.
                </p>
              </>
            ) : null}
          </div>
        ) : null}

        <p className="text-2xs text-ink-faint">
          Arrow keys change slice · +/- zoom · 0 resets · rendered at 1.0 mm/pixel
        </p>
      </div>
    </div>
  );
}
