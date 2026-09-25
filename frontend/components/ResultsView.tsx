/**
 * Results workspace.
 *
 * Desktop: viewer on the left, disc results on the right. Mobile gives the viewer
 * priority and moves the findings into a bottom sheet, rather than stacking a
 * shrunken desktop layout.
 */

"use client";

import {
  Activity,
  ChevronUp,
  Download,
  FileText,
  FlaskConical,
  Info,
  Layers,
  Loader2,
  X,
} from "lucide-react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useRouter } from "next/navigation";
import * as React from "react";

import { api, formatSeconds } from "@/lib/api";
import { PROVENANCE_STYLES, SEVERITY_STYLES, SEG_CLASSES } from "@/lib/theme";
import type {
  AnalysisResult,
  DemoStageRef,
  DiscResult,
  LongitudinalCase,
} from "@/lib/types";
import { DiscDetail, DiscList } from "./DiscPanel";

/**
 * VTK.js touches `window` and WebGL at import time, so it must not be part of a
 * server render, and it should not be in the initial bundle for users who never
 * open the 3D tab.
 */
const Mri3DViewer = dynamic(
  () => import("./Mri3DViewer").then((m) => m.Mri3DViewer),
  {
    ssr: false,
    loading: () => (
      <div className="grid h-full min-h-[320px] place-items-center bg-black">
        <p className="text-xs text-ink-muted">Loading 3D viewer…</p>
      </div>
    ),
  },
);
import { MriViewer } from "./MriViewer";
import { Badge, Button, FieldRow, Panel, PanelHeader, cn } from "./ui";

function ResearchBanner() {
  return (
    <div className="flex items-start gap-2.5 border-b border-seg-disc/25 bg-seg-disc/5 px-4 py-2.5 md:px-6">
      <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-seg-disc" aria-hidden />
      <p className="text-2xs leading-relaxed text-ink-muted">
        <span className="font-medium text-ink">
          Research / educational prototype — results require expert radiological
          review.
        </span>{" "}
        Findings are model-derived research estimates. This system does not provide
        a clinical diagnosis and is not a medical device.
      </p>
    </div>
  );
}

/**
 * Legend for the three provenance categories.
 *
 * Shown once above the results so the tags on every value are immediately
 * readable, and so "not modelled" is visibly distinct from "not detected".
 */
function ProvenanceLegend() {
  return (
    <dl className="grid gap-x-4 gap-y-1.5 px-4 py-3 sm:grid-cols-3">
      {(
        [
          ["segmentation_derived", "Measured from the predicted segmentation."],
          ["model_prediction", "Estimated by a research model."],
          ["unsupported", "No validated model — nothing is reported."],
        ] as const
      ).map(([key, hint]) => {
        const style = PROVENANCE_STYLES[key];
        return (
          <div key={key}>
            <dt>
              <Badge tone={style!.tone}>{style!.label}</Badge>
            </dt>
            <dd className="mt-1 text-2xs leading-relaxed text-ink-faint">{hint}</dd>
          </div>
        );
      })}
    </dl>
  );
}

function FindingsSummary({ result }: { result: AnalysisResult }) {
  const { summary } = result;
  return (
    <div className="space-y-3 px-4 py-3">
      <div className="grid grid-cols-3 gap-2">
        <div>
          <p className="label-caps">Discs</p>
          <p className="font-mono text-lg text-ink">{summary.disc_count}</p>
        </div>
        <div>
          <p className="label-caps">With findings</p>
          <p className="font-mono text-lg text-seg-disc">
            {summary.discs_with_findings_count}
          </p>
        </div>
        <div>
          <p className="label-caps">Mean height</p>
          <p className="font-mono text-lg text-ink">
            {summary.mean_disc_height_mm != null
              ? summary.mean_disc_height_mm.toFixed(2)
              : "—"}
            <span className="ml-1 text-2xs text-ink-faint">mm</span>
          </p>
        </div>
      </div>

      <ul className="space-y-2" role="list">
        {summary.findings.map((finding, i) => {
          const style = SEVERITY_STYLES[finding.severity];
          return (
            <li key={i} className="flex gap-2.5">
              <span
                className={cn("mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full", style.dot)}
                aria-hidden
              />
              <span className="min-w-0">
                <span className="block text-2xs font-medium uppercase tracking-wide text-ink-faint">
                  {finding.category}
                </span>
                <span className="block text-xs leading-relaxed text-ink-muted">
                  {finding.text}
                </span>
              </span>
            </li>
          );
        })}
      </ul>

      {Object.keys(summary.unsupported_targets).length > 0 ? (
        <div className="rounded border border-line-subtle bg-surface-2/50 p-2.5">
          <p className="label-caps mb-1.5">Not reported by design</p>
          <ul className="space-y-1.5" role="list">
            {Object.entries(summary.unsupported_targets).map(([name, reason]) => (
              <li key={name} className="text-2xs leading-relaxed text-ink-faint">
                <code className="text-ink-muted">{name}</code> — {reason}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function StudyMeta({ result }: { result: AnalysisResult }) {
  const { study, processing, segmentation } = result;
  return (
    <div className="px-4 py-3">
      <dl>
        <FieldRow label="File" value={study.filename} />
        <FieldRow
          label="Modality"
          value={
            study.modality ? (
              <span className="font-mono uppercase">{study.modality}</span>
            ) : (
              "not determinable"
            )
          }
        />
        <FieldRow label="Slices" value={study.slice_count} mono />
        <FieldRow
          label="Measurable slices"
          value={segmentation.informative_slice_count}
          mono
        />
        <FieldRow
          label="Native size"
          value={`${study.dimensions.join(" × ")} px`}
          mono
        />
        <FieldRow label="Timepoint" value={result.timepoint.label} />
        <FieldRow label="Model" value={processing.segmentation_source} />
        <FieldRow label="Post-processing" value={processing.postprocessing_method} />
        <FieldRow label="Pipeline version" value={result.pipeline.version} mono />
        <FieldRow label="Device" value={processing.device.toUpperCase()} mono />
        <FieldRow
          label="Duration"
          value={formatSeconds(processing.duration_seconds)}
          mono
        />
        <FieldRow label="Disc tracks" value={processing.disc_tracks_found ?? "—"} mono />
      </dl>
    </div>
  );
}

/**
 * The simulated-timeline disclaimer, shown in the actual interface.
 *
 * Deliberately not collapsible and not dismissible. Anyone looking at a stage is
 * looking at a real scan of one patient standing in for a timeline position, and
 * must be told so without having to open anything.
 */
function DemoStageBanner({ stage }: { stage: DemoStageRef }) {
  return (
    <div className="border-b border-severity-high/40 bg-severity-high/10 px-4 py-2.5 md:px-6">
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
        <span className="inline-flex items-center gap-1.5 rounded bg-severity-high/20 px-2 py-0.5 text-2xs font-semibold uppercase tracking-wide text-severity-high">
          <FlaskConical className="h-3 w-3" aria-hidden />
          Simulated Longitudinal Demo
        </span>
        <span className="text-xs font-medium text-ink">
          Demo stage: {stage.label}
        </span>
        <Badge tone="muted">{stage.research_status}</Badge>
        <Badge tone="muted">{stage.display_reference}</Badge>
      </div>
      <p className="mt-1.5 text-2xs leading-relaxed text-ink-muted">
        {stage.ui_notice}
      </p>
      <p className="mt-1 text-2xs leading-relaxed text-ink-faint">
        {stage.disclaimer}
      </p>
    </div>
  );
}

/**
 * Stage switcher. Each stage loads a *different* real study through the real
 * pipeline, so switching genuinely changes the volume rather than relabelling one.
 */
function DemoStageSwitcher({ stage }: { stage: DemoStageRef }) {
  const router = useRouter();
  const [demoCase, setDemoCase] = React.useState<LongitudinalCase | null>(null);
  const [busy, setBusy] = React.useState<string | null>(null);
  const [failed, setFailed] = React.useState<string | null>(null);

  React.useEffect(() => {
    let alive = true;
    api
      .longitudinalCase(stage.case_id)
      .then((value) => alive && setDemoCase(value))
      .catch(() => alive && setDemoCase(null));
    return () => {
      alive = false;
    };
  }, [stage.case_id]);

  const open = async (stageId: string) => {
    setBusy(stageId);
    setFailed(null);
    try {
      const created = await api.loadDemoStage(stage.case_id, stageId);
      router.push(`/analysis/${created.analysis_id}`);
    } catch (cause) {
      setFailed(cause instanceof Error ? cause.message : "Could not load that stage.");
      setBusy(null);
    }
  };

  if (!demoCase) return null;

  return (
    <div className="border-b border-line-subtle bg-surface-1/60 px-4 py-2 md:px-6">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5">
        <span className="label-caps shrink-0">Demo stages</span>
        {demoCase.stages.map((item) => {
          const current = item.stage_id === stage.stage_id;
          return (
            <button
              key={item.stage_id}
              type="button"
              disabled={!item.available || busy !== null || current}
              onClick={() => open(item.stage_id)}
              aria-current={current ? "step" : undefined}
              title={
                item.available
                  ? `${item.label} — ${item.research_status}`
                  : item.unavailable_reason ?? "Unavailable"
              }
              className={cn(
                "inline-flex items-center gap-1.5 rounded border px-2 py-1 text-2xs transition-colors",
                current
                  ? "border-accent bg-accent/15 font-medium text-accent"
                  : item.available
                    ? "border-line text-ink-muted hover:text-ink"
                    : "border-line-subtle text-ink-faint opacity-50",
              )}
            >
              <span className="font-mono">{item.order}</span>
              {item.label}
              {busy === item.stage_id ? (
                <Loader2 className="h-3 w-3 animate-spin" aria-hidden />
              ) : null}
            </button>
          );
        })}
        <span className="text-2xs text-ink-faint">
          each stage is a different SPIDER study, processed by the same pipeline
        </span>
      </div>
      {failed ? (
        <p className="mt-1 text-2xs text-severity-high">{failed}</p>
      ) : null}
    </div>
  );
}

export function ResultsView({ result }: { result: AnalysisResult }) {
  const [selected, setSelected] = React.useState<DiscResult | null>(null);
  const [sheetOpen, setSheetOpen] = React.useState(false);
  const [view, setView] = React.useState<"2d" | "3d">("2d");

  const focusSlice = selected?.representative_slice_index ?? null;

  const select = (disc: DiscResult) => {
    setSelected(disc);
    setSheetOpen(false);
  };

  return (
    <div className="flex min-h-0 flex-col">
      <ResearchBanner />
      {result.demo_stage ? <DemoStageBanner stage={result.demo_stage} /> : null}
      {result.demo_stage ? <DemoStageSwitcher stage={result.demo_stage} /> : null}

      {/* ------------------------------------------------- header row */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line-subtle px-4 py-3 md:px-6">
        <div className="min-w-0">
          <h1 className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm font-semibold text-ink">
            <Activity className="h-4 w-4 shrink-0 text-accent" aria-hidden />
            <span className="truncate">{result.study.filename}</span>
            {result.is_sample ? (
              <Badge tone="accent">Sample research study</Badge>
            ) : null}
          </h1>
          <p className="mt-0.5 flex flex-wrap items-center gap-x-2 text-2xs text-ink-faint">
            <span>{result.summary.disc_count} discs</span>
            <span aria-hidden>·</span>
            <span>{result.study.slice_count} slices</span>
            <span aria-hidden>·</span>
            <span className="font-mono text-ink-muted">
              {result.pipeline.version}
            </span>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link href={`/reports/${result.analysis_id}`}>
            <Button size="sm" variant="secondary" icon={<FileText className="h-3.5 w-3.5" />}>
              Report
            </Button>
          </Link>
          <a href={api.downloadUrl(result.analysis_id, "md")} download>
            <Button size="sm" variant="primary" icon={<Download className="h-3.5 w-3.5" />}>
              Download
            </Button>
          </a>
        </div>
      </div>

      {/* ------------------------------------------------- workspace */}
      <div className="grid min-h-0 flex-1 lg:grid-cols-[minmax(0,1fr)_380px] xl:grid-cols-[minmax(0,1fr)_420px]">
        {/* viewer */}
        <div className="flex min-h-0 flex-col border-b border-line-subtle lg:border-b-0 lg:border-r">
          <div className="flex items-center justify-between gap-2 px-4 py-2">
            <span className="flex items-center gap-1.5 text-xs text-ink-muted">
              <Layers className="h-3.5 w-3.5" aria-hidden />
              Segmentation overlay
            </span>
            <div
              className="flex rounded-md border border-line bg-surface-2 p-0.5"
              role="group"
              aria-label="Viewer dimension"
            >
              {(["2d", "3d"] as const).map((id) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setView(id)}
                  aria-pressed={view === id}
                  className={cn(
                    "rounded px-2 py-1 text-2xs font-medium uppercase transition-colors",
                    view === id
                      ? "bg-accent/15 text-accent"
                      : "text-ink-faint hover:text-ink",
                  )}
                >
                  {id === "2d" ? "2D View" : "3D View"}
                </button>
              ))}
            </div>
            <span className="flex flex-wrap items-center gap-2">
              {SEG_CLASSES.map((item) => {
                const dice = result.segmentation.classes.find(
                  (c) => c.id === item.id,
                )?.validated_test_dice;
                return (
                  <span key={item.id} className="flex items-center gap-1">
                    <span
                      aria-hidden
                      className="h-2 w-2 rounded-sm"
                      style={{ backgroundColor: item.hex }}
                    />
                    <span className="text-2xs text-ink-faint">
                      {item.label}
                      {dice != null ? (
                        <span className="ml-1 font-mono">{dice.toFixed(3)}</span>
                      ) : null}
                    </span>
                  </span>
                );
              })}
            </span>
          </div>
          <div className="min-h-[380px] flex-1">
            {/*
              Both viewers read the same `selected` disc and the same
              server-decided finding overlay, so they cannot disagree. The 2D
              viewer is mounted unchanged; only its visibility is switched, which
              preserves its slice position and prefetch cache across tab flips.
            */}
            <div className={view === "2d" ? "h-full" : "hidden"}>
              <MriViewer
                analysisId={result.analysis_id}
                sliceCount={result.study.slice_count}
                focusSlice={focusSlice}
                highlightDisc={selected?.index ?? null}
                findingOverlay={result.finding_overlay}
              />
            </div>
            {view === "3d" ? (
              <Mri3DViewer
                analysisId={result.analysis_id}
                highlightDisc={selected?.index ?? null}
                findingOverlay={result.finding_overlay}
                onSelectDisc={(index) => {
                  const disc = result.discs.find((d) => d.index === index);
                  if (disc) select(disc);
                }}
              />
            ) : null}
          </div>
        </div>

        {/* right column: desktop */}
        <aside className="hidden min-h-0 flex-col gap-3 overflow-y-auto p-3 lg:flex">
          <Panel>
            <PanelHeader
              title="Disc-level findings"
              subtitle={
                selected
                  ? `Disc ${selected.index} selected and marked in the viewer`
                  : "Select a disc for detail"
              }
            />
            <DiscList
              discs={result.discs}
              selectedIndex={selected?.index ?? null}
              onSelect={select}
              markedIndices={result.finding_overlay?.disc_indices}
            />
          </Panel>

          {selected ? (
            <DiscDetail
              disc={selected}
              allDiscs={result.discs}
              onClose={() => setSelected(null)}
            />
          ) : (
            <>
              <Panel>
                <PanelHeader
                  title="How to read these results"
                  subtitle="Every value is tagged with how it was obtained"
                />
                <ProvenanceLegend />
              </Panel>

              <Panel>
                <PanelHeader title="Observed findings" />
                <FindingsSummary result={result} />
              </Panel>

              <Panel>
                <PanelHeader title="Study & processing" />
                <StudyMeta result={result} />
              </Panel>
            </>
          )}
        </aside>
      </div>

      {/* ------------------------------------------------- mobile bottom sheet */}
      <div className="lg:hidden">
        <button
          type="button"
          onClick={() => setSheetOpen(true)}
          className="fixed inset-x-0 bottom-14 z-30 flex items-center justify-center gap-2 border-t border-line bg-surface-1/95 py-2.5 text-xs text-ink backdrop-blur"
          aria-expanded={sheetOpen}
        >
          <ChevronUp className="h-3.5 w-3.5" aria-hidden />
          {result.summary.disc_count} disc findings
        </button>

        {sheetOpen ? (
          <div className="fixed inset-0 z-50 flex flex-col justify-end">
            <div
              className="absolute inset-0 bg-surface-0/80 backdrop-blur-sm"
              onClick={() => setSheetOpen(false)}
              aria-hidden
            />
            <div className="relative max-h-[82vh] animate-fade-up overflow-y-auto rounded-t-xl border-t border-line bg-surface-1">
              <div className="sticky top-0 flex items-center justify-between border-b border-line-subtle bg-surface-1 px-4 py-3">
                <span className="text-sm font-medium text-ink">
                  Disc-level findings
                </span>
                <button
                  type="button"
                  onClick={() => setSheetOpen(false)}
                  className="rounded p-1.5 text-ink-muted hover:bg-surface-2"
                  aria-label="Close findings"
                >
                  <X className="h-4 w-4" aria-hidden />
                </button>
              </div>
              <DiscList
                discs={result.discs}
                selectedIndex={selected?.index ?? null}
                onSelect={select}
                markedIndices={result.finding_overlay?.disc_indices}
              />
              <ProvenanceLegend />
              <FindingsSummary result={result} />
              <StudyMeta result={result} />
            </div>
          </div>
        ) : null}

        {selected ? (
          <div className="fixed inset-0 z-[60] flex flex-col justify-end">
            <div
              className="absolute inset-0 bg-surface-0/80 backdrop-blur-sm"
              onClick={() => setSelected(null)}
              aria-hidden
            />
            <div className="relative max-h-[85vh] animate-fade-up overflow-hidden rounded-t-xl">
              <DiscDetail
                disc={selected}
                allDiscs={result.discs}
                onClose={() => setSelected(null)}
              />
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
