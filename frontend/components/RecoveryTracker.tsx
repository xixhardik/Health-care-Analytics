/**
 * Longitudinal Recovery Tracker.
 *
 * A demonstration of how a longitudinal imaging workflow would operate, built on
 * four *different* real SPIDER studies from four *different* patients. The imaging
 * and every measurement are real pipeline output; the timeline is simulated.
 *
 * The viewers are not reimplemented here. `MriViewer` and `Mri3DViewer` are the
 * same components the analysis workspace uses, driven by the stage's analysis id
 * and the shared selected-disc state, so a stage change updates both and the
 * disc selection stays consistent.
 */

"use client";

import {
  ArrowDown,
  Box,
  Check,
  CircleDot,
  FlaskConical,
  Image as ImageIcon,
  Layers,
  Loader2,
  Minus,
  Ruler,
  Scan,
  TrendingDown,
  TrendingUp,
  X,
} from "lucide-react";
import dynamic from "next/dynamic";
import * as React from "react";

import {
  COMPARISON_CAVEAT,
  COMPARISON_FIELDS,
  CROSS_STUDY_DISCLAIMER,
  IMAGING_STAGE_IDS,
  LONGITUDINAL_UNAVAILABLE,
  SIMULATED_BADGE,
  TIMELINE,
  computeStageMetrics,
  formatMetric,
  metricDelta,
  surgicalConsideration,
  workflowSummary,
  type StageMetrics,
  type TimelineNode,
} from "@/lib/recovery";
import { FINDING_OVERLAY } from "@/lib/theme";
import type {
  AnalysisResult,
  DiscResult,
  LongitudinalCase,
  LongitudinalStage,
} from "@/lib/types";
import { DiscList } from "./DiscPanel";
import { MriViewer } from "./MriViewer";
import { Badge, Button, Panel, PanelHeader, Skeleton, cn } from "./ui";

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

/** One stage's loaded analysis, or why it is not loaded. */
export interface StageState {
  status: "pending" | "loading" | "ready" | "failed";
  analysisId?: string;
  result?: AnalysisResult;
  error?: string;
}

export interface RecoveryTrackerProps {
  demoCase: LongitudinalCase;
  stages: Record<string, StageState>;
  activeNodeId: string;
  onSelectNode: (nodeId: string) => void;
  /** Progress text while stages are still being analysed. */
  loadingLabel?: string | null;
}

/* -------------------------------------------------------------------------- */
/* Disclaimer                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Permanently visible. Not collapsible, not dismissible, and not delegated to the
 * methodology page: anyone reading a stage must be told what these stages are.
 */
function StandingDisclaimer({ demoCase }: { demoCase: LongitudinalCase }) {
  return (
    <div className="border-b border-severity-high/40 bg-severity-high/10 px-4 py-3 md:px-6">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <span className="inline-flex items-center gap-1.5 rounded bg-severity-high/20 px-2 py-0.5 text-2xs font-semibold uppercase tracking-wide text-severity-high">
          <FlaskConical className="h-3 w-3" aria-hidden />
          {SIMULATED_BADGE}
        </span>
        <h1 className="text-sm font-semibold text-ink">
          Longitudinal Recovery Tracker
        </h1>
        <Badge tone="muted">Research prototype</Badge>
      </div>
      <p className="mt-2 max-w-4xl text-xs leading-relaxed text-ink-muted">
        {CROSS_STUDY_DISCLAIMER}
      </p>
      <p className="mt-1 max-w-4xl text-2xs leading-relaxed text-ink-faint">
        {demoCase.disclaimer}
      </p>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Timeline                                                                    */
/* -------------------------------------------------------------------------- */

function StageCard({
  node,
  stage,
  state,
  active,
  onSelect,
}: {
  node: TimelineNode;
  stage: LongitudinalStage | undefined;
  state: StageState | undefined;
  active: boolean;
  onSelect: () => void;
}) {
  const result = state?.result;
  const metrics = result ? computeStageMetrics(result) : null;

  return (
    <li className="flex flex-col items-stretch">
      <button
        type="button"
        onClick={onSelect}
        aria-current={active ? "step" : undefined}
        aria-label={`${node.label} — ${node.researchStatus}`}
        className={cn(
          "group w-full rounded-lg border p-3 text-left transition-all",
          active
            ? "border-accent bg-accent/10 shadow-[0_0_0_1px] shadow-accent/40"
            : "border-line bg-surface-1 hover:border-line-strong hover:bg-surface-2",
        )}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p
              className={cn(
                "truncate text-xs font-semibold",
                active ? "text-accent" : "text-ink",
              )}
            >
              {node.label}
            </p>
            <p className="mt-0.5 text-2xs text-ink-faint">{node.researchStatus}</p>
          </div>
          {active ? (
            <CircleDot className="h-3.5 w-3.5 shrink-0 text-accent" aria-hidden />
          ) : null}
        </div>

        {/* Non-identifying reference; the patient id stays in provenance. */}
        {stage ? (
          <p className="mt-2 text-2xs text-ink-muted">{stage.display_reference}</p>
        ) : (
          <p className="mt-2 text-2xs italic text-ink-faint">
            No separate study
          </p>
        )}

        <dl className="mt-2 space-y-0.5">
          <div className="flex items-center justify-between gap-2">
            <dt className="text-2xs text-ink-faint">MRI</dt>
            <dd className="text-2xs">
              {!node.hasOwnImaging ? (
                <span className="text-ink-faint">baseline imaging</span>
              ) : state?.status === "ready" ? (
                <span className="text-seg-canal">available</span>
              ) : state?.status === "loading" ? (
                <span className="text-ink-muted">analysing…</span>
              ) : state?.status === "failed" ? (
                <span className="text-severity-high">unavailable</span>
              ) : (
                <span className="text-ink-faint">queued</span>
              )}
            </dd>
          </div>

          {metrics ? (
            <>
              <div className="flex items-center justify-between gap-2">
                <dt className="text-2xs text-ink-faint">Segmented discs</dt>
                <dd className="font-mono text-2xs text-ink">{metrics.discCount}</dd>
              </div>
              <div className="flex items-center justify-between gap-2">
                <dt className="text-2xs text-ink-faint">Finding-associated</dt>
                <dd
                  className="font-mono text-2xs"
                  style={{ color: FINDING_OVERLAY.hex }}
                >
                  {metrics.findingDiscCount}
                </dd>
              </div>
              <div className="flex items-center justify-between gap-2">
                <dt className="text-2xs text-ink-faint">Mean disc height</dt>
                <dd className="font-mono text-2xs text-ink">
                  {metrics.meanDiscHeightMm != null
                    ? `${metrics.meanDiscHeightMm.toFixed(2)} mm`
                    : "unavailable"}
                </dd>
              </div>
            </>
          ) : state?.status === "loading" ? (
            <Skeleton className="mt-1 h-8 w-full rounded" />
          ) : null}
        </dl>
      </button>
    </li>
  );
}

function RecoveryTimeline({
  demoCase,
  stages,
  activeNodeId,
  onSelectNode,
}: Pick<RecoveryTrackerProps, "demoCase" | "stages" | "activeNodeId" | "onSelectNode">) {
  const stageById = React.useMemo(
    () => new Map(demoCase.stages.map((s) => [s.stage_id, s])),
    [demoCase.stages],
  );

  return (
    <section
      aria-label="Demonstration timeline"
      className="border-b border-line-subtle px-4 py-3 md:px-6"
    >
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <h2 className="label-caps">Demonstration timeline</h2>
        <Badge tone="muted">{TIMELINE.length} stages</Badge>
        <span className="text-2xs text-ink-faint">
          select a stage to load it in the viewers below
        </span>
      </div>

      <ol
        role="list"
        className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5"
      >
        {TIMELINE.map((node, index) => (
          <React.Fragment key={node.id}>
            <StageCard
              node={node}
              stage={node.stageId ? stageById.get(node.stageId) : undefined}
              state={node.stageId ? stages[node.stageId] : undefined}
              active={node.id === activeNodeId}
              onSelect={() => onSelectNode(node.id)}
            />
            {/* Flow arrow between cards, on wide layouts only. */}
            {index < TIMELINE.length - 1 ? (
              <li
                aria-hidden
                className="hidden items-center justify-center xl:hidden"
              >
                <ArrowDown className="h-3 w-3 text-ink-faint" />
              </li>
            ) : null}
          </React.Fragment>
        ))}
      </ol>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* Comparison                                                                  */
/* -------------------------------------------------------------------------- */

function DeltaCell({
  previous,
  current,
  field,
}: {
  previous: number | null | undefined;
  current: number | null | undefined;
  field: (typeof COMPARISON_FIELDS)[number];
}) {
  const delta = metricDelta(previous, current, field);
  const Icon =
    delta.direction === "up"
      ? TrendingUp
      : delta.direction === "down"
        ? TrendingDown
        : Minus;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 font-mono text-2xs",
        delta.direction === "unknown" ? "text-ink-faint" : "text-ink-muted",
      )}
      title="Difference between two different studies, not a change over time"
    >
      {delta.direction !== "unknown" ? (
        <Icon className="h-3 w-3" aria-hidden />
      ) : null}
      {delta.text}
    </span>
  );
}

function RecoveryComparison({
  demoCase,
  stages,
}: Pick<RecoveryTrackerProps, "demoCase" | "stages">) {
  const stageById = new Map(demoCase.stages.map((s) => [s.stage_id, s]));
  const columns = IMAGING_STAGE_IDS.map((stageId) => ({
    stageId,
    stage: stageById.get(stageId),
    state: stages[stageId],
    metrics: stages[stageId]?.result
      ? computeStageMetrics(stages[stageId]!.result!)
      : null,
  }));

  const ready = columns.filter((c) => c.metrics).length;

  return (
    <Panel>
      <PanelHeader
        title="Cross-study comparison"
        subtitle={COMPARISON_CAVEAT}
        actions={
          <Badge tone={ready === columns.length ? "muted" : "accent"}>
            {ready}/{columns.length} stages analysed
          </Badge>
        }
      />

      <div className="px-4 pb-2 pt-1">
        <p className="text-2xs leading-relaxed text-ink-faint">
          Every value below is produced by the analysis pipeline for that stage&apos;s
          own study. Because the stages are four different patients, a difference
          between columns is a difference between studies — it is not a measured
          change in anyone, and no recovery percentage is derived from it.
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-left">
          <caption className="sr-only">
            {COMPARISON_CAVEAT} {CROSS_STUDY_DISCLAIMER}
          </caption>
          <thead>
            <tr className="border-y border-line-subtle bg-surface-2/40">
              <th scope="col" className="px-4 py-2 text-2xs font-medium uppercase tracking-wide text-ink-faint">
                Measurement
              </th>
              {columns.map((column) => (
                <th
                  key={column.stageId}
                  scope="col"
                  className="whitespace-nowrap px-3 py-2 text-2xs font-medium text-ink"
                >
                  {column.stage?.label ?? column.stageId}
                  <span className="block text-2xs font-normal text-ink-faint">
                    {column.stage?.display_reference ?? ""}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {COMPARISON_FIELDS.map((field) => (
              <tr key={field.key} className="border-b border-line-subtle last:border-0">
                <th
                  scope="row"
                  className="px-4 py-2 align-top text-xs font-normal text-ink-muted"
                >
                  {field.label}
                  {field.note ? (
                    <span className="block text-2xs leading-relaxed text-ink-faint">
                      {field.note}
                    </span>
                  ) : null}
                </th>
                {columns.map((column, index) => {
                  const value = column.metrics
                    ? (column.metrics[field.key] as number | null)
                    : undefined;
                  const previous =
                    index > 0 && columns[index - 1]!.metrics
                      ? (columns[index - 1]!.metrics![field.key] as number | null)
                      : undefined;
                  return (
                    <td key={column.stageId} className="px-3 py-2 align-top">
                      {column.metrics ? (
                        <>
                          <span
                            className={cn(
                              "block font-mono text-xs",
                              value == null ? "italic text-ink-faint" : "text-ink",
                            )}
                          >
                            {formatMetric(value, field)}
                          </span>
                          {index > 0 ? (
                            <DeltaCell
                              previous={previous}
                              current={value}
                              field={field}
                            />
                          ) : (
                            <span className="text-2xs text-ink-faint">baseline</span>
                          )}
                        </>
                      ) : column.state?.status === "loading" ? (
                        <Skeleton className="h-4 w-16 rounded" />
                      ) : (
                        <span className="text-2xs italic text-ink-faint">
                          not analysed
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

/* -------------------------------------------------------------------------- */
/* Surgical evaluation consideration                                           */
/* -------------------------------------------------------------------------- */

function SurgicalEvaluation({ result }: { result: AnalysisResult }) {
  const consideration = surgicalConsideration(result);

  return (
    <Panel>
      <PanelHeader
        title={consideration.heading}
        subtitle="Derived from model-estimated imaging findings on this stage's study"
      />
      <div className="space-y-2.5 px-4 py-3">
        {consideration.warranted ? (
          <p className="text-sm text-ink">{consideration.wording}</p>
        ) : (
          <p className="text-sm text-ink-muted">
            No supported positive imaging finding was reported for this study, so
            no evaluation prompt is shown. This is not a statement that the spine
            is normal.
          </p>
        )}

        {consideration.supportingFindings.length > 0 ? (
          <div>
            <p className="label-caps mb-1">Supporting findings</p>
            <ul className="space-y-1">
              {consideration.supportingFindings.map((finding) => (
                <li
                  key={finding.text}
                  className="flex gap-2 text-xs leading-relaxed text-ink-muted"
                >
                  <Ruler
                    className="mt-0.5 h-3 w-3 shrink-0 text-ink-faint"
                    aria-hidden
                  />
                  <span>{finding.text}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        <p className="border-t border-line-subtle pt-2 text-2xs leading-relaxed text-severity-high">
          {consideration.caveat}
        </p>
      </div>
    </Panel>
  );
}

/* -------------------------------------------------------------------------- */
/* Final summary                                                               */
/* -------------------------------------------------------------------------- */

function WorkflowSummary({
  demoCase,
  stages,
}: Pick<RecoveryTrackerProps, "demoCase" | "stages">) {
  const distinct = new Set(
    demoCase.stages.map((s) => s.provenance.source_study_id),
  );
  const rows = workflowSummary({
    stageCount: TIMELINE.length,
    distinctStudyCount: distinct.size,
    sourceDataset: demoCase.source_dataset,
  });

  return (
    <Panel>
      <PanelHeader
        title="Longitudinal workflow"
        subtitle="What this demonstration does and does not include"
      />
      <dl className="divide-y divide-line-subtle">
        {rows.map((row) => (
          <div
            key={row.label}
            className="flex items-center justify-between gap-3 px-4 py-1.5"
          >
            <dt className="text-xs text-ink-muted">{row.label}</dt>
            <dd className="flex items-center gap-1.5 text-right">
              <span
                className={cn(
                  "text-xs",
                  row.state === "no" ? "text-severity-high" : "text-ink",
                )}
              >
                {row.value}
              </span>
              {row.state === "yes" ? (
                <Check className="h-3.5 w-3.5 text-seg-canal" aria-label="available" />
              ) : row.state === "no" ? (
                <X className="h-3.5 w-3.5 text-severity-high" aria-label="not available" />
              ) : null}
            </dd>
          </div>
        ))}
      </dl>
      <p className="border-t border-line-subtle px-4 py-2 text-2xs leading-relaxed text-ink-faint">
        {demoCase.interpretation}
      </p>
    </Panel>
  );
}

/* -------------------------------------------------------------------------- */
/* Tracker                                                                     */
/* -------------------------------------------------------------------------- */

export function RecoveryTracker({
  demoCase,
  stages,
  activeNodeId,
  onSelectNode,
  loadingLabel,
}: RecoveryTrackerProps) {
  const [selected, setSelected] = React.useState<DiscResult | null>(null);
  const [view, setView] = React.useState<"2d" | "3d">("3d");

  const node = TIMELINE.find((n) => n.id === activeNodeId) ?? TIMELINE[0]!;
  const state = node.stageId ? stages[node.stageId] : undefined;
  const result = state?.result;

  /*
   * A stage change can land on a study with fewer discs, so a stale selection
   * would point at a disc that does not exist. Clear it rather than carry it.
   */
  React.useEffect(() => {
    if (!result) {
      setSelected(null);
      return;
    }
    setSelected((current) => {
      if (!current) return null;
      const match = result.discs.find((disc) => disc.index === current.index);
      return match ?? null;
    });
  }, [result]);

  const stageById = new Map(demoCase.stages.map((s) => [s.stage_id, s]));
  const stageMeta = node.stageId ? stageById.get(node.stageId) : undefined;
  const metrics: StageMetrics | null = result ? computeStageMetrics(result) : null;

  return (
    <div className="flex min-h-0 flex-col">
      <StandingDisclaimer demoCase={demoCase} />

      <RecoveryTimeline
        demoCase={demoCase}
        stages={stages}
        activeNodeId={activeNodeId}
        onSelectNode={onSelectNode}
      />

      {loadingLabel ? (
        <div className="flex items-center gap-2 border-b border-line-subtle bg-surface-1/60 px-4 py-1.5 md:px-6">
          <Loader2 className="h-3 w-3 animate-spin text-accent" aria-hidden />
          <span className="text-2xs text-ink-muted">{loadingLabel}</span>
        </div>
      ) : null}

      {/* ------------------------------------------------ active stage header */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-line-subtle px-4 py-2 md:px-6">
        <span className="text-xs font-medium text-ink">{node.label}</span>
        <Badge tone="accent">{node.researchStatus}</Badge>
        {stageMeta ? <Badge tone="muted">{stageMeta.display_reference}</Badge> : null}
        {!node.hasOwnImaging ? (
          <Badge tone="warn">Assessment of baseline imaging</Badge>
        ) : null}
        <span className="text-2xs text-ink-faint">{node.description}</span>
      </div>

      <div className="grid min-h-0 flex-1 gap-3 p-3 xl:grid-cols-[minmax(0,1fr)_380px]">
        {/* ------------------------------------------------ viewers */}
        <div className="flex min-h-0 flex-col gap-3">
          <Panel className="flex min-h-[420px] flex-col">
            <div className="flex flex-wrap items-center gap-2 border-b border-line-subtle px-3 py-2">
              <span className="flex items-center gap-1.5 text-xs text-ink-muted">
                <Scan className="h-3.5 w-3.5" aria-hidden />
                Imaging
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
                      "inline-flex items-center gap-1 rounded px-2 py-1 text-2xs font-medium uppercase transition-colors",
                      view === id
                        ? "bg-accent/15 text-accent"
                        : "text-ink-faint hover:text-ink",
                    )}
                  >
                    {id === "2d" ? (
                      <ImageIcon className="h-3 w-3" aria-hidden />
                    ) : (
                      <Box className="h-3 w-3" aria-hidden />
                    )}
                    {id === "2d" ? "2D View" : "3D View"}
                  </button>
                ))}
              </div>
              {selected ? (
                <Badge tone="accent">Disc {selected.index} selected</Badge>
              ) : null}
            </div>

            <div className="min-h-[360px] flex-1">
              {state?.status === "ready" && result ? (
                <>
                  <div className={view === "2d" ? "h-full" : "hidden"}>
                    <MriViewer
                      analysisId={result.analysis_id}
                      sliceCount={result.study.slice_count}
                      focusSlice={selected?.representative_slice_index ?? null}
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
                        if (disc) setSelected(disc);
                      }}
                    />
                  ) : null}
                </>
              ) : (
                <div className="grid h-full place-items-center px-6">
                  <div className="text-center">
                    {state?.status === "failed" ? (
                      <>
                        <p className="text-sm text-ink">Stage unavailable</p>
                        <p className="mt-1 text-xs leading-relaxed text-ink-faint">
                          {state.error}
                        </p>
                      </>
                    ) : (
                      <>
                        <Loader2
                          className="mx-auto h-5 w-5 animate-spin text-accent"
                          aria-hidden
                        />
                        <p className="mt-2 text-xs text-ink-muted">
                          Running this stage&apos;s study through the pipeline…
                        </p>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          </Panel>

          <RecoveryComparison demoCase={demoCase} stages={stages} />
        </div>

        {/* ------------------------------------------------ right column */}
        <aside className="flex min-h-0 flex-col gap-3 overflow-y-auto">
          {result ? <SurgicalEvaluation result={result} /> : null}

          {result ? (
            <Panel>
              <PanelHeader
                title="Stage measurements"
                subtitle="Produced by the pipeline for this stage's own study"
              />
              <dl className="divide-y divide-line-subtle">
                {[
                  ["Segmented discs", String(metrics!.discCount)],
                  ["Finding-associated discs", String(metrics!.findingDiscCount)],
                  [
                    "Mean disc height",
                    metrics!.meanDiscHeightMm != null
                      ? `${metrics!.meanDiscHeightMm.toFixed(2)} mm`
                      : "unavailable",
                  ],
                  [
                    "Highest Pfirrmann grade",
                    metrics!.maxPfirrmannGrade != null
                      ? String(metrics!.maxPfirrmannGrade)
                      : "unavailable",
                  ],
                  ["Sagittal slices", String(metrics!.sliceCount)],
                  ["Measurable slices", String(metrics!.informativeSliceCount)],
                ].map(([label, value]) => (
                  <div
                    key={label}
                    className="flex items-center justify-between gap-3 px-4 py-1.5"
                  >
                    <dt className="text-xs text-ink-muted">{label}</dt>
                    <dd className="font-mono text-xs text-ink">{value}</dd>
                  </div>
                ))}
              </dl>
            </Panel>
          ) : null}

          {result ? (
            <Panel>
              <PanelHeader
                title="Discs"
                subtitle={
                  selected
                    ? `Disc ${selected.index} selected and marked in both viewers`
                    : "Select a disc to highlight it in 2D and 3D"
                }
                actions={
                  selected ? (
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => setSelected(null)}
                    >
                      Clear
                    </Button>
                  ) : null
                }
              />
              <DiscList
                discs={result.discs}
                selectedIndex={selected?.index ?? null}
                onSelect={setSelected}
                markedIndices={result.finding_overlay?.disc_indices}
              />
            </Panel>
          ) : null}

          <WorkflowSummary demoCase={demoCase} stages={stages} />

          <Panel>
            <PanelHeader title="Provenance" subtitle="Preserved for every stage" />
            <dl className="divide-y divide-line-subtle">
              {demoCase.stages.map((stage) => (
                <div key={stage.stage_id} className="px-4 py-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <dt className="text-xs text-ink-muted">{stage.label}</dt>
                    <dd className="text-2xs text-ink-faint">
                      {stage.display_reference}
                    </dd>
                  </div>
                  <p className="mt-0.5 text-2xs text-ink-faint">
                    {stage.provenance.source_dataset} ·{" "}
                    {stage.provenance.source_split ?? "unknown"} split ·{" "}
                    {stage.provenance.measurement_source === "real_pipeline"
                      ? "real pipeline measurements"
                      : "simulated demo value"}{" "}
                    · not a follow-up scan
                  </p>
                </div>
              ))}
            </dl>
          </Panel>
        </aside>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Real single-study state                                                     */
/* -------------------------------------------------------------------------- */

/**
 * Shown for an ordinary SPIDER analysis. No postoperative stage is invented; the
 * absence of follow-up is stated instead.
 */
export function LongitudinalUnavailable({
  analysisId,
}: {
  analysisId?: string;
}) {
  return (
    <Panel>
      <PanelHeader
        title={LONGITUDINAL_UNAVAILABLE}
        subtitle="Single-study analysis"
      />
      <div className="space-y-2 px-4 py-3">
        <p className="flex items-start gap-2 text-xs leading-relaxed text-ink-muted">
          <Layers className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-faint" aria-hidden />
          <span>
            This study has a single timepoint. The dataset this system was
            validated on contains no postoperative imaging, no surgical record and
            no follow-up assessment, so no change over time is measured and none is
            claimed.
          </span>
        </p>
        {analysisId ? (
          <p className="text-2xs text-ink-faint">
            Analysis <span className="font-mono">{analysisId}</span>
          </p>
        ) : null}
      </div>
    </Panel>
  );
}

export default RecoveryTracker;
