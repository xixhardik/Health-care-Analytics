/**
 * Recovery Tracker logic.
 *
 * Pure functions, kept out of the components so the timeline, the comparison
 * panel and the tests all read from one definition.
 *
 * The single rule this module enforces: **every number it produces comes from the
 * analysis pipeline.** There is no fabricated measurement, no interpolation
 * between stages and no composite recovery score. Where a value is unavailable it
 * is `null` and the UI says so.
 *
 * What is simulated is the *arrangement*: the timeline positions below are filled
 * by four different SPIDER studies from four different patients. A trend across
 * them is a consequence of which studies were selected, not an observed recovery
 * trajectory, and the wording exported here says exactly that.
 */

import type { AnalysisResult } from "./types";

/* -------------------------------------------------------------------------- */
/* Wording                                                                     */
/* -------------------------------------------------------------------------- */

export const SIMULATED_BADGE = "Simulated Longitudinal Demo";

export const CROSS_STUDY_DISCLAIMER =
  "These stages use different SPIDER studies to demonstrate the longitudinal " +
  "workflow. They are not postoperative follow-up scans from the same patient.";

export const COMPARISON_CAVEAT =
  "Cross-study demonstration trend — not patient recovery.";

export const LONGITUDINAL_UNAVAILABLE =
  "Longitudinal follow-up unavailable for this study.";

export const SINGLE_STUDY_LABEL = "Single-study analysis";

export const SURGICAL_CONSIDERATION_HEADING = "Surgical Evaluation Consideration";

export const SURGICAL_CONSIDERATION_WORDING =
  "Imaging findings may warrant surgical evaluation.";

export const SURGICAL_CONSIDERATION_CAVEAT =
  "This is a research prototype and not a clinical recommendation.";

export const WORKFLOW_TYPE = "Simulated longitudinal workflow";

/* -------------------------------------------------------------------------- */
/* Timeline                                                                    */
/* -------------------------------------------------------------------------- */

export interface TimelineNode {
  id: string;
  label: string;
  /** Demo stage whose study backs this node, or null when it has no imaging. */
  stageId: string | null;
  researchStatus: string;
  /**
   * False for Surgical Evaluation: it is an assessment of the Diagnosis imaging,
   * not a separate scan. Inventing a scan for it would be inventing imaging.
   */
  hasOwnImaging: boolean;
  description: string;
}

export const TIMELINE: readonly TimelineNode[] = [
  {
    id: "diagnosis",
    label: "Diagnosis",
    stageId: "pre_surgery",
    researchStatus: "Baseline",
    hasOwnImaging: true,
    description:
      "Baseline imaging for the demonstration, from the first demonstration study.",
  },
  {
    id: "surgical_evaluation",
    label: "Surgical Evaluation",
    stageId: "pre_surgery",
    researchStatus: "Surgical Evaluation",
    hasOwnImaging: false,
    description:
      "An assessment step over the baseline imaging, not a separate scan. No " +
      "additional study exists for this position.",
  },
  {
    id: "post_surgery",
    label: "Post-Surgery",
    stageId: "post_surgery",
    researchStatus: "Postoperative Demonstration",
    hasOwnImaging: true,
    description:
      "A different demonstration study occupying the postoperative position. No " +
      "surgery is depicted; this dataset contains no postoperative imaging.",
  },
  {
    id: "month_3",
    label: "3-Month Recovery",
    stageId: "month_3",
    researchStatus: "Recovery Monitoring",
    hasOwnImaging: true,
    description:
      "A third demonstration study, deliberately from a different scanner and " +
      "field strength.",
  },
  {
    id: "month_6",
    label: "6-Month Recovery",
    stageId: "month_6",
    researchStatus: "Final Follow-up Demonstration",
    hasOwnImaging: true,
    description:
      "A fourth demonstration study occupying the final position in the " +
      "demonstration timeline.",
  },
] as const;

/** Stage ids that actually have imaging, in timeline order, without duplicates. */
export const IMAGING_STAGE_IDS: readonly string[] = TIMELINE.filter(
  (node) => node.hasOwnImaging && node.stageId,
).map((node) => node.stageId as string);

export function nodeById(id: string): TimelineNode | undefined {
  return TIMELINE.find((node) => node.id === id);
}

/* -------------------------------------------------------------------------- */
/* Metrics                                                                     */
/* -------------------------------------------------------------------------- */

export interface StageMetrics {
  discCount: number;
  findingDiscCount: number;
  discsWithFindings: number;
  meanDiscHeightMm: number | null;
  maxPfirrmannGrade: number | null;
  gradedDiscCount: number;
  narrowingCount: number;
  bulgingCount: number;
  herniationCount: number;
  modicCount: number;
  meanCanalWidthMm: number | null;
  meanDiscAreaMm2: number | null;
  sliceCount: number;
  informativeSliceCount: number;
}

function meanOf(values: (number | null | undefined)[]): number | null {
  const usable = values.filter(
    (v): v is number => typeof v === "number" && Number.isFinite(v),
  );
  if (usable.length === 0) return null;
  return usable.reduce((sum, v) => sum + v, 0) / usable.length;
}

function countTrue(
  discs: AnalysisResult["discs"],
  key: "narrowing" | "bulging" | "herniation" | "modic_any",
): number {
  // Strictly `true`: a null means "not reported", which is not a negative.
  return discs.filter((disc) => disc[key] === true).length;
}

/**
 * Everything the comparison panel can show for one stage, read straight from the
 * pipeline's own result. Absent values are null, never zero-filled — a missing
 * measurement and a measurement of zero are different statements.
 */
export function computeStageMetrics(result: AnalysisResult): StageMetrics {
  const discs = result.discs ?? [];
  const grades = discs
    .map((disc) => disc.pfirrmann_grade)
    .filter((g): g is number => typeof g === "number");

  return {
    discCount: result.summary.disc_count,
    findingDiscCount: result.finding_overlay?.disc_indices?.length ?? 0,
    discsWithFindings: result.summary.discs_with_findings_count,
    meanDiscHeightMm: result.summary.mean_disc_height_mm ?? null,
    maxPfirrmannGrade: grades.length > 0 ? Math.max(...grades) : null,
    gradedDiscCount: grades.length,
    narrowingCount: countTrue(discs, "narrowing"),
    bulgingCount: countTrue(discs, "bulging"),
    herniationCount: countTrue(discs, "herniation"),
    modicCount: countTrue(discs, "modic_any"),
    meanCanalWidthMm: meanOf(
      discs.map((disc) => disc.measurements.canal_width_at_disc_mm),
    ),
    meanDiscAreaMm2: meanOf(discs.map((disc) => disc.measurements.area_mm2)),
    sliceCount: result.study.slice_count,
    informativeSliceCount: result.segmentation.informative_slice_count,
  };
}

export interface ComparisonField {
  key: keyof StageMetrics;
  label: string;
  unit?: string;
  decimals?: number;
  /** How to read a rise: neutral fields get no directional styling at all. */
  interpretation: "higher-is-looser-degeneration" | "count-of-findings" | "neutral";
  note?: string;
}

/**
 * Fields the panel compares. Every one is a direct pipeline output.
 *
 * `interpretation` drives nothing clinical - it only decides whether an arrow is
 * tinted, and even that is captioned as a cross-study difference.
 */
export const COMPARISON_FIELDS: readonly ComparisonField[] = [
  {
    key: "discCount",
    label: "Segmented discs",
    interpretation: "neutral",
    note: "Anatomy differs between studies, so this is not expected to be constant.",
  },
  {
    key: "meanDiscHeightMm",
    label: "Mean disc height",
    unit: "mm",
    decimals: 2,
    interpretation: "higher-is-looser-degeneration",
  },
  {
    key: "findingDiscCount",
    label: "Finding-associated discs",
    interpretation: "count-of-findings",
  },
  {
    key: "maxPfirrmannGrade",
    label: "Highest Pfirrmann grade",
    interpretation: "count-of-findings",
    note: "Model-estimated, ordinal 1-5. Withheld on non-T2 studies.",
  },
  { key: "narrowingCount", label: "Discs with narrowing", interpretation: "count-of-findings" },
  { key: "bulgingCount", label: "Discs with bulging", interpretation: "count-of-findings" },
  { key: "herniationCount", label: "Discs with herniation", interpretation: "count-of-findings", note: "Weakly validated target; treat with particular caution." },
  { key: "modicCount", label: "Discs with Modic change", interpretation: "count-of-findings" },
  {
    key: "meanCanalWidthMm",
    label: "Mean canal width at disc",
    unit: "mm",
    decimals: 2,
    interpretation: "neutral",
  },
  {
    key: "meanDiscAreaMm2",
    label: "Mean disc area",
    unit: "mm²",
    decimals: 1,
    interpretation: "neutral",
  },
] as const;

export function formatMetric(
  value: number | null | undefined,
  field: ComparisonField,
): string {
  if (value == null || !Number.isFinite(value)) return "unavailable";
  const text =
    field.decimals != null ? value.toFixed(field.decimals) : String(value);
  return field.unit ? `${text} ${field.unit}` : text;
}

export interface MetricDelta {
  absolute: number | null;
  text: string;
  /** Tint only; carries no clinical meaning. */
  direction: "up" | "down" | "flat" | "unknown";
}

/**
 * Difference between two consecutive demonstration stages.
 *
 * Named "difference" rather than "change" deliberately: these are two different
 * patients, so nothing changed between them.
 */
export function metricDelta(
  previous: number | null | undefined,
  current: number | null | undefined,
  field: ComparisonField,
): MetricDelta {
  if (
    previous == null || current == null ||
    !Number.isFinite(previous) || !Number.isFinite(current)
  ) {
    return { absolute: null, text: "not comparable", direction: "unknown" };
  }
  const absolute = current - previous;
  if (Math.abs(absolute) < 1e-9) {
    return { absolute: 0, text: "no difference", direction: "flat" };
  }
  const decimals = field.decimals ?? 0;
  const sign = absolute > 0 ? "+" : "−";
  const magnitude = Math.abs(absolute).toFixed(decimals);
  return {
    absolute,
    text: `${sign}${magnitude}${field.unit ? ` ${field.unit}` : ""}`,
    direction: absolute > 0 ? "up" : "down",
  };
}

/* -------------------------------------------------------------------------- */
/* Surgical evaluation consideration                                           */
/* -------------------------------------------------------------------------- */

export interface SurgicalConsideration {
  /** True when the pipeline reported at least one supported positive finding. */
  warranted: boolean;
  /** The actual detected findings, verbatim from the analysis summary. */
  supportingFindings: { text: string; discIndices: number[] }[];
  heading: string;
  wording: string;
  caveat: string;
}

/**
 * Build the surgical-evaluation section from findings the pipeline actually
 * reported.
 *
 * This deliberately makes no recommendation and invents no indication. It states
 * that findings *may warrant evaluation* and then lists the findings, so a reader
 * can see precisely what prompted the statement.
 */
export function surgicalConsideration(
  result: AnalysisResult,
): SurgicalConsideration {
  const supporting = (result.summary.findings ?? [])
    .filter((finding) => finding.category === "Disc-level finding")
    .map((finding) => ({
      text: finding.text,
      discIndices: finding.disc_indices ?? [],
    }));

  return {
    warranted: supporting.length > 0,
    supportingFindings: supporting,
    heading: SURGICAL_CONSIDERATION_HEADING,
    wording: SURGICAL_CONSIDERATION_WORDING,
    caveat: SURGICAL_CONSIDERATION_CAVEAT,
  };
}

/* -------------------------------------------------------------------------- */
/* Final demonstration summary                                                 */
/* -------------------------------------------------------------------------- */

export interface WorkflowSummaryRow {
  label: string;
  value: string;
  state: "yes" | "no" | "info";
}

/**
 * The closing card. `trueLongitudinalFollowUp` is hard-coded absent because the
 * dataset has none; it is shown rather than omitted so the limitation is part of
 * the summary a viewer reads last.
 */
export function workflowSummary(options: {
  stageCount: number;
  distinctStudyCount: number;
  sourceDataset: string;
}): WorkflowSummaryRow[] {
  return [
    {
      label: "Demonstration stages",
      value: String(options.stageCount),
      state: "info",
    },
    {
      label: "Distinct source studies",
      value: `${options.distinctStudyCount} distinct SPIDER studies`,
      state: "info",
    },
    { label: "Source dataset", value: options.sourceDataset, state: "info" },
    { label: "2D MRI", value: "Available", state: "yes" },
    { label: "3D MRI", value: "Available", state: "yes" },
    { label: "Segmentation", value: "Available", state: "yes" },
    { label: "Finding analysis", value: "Available", state: "yes" },
    {
      label: "True longitudinal follow-up",
      value: "Not available in this dataset",
      state: "no",
    },
    { label: "Workflow type", value: WORKFLOW_TYPE, state: "info" },
  ];
}
