/**
 * Disc-level results: the list, and the detail panel for a selected disc.
 *
 * Every value shown here comes from the backend result. Where the backend reports
 * a finding as unsupported or not computed, this renders "not reported" with the
 * reason rather than an empty badge that could read as a negative result.
 */

"use client";

import { ChevronRight, Info, Ruler, X } from "lucide-react";
import * as React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  FINDING_OVERLAY,
  PFIRRMANN_COLOURS,
  PROVENANCE_STYLES,
  STRENGTH_LABELS,
} from "@/lib/theme";
import type { DiscResult, FindingEstimate, StructuredLine } from "@/lib/types";
import { Badge, Button, FieldRow, Panel, PanelHeader, cn } from "./ui";

/** Provenance chip. Rendered beside every value, including unavailable ones. */
function ProvenanceTag({ provenance }: { provenance: string }) {
  const style = PROVENANCE_STYLES[provenance];
  if (!style) return null;
  return (
    <Badge tone={style.tone} className="shrink-0" title={style.hint}>
      {style.label}
    </Badge>
  );
}

/**
 * One structured line: label, value, provenance.
 *
 * An unavailable value renders the word "unavailable" in a muted tone with its
 * reason — never 0, "No" or "Normal", because absence of a model is not a
 * negative finding.
 */
function StructuredRow({ line }: { line: StructuredLine }) {
  const strength = line.strength ? STRENGTH_LABELS[line.strength] : undefined;
  const detected = line.value === "detected";

  return (
    <div className="border-b border-line-subtle py-2 last:border-0">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <span className="text-xs text-ink-muted">{line.label}</span>
        <span className="flex items-center gap-2">
          {strength ? (
            <span className={cn("text-2xs", strength.tone)}>{strength.label}</span>
          ) : null}
          <span
            className={cn(
              "font-mono text-xs",
              !line.available
                ? "italic text-ink-faint"
                : detected
                  ? "text-seg-disc"
                  : "text-ink",
            )}
          >
            {line.value}
          </span>
          <ProvenanceTag provenance={line.provenance} />
        </span>
      </div>
      {!line.available && line.unavailable_reason ? (
        <p className="mt-1 text-2xs leading-relaxed text-ink-faint">
          {line.unavailable_reason}
        </p>
      ) : null}
    </div>
  );
}

/** Findings shown as compact badges on the list rows. */
const ROW_FINDINGS: { key: keyof DiscResult; short: string }[] = [
  { key: "narrowing", short: "Narrowing" },
  { key: "bulging", short: "Bulging" },
  { key: "modic_any", short: "Modic" },
  { key: "herniation", short: "Herniation" },
];

function GradeChip({ grade }: { grade: number | null | undefined }) {
  if (grade == null) {
    return (
      <span className="font-mono text-2xs text-ink-faint" title="Not reported">
        —
      </span>
    );
  }
  return (
    <span
      className="inline-grid h-5 w-5 place-items-center rounded font-mono text-2xs font-semibold text-surface-0"
      style={{ backgroundColor: PFIRRMANN_COLOURS[grade] ?? "#64748b" }}
      title={`Model-estimated Pfirrmann grade ${grade}`}
    >
      {grade}
    </span>
  );
}

export function DiscList({
  discs,
  selectedIndex,
  onSelect,
  markedIndices,
}: {
  discs: DiscResult[];
  selectedIndex: number | null;
  onSelect: (disc: DiscResult) => void;
  /**
   * Discs the findings overlay marks, as decided by the backend. Passed in rather
   * than derived here so the row marker and the rendered slice agree by
   * construction.
   */
  markedIndices?: number[];
}) {
  const marked = new Set(markedIndices ?? []);
  if (discs.length === 0) {
    return (
      <div className="px-4 py-8 text-center">
        <p className="text-sm text-ink">No disc was identified</p>
        <p className="mt-1.5 text-xs leading-relaxed text-ink-faint">
          The pipeline found no intervertebral disc in this study. This can happen
          when the field of view does not cover the lumbar spine.
        </p>
      </div>
    );
  }

  return (
    <ul className="divide-y divide-line-subtle" role="list">
      {discs.map((disc) => {
        const active = selectedIndex === disc.index;
        return (
          <li key={disc.index}>
            <button
              type="button"
              onClick={() => onSelect(disc)}
              aria-current={active ? "true" : undefined}
              className={cn(
                "flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors",
                active ? "bg-accent/10" : "hover:bg-surface-2",
              )}
            >
              <span className="w-12 shrink-0">
                <span className="label-caps">Disc</span>
                <span className="flex items-center gap-1.5">
                  <span className="font-mono text-sm text-ink">
                    {disc.index}
                  </span>
                  {marked.has(disc.index) ? (
                    <span
                      className="h-1.5 w-1.5 shrink-0 rounded-full"
                      style={{ backgroundColor: FINDING_OVERLAY.hex }}
                      title={`${FINDING_OVERLAY.legend} — marked by the findings overlay`}
                    />
                  ) : null}
                </span>
              </span>

              <GradeChip grade={disc.pfirrmann_grade} />

              <span className="flex min-w-0 flex-1 flex-wrap gap-1">
                {ROW_FINDINGS.map(({ key, short }) => {
                  const value = disc[key] as boolean | null | undefined;
                  if (value !== true) return null;
                  return (
                    <Badge key={short} tone="warn">
                      {short}
                    </Badge>
                  );
                })}
                {ROW_FINDINGS.every(
                  ({ key }) => (disc[key] as boolean | null) !== true,
                ) ? (
                  <span className="text-2xs text-ink-faint">
                    No positive finding
                  </span>
                ) : null}
              </span>

              <span className="hidden shrink-0 text-right font-mono text-2xs text-ink-muted sm:block">
                {disc.measurements.height_mm_central != null
                  ? `${disc.measurements.height_mm_central.toFixed(2)} mm`
                  : "—"}
              </span>

              <ChevronRight
                className="h-3.5 w-3.5 shrink-0 text-ink-faint"
                aria-hidden
              />
            </button>
          </li>
        );
      })}
    </ul>
  );
}

function FindingRow({ finding }: { finding: FindingEstimate }) {
  const strength = finding.strength
    ? STRENGTH_LABELS[finding.strength]
    : undefined;

  if (finding.unavailable_reason) {
    return (
      <div className="border-b border-line-subtle py-2 last:border-0">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-ink-muted">{finding.label}</span>
          <Badge tone="muted">Not reported</Badge>
        </div>
        <p className="mt-1 text-2xs leading-relaxed text-ink-faint">
          {finding.unavailable_reason}
        </p>
      </div>
    );
  }

  const positive = finding.value === true;
  return (
    <div className="border-b border-line-subtle py-2 last:border-0">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs text-ink-muted">{finding.label}</span>
        <span className="flex items-center gap-1.5">
          {strength ? (
            <span className={cn("text-2xs", strength.tone)} title="Validated strength">
              {strength.label}
            </span>
          ) : null}
          {typeof finding.value === "number" ? (
            <GradeChip grade={finding.value} />
          ) : (
            <Badge tone={positive ? "warn" : "muted"}>
              {positive ? "Detected" : "Not detected"}
            </Badge>
          )}
        </span>
      </div>
      <div className="mt-1 flex items-center gap-2">
        {finding.probability != null ? (
          <>
            <div className="h-1 flex-1 overflow-hidden rounded-full bg-surface-3">
              <div
                className={cn(
                  "h-full rounded-full",
                  positive ? "bg-seg-disc" : "bg-line-strong",
                )}
                style={{ width: `${Math.round(finding.probability * 100)}%` }}
              />
            </div>
            <span className="w-8 shrink-0 text-right font-mono text-2xs text-ink-faint">
              {Math.round(finding.probability * 100)}%
            </span>
          </>
        ) : null}
      </div>
      {finding.validated_value != null ? (
        <p className="mt-1 text-2xs text-ink-faint">
          Validated {finding.validated_metric?.replace(/_/g, " ")}{" "}
          {finding.validated_value}
          {finding.prevalence_baseline != null
            ? ` against a prevalence baseline of ${finding.prevalence_baseline}`
            : ""}
        </p>
      ) : null}
      {finding.caveat ? (
        <p className="mt-1 text-2xs leading-relaxed text-seg-disc">
          {finding.caveat}
        </p>
      ) : null}
    </div>
  );
}

export function DiscDetail({
  disc,
  allDiscs,
  onClose,
}: {
  disc: DiscResult;
  allDiscs: DiscResult[];
  onClose: () => void;
}) {
  const m = disc.measurements;
  const structured = disc.structured ?? [];

  // Height across the series, so a single disc is read in context rather than
  // against an invented reference range.
  const heightSeries = React.useMemo(
    () =>
      allDiscs
        .filter((d) => d.measurements.height_mm_central != null)
        .map((d) => ({
          disc: d.index,
          height: d.measurements.height_mm_central as number,
          current: d.index === disc.index,
        })),
    [allDiscs, disc.index],
  );

  const gradeSpread = React.useMemo(() => {
    const entries = disc.findings.find((f) => f.name === "pfirrmann_grade")
      ?.probabilities;
    if (!entries) return [];
    return Object.entries(entries).map(([grade, probability]) => ({
      grade,
      probability: Number((probability * 100).toFixed(1)),
    }));
  }, [disc.findings]);

  return (
    <Panel className="flex h-full min-h-0 flex-col">
      <PanelHeader
        title={`Disc ${disc.index}`}
        subtitle={
          disc.identity_confidence != null
            ? `Identity support ${(disc.identity_confidence * 100).toFixed(0)}% of slices`
            : "Identity support not available"
        }
        actions={
          <Button
            size="sm"
            variant="ghost"
            onClick={onClose}
            aria-label="Close disc detail"
          >
            <X className="h-3.5 w-3.5" aria-hidden />
          </Button>
        }
      />

      <div className="min-h-0 flex-1 overflow-y-auto">
        {/* ---------------------------------- structured summary */}
        <section className="px-4 py-3">
          <h3 className="label-caps mb-1.5 flex items-center gap-1.5">
            <Ruler className="h-3 w-3" aria-hidden />
            Findings and measurements
          </h3>
          {structured.length > 0 ? (
            <div>
              {structured.map((line) => (
                <StructuredRow key={line.label} line={line} />
              ))}
            </div>
          ) : (
            <dl>
              <FieldRow
                label="Central height"
                value={
                  m.height_mm_central != null
                    ? `${m.height_mm_central.toFixed(3)} mm`
                    : "unavailable"
                }
                mono
              />
              <FieldRow
                label="Area"
                value={m.area_mm2 != null ? `${m.area_mm2.toFixed(1)} mm²` : "unavailable"}
                mono
              />
            </dl>
          )}
          <div className="mt-2 flex items-baseline justify-between gap-3">
            <span className="text-xs text-ink-faint">Slices contributing</span>
            <span className="font-mono text-xs text-ink">
              {disc.slices_present ?? "—"}
            </span>
          </div>
        </section>

        {/* ------------------------------------------- height context chart */}
        {heightSeries.length > 1 ? (
          <section className="border-t border-line-subtle px-4 py-3">
            <h3 className="label-caps mb-2">Central height across this study</h3>
            <div className="h-32">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={heightSeries} margin={{ top: 4, right: 4, bottom: 0, left: -18 }}>
                  <CartesianGrid stroke="#1c2942" strokeDasharray="2 4" />
                  <XAxis
                    dataKey="disc"
                    stroke="#6b7c9e"
                    tick={{ fontSize: 10 }}
                    tickLine={false}
                  />
                  <YAxis
                    stroke="#6b7c9e"
                    tick={{ fontSize: 10 }}
                    tickLine={false}
                    width={38}
                    unit=""
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#111b2e",
                      border: "1px solid #243352",
                      borderRadius: 6,
                      fontSize: 11,
                    }}
                    labelFormatter={(label) => `Disc ${label}`}
                    formatter={(value) => [`${Number(value).toFixed(2)} mm`, "Height"]}
                  />
                  <Line
                    type="monotone"
                    dataKey="height"
                    stroke="#22d3ee"
                    strokeWidth={1.5}
                    dot={{ r: 2.5, fill: "#22d3ee" }}
                    activeDot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-1.5 text-2xs leading-relaxed text-ink-faint">
              Shown for context within this study only. No clinical reference range
              is implied.
            </p>
          </section>
        ) : null}

        {/* ------------------------------------------- grade distribution */}
        {gradeSpread.length > 0 ? (
          <section className="border-t border-line-subtle px-4 py-3">
            <h3 className="label-caps mb-2">
              Model-estimated Pfirrmann distribution
            </h3>
            <div className="h-28">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={gradeSpread} margin={{ top: 4, right: 4, bottom: 0, left: -18 }}>
                  <CartesianGrid stroke="#1c2942" strokeDasharray="2 4" vertical={false} />
                  <XAxis dataKey="grade" stroke="#6b7c9e" tick={{ fontSize: 10 }} tickLine={false} />
                  <YAxis stroke="#6b7c9e" tick={{ fontSize: 10 }} tickLine={false} width={38} unit="%" />
                  <Tooltip
                    contentStyle={{
                      background: "#111b2e",
                      border: "1px solid #243352",
                      borderRadius: 6,
                      fontSize: 11,
                    }}
                    labelFormatter={(label) => `Grade ${label}`}
                    formatter={(value) => [`${Number(value)}%`, "Probability"]}
                  />
                  <Bar dataKey="probability" radius={[2, 2, 0, 0]}>
                    {gradeSpread.map((entry) => (
                      <Cell
                        key={entry.grade}
                        fill={PFIRRMANN_COLOURS[Number(entry.grade)] ?? "#64748b"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
        ) : null}

        {/* ------------------------------------------- model evidence */}
        <section className="border-t border-line-subtle px-4 py-3">
          <h3 className="label-caps mb-1.5">
            Model evidence and validated performance
          </h3>
          {disc.findings.length === 0 ? (
            <p className="text-xs text-ink-faint">No finding estimate available.</p>
          ) : (
            <div>
              {disc.findings
                .slice()
                .sort((a, b) => {
                  const rank = (f: FindingEstimate) =>
                    f.unavailable_reason ? 1 : 0;
                  return rank(a) - rank(b) || a.label.localeCompare(b.label);
                })
                .map((finding) => (
                  <FindingRow key={finding.name} finding={finding} />
                ))}
            </div>
          )}
        </section>

        <section className="border-t border-line-subtle px-4 py-3">
          <p className="flex gap-2 text-2xs leading-relaxed text-ink-faint">
            <Info className="mt-0.5 h-3 w-3 shrink-0" aria-hidden />
            <span>
              Identity support describes how consistently this disc was tracked
              across the study&apos;s slices. It is not a clinical confidence.
              All findings are research estimates requiring expert review.
            </span>
          </p>
        </section>
      </div>
    </Panel>
  );
}
