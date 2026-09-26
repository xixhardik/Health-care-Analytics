"use client";

import { ArrowDown, CheckCircle2, FlaskConical, XCircle } from "lucide-react";
import * as React from "react";

import { PageHeader } from "@/components/AppShell";
import {
  Badge,
  ErrorState,
  Panel,
  PanelHeader,
  Skeleton,
  cn,
} from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";

/** The pipeline stages, with a one-line description each. */
const FLOW = [
  {
    stage: "Data",
    detail:
      "A sagittal lumbar spine MRI volume. Nothing about the study is assumed: geometry, spacing and intensity convention are all read from the file.",
  },
  {
    stage: "Preprocessing",
    detail:
      "Resample to 1.0 mm/pixel, centre crop or pad to 352×256, normalise on per-volume foreground percentiles, median denoise, contrast equalise with CLAHE.",
  },
  {
    stage: "U-Net Segmentation",
    detail:
      "A 16-channel U-Net predicts four classes per slice: background, vertebra, intervertebral disc, spinal canal.",
  },
  {
    stage: "Disc Indexing",
    detail:
      "Disc identity is assigned once per series by clustering candidates into row-aligned tracks and ordering them from the most inferior disc upward, then propagating that identity to every slice. Vertebral bodies are separated from posterior elements using the spinal canal.",
  },
  {
    stage: "Feature Extraction",
    detail:
      "Per-disc height, area, AP extent, signal ratios and canal width, aggregated across the slices each disc appears on.",
  },
  {
    stage: "Radiological Finding Estimation",
    detail:
      "An ordinal model estimates the Pfirrmann grade; separate binary models estimate narrowing, bulging, Modic change and endplate change. Targets without validated support are not estimated.",
  },
  {
    stage: "Results",
    detail:
      "A structured result: segmentation overlays, disc identities, quantitative measurements, model-derived findings and a downloadable report.",
  },
] as const;

export default function MethodologyPage() {
  const [health, setHealth] = React.useState<HealthResponse | null>(null);
  const [error, setError] = React.useState<ApiError | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    void api
      .health()
      .then((next) => {
        if (!cancelled) setHealth(next);
      })
      .catch((caught) => {
        if (!cancelled)
          setError(
            caught instanceof ApiError
              ? caught
              : new ApiError("UNKNOWN", "Could not reach the API.", 0),
          );
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const segmentation = health?.metric_table.filter((m) => m.kind === "segmentation");
  const postprocessing = health?.metric_table.filter(
    (m) => m.kind === "postprocessing",
  );

  return (
    <div>
      <PageHeader
        title="Methodology"
        description="How a study is processed, what was measured, and which experiments were kept or rejected."
      />

      <div className="mx-auto max-w-5xl space-y-5 p-4 md:p-6">
        {error ? <ErrorState message={error.message} code={error.code} /> : null}

        {/* ------------------------------------------------ pipeline flow */}
        <Panel>
          <PanelHeader
            title="Pipeline"
            subtitle={
              health
                ? `${health.pipeline.version} — frozen: preprocessing ${health.pipeline.preprocessing}, segmentation ${health.pipeline.segmentation}, post-processing ${health.pipeline.postprocessing}`
                : "Loading"
            }
            actions={
              health ? <Badge tone="accent">{health.pipeline.version}</Badge> : null
            }
          />
          <ol className="px-4 py-4">
            {FLOW.map((item, i) => (
              <li key={item.stage}>
                <div className="flex gap-3">
                  <span className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full border border-line bg-surface-2 font-mono text-2xs text-ink-muted">
                    {i + 1}
                  </span>
                  <div className="min-w-0 pb-1">
                    <p className="text-sm font-medium text-ink">{item.stage}</p>
                    <p className="mt-1 text-xs leading-relaxed text-ink-muted">
                      {item.detail}
                    </p>
                  </div>
                </div>
                {i < FLOW.length - 1 ? (
                  <div
                    className="ml-3 flex h-5 w-6 items-center justify-center"
                    aria-hidden
                  >
                    <ArrowDown className="h-3.5 w-3.5 text-line-strong" />
                  </div>
                ) : null}
              </li>
            ))}
          </ol>
        </Panel>

        {/* ------------------------------------------------ research results */}
        <Panel>
          <PanelHeader
            title="Research evaluation results"
            subtitle={
              health
                ? `Measured on a ${health.validated_metrics.test_patients}-patient held-out test split — not on any study you upload`
                : "Loading"
            }
          />
          {!health ? (
            <div className="space-y-2 p-4">
              {Array.from({ length: 8 }).map((_, i) => (
                <Skeleton key={i} className="h-6" />
              ))}
            </div>
          ) : (
            <div className="grid gap-x-8 sm:grid-cols-2">
              <div className="px-4 py-3">
                <p className="label-caps mb-2">
                  <Badge tone="accent">Segmentation quality</Badge>
                </p>
                <dl>
                  {segmentation?.map((row) => (
                    <div
                      key={row.label}
                      className="flex items-baseline justify-between gap-3 border-b border-line-subtle py-1.5 last:border-0"
                    >
                      <dt className="text-xs text-ink-muted">{row.label}</dt>
                      <dd className="flex shrink-0 items-baseline gap-2">
                        <span className="font-mono text-xs text-ink">{row.value}</span>
                        <span className="text-2xs text-ink-faint">{row.sprint}</span>
                      </dd>
                    </div>
                  ))}
                </dl>
              </div>
              <div className="px-4 py-3">
                <p className="label-caps mb-2">
                  <Badge tone="positive">Disc identity and measurement</Badge>
                </p>
                <dl>
                  {postprocessing?.map((row) => (
                    <div
                      key={row.label}
                      className="flex items-baseline justify-between gap-3 border-b border-line-subtle py-1.5 last:border-0"
                    >
                      <dt className="text-xs text-ink-muted">{row.label}</dt>
                      <dd className="flex shrink-0 items-baseline gap-2">
                        <span className="font-mono text-xs text-ink">{row.value}</span>
                        <span className="text-2xs text-ink-faint">{row.sprint}</span>
                      </dd>
                    </div>
                  ))}
                </dl>
              </div>
            </div>
          )}
          {health ? (
            <p className="border-t border-line-subtle px-4 py-3 text-2xs leading-relaxed text-ink-faint">
              {health.validated_metrics.note}
            </p>
          ) : null}
        </Panel>

        {/* ------------------------------------------------ research progress */}
        <Panel>
          <PanelHeader
            title="Research progress"
            subtitle="Including the experiment that was rejected"
          />
          {!health ? (
            <div className="space-y-2 p-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-20" />
              ))}
            </div>
          ) : (
            <ol className="divide-y divide-line-subtle">
              {health.research_progress.map((stage) => {
                const rejected = stage.outcome === "rejected";
                return (
                  <li key={stage.sprint} className="flex gap-3 px-4 py-3">
                    <span
                      className={cn(
                        "mt-0.5 shrink-0",
                        rejected ? "text-severity-high" : "text-seg-canal",
                      )}
                      aria-hidden
                    >
                      {rejected ? (
                        <XCircle className="h-4 w-4" />
                      ) : (
                        <CheckCircle2 className="h-4 w-4" />
                      )}
                    </span>
                    <div className="min-w-0">
                      <p className="flex flex-wrap items-center gap-2">
                        <span className="text-sm font-medium text-ink">
                          {stage.sprint} — {stage.title}
                        </span>
                        <Badge tone={rejected ? "danger" : "positive"}>
                          {rejected ? "not selected" : "selected"}
                        </Badge>
                      </p>
                      <p className="mt-1 font-mono text-2xs text-ink-faint">
                        {stage.headline}
                      </p>
                      <p className="mt-1.5 text-xs leading-relaxed text-ink-muted">
                        {stage.summary}
                      </p>
                    </div>
                  </li>
                );
              })}
            </ol>
          )}
          <p className="border-t border-line-subtle px-4 py-3 text-2xs leading-relaxed text-ink-faint">
            Negative results are shown deliberately. An experiment that failed is
            what rules an option out, and hiding it would misrepresent how the
            served pipeline was chosen.
          </p>
        </Panel>

        {/* ------------------------------------------------ what is not served */}
        <Panel>
          <PanelHeader title="Deliberately not served" />
          <div className="space-y-2.5 px-4 py-3">
            {health
              ? Object.entries(health.pipeline.excluded).map(([name, reason]) => (
                  <div key={name} className="flex gap-2.5">
                    <FlaskConical
                      className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-faint"
                      aria-hidden
                    />
                    <p className="text-xs leading-relaxed text-ink-muted">
                      <span className="text-ink">{name}:</span> {reason}
                    </p>
                  </div>
                ))
              : null}
            {health
              ? Object.entries(health.finding_models.unsupported_targets).map(
                  ([name, reason]) => (
                    <div key={name} className="flex gap-2.5">
                      <FlaskConical
                        className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-faint"
                        aria-hidden
                      />
                      <p className="text-xs leading-relaxed text-ink-muted">
                        <code className="text-ink">{name}</code>: {reason}
                      </p>
                    </div>
                  ),
                )
              : null}
          </div>
        </Panel>

        {/* ------------------------------------ real data vs simulated demo */}
        <Panel>
          <PanelHeader
            title="Real data vs simulated demonstration"
            subtitle="What is measured, and what is only a workflow demonstration"
          />
          <div className="space-y-3 px-4 py-3">
            <p className="max-w-3xl text-xs leading-relaxed text-ink-muted">
              The current SPIDER dataset provides cross-sectional studies rather
              than true longitudinal postoperative follow-up. The longitudinal
              tracker therefore uses a simulated demonstration workflow and does
              not claim postoperative recovery outcomes from SPIDER.
            </p>

            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-md border border-line-subtle bg-surface-2/40 p-3">
                <p className="label-caps mb-1.5 flex items-center gap-1.5">
                  <CheckCircle2 className="h-3 w-3 text-seg-canal" aria-hidden />
                  Real data
                </p>
                <ul className="space-y-1 text-2xs leading-relaxed text-ink-muted">
                  <li>Every uploaded or sample study analysed by the frozen pipeline</li>
                  <li>All segmentation, disc indexing and measurements on screen</li>
                  <li>All model-estimated findings and their validated metrics</li>
                  <li>The benchmark figures above, on a 33-patient held-out split</li>
                  <li>
                    The four studies behind the demonstration stages — real SPIDER
                    volumes, real pipeline output
                  </li>
                </ul>
              </div>

              <div className="rounded-md border border-severity-high/40 bg-severity-high/10 p-3">
                <p className="label-caps mb-1.5 flex items-center gap-1.5">
                  <XCircle className="h-3 w-3 text-severity-high" aria-hidden />
                  Simulated demonstration workflow
                </p>
                <ul className="space-y-1 text-2xs leading-relaxed text-ink-muted">
                  <li>
                    The <strong>timeline</strong> in the Recovery Tracker: four
                    different studies from four different patients, arranged into
                    stage positions
                  </li>
                  <li>
                    Stage names such as Post-Surgery and 3-Month — positions in a
                    demonstration, not observed timepoints
                  </li>
                  <li>
                    Any trend across stages, which follows from which studies were
                    selected
                  </li>
                </ul>
                <p className="mt-2 text-2xs leading-relaxed text-severity-high">
                  These stages are not postoperative follow-up scans from the same
                  patient, and no recovery outcome is claimed.
                </p>
              </div>
            </div>

            <div>
              <p className="label-caps mb-1">Why the demonstration is not real longitudinal data</p>
              <p className="max-w-3xl text-2xs leading-relaxed text-ink-faint">
                SPIDER images each patient once. It contains no postoperative
                imaging, no surgical record, no intervention date and no follow-up
                assessment; the time fields exist in the schema but are null in the
                values. With no second timepoint and no outcome label there is
                nothing to train a progression model against and nothing to
                validate one with.
              </p>
            </div>

            <div>
              <p className="label-caps mb-1">How real longitudinal data would replace it</p>
              <p className="max-w-3xl text-2xs leading-relaxed text-ink-faint">
                The demonstration is defined entirely by{" "}
                <code className="text-ink">demo/longitudinal_cases/&lt;case&gt;/manifest.json</code>.
                A genuine dataset would replace that manifest with one whose stages
                point at the same patient&apos;s successive studies, set real
                <code className="text-ink"> acquisition_timestamp</code> values, and
                declare a case type other than{" "}
                <code className="text-ink">SIMULATED_LONGITUDINAL_DEMO</code>. The
                loader deliberately refuses that type today, so enabling real
                follow-up is an explicit, reviewable change rather than something
                that could happen by accident. The disc-level analysis, viewers and
                comparison panel would work unchanged, because they already read
                per-stage pipeline output rather than anything stage-specific.
              </p>
            </div>

            <div>
              <p className="label-caps mb-1">2D / 3D viewing</p>
              <p className="max-w-3xl text-2xs leading-relaxed text-ink-faint">
                Slices are composed server-side and sent as PNGs, so a hidden class
                is never transmitted. The 3D view is genuine WebGL volume rendering
                (VTK.js) over a single cached volume request per analysis; adding it
                means voxels now do cross the network on that one path, which is a
                deliberate trade recorded in the architecture document. Both views
                read one selected-disc state and one server-side finding decision,
                so they cannot disagree.
              </p>
            </div>

            <div>
              <p className="label-caps mb-1">Current limitations</p>
              <ul className="max-w-3xl space-y-1 text-2xs leading-relaxed text-ink-faint">
                <li>No true longitudinal or postoperative capability</li>
                <li>Single dataset, no external validation; 33 test patients</li>
                <li>2-D slice-wise segmentation; no 3-D context in the model</li>
                <li>Disc indexing is 93.84% correct, not 100%</li>
                <li>Pfirrmann restricted to T2 and T2-SPACE series</li>
                <li>Herniation is weakly validated on 18 positive test discs</li>
                <li>No composite severity or recovery score, by design</li>
                <li>Local deployment with no authentication</li>
              </ul>
            </div>
          </div>
        </Panel>
      </div>
    </div>
  );
}
