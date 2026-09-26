"use client";

import {
  ArrowRight,
  CheckCircle2,
  FileScan,
  FilePlus2,
  FlaskConical,
  Loader2,
  XCircle,
} from "lucide-react";
import Link from "next/link";
import * as React from "react";

import { PageHeader } from "@/components/AppShell";
import {
  Badge,
  Button,
  EmptyState,
  ErrorState,
  Panel,
  PanelHeader,
  Skeleton,
  StatCard,
} from "@/components/ui";
import { ApiError, api, formatDate } from "@/lib/api";
import { CROSS_STUDY_DISCLAIMER } from "@/lib/recovery";
import type { HealthResponse, HistoryResponse } from "@/lib/types";

function StatusBadge({ status }: { status: string }) {
  switch (status) {
    case "completed":
      return <Badge tone="positive">Completed</Badge>;
    case "processing":
      return <Badge tone="accent">Processing</Badge>;
    case "failed":
      return <Badge tone="danger">Failed</Badge>;
    default:
      return <Badge tone="muted">Queued</Badge>;
  }
}

export default function DashboardPage() {
  const [data, setData] = React.useState<HistoryResponse | null>(null);
  const [health, setHealth] = React.useState<HealthResponse | null>(null);
  const [error, setError] = React.useState<ApiError | null>(null);
  const [loading, setLoading] = React.useState(true);

  // Pipeline identity and every research figure come from the API, so the UI
  // holds no hard-coded experimental number.
  React.useEffect(() => {
    let cancelled = false;
    void api
      .health()
      .then((next) => {
        if (!cancelled) setHealth(next);
      })
      .catch(() => {
        /* the health badge in the shell already reports an offline API */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const load = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await api.history(10));
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught
          : new ApiError("UNKNOWN", "Could not load analyses.", 0),
      );
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    void load();
  }, [load]);

  const counts = data?.counts ?? {};
  const total = data?.total ?? 0;

  return (
    <div>
      <PageHeader
        title="Overview"
        description="AI-assisted lumbar spine MRI research analysis. Upload a sagittal study to run segmentation, disc indexing and quantitative measurement."
        actions={
          <Link href="/new">
            <Button variant="primary" icon={<FilePlus2 className="h-4 w-4" />}>
              New Analysis
            </Button>
          </Link>
        }
      />

      <div className="space-y-5 p-4 md:p-6">
        {/* ------------------------------------------------ stats */}
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {loading && !data ? (
            Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-[92px]" />
            ))
          ) : (
            <>
              <StatCard label="Total analyses" value={total} />
              <StatCard
                label="Completed"
                value={counts.completed ?? 0}
                tone="positive"
              />
              <StatCard
                label="Processing"
                value={(counts.processing ?? 0) + (counts.queued ?? 0)}
                tone="accent"
              />
              <StatCard
                label="Failed"
                value={counts.failed ?? 0}
                tone={counts.failed ? "danger" : "neutral"}
              />
            </>
          )}
        </div>

        {error ? (
          <ErrorState
            title="Cannot load analyses"
            message={error.message}
            code={error.code}
            onRetry={load}
          />
        ) : null}

        {/* ---------------------------------- longitudinal demonstration */}
        <Panel>
          <PanelHeader
            title="Longitudinal workflow demonstration"
            subtitle="Simulated timeline across four different research studies"
            actions={
              <Link href="/recovery">
                <Button
                  size="sm"
                  variant="secondary"
                  icon={<FlaskConical className="h-3.5 w-3.5" />}
                >
                  Load Longitudinal Demo
                </Button>
              </Link>
            }
          />
          <div className="px-4 py-3">
            <p className="max-w-3xl text-xs leading-relaxed text-ink-muted">
              The Recovery Tracker demonstrates how a longitudinal imaging workflow
              would operate. It stages four different SPIDER studies from four
              different patients; the imaging and every measurement are real
              pipeline output, but the timeline is simulated.
            </p>
            <p className="mt-1.5 max-w-3xl text-2xs leading-relaxed text-ink-faint">
              {CROSS_STUDY_DISCLAIMER}
            </p>
          </div>
        </Panel>

        {/* ------------------------------------------------ recent */}
        <Panel>
          <PanelHeader
            title="Recent MRI analyses"
            subtitle="Most recent first"
            actions={
              total > 0 ? (
                <Link href="/history">
                  <Button size="sm" variant="ghost">
                    View all
                    <ArrowRight className="h-3.5 w-3.5" aria-hidden />
                  </Button>
                </Link>
              ) : null
            }
          />

          {loading && !data ? (
            <div className="space-y-2 p-4">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} className="h-10" />
              ))}
            </div>
          ) : !data || data.items.length === 0 ? (
            <EmptyState
              icon={<FileScan className="h-5 w-5" aria-hidden />}
              title="No MRI analyses yet"
              description="Upload a lumbar spine MRI volume to run the validated segmentation and disc-indexing pipeline. Nothing is analysed until you start it."
              action={
                <Link href="/new">
                  <Button variant="primary" icon={<FilePlus2 className="h-4 w-4" />}>
                    Start New Analysis
                  </Button>
                </Link>
              }
            />
          ) : (
            <>
              {/* Desktop table */}
              <div className="hidden overflow-x-auto md:block">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-line-subtle">
                      <th scope="col" className="px-4 py-2 font-medium text-ink-faint">Study</th>
                      <th scope="col" className="px-4 py-2 font-medium text-ink-faint">Date</th>
                      <th scope="col" className="px-4 py-2 font-medium text-ink-faint">Status</th>
                      <th scope="col" className="px-4 py-2 text-right font-medium text-ink-faint">Slices</th>
                      <th scope="col" className="px-4 py-2 text-right font-medium text-ink-faint">Discs</th>
                      <th scope="col" className="px-4 py-2 text-right font-medium text-ink-faint">Findings</th>
                      <th scope="col" className="px-4 py-2 text-right font-medium text-ink-faint">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.items.map((item) => (
                      <tr key={item.analysis_id} className="table-row">
                        <td className="max-w-[220px] truncate px-4 py-2.5 text-ink">
                          {item.filename}
                          {item.modality ? (
                            <span className="ml-2 font-mono text-2xs uppercase text-ink-faint">
                              {item.modality}
                            </span>
                          ) : null}
                        </td>
                        <td className="whitespace-nowrap px-4 py-2.5 text-ink-muted">
                          {formatDate(item.created_at)}
                        </td>
                        <td className="px-4 py-2.5">
                          <StatusBadge status={item.status} />
                        </td>
                        <td className="px-4 py-2.5 text-right font-mono text-ink-muted">
                          {item.slice_count ?? "—"}
                        </td>
                        <td className="px-4 py-2.5 text-right font-mono text-ink-muted">
                          {item.disc_count ?? "—"}
                        </td>
                        <td className="px-4 py-2.5 text-right font-mono text-ink-muted">
                          {item.findings_count ?? "—"}
                        </td>
                        <td className="px-4 py-2.5 text-right">
                          <Link href={`/analysis/${item.analysis_id}`}>
                            <Button size="sm" variant="ghost">
                              Open
                            </Button>
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Mobile cards */}
              <ul className="divide-y divide-line-subtle md:hidden" role="list">
                {data.items.map((item) => (
                  <li key={item.analysis_id}>
                    <Link
                      href={`/analysis/${item.analysis_id}`}
                      className="flex items-center gap-3 px-4 py-3 hover:bg-surface-2"
                    >
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-xs text-ink">
                          {item.filename}
                        </span>
                        <span className="mt-1 flex items-center gap-2">
                          <StatusBadge status={item.status} />
                          <span className="font-mono text-2xs text-ink-faint">
                            {item.slice_count ?? "—"} slices
                            {item.disc_count != null
                              ? ` · ${item.disc_count} discs`
                              : ""}
                          </span>
                        </span>
                      </span>
                      <ArrowRight className="h-4 w-4 shrink-0 text-ink-faint" aria-hidden />
                    </Link>
                  </li>
                ))}
              </ul>
            </>
          )}
        </Panel>

        {/* ---------------------------------------- pipeline, served by the API */}
        <Panel>
          <PanelHeader
            title="Active pipeline"
            subtitle={
              health
                ? `${health.pipeline.version} · research evaluation results, measured on a ${health.validated_metrics.test_patients}-patient held-out test split`
                : "Loading pipeline information"
            }
            actions={
              health ? (
                <Link href="/methodology">
                  <Button size="sm" variant="ghost">
                    Methodology
                    <ArrowRight className="h-3.5 w-3.5" aria-hidden />
                  </Button>
                </Link>
              ) : null
            }
          />
          {!health ? (
            <div className="space-y-2 p-4">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-6" />
              ))}
            </div>
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-2 border-b border-line-subtle px-4 py-3">
                <Badge tone="accent">{health.pipeline.version}</Badge>
                <Badge tone="muted">
                  Segmentation {health.pipeline.segmentation}
                </Badge>
                <Badge tone="muted">
                  Post-processing {health.pipeline.postprocessing}
                </Badge>
              </div>
              <div className="grid gap-x-8 gap-y-1 px-4 py-3 sm:grid-cols-2">
                {health.metric_table.map((row) => (
                  <div
                    key={row.label}
                    className="flex items-baseline justify-between gap-3 border-b border-line-subtle py-1.5 last:border-0 sm:last:border-b"
                  >
                    <span className="min-w-0 truncate text-xs text-ink-muted">
                      {row.label}
                    </span>
                    <span className="flex shrink-0 items-baseline gap-2">
                      <span className="font-mono text-xs text-ink">{row.value}</span>
                      <Badge tone={row.kind === "segmentation" ? "accent" : "positive"}>
                        {row.sprint}
                      </Badge>
                    </span>
                  </div>
                ))}
              </div>
              <p className="border-t border-line-subtle px-4 py-3 text-2xs leading-relaxed text-ink-faint">
                {health.validated_metrics.note}
              </p>
            </>
          )}
        </Panel>
      </div>
    </div>
  );
}
