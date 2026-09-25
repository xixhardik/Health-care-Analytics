"use client";

import { Download, FileText } from "lucide-react";
import Link from "next/link";
import * as React from "react";

import { PageHeader } from "@/components/AppShell";
import {
  Badge,
  Button,
  EmptyState,
  ErrorState,
  Panel,
  Skeleton,
} from "@/components/ui";
import { ApiError, api, formatDate } from "@/lib/api";
import type { HistoryResponse } from "@/lib/types";

export default function ReportsPage() {
  const [data, setData] = React.useState<HistoryResponse | null>(null);
  const [error, setError] = React.useState<ApiError | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const next = await api.history(200);
        if (!cancelled) setData(next);
      } catch (caught) {
        if (!cancelled)
          setError(
            caught instanceof ApiError
              ? caught
              : new ApiError("UNKNOWN", "Could not load reports.", 0),
          );
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Only completed analyses have a report.
  const reports = (data?.items ?? []).filter((i) => i.status === "completed");

  return (
    <div>
      <PageHeader
        title="Reports"
        description="Structured reports for completed analyses. Every report is generated from the stored result, so it always matches what the pipeline produced."
      />

      <div className="p-4 md:p-6">
        {error ? (
          <div className="mb-4">
            <ErrorState title="Cannot load reports" message={error.message} code={error.code} />
          </div>
        ) : null}

        <Panel>
          {loading ? (
            <div className="space-y-2 p-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-11" />
              ))}
            </div>
          ) : reports.length === 0 ? (
            <EmptyState
              icon={<FileText className="h-5 w-5" aria-hidden />}
              title="No reports yet"
              description="A report becomes available as soon as an analysis completes."
              action={
                <Link href="/new">
                  <Button variant="primary">Start New Analysis</Button>
                </Link>
              }
            />
          ) : (
            <ul className="divide-y divide-line-subtle" role="list">
              {reports.map((item) => (
                <li
                  key={item.analysis_id}
                  className="flex flex-wrap items-center gap-3 px-4 py-3 hover:bg-surface-2/50"
                >
                  <Link href={`/reports/${item.analysis_id}`} className="min-w-0 flex-1">
                    <span className="block truncate text-xs text-ink">
                      {item.filename}
                    </span>
                    <span className="mt-0.5 block font-mono text-2xs text-ink-faint">
                      {formatDate(item.created_at)} · {item.analysis_id}
                    </span>
                  </Link>
                  <Badge tone="muted">
                    {item.disc_count ?? "—"} discs
                  </Badge>
                  <Link href={`/reports/${item.analysis_id}`}>
                    <Button size="sm" variant="secondary" icon={<FileText className="h-3.5 w-3.5" />}>
                      View
                    </Button>
                  </Link>
                  <a href={api.downloadUrl(item.analysis_id, "md")} download>
                    <Button size="sm" variant="ghost" icon={<Download className="h-3.5 w-3.5" />}>
                      Download
                    </Button>
                  </a>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </div>
  );
}
