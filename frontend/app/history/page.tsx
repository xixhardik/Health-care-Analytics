"use client";

import { FileScan, FilePlus2, Trash2 } from "lucide-react";
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

export default function HistoryPage() {
  const [data, setData] = React.useState<HistoryResponse | null>(null);
  const [error, setError] = React.useState<ApiError | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [removing, setRemoving] = React.useState<string | null>(null);

  const load = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await api.history(200));
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught
          : new ApiError("UNKNOWN", "Could not load history.", 0),
      );
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    void load();
  }, [load]);

  const remove = async (id: string) => {
    setRemoving(id);
    try {
      await api.remove(id);
      await load();
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught
          : new ApiError("DELETE_FAILED", "Could not delete the analysis.", 0),
      );
    } finally {
      setRemoving(null);
    }
  };

  return (
    <div>
      <PageHeader
        title="Analysis History"
        description="Analyses are stored locally by the backend. Deleting one removes its upload, result and cached slice images."
        actions={
          <Link href="/new">
            <Button variant="primary" icon={<FilePlus2 className="h-4 w-4" />}>
              New Analysis
            </Button>
          </Link>
        }
      />

      <div className="p-4 md:p-6">
        {error ? (
          <div className="mb-4">
            <ErrorState
              title="Cannot load history"
              message={error.message}
              code={error.code}
              onRetry={load}
            />
          </div>
        ) : null}

        <Panel>
          {loading && !data ? (
            <div className="space-y-2 p-4">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-11" />
              ))}
            </div>
          ) : !data || data.items.length === 0 ? (
            <EmptyState
              icon={<FileScan className="h-5 w-5" aria-hidden />}
              title="No MRI analyses yet"
              description="Once you analyse a study it will appear here with its status, slice count and disc findings."
              action={
                <Link href="/new">
                  <Button variant="primary">Start New Analysis</Button>
                </Link>
              }
            />
          ) : (
            <ul className="divide-y divide-line-subtle" role="list">
              {data.items.map((item) => (
                <li
                  key={item.analysis_id}
                  className="flex flex-wrap items-center gap-3 px-4 py-3 hover:bg-surface-2/50"
                >
                  <Link
                    href={`/analysis/${item.analysis_id}`}
                    className="min-w-0 flex-1"
                  >
                    <span className="block truncate text-xs text-ink">
                      {item.filename}
                    </span>
                    <span className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-2xs text-ink-faint">
                      <span>{formatDate(item.created_at)}</span>
                      <span>{item.analysis_id}</span>
                      {item.modality ? (
                        <span className="uppercase">{item.modality}</span>
                      ) : null}
                    </span>
                  </Link>

                  <span className="flex items-center gap-4 font-mono text-2xs text-ink-muted">
                    <span title="Slices">{item.slice_count ?? "—"} sl</span>
                    <span title="Discs identified">{item.disc_count ?? "—"} discs</span>
                    <span title="Discs with a positive finding">
                      {item.findings_count ?? "—"} find
                    </span>
                  </span>

                  <Badge
                    tone={
                      item.status === "completed"
                        ? "positive"
                        : item.status === "failed"
                          ? "danger"
                          : item.status === "processing"
                            ? "accent"
                            : "muted"
                    }
                  >
                    {item.status}
                  </Badge>

                  <Button
                    size="sm"
                    variant="ghost"
                    aria-label={`Delete analysis ${item.analysis_id}`}
                    state={removing === item.analysis_id ? "loading" : "idle"}
                    onClick={() => void remove(item.analysis_id)}
                  >
                    <Trash2 className="h-3.5 w-3.5" aria-hidden />
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </div>
  );
}
