"use client";

import { Play } from "lucide-react";
import { useParams, useRouter } from "next/navigation";
import * as React from "react";

import { ProcessingView } from "@/components/ProcessingView";
import { ResultsView } from "@/components/ResultsView";
import { Button, ErrorState, Panel, Skeleton } from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import type { AnalysisResult, StatusResponse } from "@/lib/types";

/** Poll interval while a job is running. */
const POLL_MS = 1200;

export default function AnalysisPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const id = params?.id;

  const [status, setStatus] = React.useState<StatusResponse | null>(null);
  const [result, setResult] = React.useState<AnalysisResult | null>(null);
  const [error, setError] = React.useState<ApiError | null>(null);
  const [starting, setStarting] = React.useState(false);

  // Poll status until the job reaches a terminal state, then fetch the result
  // once. Polling stops as soon as it is no longer needed.
  React.useEffect(() => {
    if (!id) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const tick = async () => {
      try {
        const next = await api.status(id);
        if (cancelled) return;
        setStatus(next);
        setError(null);

        if (next.status === "completed") {
          const payload = await api.result(id);
          if (!cancelled) setResult(payload);
          return; // terminal: stop polling
        }
        if (next.status === "failed") return; // terminal

        timer = setTimeout(tick, POLL_MS);
      } catch (caught) {
        if (cancelled) return;
        setError(
          caught instanceof ApiError
            ? caught
            : new ApiError("UNKNOWN", "Could not load this analysis.", 0),
        );
      }
    };

    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [id]);

  const start = async () => {
    if (!id) return;
    setStarting(true);
    try {
      await api.run(id);
      const next = await api.status(id);
      setStatus(next);
      // Re-enter the polling effect by nudging state; the effect keys on id, so
      // restart it manually here.
      setTimeout(() => void api.status(id).then(setStatus), POLL_MS);
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught
          : new ApiError("RUN_FAILED", "Could not start the analysis.", 0),
      );
    } finally {
      setStarting(false);
    }
  };

  if (!id) return null;

  if (error && !status) {
    return (
      <div className="mx-auto max-w-2xl p-4 md:p-6">
        <ErrorState
          title="Analysis unavailable"
          message={error.message}
          code={error.code}
          onRetry={() => router.refresh()}
        />
        <div className="mt-3">
          <Button variant="secondary" onClick={() => router.push("/history")}>
            Back to history
          </Button>
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="mx-auto max-w-2xl space-y-3 p-4 md:p-6">
        <Skeleton className="h-8 w-48" />
        <Panel className="p-4">
          <Skeleton className="h-2 w-full" />
          <div className="mt-4 space-y-2">
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="h-8" />
            ))}
          </div>
        </Panel>
      </div>
    );
  }

  // Uploaded but never started: offer to start rather than polling forever.
  if (status.status === "queued" && status.progress === 0 && !status.started_at) {
    return (
      <div className="mx-auto max-w-lg p-4 md:p-6">
        <Panel className="p-6 text-center">
          <h1 className="text-sm font-medium text-ink">Ready to analyse</h1>
          <p className="mx-auto mt-1.5 max-w-sm text-xs leading-relaxed text-ink-faint">
            This study has been uploaded and validated but not yet processed.
          </p>
          <Button
            variant="primary"
            className="mt-5"
            icon={<Play className="h-4 w-4" />}
            state={starting ? "loading" : "idle"}
            onClick={start}
          >
            Start analysis
          </Button>
        </Panel>
      </div>
    );
  }

  if (status.status === "completed" && result) {
    return <ResultsView result={result} />;
  }

  return <ProcessingView status={status} onRetry={start} />;
}
