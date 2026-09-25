/**
 * Processing view.
 *
 * Every number here comes from the backend's status endpoint. There is no
 * client-side timer driving a fake bar: if the backend reports 60%, this shows
 * 60% and stays there until the backend moves.
 */

"use client";

import { AlertTriangle, Check, Loader2 } from "lucide-react";
import * as React from "react";

import { PROCESSING_STAGES } from "@/lib/theme";
import type { StatusResponse } from "@/lib/types";
import { Button, ErrorState, Panel, PanelHeader, ProgressBar, cn } from "./ui";
import { formatSeconds } from "@/lib/api";

/** Stages shown in the UI, mapped from the backend's stage names. */
const DISPLAY_STAGES = [
  { key: "Upload validated", label: "Upload" },
  { key: "Preprocessing", label: "Preprocessing" },
  { key: "Segmentation", label: "Segmentation" },
  { key: "Disc indexing", label: "Disc Indexing" },
  { key: "Feature extraction", label: "Feature Extraction" },
  { key: "Radiological analysis", label: "Report Generation" },
] as const;

function stagePosition(stage: string): number {
  const index = PROCESSING_STAGES.indexOf(stage as never);
  return index < 0 ? 0 : index;
}

export function ProcessingView({
  status,
  onRetry,
}: {
  status: StatusResponse;
  onRetry?: () => void;
}) {
  const failed = status.status === "failed";
  const currentPosition = stagePosition(status.stage);

  return (
    <div className="mx-auto max-w-2xl space-y-4 p-4 md:p-6">
      <Panel>
        <PanelHeader
          title={failed ? "Analysis failed" : "Analysing study"}
          subtitle={
            failed
              ? "The pipeline stopped before completing"
              : "Running the validated segmentation and indexing pipeline"
          }
        />

        <div className="space-y-5 px-4 py-5">
          {/* ------------------------------------------ overall progress */}
          <div>
            <div className="mb-2 flex items-baseline justify-between gap-3">
              <span className="text-sm text-ink">
                {failed ? "Stopped" : status.stage}
              </span>
              <span className="font-mono text-sm text-accent">
                {status.progress}%
              </span>
            </div>
            <ProgressBar
              value={status.progress}
              label="Analysis progress"
              className={failed ? "opacity-40" : undefined}
            />
            <div className="mt-2 flex items-baseline justify-between text-2xs text-ink-faint">
              <span>
                Elapsed {formatSeconds(status.elapsed_seconds)}
              </span>
              <span className="capitalize">{status.status}</span>
            </div>
          </div>

          {/* ------------------------------------------ stage list */}
          <ol className="space-y-1" role="list">
            {DISPLAY_STAGES.map((stage) => {
              const position = stagePosition(stage.key);
              const done =
                status.stages_completed.includes(stage.key) ||
                status.progress === 100 ||
                position < currentPosition;
              const active = !failed && !done && stage.key === status.stage;
              const pending = !done && !active;

              return (
                <li
                  key={stage.key}
                  className={cn(
                    "flex items-center gap-3 rounded-md px-2.5 py-2 transition-colors",
                    active && "bg-accent/10",
                  )}
                >
                  <span
                    className={cn(
                      "grid h-5 w-5 shrink-0 place-items-center rounded-full border",
                      done
                        ? "border-seg-canal/40 bg-seg-canal/15 text-seg-canal"
                        : active
                          ? "border-accent/40 bg-accent/15 text-accent"
                          : "border-line bg-surface-2 text-ink-faint",
                    )}
                  >
                    {done ? (
                      <Check className="h-3 w-3" aria-hidden />
                    ) : active ? (
                      <Loader2 className="h-3 w-3 animate-spin" aria-hidden />
                    ) : (
                      <span className="h-1 w-1 rounded-full bg-current" aria-hidden />
                    )}
                  </span>
                  <span
                    className={cn(
                      "text-xs",
                      done
                        ? "text-ink-muted"
                        : active
                          ? "text-ink"
                          : "text-ink-faint",
                    )}
                  >
                    {stage.label}
                  </span>
                  {active ? (
                    <span className="ml-auto text-2xs text-accent">
                      in progress
                    </span>
                  ) : done ? (
                    <span className="ml-auto text-2xs text-ink-faint">done</span>
                  ) : (
                    <span className="ml-auto text-2xs text-ink-faint" aria-hidden>
                      pending
                    </span>
                  )}
                </li>
              );
            })}
          </ol>

          {failed && status.error ? (
            <ErrorState
              title={status.error.message}
              message={
                status.error.details &&
                typeof status.error.details === "object" &&
                "stage" in status.error.details
                  ? `The pipeline stopped during ${String(
                      (status.error.details as Record<string, unknown>).stage,
                    ).toLowerCase()}. No partial result is reported.`
                  : "No partial result is reported."
              }
              code={status.error.code}
              onRetry={onRetry}
            />
          ) : null}

          {!failed ? (
            <p className="flex gap-2 text-2xs leading-relaxed text-ink-faint">
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" aria-hidden />
              <span>
                Progress reflects the real pipeline stage reported by the backend.
                Inference runs on CPU by default, so a large study takes longer.
              </span>
            </p>
          ) : null}
        </div>
      </Panel>
    </div>
  );
}
