"use client";

/**
 * Recovery Tracker page.
 *
 * Loads the simulated demonstration case, then runs each stage's real source study
 * through the real pipeline. Nothing is pre-computed and no result is cached on
 * the server, so the first visit genuinely analyses four studies - roughly three
 * seconds each on CPU. Progress is reported rather than hidden.
 *
 * Analysis ids are remembered in sessionStorage so navigating back does not
 * re-analyse. A remembered id is verified before use; if the analysis has been
 * deleted the stage is simply analysed again.
 */

import { AlertTriangle, FlaskConical, Loader2 } from "lucide-react";
import Link from "next/link";
import * as React from "react";

import {
  RecoveryTracker,
  type StageState,
} from "@/components/RecoveryTracker";
import { Button, Panel, PanelHeader } from "@/components/ui";
import { api } from "@/lib/api";
import { IMAGING_STAGE_IDS, TIMELINE } from "@/lib/recovery";
import type { LongitudinalCase } from "@/lib/types";

const CASE_ID = "LS-DEMO-001";
const CACHE_KEY = `lumbar.recovery.${CASE_ID}`;

function readCache(): Record<string, string> {
  if (typeof window === "undefined") return {};
  try {
    return JSON.parse(window.sessionStorage.getItem(CACHE_KEY) ?? "{}");
  } catch {
    return {};
  }
}

function writeCache(value: Record<string, string>) {
  try {
    window.sessionStorage.setItem(CACHE_KEY, JSON.stringify(value));
  } catch {
    // A full or disabled sessionStorage is not worth failing the page over.
  }
}

export default function RecoveryPage() {
  const [demoCase, setDemoCase] = React.useState<LongitudinalCase | null>(null);
  const [caseError, setCaseError] = React.useState<string | null>(null);
  const [stages, setStages] = React.useState<Record<string, StageState>>({});
  const [activeNodeId, setActiveNodeId] = React.useState(TIMELINE[0]!.id);
  const [progress, setProgress] = React.useState<string | null>(null);

  /* ------------------------------------------------ load the case manifest */
  React.useEffect(() => {
    let alive = true;
    api
      .longitudinalCase(CASE_ID)
      .then((value) => alive && setDemoCase(value))
      .catch((cause) =>
        alive &&
        setCaseError(
          cause instanceof Error
            ? cause.message
            : "The demonstration case could not be loaded.",
        ),
      );
    return () => {
      alive = false;
    };
  }, []);

  /* ------------------------------------------------ analyse each stage */
  React.useEffect(() => {
    if (!demoCase) return;
    let alive = true;
    const cache = readCache();

    async function ensureStage(stageId: string): Promise<void> {
      const stage = demoCase!.stages.find((s) => s.stage_id === stageId);
      if (!stage) return;

      if (!stage.available) {
        setStages((current) => ({
          ...current,
          [stageId]: {
            status: "failed",
            error:
              stage.unavailable_reason ??
              "This stage's source study is not present on this machine.",
          },
        }));
        return;
      }

      setStages((current) => ({ ...current, [stageId]: { status: "loading" } }));

      // A remembered analysis is reused only if it still resolves.
      const remembered = cache[stageId];
      if (remembered) {
        try {
          const result = await api.result(remembered);
          if (!alive) return;
          setStages((current) => ({
            ...current,
            [stageId]: { status: "ready", analysisId: remembered, result },
          }));
          return;
        } catch {
          delete cache[stageId];
        }
      }

      try {
        const created = await api.loadDemoStage(CASE_ID, stageId);
        if (!alive) return;
        await api.run(created.analysis_id);

        // Poll the real progress rather than guessing a duration.
        for (;;) {
          if (!alive) return;
          const state = await api.status(created.analysis_id);
          if (state.status === "completed") break;
          if (state.status === "failed") {
            throw new Error(
              state.error?.message ?? "The pipeline failed on this study.",
            );
          }
          setProgress(
            `${stage.label}: ${state.stage} ${state.progress}%`,
          );
          await new Promise((resolve) => setTimeout(resolve, 350));
        }

        const result = await api.result(created.analysis_id);
        if (!alive) return;
        cache[stageId] = created.analysis_id;
        writeCache(cache);
        setStages((current) => ({
          ...current,
          [stageId]: {
            status: "ready",
            analysisId: created.analysis_id,
            result,
          },
        }));
      } catch (cause) {
        if (!alive) return;
        setStages((current) => ({
          ...current,
          [stageId]: {
            status: "failed",
            error:
              cause instanceof Error
                ? cause.message
                : "This stage could not be analysed.",
          },
        }));
      }
    }

    (async () => {
      // The active stage first, so the viewers fill as soon as possible; the rest
      // follow because the comparison panel needs all of them.
      const activeStageId =
        TIMELINE.find((n) => n.id === activeNodeId)?.stageId ??
        IMAGING_STAGE_IDS[0]!;
      const order = [
        activeStageId,
        ...IMAGING_STAGE_IDS.filter((id) => id !== activeStageId),
      ];
      for (const stageId of order) {
        if (!alive) return;
        await ensureStage(stageId);
      }
      if (alive) setProgress(null);
    })();

    return () => {
      alive = false;
    };
    // Intentionally keyed on the case only: re-running per stage selection would
    // restart the queue every time the user clicks a stage.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demoCase]);

  if (caseError) {
    return (
      <>
        <div className="p-4 md:p-6">
          <Panel>
            <PanelHeader
              title="Demonstration unavailable"
              subtitle="The simulated longitudinal case could not be loaded"
            />
            <div className="space-y-3 px-4 py-4">
              <p className="flex items-start gap-2 text-xs leading-relaxed text-ink-muted">
                <AlertTriangle
                  className="mt-0.5 h-4 w-4 shrink-0 text-severity-high"
                  aria-hidden
                />
                <span>{caseError}</span>
              </p>
              <p className="text-2xs leading-relaxed text-ink-faint">
                The demonstration needs the SPIDER dataset extracted locally.
                Run <span className="font-mono">scripts/01_extract.py</span>, or
                analyse a single study instead.
              </p>
              <Link href="/new">
                <Button size="sm" variant="secondary">
                  Go to New Analysis
                </Button>
              </Link>
            </div>
          </Panel>
        </div>
      </>
    );
  }

  if (!demoCase) {
    return (
      <div className="grid min-h-[60vh] place-items-center p-6">
        <div className="text-center">
          <FlaskConical className="mx-auto h-6 w-6 text-accent" aria-hidden />
          <p className="mt-2 flex items-center gap-2 text-sm text-ink">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            Loading the simulated longitudinal demonstration…
          </p>
        </div>
      </div>
    );
  }

  return (
    <RecoveryTracker
      demoCase={demoCase}
      stages={stages}
      activeNodeId={activeNodeId}
      onSelectNode={setActiveNodeId}
      loadingLabel={progress}
    />
  );
}
