"use client";

import Link from "next/link";
import * as React from "react";

import { PageHeader } from "@/components/AppShell";
import {
  Badge,
  Button,
  ErrorState,
  FieldRow,
  Panel,
  PanelHeader,
  Skeleton,
} from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";

const TECHNOLOGY = [
  { group: "Frontend", items: ["Next.js", "React", "TypeScript", "Tailwind CSS", "Recharts"] },
  { group: "Backend", items: ["FastAPI", "Pydantic", "Uvicorn"] },
  { group: "Machine learning", items: ["PyTorch", "scikit-learn", "SciPy", "NumPy"] },
  { group: "Medical imaging", items: ["SimpleITK", "OpenCV", "Pillow"] },
];

export default function AboutPage() {
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

  return (
    <div>
      <PageHeader
        title="About"
        description="Project scope, technology, and what this system deliberately does not claim."
      />

      <div className="mx-auto max-w-4xl space-y-5 p-4 md:p-6">
        {/* ------------------------------------------------ title card */}
        <Panel className="p-5">
          <p className="label-caps">Project</p>
          <h2 className="mt-2 text-lg font-semibold leading-snug tracking-tight text-ink">
            Automated Lumbar Spine MRI Segmentation and Disc-Level Abnormality
            Analysis
          </h2>
          <p className="mt-2.5 max-w-2xl text-xs leading-relaxed text-ink-muted">
            A research pipeline that segments vertebrae, intervertebral discs and
            the spinal canal in sagittal lumbar MRI, recovers disc identity across
            a series, measures each disc in millimetres, and estimates
            disc-level radiological findings where a validated model exists.
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Badge tone="accent">Academic / research prototype</Badge>
            {health ? <Badge tone="muted">{health.pipeline.version}</Badge> : null}
            {health ? <Badge tone="muted">{health.pipeline.dataset}</Badge> : null}
          </div>
        </Panel>

        {/* ------------------------------------------------ disclaimer */}
        <div className="rounded-lg border border-seg-disc/25 bg-seg-disc/5 px-4 py-3">
          <p className="text-xs font-medium text-ink">
            This application is an academic / research prototype.
          </p>
          <p className="mt-1 text-2xs leading-relaxed text-ink-muted">
            Results require expert radiological review. It does not provide a
            clinical diagnosis, it does not replace a radiologist, and it is not a
            medical device. It must not be used to make care decisions.
          </p>
        </div>

        {/* ------------------------------------------------ technology */}
        <Panel>
          <PanelHeader title="Technology" />
          <div className="grid gap-x-8 gap-y-3 px-4 py-3 sm:grid-cols-2">
            {TECHNOLOGY.map((group) => (
              <div key={group.group}>
                <p className="label-caps mb-1.5">{group.group}</p>
                <div className="flex flex-wrap gap-1.5">
                  {group.items.map((item) => (
                    <Badge key={item} tone="neutral">
                      {item}
                    </Badge>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Panel>

        {/* ------------------------------------------------ research pipeline */}
        <Panel>
          <PanelHeader
            title="Research pipeline"
            subtitle="Full detail on the Methodology page"
            actions={
              <Link href="/methodology">
                <Button size="sm" variant="secondary">
                  Methodology
                </Button>
              </Link>
            }
          />
          <div className="px-4 py-3">
            {health ? (
              <dl>
                <FieldRow label="Dataset" value={health.pipeline.dataset} />
                <FieldRow label="Pipeline version" value={health.pipeline.version} mono />
                <FieldRow
                  label="Preprocessing"
                  value={`${health.pipeline.preprocessing} — ${health.pipeline.preprocessing_detail}`}
                />
                <FieldRow
                  label="Segmentation"
                  value={`${health.pipeline.segmentation} — ${health.pipeline.segmentation_detail}`}
                />
                <FieldRow
                  label="Post-processing"
                  value={`${health.pipeline.postprocessing} — ${health.pipeline.postprocessing_detail}`}
                />
              </dl>
            ) : (
              <div className="space-y-2">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Skeleton key={i} className="h-6" />
                ))}
              </div>
            )}
          </div>
        </Panel>

        {/* ------------------------------------------------ limitations */}
        <Panel>
          <PanelHeader title="Deliberate limitations" />
          <ul className="space-y-2.5 px-4 py-3 text-xs leading-relaxed text-ink-muted">
            <li>
              <span className="text-ink">No anatomical level names.</span> Discs are
              reported by integer index counting upward from the most inferior disc.
              The source dataset does not state which vertebra is L5, so no L1–L5 or
              S1 label is asserted.
            </li>
            <li>
              <span className="text-ink">No Modic type.</span> Only a binary
              &ldquo;any Modic change&rdquo; target was validated. The nominal type
              is reported as unavailable rather than guessed.
            </li>
            <li>
              <span className="text-ink">No spondylolisthesis estimate.</span> Five
              positive cases in the test split is too few to validate.
            </li>
            <li>
              <span className="text-ink">No numeric confidence.</span> Evidence
              strength is qualitative. Converting a model probability into a
              percentage confidence would invent a quantity that was never
              validated.
            </li>
            <li>
              <span className="text-ink">No longitudinal or postoperative
              analysis.</span> The data this system was validated on contains no
              postoperative follow-up, so no healing, recovery or change-over-time
              assessment is offered. The result schema carries a single baseline
              timepoint so such a module could be added later without altering
              existing fields.
            </li>
            <li>
              <span className="text-ink">No composite severity score.</span> There is
              deliberately no single &ldquo;percentage damage&rdquo; number, because
              no such quantity was defined or validated.
            </li>
            <li>
              <span className="text-ink">No generated prose.</span> Summary text comes
              from deterministic templates filled with measured values, so it cannot
              drift from the numbers. No language model is involved.
            </li>
          </ul>
        </Panel>

        {/* ------------------------------------------------ security */}
        <Panel>
          <PanelHeader title="Deployment and privacy" />
          <ul className="space-y-2.5 px-4 py-3 text-xs leading-relaxed text-ink-muted">
            <li>
              <span className="text-ink">Local prototype, not deployed.</span> Intended
              to run on localhost for academic demonstration.
            </li>
            <li>
              <span className="text-ink">No authentication.</span> There are no user
              accounts and no access control, so the API must not be exposed to a
              network.
            </li>
            <li>
              <span className="text-ink">Local filesystem storage.</span> Uploads,
              results and cached slice images are written to a local directory. No
              external service receives any data.
            </li>
            <li>
              <span className="text-ink">Research data only.</span> Analysis ids are
              random tokens rather than derived from filenames, filenames are
              sanitised, and no patient identifier is used or displayed.
            </li>
            <li>
              <span className="text-ink">Not production clinical software.</span> It
              has not been through any clinical validation, regulatory assessment or
              security review.
            </li>
          </ul>
        </Panel>

        {/* ------------------------------------------------ runtime */}
        <Panel>
          <PanelHeader title="Runtime" subtitle="Reported live by the backend" />
          <div className="px-4 py-3">
            {error ? (
              <ErrorState message={error.message} code={error.code} />
            ) : !health ? (
              <div className="space-y-2">
                {Array.from({ length: 6 }).map((_, i) => (
                  <Skeleton key={i} className="h-6" />
                ))}
              </div>
            ) : (
              <dl>
                <FieldRow label="API version" value={health.api_version} mono />
                <FieldRow
                  label="Status"
                  value={
                    <Badge tone={health.status === "ok" ? "positive" : "warn"}>
                      {health.status}
                    </Badge>
                  }
                />
                <FieldRow
                  label="Architecture"
                  value={health.segmentation_model.architecture ?? "not loaded"}
                  mono
                />
                <FieldRow
                  label="Checkpoint"
                  value={health.segmentation_model.checkpoint_name ?? "—"}
                  mono
                />
                <FieldRow
                  label="Selection criterion"
                  value={health.segmentation_model.selection ?? "—"}
                />
                <FieldRow
                  label="Device"
                  value={health.segmentation_model.device.toUpperCase()}
                  mono
                />
                <FieldRow
                  label="Finding models served"
                  value={health.finding_models.supported_targets.join(", ")}
                />
                <FieldRow
                  label="Not served"
                  value={
                    Object.keys(health.finding_models.unsupported_targets).join(", ") ||
                    "—"
                  }
                />
                <FieldRow
                  label="Sample study available"
                  value={health.sample_study_available ? "yes" : "no"}
                />
              </dl>
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}
