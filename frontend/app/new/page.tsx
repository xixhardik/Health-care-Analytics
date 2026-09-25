"use client";

import {
  CheckCircle2,
  FileUp,
  FlaskConical,
  Play,
  ShieldAlert,
  Trash2,
  UploadCloud,
} from "lucide-react";
import { useRouter } from "next/navigation";
import * as React from "react";

import { PageHeader } from "@/components/AppShell";
import {
  Badge,
  Button,
  ErrorState,
  FieldRow,
  Panel,
  PanelHeader,
  cn,
} from "@/components/ui";
import { ApiError, api, formatBytes } from "@/lib/api";
import type { UploadResponse } from "@/lib/types";

const ACCEPTED = [".mha", ".mhd", ".nii", ".nii.gz"];

type Phase =
  | "select"
  | "uploading"
  | "sampling"
  | "validated"
  | "starting"
  | "error";

export default function NewAnalysisPage() {
  const router = useRouter();
  const inputRef = React.useRef<HTMLInputElement>(null);
  const [file, setFile] = React.useState<File | null>(null);
  const [phase, setPhase] = React.useState<Phase>("select");
  const [dragging, setDragging] = React.useState(false);
  const [uploaded, setUploaded] = React.useState<UploadResponse | null>(null);
  const [error, setError] = React.useState<ApiError | null>(null);
  const [sampleAvailable, setSampleAvailable] = React.useState(false);

  // The sample button only appears when the backend confirms the volume exists,
  // so it can never be clicked into a dead end.
  React.useEffect(() => {
    let cancelled = false;
    void api
      .health()
      .then((health) => {
        if (!cancelled) setSampleAvailable(health.sample_study_available);
      })
      .catch(() => {
        if (!cancelled) setSampleAvailable(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const hasAcceptedExtension = (name: string) =>
    ACCEPTED.some((ext) => name.toLowerCase().endsWith(ext));

  const reset = () => {
    setFile(null);
    setUploaded(null);
    setError(null);
    setPhase("select");
    if (inputRef.current) inputRef.current.value = "";
  };

  const beginUpload = React.useCallback(async (candidate: File) => {
    setError(null);
    setUploaded(null);

    // Client-side extension check first: no point spending a large upload to be
    // told the format is wrong. The backend re-validates regardless.
    if (!hasAcceptedExtension(candidate.name)) {
      setPhase("error");
      setError(
        new ApiError(
          "UNSUPPORTED_FORMAT",
          `'${candidate.name}' is not a supported MRI volume. Accepted formats: ${ACCEPTED.join(", ")}.`,
          0,
        ),
      );
      return;
    }

    setFile(candidate);
    setPhase("uploading");
    try {
      const response = await api.upload(candidate);
      setUploaded(response);
      setPhase("validated");
    } catch (caught) {
      setPhase("error");
      setError(
        caught instanceof ApiError
          ? caught
          : new ApiError("UPLOAD_FAILED", "The upload failed.", 0),
      );
    }
  }, []);

  const loadSample = async () => {
    setError(null);
    setUploaded(null);
    setFile(null);
    setPhase("sampling");
    try {
      const response = await api.loadSample();
      setUploaded(response);
      setPhase("validated");
    } catch (caught) {
      setPhase("error");
      setError(
        caught instanceof ApiError
          ? caught
          : new ApiError(
              "SAMPLE_UNAVAILABLE",
              "The sample study could not be loaded.",
              0,
            ),
      );
    }
  };

  const startAnalysis = async () => {
    if (!uploaded) return;
    setPhase("starting");
    try {
      await api.run(uploaded.analysis_id);
      router.push(`/analysis/${uploaded.analysis_id}`);
    } catch (caught) {
      setPhase("error");
      setError(
        caught instanceof ApiError
          ? caught
          : new ApiError("RUN_FAILED", "The analysis could not be started.", 0),
      );
    }
  };

  const onDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setDragging(false);
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) void beginUpload(dropped);
  };

  const study = uploaded?.study;

  return (
    <div>
      <PageHeader
        title="New Analysis"
        description="Upload a sagittal lumbar spine MRI volume. The file is validated on the server before any analysis runs."
      />

      <div className="grid gap-5 p-4 md:p-6 lg:grid-cols-[1fr_360px]">
        {/* ---------------------------------------------- dropzone */}
        <div className="space-y-4">
          <div
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className={cn(
              "rounded-lg border-2 border-dashed transition-colors",
              dragging
                ? "border-accent bg-accent/5"
                : "border-line bg-surface-1 hover:border-line-strong",
            )}
          >
            <div className="flex flex-col items-center px-6 py-12 text-center">
              <span
                className={cn(
                  "grid h-12 w-12 place-items-center rounded-full border",
                  dragging
                    ? "border-accent/40 bg-accent/10 text-accent"
                    : "border-line bg-surface-2 text-ink-faint",
                )}
              >
                <UploadCloud className="h-5 w-5" aria-hidden />
              </span>

              <h2 className="mt-4 text-sm font-medium text-ink">
                Upload Lumbar MRI
              </h2>
              <p className="mt-1.5 max-w-sm text-xs leading-relaxed text-ink-faint">
                Drag a volume here, or browse. Supported formats:{" "}
                <span className="font-mono text-ink-muted">
                  {ACCEPTED.join(" · ")}
                </span>
                . Maximum 300 MB.
              </p>

              <input
                ref={inputRef}
                type="file"
                accept={ACCEPTED.join(",")}
                className="sr-only"
                id="mri-file"
                onChange={(event) => {
                  const chosen = event.target.files?.[0];
                  if (chosen) void beginUpload(chosen);
                }}
              />
              <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
                <Button
                  variant="primary"
                  icon={<FileUp className="h-4 w-4" />}
                  state={phase === "uploading" ? "loading" : "idle"}
                  onClick={() => inputRef.current?.click()}
                >
                  {phase === "uploading" ? "Uploading…" : "Browse files"}
                </Button>
                {sampleAvailable ? (
                  <Button
                    variant="secondary"
                    icon={<FlaskConical className="h-4 w-4" />}
                    state={phase === "sampling" ? "loading" : "idle"}
                    onClick={loadSample}
                  >
                    Load Sample Study
                  </Button>
                ) : null}
                {file ? (
                  <Button
                    variant="ghost"
                    icon={<Trash2 className="h-4 w-4" />}
                    onClick={reset}
                    disabled={phase === "uploading" || phase === "starting"}
                  >
                    Remove
                  </Button>
                ) : null}
              </div>
              {sampleAvailable ? (
                <p className="mt-3 max-w-md text-2xs leading-relaxed text-ink-faint">
                  The sample study is a real volume from the research dataset,
                  processed by the identical pipeline. No result is pre-computed.
                </p>
              ) : null}
            </div>

            {file ? (
              <div className="flex items-center gap-3 border-t border-line-subtle px-4 py-3">
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-xs text-ink">
                    {file.name}
                  </span>
                  <span className="font-mono text-2xs text-ink-faint">
                    {formatBytes(file.size)}
                  </span>
                </span>
                {phase === "validated" ? (
                  <Badge tone="positive">
                    <CheckCircle2 className="h-3 w-3" aria-hidden />
                    Validated
                  </Badge>
                ) : phase === "uploading" ? (
                  <Badge tone="accent">Validating…</Badge>
                ) : phase === "error" ? (
                  <Badge tone="danger">Rejected</Badge>
                ) : null}
              </div>
            ) : null}
          </div>

          {error ? (
            <ErrorState
              title="Upload rejected"
              message={error.message}
              code={error.code}
              onRetry={reset}
            />
          ) : null}

          {(phase === "validated" || phase === "starting") && uploaded ? (
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-accent/30 bg-accent/5 p-4">
              <div>
                <p className="text-sm font-medium text-ink">
                  Ready to analyse
                </p>
                <p className="mt-0.5 text-xs text-ink-muted">
                  {uploaded.message}
                </p>
                {uploaded.is_sample ? (
                  <p className="mt-1 text-2xs text-accent">
                    Sample research study — processed by the real pipeline.
                  </p>
                ) : null}
              </div>
              <Button
                variant="primary"
                icon={<Play className="h-4 w-4" />}
                state={phase === "starting" ? "loading" : "idle"}
                onClick={startAnalysis}
              >
                Start analysis
              </Button>
            </div>
          ) : null}
        </div>

        {/* ---------------------------------------------- volume info */}
        <div className="space-y-4">
          <Panel>
            <PanelHeader title="Volume information" />
            <div className="px-4 py-3">
              {study ? (
                <dl>
                  <FieldRow label="File format" value={study.format} mono />
                  <FieldRow label="Size" value={formatBytes(study.size_bytes)} mono />
                  <FieldRow
                    label="Sagittal slices"
                    value={study.slice_count}
                    mono
                  />
                  <FieldRow
                    label="In-plane size"
                    value={`${study.dimensions.join(" × ")} px`}
                    mono
                  />
                  <FieldRow
                    label="In-plane spacing"
                    value={`${study.in_plane_spacing_mm
                      .map((v) => v.toFixed(4))
                      .join(" × ")} mm`}
                    mono
                  />
                  <FieldRow
                    label="Slice spacing"
                    value={`${study.slice_spacing_mm} mm`}
                    mono
                  />
                  <FieldRow
                    label="Orientation"
                    value={study.native_orientation}
                    mono
                  />
                  <FieldRow
                    label="Modality"
                    value={
                      study.modality ? (
                        <span className="inline-flex items-center gap-1.5">
                          <span className="font-mono uppercase">{study.modality}</span>
                          <Badge tone="muted">{study.modality_source.replace(/_/g, " ")}</Badge>
                        </span>
                      ) : (
                        "not determinable"
                      )
                    }
                  />
                </dl>
              ) : (
                <p className="py-4 text-xs leading-relaxed text-ink-faint">
                  Volume details appear here once the server has read and validated
                  the file. Nothing is inferred before then.
                </p>
              )}
            </div>
          </Panel>

          {study && !study.modality ? (
            <div className="flex gap-2.5 rounded-lg border border-seg-disc/30 bg-seg-disc/5 p-3">
              <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-seg-disc" aria-hidden />
              <p className="text-2xs leading-relaxed text-ink-muted">
                The modality could not be determined from the filename. Pfirrmann
                grading was validated on T2 and T2-SPACE series only, so grades will
                be reported with an explicit caution.
              </p>
            </div>
          ) : null}

          <Panel>
            <PanelHeader title="What will run" />
            <ol className="space-y-2 px-4 py-3">
              {[
                ["Preprocessing", "Resample to 1.0 mm/px, crop or pad to 352×256, normalise, denoise, CLAHE"],
                ["Segmentation", "16-channel U-Net, 4 classes"],
                ["Disc indexing", "Series-level ordering with vertebral-body separation"],
                ["Measurements", "Height, area, AP extent, signal ratios per disc"],
                ["Findings", "Pfirrmann grade plus validated binary findings"],
              ].map(([title, detail], i) => (
                <li key={title} className="flex gap-2.5">
                  <span className="mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded-full bg-surface-3 font-mono text-[9px] text-ink-muted">
                    {i + 1}
                  </span>
                  <span>
                    <span className="block text-xs text-ink">{title}</span>
                    <span className="block text-2xs leading-relaxed text-ink-faint">
                      {detail}
                    </span>
                  </span>
                </li>
              ))}
            </ol>
          </Panel>
        </div>
      </div>
    </div>
  );
}
