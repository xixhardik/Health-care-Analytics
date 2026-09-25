/**
 * Small component primitives.
 *
 * Hand-rolled rather than pulled from a component library: the set needed here is
 * narrow, and owning it keeps the visual language consistent with the viewer
 * without dragging in a dependency and its theme layer.
 */

"use client";

import { clsx, type ClassValue } from "clsx";
import { AlertTriangle, Check, Loader2 } from "lucide-react";
import * as React from "react";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/* -------------------------------------------------------------------------- */
/* Button                                                                      */
/* -------------------------------------------------------------------------- */

type ButtonState = "idle" | "loading" | "success" | "error";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md" | "lg";
  state?: ButtonState;
  icon?: React.ReactNode;
}

const VARIANTS: Record<string, string> = {
  primary:
    "bg-accent text-surface-0 hover:bg-accent/90 disabled:bg-accent/40 font-medium",
  secondary:
    "bg-surface-3 text-ink hover:bg-surface-4 border border-line disabled:text-ink-faint",
  ghost: "text-ink-muted hover:bg-surface-2 hover:text-ink",
  danger:
    "bg-severity-high/15 text-severity-high border border-severity-high/30 hover:bg-severity-high/25",
};

const SIZES: Record<string, string> = {
  sm: "h-8 px-3 text-xs gap-1.5",
  md: "h-9 px-4 text-sm gap-2",
  lg: "h-11 px-6 text-sm gap-2",
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  function Button(
    {
      className,
      variant = "secondary",
      size = "md",
      state = "idle",
      icon,
      children,
      disabled,
      ...props
    },
    ref,
  ) {
    const busy = state === "loading";
    return (
      <button
        ref={ref}
        // Disabled while busy so a slow request cannot be fired twice.
        disabled={disabled || busy}
        aria-busy={busy || undefined}
        className={cn(
          "inline-flex items-center justify-center rounded-md transition-colors",
          "disabled:cursor-not-allowed disabled:opacity-60",
          VARIANTS[variant],
          SIZES[size],
          state === "success" && "bg-seg-canal/20 text-seg-canal",
          state === "error" && "bg-severity-high/20 text-severity-high",
          className,
        )}
        {...props}
      >
        {busy ? (
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
        ) : state === "success" ? (
          <Check className="h-4 w-4" aria-hidden />
        ) : state === "error" ? (
          <AlertTriangle className="h-4 w-4" aria-hidden />
        ) : (
          icon
        )}
        {children}
      </button>
    );
  },
);

/* -------------------------------------------------------------------------- */
/* Surfaces                                                                    */
/* -------------------------------------------------------------------------- */

export function Panel({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("panel", className)} {...props}>
      {children}
    </div>
  );
}

export function PanelHeader({
  title,
  subtitle,
  actions,
}: {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <div className="panel-header">
      <div className="min-w-0">
        <h2 className="truncate text-sm font-medium text-ink">{title}</h2>
        {subtitle ? (
          <p className="mt-0.5 truncate text-xs text-ink-faint">{subtitle}</p>
        ) : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Badge                                                                       */
/* -------------------------------------------------------------------------- */

export function Badge({
  children,
  tone = "neutral",
  className,
  title,
}: {
  children: React.ReactNode;
  tone?: "neutral" | "accent" | "positive" | "warn" | "danger" | "muted";
  className?: string;
  /** Tooltip text, used to explain provenance tags without crowding the label. */
  title?: string;
}) {
  const tones: Record<string, string> = {
    neutral: "bg-surface-3 text-ink-muted border-line",
    accent: "bg-accent/10 text-accent border-accent/30",
    positive: "bg-seg-canal/10 text-seg-canal border-seg-canal/30",
    warn: "bg-seg-disc/10 text-seg-disc border-seg-disc/30",
    danger: "bg-severity-high/10 text-severity-high border-severity-high/30",
    muted: "bg-surface-2 text-ink-faint border-line-subtle",
  };
  return (
    <span
      title={title}
      className={cn(
        "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-2xs font-medium",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/* -------------------------------------------------------------------------- */
/* States                                                                      */
/* -------------------------------------------------------------------------- */

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("skeleton", className)} aria-hidden />;
}

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
      {icon ? (
        <div className="mb-4 rounded-full border border-line-subtle bg-surface-2 p-3 text-ink-faint">
          {icon}
        </div>
      ) : null}
      <h3 className="text-sm font-medium text-ink">{title}</h3>
      {description ? (
        <p className="mt-1.5 max-w-sm text-xs leading-relaxed text-ink-faint">
          {description}
        </p>
      ) : null}
      {action ? <div className="mt-5">{action}</div> : null}
    </div>
  );
}

export function ErrorState({
  title = "Something went wrong",
  message,
  code,
  onRetry,
}: {
  title?: string;
  message: string;
  code?: string;
  onRetry?: () => void;
}) {
  return (
    <div
      role="alert"
      className="flex flex-col items-start gap-3 rounded-lg border border-severity-high/30 bg-severity-high/5 p-4"
    >
      <div className="flex items-start gap-3">
        <AlertTriangle
          className="mt-0.5 h-4 w-4 shrink-0 text-severity-high"
          aria-hidden
        />
        <div>
          <p className="text-sm font-medium text-ink">{title}</p>
          <p className="mt-1 text-xs leading-relaxed text-ink-muted">{message}</p>
          {code ? (
            <p className="mt-2 font-mono text-2xs text-ink-faint">{code}</p>
          ) : null}
        </div>
      </div>
      {onRetry ? (
        <Button size="sm" variant="secondary" onClick={onRetry}>
          Try again
        </Button>
      ) : null}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Progress                                                                    */
/* -------------------------------------------------------------------------- */

export function ProgressBar({
  value,
  label,
  className,
}: {
  value: number;
  label?: string;
  className?: string;
}) {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <div
      role="progressbar"
      aria-valuenow={clamped}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label ?? "Progress"}
      className={cn("h-1.5 w-full overflow-hidden rounded-full bg-surface-3", className)}
    >
      <div
        className="h-full rounded-full bg-accent transition-[width] duration-500 ease-out"
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Field rows                                                                  */
/* -------------------------------------------------------------------------- */

export function FieldRow({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5">
      <dt className="shrink-0 text-xs text-ink-faint">{label}</dt>
      <dd
        className={cn(
          "min-w-0 truncate text-right text-xs text-ink",
          mono && "font-mono",
        )}
      >
        {value}
      </dd>
    </div>
  );
}

/** Small labelled statistic used on the dashboard. */
export function StatCard({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
  tone?: "neutral" | "accent" | "positive" | "warn" | "danger";
}) {
  const accents: Record<string, string> = {
    neutral: "text-ink",
    accent: "text-accent",
    positive: "text-seg-canal",
    warn: "text-seg-disc",
    danger: "text-severity-high",
  };
  return (
    <div className="panel p-4">
      <p className="label-caps">{label}</p>
      <p className={cn("metric-value mt-2", accents[tone])}>{value}</p>
      {hint ? <p className="mt-1.5 text-2xs text-ink-faint">{hint}</p> : null}
    </div>
  );
}

/** Accessible on/off control used for overlay class visibility. */
export function Toggle({
  checked,
  onChange,
  label,
  swatch,
  disabled,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
  swatch?: string;
  disabled?: boolean;
}) {
  return (
    <label
      className={cn(
        "flex cursor-pointer select-none items-center gap-2 rounded px-1.5 py-1 text-xs transition-colors",
        disabled ? "cursor-not-allowed opacity-50" : "hover:bg-surface-2",
      )}
    >
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
        className="h-3.5 w-3.5 rounded border-line bg-surface-2 text-accent focus-visible:ring-2 focus-visible:ring-accent"
      />
      {swatch ? (
        <span
          aria-hidden
          className="h-2.5 w-2.5 shrink-0 rounded-sm"
          style={{ backgroundColor: swatch }}
        />
      ) : null}
      <span className="text-ink-muted">{label}</span>
    </label>
  );
}
