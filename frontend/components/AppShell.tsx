/**
 * Application shell: sidebar, top bar, and the responsive behaviour.
 *
 * Desktop keeps a persistent sidebar. Tablet collapses it to icons. Mobile drops
 * it entirely in favour of a bottom bar, so the viewer gets the full width rather
 * than a shrunken desktop layout.
 */

"use client";

import {
  Activity,
  CircleDot,
  FilePlus2,
  FileText,
  History,
  Info,
  LayoutDashboard,
  Menu,
  Workflow,
  X,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import * as React from "react";

import { api } from "@/lib/api";
import { RESEARCH_NOTICE } from "@/lib/theme";
import type { HealthResponse } from "@/lib/types";
import { Badge, cn } from "./ui";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/new", label: "New Analysis", icon: FilePlus2 },
  { href: "/history", label: "Analysis History", icon: History },
  { href: "/reports", label: "Reports", icon: FileText },
  { href: "/methodology", label: "Methodology", icon: Workflow },
  { href: "/about", label: "About", icon: Info },
] as const;

function useHealth() {
  const [health, setHealth] = React.useState<HealthResponse | null>(null);
  const [offline, setOffline] = React.useState(false);

  React.useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const next = await api.health();
        if (!cancelled) {
          setHealth(next);
          setOffline(false);
        }
      } catch {
        if (!cancelled) setOffline(true);
      }
    };
    void poll();
    const timer = setInterval(poll, 30_000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  return { health, offline };
}

function SystemStatus({
  health,
  offline,
}: {
  health: HealthResponse | null;
  offline: boolean;
}) {
  const tone = offline ? "danger" : health?.status === "ok" ? "positive" : "warn";
  const text = offline
    ? "API offline"
    : health?.status === "ok"
      ? `Model ready · ${health.segmentation_model.device.toUpperCase()}`
      : "Model degraded";
  return (
    <Badge tone={tone}>
      <CircleDot className="h-3 w-3" aria-hidden />
      {text}
    </Badge>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [drawerOpen, setDrawerOpen] = React.useState(false);
  const { health, offline } = useHealth();

  // Any navigation closes the mobile drawer.
  React.useEffect(() => setDrawerOpen(false), [pathname]);

  const analysisId = React.useMemo(() => {
    const match = pathname?.match(/^\/analysis\/([0-9a-f]+)/);
    return match?.[1] ?? null;
  }, [pathname]);

  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : Boolean(pathname?.startsWith(href));

  return (
    <div className="flex min-h-screen flex-col bg-surface-0">
      {/* -------------------------------------------------- top bar */}
      <header className="sticky top-0 z-40 flex h-14 shrink-0 items-center gap-3 border-b border-line-subtle bg-surface-1/95 px-3 backdrop-blur md:px-4">
        <button
          type="button"
          onClick={() => setDrawerOpen(true)}
          className="rounded-md p-2 text-ink-muted hover:bg-surface-2 hover:text-ink lg:hidden"
          aria-label="Open navigation"
          aria-expanded={drawerOpen}
        >
          <Menu className="h-4 w-4" aria-hidden />
        </button>

        <Link href="/" className="flex min-w-0 items-center gap-2.5">
          <span className="grid h-7 w-7 shrink-0 place-items-center rounded bg-accent/15 text-accent">
            <Activity className="h-4 w-4" aria-hidden />
          </span>
          <span className="min-w-0">
            <span className="block truncate text-sm font-semibold tracking-tight text-ink">
              Lumbar MRI Analysis
            </span>
            <span className="hidden text-2xs text-ink-faint sm:block">
              AI-assisted research workstation
            </span>
          </span>
        </Link>

        <div className="ml-auto flex items-center gap-2 md:gap-3">
          {analysisId ? (
            <span className="hidden items-center gap-1.5 md:flex">
              <span className="label-caps">Analysis</span>
              <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-2xs text-ink-muted">
                {analysisId}
              </code>
            </span>
          ) : null}
          {health ? (
            <Badge tone="muted" className="hidden sm:inline-flex">
              {health.pipeline.version}
            </Badge>
          ) : null}
          <SystemStatus health={health} offline={offline} />
          <span className="hidden h-7 w-7 place-items-center rounded-full border border-line bg-surface-2 text-2xs font-medium text-ink-muted sm:grid">
            RS
          </span>
        </div>
      </header>

      <div className="flex flex-1">
        {/* ---------------------------------------------- sidebar */}
        <nav
          aria-label="Primary"
          className="sticky top-14 hidden h-[calc(100vh-3.5rem)] w-14 shrink-0 flex-col gap-1 border-r border-line-subtle bg-surface-1 p-2 md:flex lg:w-56"
        >
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = isActive(href);
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                title={label}
                className={cn(
                  "flex items-center gap-2.5 rounded-md px-2.5 py-2 text-xs transition-colors",
                  active
                    ? "bg-accent/10 text-accent"
                    : "text-ink-muted hover:bg-surface-2 hover:text-ink",
                )}
              >
                <Icon className="h-4 w-4 shrink-0" aria-hidden />
                <span className="hidden truncate lg:inline">{label}</span>
              </Link>
            );
          })}

          <p className="mt-auto hidden px-2.5 text-2xs leading-relaxed text-ink-faint lg:block">
            {RESEARCH_NOTICE}
          </p>
        </nav>

        {/* ---------------------------------------------- mobile drawer */}
        {drawerOpen ? (
          <div className="fixed inset-0 z-50 md:hidden">
            <div
              className="absolute inset-0 bg-surface-0/80 backdrop-blur-sm"
              onClick={() => setDrawerOpen(false)}
              aria-hidden
            />
            <nav
              aria-label="Primary"
              className="absolute inset-y-0 left-0 flex w-64 animate-fade-up flex-col gap-1 border-r border-line bg-surface-1 p-3"
            >
              <div className="mb-2 flex items-center justify-between">
                <span className="text-sm font-medium text-ink">Navigation</span>
                <button
                  type="button"
                  onClick={() => setDrawerOpen(false)}
                  className="rounded p-1.5 text-ink-muted hover:bg-surface-2 hover:text-ink"
                  aria-label="Close navigation"
                >
                  <X className="h-4 w-4" aria-hidden />
                </button>
              </div>
              {NAV.map(({ href, label, icon: Icon }) => (
                <Link
                  key={href}
                  href={href}
                  aria-current={isActive(href) ? "page" : undefined}
                  className={cn(
                    "flex items-center gap-2.5 rounded-md px-2.5 py-2.5 text-sm",
                    isActive(href)
                      ? "bg-accent/10 text-accent"
                      : "text-ink-muted hover:bg-surface-2 hover:text-ink",
                  )}
                >
                  <Icon className="h-4 w-4" aria-hidden />
                  {label}
                </Link>
              ))}
              <p className="mt-auto text-2xs leading-relaxed text-ink-faint">
                {RESEARCH_NOTICE}
              </p>
            </nav>
          </div>
        ) : null}

        {/* ---------------------------------------------- main */}
        <main className="min-w-0 flex-1 pb-16 md:pb-0">{children}</main>
      </div>

      {/* -------------------------------------------------- mobile bottom bar */}
      <nav
        aria-label="Primary"
        className="fixed inset-x-0 bottom-0 z-40 flex h-14 items-stretch border-t border-line-subtle bg-surface-1/95 backdrop-blur md:hidden"
      >
        {NAV.slice(0, 4).map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            aria-current={isActive(href) ? "page" : undefined}
            className={cn(
              "flex flex-1 flex-col items-center justify-center gap-0.5 text-2xs",
              isActive(href) ? "text-accent" : "text-ink-faint",
            )}
          >
            <Icon className="h-4 w-4" aria-hidden />
            <span className="truncate px-1">{label.split(" ")[0]}</span>
          </Link>
        ))}
      </nav>
    </div>
  );
}

/** Page heading used consistently across routes. */
export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: string;
  actions?: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3 border-b border-line-subtle px-4 py-4 md:px-6">
      <div className="min-w-0">
        <h1 className="text-lg font-semibold tracking-tight text-ink">{title}</h1>
        {description ? (
          <p className="mt-1 max-w-2xl text-xs leading-relaxed text-ink-muted">
            {description}
          </p>
        ) : null}
      </div>
      {actions ? <div className="flex items-center gap-2">{actions}</div> : null}
    </div>
  );
}
