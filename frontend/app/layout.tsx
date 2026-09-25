import type { Metadata, Viewport } from "next";

import { AppShell } from "@/components/AppShell";
import { RESEARCH_NOTICE } from "@/lib/theme";
import "./globals.css";

export const metadata: Metadata = {
  title: "Lumbar MRI Analysis — AI-assisted research workstation",
  description:
    "Research prototype for AI-assisted lumbar spine MRI analysis: segmentation, " +
    "disc indexing, quantitative measurements and model-derived findings. " +
    RESEARCH_NOTICE,
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  themeColor: "#070b14",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body>
        {/* Skip link: the viewer page is dense, so keyboard users need a direct
            route past the navigation. */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-[60] focus:rounded focus:bg-accent focus:px-3 focus:py-2 focus:text-xs focus:font-medium focus:text-surface-0"
        >
          Skip to content
        </a>
        <AppShell>
          <div id="main-content">{children}</div>
        </AppShell>
      </body>
    </html>
  );
}
