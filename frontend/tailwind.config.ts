import type { Config } from "tailwindcss";

/**
 * Radiology-workstation palette.
 *
 * Deep slate foundation with a single cyan accent, because a viewer that shows
 * clinical overlays should not compete with them for attention. Segmentation
 * class colours are defined once in lib/theme.ts and referenced here so the
 * legend, the badges and the server-rendered overlay agree.
 */
const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          0: "#070b14",
          1: "#0b1220",
          2: "#111b2e",
          3: "#17243c",
          4: "#1f2f4d",
        },
        line: {
          subtle: "#1c2942",
          DEFAULT: "#243352",
          strong: "#31456b",
        },
        ink: {
          DEFAULT: "#e8eefc",
          muted: "#97a6c4",
          faint: "#6b7c9e",
        },
        accent: {
          DEFAULT: "#22d3ee",
          soft: "#0e7490",
          dim: "#155e75",
        },
        // Segmentation classes. Mirrors backend/app/services/imaging.py.
        seg: {
          vertebra: "#38bdf8",
          disc: "#fbbf24",
          canal: "#34d399",
        },
        severity: {
          info: "#64748b",
          low: "#38bdf8",
          moderate: "#fbbf24",
          high: "#fb7185",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      boxShadow: {
        panel: "0 1px 2px rgba(0,0,0,0.4), 0 8px 24px -12px rgba(0,0,0,0.6)",
        focus: "0 0 0 2px #070b14, 0 0 0 4px #22d3ee",
      },
      keyframes: {
        shimmer: {
          "100%": { transform: "translateX(100%)" },
        },
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        shimmer: "shimmer 1.6s infinite",
        "fade-up": "fade-up 0.18s ease-out",
      },
    },
  },
  plugins: [],
};

export default config;
