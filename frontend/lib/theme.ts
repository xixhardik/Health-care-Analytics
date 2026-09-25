/**
 * Centralised display configuration.
 *
 * Segmentation colours are defined once here and must match
 * `CLASS_COLOURS` in backend/app/services/imaging.py, since the server renders
 * the overlay and the client renders the legend. Keeping them in one module per
 * side makes that agreement auditable instead of scattered through components.
 */

import type { Severity } from "./types";

export interface SegClass {
  id: number;
  name: string;
  label: string;
  hex: string;
  /** Tailwind text colour token, for legends and badges. */
  text: string;
}

export const SEG_CLASSES: SegClass[] = [
  { id: 1, name: "vertebra", label: "Vertebrae", hex: "#38bdf8", text: "text-seg-vertebra" },
  { id: 2, name: "intervertebral_disc", label: "Intervertebral discs", hex: "#fbbf24", text: "text-seg-disc" },
  { id: 3, name: "spinal_canal", label: "Spinal canal", hex: "#34d399", text: "text-seg-canal" },
];

export const SEVERITY_STYLES: Record<Severity, { dot: string; text: string; border: string }> = {
  info: { dot: "bg-severity-info", text: "text-ink-muted", border: "border-line" },
  low: { dot: "bg-severity-low", text: "text-seg-vertebra", border: "border-seg-vertebra/30" },
  moderate: { dot: "bg-severity-moderate", text: "text-seg-disc", border: "border-seg-disc/30" },
  high: { dot: "bg-severity-high", text: "text-severity-high", border: "border-severity-high/30" },
};

/**
 * Evidence strength per validated category. The wording is served by the backend
 * (`evidence_labels`); these are only the display tones, so a weak target is
 * never shown with the same visual weight as a strong one.
 *
 * No numeric confidence is displayed anywhere. Converting a model probability
 * into a percentage "confidence" would invent a quantity that was never
 * validated, so strength stays qualitative.
 */
export const STRENGTH_LABELS: Record<string, { label: string; tone: string }> = {
  strong: { label: "Strong evidence", tone: "text-seg-canal" },
  moderate: { label: "Moderate evidence", tone: "text-seg-vertebra" },
  modest: { label: "Limited evidence", tone: "text-seg-disc" },
  weak: { label: "Weak evidence", tone: "text-severity-high" },
};

/**
 * The findings overlay layer.
 *
 * `hex` must match `FINDING_COLOUR` in backend/app/services/imaging.py, since the
 * server paints the overlay and the client draws the legend for it.
 *
 * The wording is deliberate. The overlay marks the *disc region* associated with
 * a model-estimated finding, drawn from the validated disc segmentation. There is
 * no pixel-level pathology model in this system, so nothing here may be described
 * as damaged tissue, a lesion, or a localisation of disease.
 */
export const FINDING_OVERLAY = {
  hex: "#ef4444",
  label: "Model-estimated finding",
  legend: "Finding-associated disc region",
} as const;

/**
 * How a value was obtained. Rendered next to every number so a measurement is
 * never mistaken for a model estimate, and a missing value is never mistaken for
 * a negative finding.
 */
export const PROVENANCE_STYLES: Record<
  string,
  { label: string; tone: "accent" | "warn" | "muted"; hint: string }
> = {
  segmentation_derived: {
    label: "Segmentation-derived",
    tone: "accent",
    hint: "Measured from the predicted segmentation at 1.0 mm/pixel.",
  },
  model_prediction: {
    label: "Model-estimated",
    tone: "warn",
    hint: "Estimated by a research model from segmentation-derived features.",
  },
  unsupported: {
    label: "Not modelled",
    tone: "muted",
    hint: "No validated model exists for this target, so nothing is reported.",
  },
};

/** Pfirrmann grade 1-5 shown on a restrained ramp, not a traffic light. */
export const PFIRRMANN_COLOURS: Record<number, string> = {
  1: "#34d399",
  2: "#7dd3fc",
  3: "#fbbf24",
  4: "#fb923c",
  5: "#fb7185",
};

export const PROCESSING_STAGES = [
  "Upload validated",
  "Loading MRI",
  "Preprocessing",
  "Segmentation",
  "Disc indexing",
  "Feature extraction",
  "Radiological analysis",
  "Complete",
] as const;

export const RESEARCH_NOTICE =
  "Research / educational prototype — results require expert radiological review.";
