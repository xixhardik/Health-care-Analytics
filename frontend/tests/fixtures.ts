/**
 * A result fixture shaped exactly like the backend's response.
 *
 * Copied from a real `GET /api/analysis/{id}/result` payload rather than
 * hand-invented, so a schema drift between frontend and backend shows up as a
 * failing test instead of a runtime surprise.
 */

import type {
  AnalysisResult,
  DemoStageRef,
  HealthResponse,
  LongitudinalCase,
  StatusResponse,
} from "@/lib/types";

export const statusProcessing: StatusResponse = {
  analysis_id: "abc123abc123abcd",
  status: "processing",
  progress: 60,
  stage: "Disc indexing",
  stages_completed: ["Upload validated", "Preprocessing", "Segmentation"],
  elapsed_seconds: 4.2,
  started_at: "2026-09-24T20:00:00Z",
  finished_at: null,
  error: null,
};

export const statusFailed: StatusResponse = {
  analysis_id: "abc123abc123abcd",
  status: "failed",
  progress: 40,
  stage: "Segmentation",
  stages_completed: ["Upload validated", "Preprocessing"],
  elapsed_seconds: 2.1,
  started_at: "2026-09-24T20:00:00Z",
  finished_at: "2026-09-24T20:00:02Z",
  error: {
    code: "ANALYSIS_FAILED",
    message: "Analysis failed during segmentation.",
    details: { stage: "Segmentation", error_type: "RuntimeError" },
  },
};

export const result: AnalysisResult = {
  analysis_id: "abc123abc123abcd",
  status: "completed",
  created_at: "2026-09-24T20:00:00Z",
  completed_at: "2026-09-24T20:00:04Z",
  is_sample: false,
  pipeline: {
    version: "SPIDER-Lumbar-v1",
    preprocessing: "Sprint 1",
    segmentation: "Sprint 3",
    postprocessing: "Sprint 5",
    preprocessing_detail: "Resample to 1.0 mm/px, crop or pad to 352x256",
    segmentation_detail: "16-channel U-Net, depth 4, bilinear",
    postprocessing_detail: "Series-level disc ordering + vertebral-body separation",
    excluded: {
      "Sprint 4":
        "32-channel capacity experiment did not improve the primary segmentation metrics and was therefore not selected.",
    },
    dataset: "SPIDER lumbar spine MRI (research dataset)",
  },
  study: {
    filename: "33_t2.mha",
    size_bytes: 1_268_000,
    format: ".mha",
    modality: "t2",
    modality_source: "filename_heuristic",
    slice_count: 24,
    dimensions: [242, 305],
    in_plane_spacing_mm: [1.0982, 0.866],
    slice_spacing_mm: 4.6731,
    native_orientation: "LPS",
  },
  timepoint: {
    id: "baseline",
    label: "Baseline",
    acquired_at: null,
    is_baseline: true,
  },
  segmentation: {
    classes: [
      { id: 0, name: "background", display_name: "Background", validated_test_dice: null },
      { id: 1, name: "vertebra", display_name: "Vertebrae", validated_test_dice: 0.91663 },
      { id: 2, name: "intervertebral_disc", display_name: "Intervertebral discs", validated_test_dice: 0.87827 },
      { id: 3, name: "spinal_canal", display_name: "Spinal canal", validated_test_dice: 0.90514 },
    ],
    slice_count: 24,
    informative_slice_count: 24,
    macro_foreground_dice_validated: 0.90001,
    note: "Every sagittal slice of the upload was segmented.",
  },
  processing: {
    preprocessing: "Sprint 1: resample to 1.0 mm/px",
    segmentation_model: "UNet(in=1, classes=4, base=16, depth=4, bilinear=True) 1,963,860 params",
    segmentation_source: "Sprint 3 Coverage (best validated model)",
    postprocessing_method: "Sprint 5 5A+5D",
    postprocessing_params: { cluster_px: 8.0 },
    device: "cpu",
    duration_seconds: 3.67,
    stage_durations: { preprocessing: 0.6, segmentation: 1.9 },
    disc_tracks_found: 6,
    components_rejected: 0,
    components_unassigned: 0,
  },
  discs: [
    {
      index: 1,
      structured: [
        {
          label: "Pfirrmann grade",
          value: "5",
          provenance: "model_prediction",
          available: true,
          strength: "moderate",
        },
        {
          label: "Narrowing",
          value: "detected",
          provenance: "model_prediction",
          available: true,
          strength: "strong",
        },
        {
          label: "Modic type",
          value: "unavailable",
          provenance: "unsupported",
          available: false,
          unavailable_reason:
            "Not served. The nominal Modic type was never modelled.",
        },
        {
          label: "Spondylolisthesis",
          value: "unavailable",
          provenance: "unsupported",
          available: false,
          unavailable_reason: "Not served. Too few positive cases to validate.",
        },
        {
          label: "Disc height (central)",
          value: "5.90 mm",
          provenance: "segmentation_derived",
          available: true,
        },
        {
          label: "Disc area",
          value: "231.0 mm²",
          provenance: "segmentation_derived",
          available: true,
        },
      ],
      identity_confidence: 0.4583,
      representative_slice_id: "study_abc_s011",
      representative_slice_index: 11,
      slices_present: 11,
      measurements: {
        height_mm_central: 5.9,
        height_mm_anterior: 6.1,
        height_mm_posterior: 4.4,
        area_mm2: 231,
        ap_extent_mm: 33.5,
        intensity_disc_vertebra_ratio: 0.10489,
        intensity_nucleus_annulus_ratio: 0.9,
        canal_width_at_disc_mm: 11.2,
        disc_to_vertebra_height_ratio: 0.19,
        height_ratio_to_series_median: 0.8,
        source: "segmentation_derived_measurement",
      },
      findings: [
        {
          name: "narrowing",
          label: "Disc narrowing",
          value: true,
          probability: 0.997,
          strength: "strong",
          validated_metric: "test_pr_auc",
          validated_value: 0.8779,
          prevalence_baseline: 0.323,
          source: "model_prediction",
          unavailable_reason: null,
        },
        {
          name: "pfirrmann_grade",
          label: "Pfirrmann grade",
          value: 5,
          expected: 4.692,
          probabilities: { "1": 0.01, "2": 0.02, "3": 0.05, "4": 0.2, "5": 0.72 },
          strength: "moderate",
          validated_metric: "end_to_end_quadratic_weighted_kappa",
          validated_value: 0.6543,
          within_one_grade: 0.8531,
          source: "model_prediction",
          unavailable_reason: null,
        },
        {
          name: "spondylolisthesis",
          label: "spondylolisthesis",
          value: null,
          probability: null,
          source: "unsupported",
          unavailable_reason:
            "Not served. Only 5 positive discs exist in the held-out test split.",
        },
        {
          name: "modic_type",
          label: "modic_type",
          value: null,
          probability: null,
          source: "unsupported",
          unavailable_reason:
            "Not served. The nominal Modic type was never modelled.",
        },
      ],
      pfirrmann_grade: 5,
      modic_type: null,
      modic_any: true,
      bulging: true,
      narrowing: true,
      herniation: true,
      spondylolisthesis: null,
      upper_endplate: true,
      lower_endplate: true,
    },
    {
      index: 2,
      structured: [
        {
          label: "Pfirrmann grade",
          value: "2",
          provenance: "model_prediction",
          available: true,
          strength: "moderate",
        },
        {
          label: "Narrowing",
          value: "not detected",
          provenance: "model_prediction",
          available: true,
          strength: "strong",
        },
        {
          label: "Disc height (central)",
          value: "9.57 mm",
          provenance: "segmentation_derived",
          available: true,
        },
      ],
      identity_confidence: 0.4583,
      representative_slice_id: "study_abc_s012",
      representative_slice_index: 12,
      slices_present: 11,
      measurements: {
        height_mm_central: 9.571,
        height_mm_anterior: 9.9,
        height_mm_posterior: 7.2,
        area_mm2: 347,
        ap_extent_mm: 35.1,
        intensity_disc_vertebra_ratio: 0.30603,
        intensity_nucleus_annulus_ratio: 1.1,
        canal_width_at_disc_mm: 12.0,
        disc_to_vertebra_height_ratio: 0.31,
        height_ratio_to_series_median: 1.1,
        source: "segmentation_derived_measurement",
      },
      findings: [
        {
          name: "narrowing",
          label: "Disc narrowing",
          value: false,
          probability: 0.24,
          strength: "strong",
          validated_metric: "test_pr_auc",
          validated_value: 0.8779,
          prevalence_baseline: 0.323,
          source: "model_prediction",
          unavailable_reason: null,
        },
      ],
      pfirrmann_grade: 2,
      modic_type: null,
      modic_any: false,
      bulging: true,
      narrowing: false,
      herniation: true,
      spondylolisthesis: null,
      upper_endplate: false,
      lower_endplate: false,
    },
  ],
  summary: {
    disc_count: 2,
    discs_with_findings_count: 2,
    mean_disc_height_mm: 7.736,
    findings: [
      {
        category: "Detection",
        text: "2 intervertebral disc levels were identified and indexed from the most inferior disc upward. No anatomical level name is asserted.",
        disc_indices: [1, 2],
        severity: "info",
      },
      {
        category: "Disc-level finding",
        text: "Disc narrowing: model-estimated at disc 1.",
        disc_indices: [1],
        severity: "moderate",
      },
      {
        category: "Scope",
        text: "Findings above are model-derived research estimates from a single timepoint. No postoperative assessment is provided.",
        disc_indices: [],
        severity: "info",
      },
    ],
    unsupported_targets: {
      spondylolisthesis: "Not served. Too few positive cases to validate.",
      modic_type: "Not served. The nominal Modic type was never modelled.",
    },
  },
  // Server-decided: disc 1 carries a positive supported narrowing finding, disc 2
  // does not (its narrowing value is false), so only disc 1 may be marked red.
  // Ordinary studies are not demonstration stages. Overridden per-test where a
  // simulated timeline is under test.
  demo_stage: null,
  finding_overlay: {
    disc_indices: [1],
    discs: [{ index: 1, findings: ["Disc narrowing"] }],
    eligible_findings: [
      "narrowing",
      "bulging",
      "herniation",
      "any_modic",
      "upper_endplate",
      "lower_endplate",
    ],
    label: "Model-estimated finding",
    note:
      "The red overlay marks disc regions associated with a model-estimated " +
      "finding. It is drawn from the validated disc segmentation for that disc, " +
      "not from a pixel-level pathology model, and it is not a diagnosis of " +
      "damaged tissue.",
  },
  provenance_labels: {
    segmentation_derived: "Segmentation-derived",
    model_prediction: "Model-estimated",
    unsupported: "Not modelled",
  },
  evidence_labels: {
    strong: "Strong evidence",
    moderate: "Moderate evidence",
    modest: "Limited evidence",
    weak: "Weak evidence",
  },
  validated_metrics: {
    segmentation_macro_foreground_dice: 0.90001,
    segmentation_vertebra_dice: 0.91663,
    segmentation_disc_dice: 0.87827,
    segmentation_canal_dice: 0.90514,
    disc_indexing_percent: 93.84,
    disc_indexing_baseline_percent: 84.08,
    disc_height_mae_mm: 0.6492,
    disc_area_mae_mm2: 21.4831,
    intensity_ratio_pearson_r: 0.9726,
    pfirrmann_qwk_end_to_end: 0.6543,
    test_patients: 33,
    // Verbatim from backend/app/pipeline_info.py: the sentence that keeps a
    // research benchmark from reading as this study's accuracy.
    note:
      "Research evaluation results measured on a 33-patient held-out test " +
      "split. Segmentation figures are Sprint 3; disc indexing, measurement " +
      "and Pfirrmann figures are Sprint 5 post-processing applied on top of " +
      "it. They describe the pipeline, not this study.",
  },
  disclaimer:
    "Research and educational prototype. Results are model-derived research estimates and require review by a qualified radiologist.",
};

/**
 * A `GET /api/health` fixture.
 *
 * Both the dashboard and the New Analysis page read health on mount: the
 * dashboard to render the pipeline identity and the served metric table, the
 * upload page to decide whether the sample-study button can exist. The shared
 * fields below are taken from the result fixture rather than retyped, so the two
 * fixtures cannot disagree about the pipeline version or the research figures.
 *
 * Mirrors backend/app/pipeline_info.py and backend/app/routers/health.py.
 */
export const health: HealthResponse = {
  status: "ok",
  api_version: "1.0.0",
  pipeline: result.pipeline,
  segmentation_model: {
    loaded: true,
    architecture: "UNet(in=1, classes=4, base=16, depth=4, bilinear=True)",
    parameters: 1_963_860,
    checkpoint_name: "best_val_dice.pt",
    selection: "coverage",
    epoch: 57,
    device: "cpu",
    validated_macro_foreground_dice: 0.90001,
    source_sprint: "Sprint 3 Coverage",
  },
  finding_models: {
    supported_targets: [
      "pfirrmann_grade",
      "narrowing",
      "bulging",
      "herniation",
      "any_modic",
      "upper_endplate",
      "lower_endplate",
    ],
    unsupported_targets: {
      spondylolisthesis: "Not served. Too few positive cases to validate.",
      modic_type: "Not served. The nominal Modic type was never modelled.",
    },
    loaded: true,
  },
  postprocessing_method: "Sprint 5 5A+5D",
  validated_metrics: result.validated_metrics,
  // Served, never hard-coded in the UI: the dashboard renders exactly these rows.
  metric_table: [
    { label: "Macro foreground Dice", value: "0.90001", kind: "segmentation", sprint: "Sprint 3" },
    { label: "Vertebra Dice", value: "0.91663", kind: "segmentation", sprint: "Sprint 3" },
    { label: "Intervertebral disc Dice", value: "0.87827", kind: "segmentation", sprint: "Sprint 3" },
    { label: "Spinal canal Dice", value: "0.90514", kind: "segmentation", sprint: "Sprint 3" },
    { label: "Disc indexing, per-slice ordering", value: "84.08%", kind: "postprocessing", sprint: "Sprint 3" },
    { label: "Disc indexing, series-level ordering", value: "93.84%", kind: "postprocessing", sprint: "Sprint 5" },
    { label: "Disc height mean absolute error", value: "0.6492 mm", kind: "postprocessing", sprint: "Sprint 5" },
    { label: "Disc area mean absolute error", value: "21.4831 mm²", kind: "postprocessing", sprint: "Sprint 5" },
    { label: "Disc signal ratio correlation", value: "r = 0.9726", kind: "postprocessing", sprint: "Sprint 5" },
    { label: "End-to-end Pfirrmann agreement", value: "QWK 0.6543", kind: "postprocessing", sprint: "Sprint 5" },
  ],
  // Includes the rejected Sprint 4 experiment: negative results are served too.
  research_progress: [
    {
      sprint: "Sprint 2",
      title: "Baseline U-Net",
      outcome: "selected",
      summary:
        "First working 16-channel U-Net plus the disc-level measurement and finding estimators.",
      headline: "Macro foreground Dice 0.8979, disc indexing 82.6%",
    },
    {
      sprint: "Sprint 3",
      title: "100% training-data coverage",
      outcome: "selected",
      summary:
        "A rotating sampler raised training-slice coverage from 28% to 100%. This run produced the best segmentation and is the one served.",
      headline: "Macro foreground Dice 0.90001, disc indexing 84.08%",
    },
    {
      sprint: "Sprint 4",
      title: "32-channel capacity experiment",
      outcome: "rejected",
      summary:
        "32-channel capacity experiment did not improve the primary segmentation metrics and was therefore not selected.",
      headline: "Macro foreground Dice 0.89393, disc indexing 82.70% — worse",
    },
    {
      sprint: "Sprint 5",
      title: "Series-level post-processing",
      outcome: "selected",
      summary:
        "Disc identity is assigned once per series from row-aligned tracks instead of independently per slice.",
      headline: "Disc indexing 84.08% → 93.84%, height MAE 0.6492 mm",
    },
  ],
  provenance_labels: result.provenance_labels,
  evidence_labels: result.evidence_labels,
  // The New Analysis page only renders "Load Sample Study" when this is true.
  sample_study_available: true,
  disclaimer: result.disclaimer,
  research_notice:
    "Research / educational prototype — results require expert radiological review.",
};

/* -------------------------------------------------------------------------- */
/* Sprint 8: simulated longitudinal demonstration                             */
/* -------------------------------------------------------------------------- */

/**
 * A demonstration stage, as the API attaches it to an analysis.
 *
 * Mirrors `DemoStageRef` in backend/app/schemas.py. The wording is the real
 * wording: these strings are what the interface renders, so a test asserting on
 * them is asserting on what a viewer actually reads.
 */
export const demoStage: DemoStageRef = {
  case_id: "LS-DEMO-001",
  stage_id: "pre_surgery",
  label: "Pre-Surgery",
  research_status: "Baseline",
  display_reference: "Demonstration study A",
  order: 1,
  stage_note:
    "Most degenerate of the four studies: nine thin discs, findings on eight of them.",
  provenance: {
    source_study_id: "177_t2",
    source_patient_id: 177,
    source_dataset: "SPIDER",
    source_split: "test",
    is_true_followup: false,
    measurement_source: "real_pipeline",
  },
  disclaimer:
    "Simulated longitudinal demonstration. The available SPIDER dataset does " +
    "not contain true postoperative longitudinal follow-up for this case.",
  ui_notice:
    "Visual stages use different SPIDER studies to demonstrate the " +
    "longitudinal workflow. They are not postoperative follow-up scans from " +
    "the same patient.",
  is_simulated_timeline: true,
};

/** The same result, presented as a demonstration stage. */
export const demoStageResult: AnalysisResult = {
  ...result,
  demo_stage: demoStage,
};

/** Four stages, four different patients. */
export const demoCase: LongitudinalCase = {
  case_id: "LS-DEMO-001",
  type: "SIMULATED_LONGITUDINAL_DEMO",
  title: "Simulated longitudinal workflow demonstration",
  is_true_followup: false,
  is_same_patient: false,
  source_dataset: "SPIDER lumbar spine MRI (research dataset)",
  disclaimer: demoStage.disclaimer,
  ui_notice: demoStage.ui_notice,
  interpretation:
    "Four real SPIDER studies from four different patients, arranged into a " +
    "simulated timeline to demonstrate how a longitudinal system would operate.",
  measurement_policy: { source: "real_pipeline" },
  selection: { split: "test" },
  stage_count: 4,
  available_stage_count: 4,
  stages: [
    {
      order: 1, stage_id: "pre_surgery", label: "Pre-Surgery",
      research_status: "Baseline", display_reference: "Demonstration study A",
      stage_note: null, scanner: "SIEMENS 1.5T", acquisition_timestamp: null,
      available: true, unavailable_reason: null,
      provenance: demoStage.provenance,
      expected: { discs: 9, mean_disc_height_mm: 4.64 },
    },
    {
      order: 2, stage_id: "post_surgery", label: "Post-Surgery",
      research_status: "Early postoperative",
      display_reference: "Demonstration study B",
      stage_note: null, scanner: "SIEMENS 1.5T", acquisition_timestamp: null,
      available: true, unavailable_reason: null,
      provenance: {
        source_study_id: "106_t2", source_patient_id: 106,
        source_dataset: "SPIDER", source_split: "test",
        is_true_followup: false, measurement_source: "real_pipeline",
      },
      expected: { discs: 7, mean_disc_height_mm: 5.8 },
    },
    {
      order: 3, stage_id: "month_3", label: "3-Month Recovery",
      research_status: "Recovery monitoring",
      display_reference: "Demonstration study C",
      stage_note: null, scanner: "Philips Healthcare 3.0T",
      acquisition_timestamp: null, available: true, unavailable_reason: null,
      provenance: {
        source_study_id: "16_t2", source_patient_id: 16,
        source_dataset: "SPIDER", source_split: "test",
        is_true_followup: false, measurement_source: "real_pipeline",
      },
      expected: { discs: 7, mean_disc_height_mm: 7.79 },
    },
    {
      order: 4, stage_id: "month_6", label: "6-Month Recovery",
      research_status: "Final follow-up demonstration",
      display_reference: "Demonstration study D",
      stage_note: null, scanner: "SIEMENS 1.5T", acquisition_timestamp: null,
      available: true, unavailable_reason: null,
      provenance: {
        source_study_id: "6_t2", source_patient_id: 6,
        source_dataset: "SPIDER", source_split: "test",
        is_true_followup: false, measurement_source: "real_pipeline",
      },
      expected: { discs: 6, mean_disc_height_mm: 8.49 },
    },
  ],
};

/**
 * Build a volume payload in exactly the wire format the backend emits, so the
 * client parser is tested against the real layout rather than a convenient one.
 */
export function buildVolumePayload(options: {
  slices?: number;
  rows?: number;
  cols?: number;
  findingDiscs?: number[];
  formatVersion?: number;
  discInstanceOffset?: number;
  truncate?: number;
} = {}): ArrayBuffer {
  const slices = options.slices ?? 3;
  const rows = options.rows ?? 4;
  const cols = options.cols ?? 5;
  const offset = options.discInstanceOffset ?? 10;
  const voxels = slices * rows * cols;

  const image = new Uint8Array(voxels);
  const semantic = new Uint8Array(voxels);
  const instance = new Uint8Array(voxels);
  for (let i = 0; i < voxels; i += 1) image[i] = i % 256;
  // A disc-2 region and a vertebra region, so label handling is exercised.
  for (let i = 0; i < voxels; i += 1) {
    if (i % 7 === 0) {
      semantic[i] = 2;
      instance[i] = offset + 2;
    } else if (i % 11 === 0) {
      semantic[i] = 1;
      instance[i] = 1;
    }
  }

  const header = {
    format_version: options.formatVersion ?? 1,
    analysis_id: "abc123abc123abcd",
    pipeline_version: "SPIDER-Lumbar-v1",
    axis_order: ["slice", "row", "col"],
    dimensions: { slices, rows, cols },
    spacing_mm: { row: 1.0, col: 1.0, slice: 4.0 },
    native_in_plane_spacing_mm: [0.6, 0.6],
    channels: [
      { name: "image", offset: 0, length: voxels, dtype: "uint8" },
      { name: "semantic", offset: voxels, length: voxels, dtype: "uint8" },
      { name: "instance", offset: voxels * 2, length: voxels, dtype: "uint8" },
    ],
    semantic_labels: { "1": "vertebra", "2": "intervertebral_disc", "3": "spinal_canal" },
    disc_instance_offset: offset,
    finding_discs: options.findingDiscs ?? [2],
    slice_ids: Array.from({ length: slices }, (_, i) => `s${i}`),
    total_bytes: voxels * 3,
    notice: "Voxel data for browser-side rendering.",
  };

  const encoded = new TextEncoder().encode(JSON.stringify(header));
  const total = 4 + encoded.length + voxels * 3;
  const buffer = new ArrayBuffer(options.truncate ?? total);
  const bytes = new Uint8Array(buffer);
  new DataView(buffer).setUint32(0, encoded.length, true);
  bytes.set(encoded.subarray(0, Math.max(0, bytes.length - 4)), 4);
  let cursor = 4 + encoded.length;
  for (const channel of [image, semantic, instance]) {
    if (cursor >= bytes.length) break;
    bytes.set(channel.subarray(0, Math.max(0, bytes.length - cursor)), cursor);
    cursor += channel.length;
  }
  return buffer;
}
