/**
 * TypeScript mirrors of the backend Pydantic schemas.
 *
 * These are kept deliberately thin and are generated from one source of truth:
 * the FastAPI OpenAPI document at /api/openapi.json. When the backend schema
 * changes, run `npm run typecheck` against a running backend and update here -
 * the shapes below are the contract documented in docs/api.md.
 */

export type AnalysisStatus = "queued" | "processing" | "completed" | "failed";

export interface ApiErrorDetail {
  code: string;
  message: string;
  details?: Record<string, unknown> | null;
}

export interface ApiErrorBody {
  error: ApiErrorDetail;
}

export interface ModelInfo {
  loaded: boolean;
  architecture?: string | null;
  parameters?: number | null;
  checkpoint_name?: string | null;
  selection?: string | null;
  epoch?: number | null;
  device: string;
  validated_macro_foreground_dice?: number | null;
  source_sprint?: string | null;
}

export interface FindingModelInfo {
  supported_targets: string[];
  unsupported_targets: Record<string, string>;
  loaded: boolean;
}

export interface PipelineInfo {
  version: string;
  preprocessing: string;
  segmentation: string;
  postprocessing: string;
  preprocessing_detail: string;
  segmentation_detail: string;
  postprocessing_detail: string;
  excluded: Record<string, string>;
  dataset: string;
}

export interface MetricRow {
  label: string;
  value: string;
  kind: "segmentation" | "postprocessing";
  sprint: string;
}

export interface ResearchStage {
  sprint: string;
  title: string;
  outcome: "selected" | "rejected";
  summary: string;
  headline: string;
}

export type Provenance =
  | "segmentation_derived"
  | "model_prediction"
  | "unsupported";

export interface HealthResponse {
  status: "ok" | "degraded";
  api_version: string;
  pipeline: PipelineInfo;
  segmentation_model: ModelInfo;
  finding_models: FindingModelInfo;
  postprocessing_method?: string | null;
  validated_metrics: ValidatedMetrics;
  metric_table: MetricRow[];
  research_progress: ResearchStage[];
  provenance_labels: Record<string, string>;
  evidence_labels: Record<string, string>;
  sample_study_available: boolean;
  disclaimer: string;
  research_notice: string;
}

export interface StudyInfo {
  filename: string;
  size_bytes: number;
  format: string;
  modality?: string | null;
  modality_source: string;
  slice_count: number;
  dimensions: number[];
  in_plane_spacing_mm: number[];
  slice_spacing_mm: number;
  native_orientation: string;
}

export interface UploadResponse {
  analysis_id: string;
  status: "queued";
  study: StudyInfo;
  created_at: string;
  message: string;
  is_sample: boolean;
}

export interface RunResponse {
  analysis_id: string;
  status: AnalysisStatus;
}

export interface StatusResponse {
  analysis_id: string;
  status: AnalysisStatus;
  progress: number;
  stage: string;
  stages_completed: string[];
  elapsed_seconds: number;
  started_at?: string | null;
  finished_at?: string | null;
  error?: ApiErrorDetail | null;
}

export interface SegmentationClass {
  id: number;
  name: string;
  display_name: string;
  validated_test_dice?: number | null;
}

export interface SegmentationInfo {
  classes: SegmentationClass[];
  slice_count: number;
  informative_slice_count: number;
  macro_foreground_dice_validated?: number | null;
  note: string;
}

export interface FindingEstimate {
  name: string;
  label: string;
  value?: boolean | number | null;
  probability?: number | null;
  expected?: number | null;
  probabilities?: Record<string, number> | null;
  strength?: string | null;
  validated_metric?: string | null;
  validated_value?: number | null;
  prevalence_baseline?: number | null;
  within_one_grade?: number | null;
  source: string;
  caveat?: string | null;
  unavailable_reason?: string | null;
}

export interface DiscMeasurements {
  height_mm_central?: number | null;
  height_mm_anterior?: number | null;
  height_mm_posterior?: number | null;
  area_mm2?: number | null;
  ap_extent_mm?: number | null;
  intensity_disc_vertebra_ratio?: number | null;
  intensity_nucleus_annulus_ratio?: number | null;
  canal_width_at_disc_mm?: number | null;
  disc_to_vertebra_height_ratio?: number | null;
  height_ratio_to_series_median?: number | null;
  source: string;
}

export interface StructuredLine {
  label: string;
  value: string;
  provenance: Provenance;
  available: boolean;
  strength?: string | null;
  unavailable_reason?: string | null;
}

export interface DiscResult {
  index: number;
  structured: StructuredLine[];
  identity_confidence?: number | null;
  representative_slice_id?: string | null;
  representative_slice_index?: number | null;
  slices_present?: number | null;
  measurements: DiscMeasurements;
  findings: FindingEstimate[];
  pfirrmann_grade?: number | null;
  modic_type?: number | null;
  modic_any?: boolean | null;
  bulging?: boolean | null;
  narrowing?: boolean | null;
  herniation?: boolean | null;
  spondylolisthesis?: boolean | null;
  upper_endplate?: boolean | null;
  lower_endplate?: boolean | null;
}

export type Severity = "info" | "low" | "moderate" | "high";

export interface ObservedFinding {
  category: string;
  text: string;
  disc_indices: number[];
  severity: Severity;
}

export interface AnalysisSummary {
  disc_count: number;
  discs_with_findings_count: number;
  mean_disc_height_mm?: number | null;
  findings: ObservedFinding[];
  unsupported_targets: Record<string, string>;
}

export interface ProcessingInfo {
  preprocessing: string;
  segmentation_model: string;
  segmentation_source: string;
  postprocessing_method: string;
  postprocessing_params: Record<string, unknown>;
  device: string;
  duration_seconds: number;
  stage_durations: Record<string, number>;
  disc_tracks_found?: number | null;
  components_rejected?: number | null;
  components_unassigned?: number | null;
}

export interface Timepoint {
  id: string;
  label: string;
  acquired_at?: string | null;
  is_baseline: boolean;
}

export interface ValidatedMetrics {
  segmentation_macro_foreground_dice: number;
  segmentation_vertebra_dice: number;
  segmentation_disc_dice: number;
  segmentation_canal_dice: number;
  disc_indexing_percent: number;
  disc_indexing_baseline_percent: number;
  disc_height_mae_mm: number;
  disc_area_mae_mm2: number;
  intensity_ratio_pearson_r: number;
  pfirrmann_qwk_end_to_end: number;
  test_patients: number;
  note: string;
}

export interface FindingOverlayDisc {
  index: number;
  findings: string[];
}

/**
 * Which disc regions the findings overlay marks.
 *
 * Served by the backend so the interface can mark exactly the discs the slice
 * renderer tints, without re-deriving the rule in the browser. The marked region
 * is the validated disc segmentation, not a pathology mask.
 */
export interface FindingOverlayInfo {
  disc_indices: number[];
  discs: FindingOverlayDisc[];
  eligible_findings: string[];
  label: string;
  note: string;
}

export interface AnalysisResult {
  analysis_id: string;
  status: AnalysisStatus;
  created_at: string;
  completed_at?: string | null;
  is_sample: boolean;
  pipeline: PipelineInfo;
  study: StudyInfo;
  timepoint: Timepoint;
  segmentation: SegmentationInfo;
  processing: ProcessingInfo;
  discs: DiscResult[];
  summary: AnalysisSummary;
  validated_metrics: ValidatedMetrics;
  provenance_labels: Record<string, string>;
  evidence_labels: Record<string, string>;
  finding_overlay: FindingOverlayInfo;
  disclaimer: string;
}

export interface HistoryItem {
  analysis_id: string;
  filename: string;
  created_at: string;
  status: AnalysisStatus;
  slice_count?: number | null;
  disc_count?: number | null;
  findings_count?: number | null;
  modality?: string | null;
  is_sample?: boolean;
}

export interface HistoryResponse {
  items: HistoryItem[];
  total: number;
  counts: Record<string, number>;
}

export type SliceMode = "original" | "mask" | "overlay";

/**
 * How the slice endpoint interprets a highlight request.
 *
 * `disc` is the selected-disc marker alone; `findings` additionally tints the
 * disc regions the backend decided are associated with a model-estimated
 * finding. The decision is the server's - see backend findings_overlay.py.
 */
export type HighlightMode = "none" | "disc" | "findings";
