/**
 * The single place the frontend talks to the backend.
 *
 * Every call funnels through `request`, so the backend's error envelope is
 * translated into a typed `ApiError` exactly once and components never have to
 * inspect response shapes.
 */

import type {
  AnalysisResult,
  ApiErrorBody,
  HealthResponse,
  HighlightMode,
  HistoryResponse,
  RunResponse,
  SliceMode,
  StatusResponse,
  UploadResponse,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details?: Record<string, unknown> | null;

  constructor(
    code: string,
    message: string,
    status: number,
    details?: Record<string, unknown> | null,
  ) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

/** Message shown when the backend cannot be reached at all. */
const OFFLINE_MESSAGE =
  "Cannot reach the analysis API. Confirm the backend is running on " +
  `${API_BASE} and try again.`;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
    });
  } catch {
    throw new ApiError("NETWORK_ERROR", OFFLINE_MESSAGE, 0);
  }

  if (!response.ok) {
    let code = "REQUEST_FAILED";
    let message = `The request failed with status ${response.status}.`;
    let details: Record<string, unknown> | null | undefined;
    try {
      const body = (await response.json()) as ApiErrorBody;
      if (body?.error) {
        code = body.error.code ?? code;
        message = body.error.message ?? message;
        details = body.error.details;
      }
    } catch {
      // A non-JSON error body is not worth surfacing verbatim.
    }
    throw new ApiError(code, message, response.status, details);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),

  upload: async (file: File, signal?: AbortSignal) => {
    const body = new FormData();
    body.append("file", file);
    // Not routed through `request`: FormData must not get a JSON content-type.
    let response: Response;
    try {
      response = await fetch(`${API_BASE}/api/analysis/upload`, {
        method: "POST",
        body,
        signal,
      });
    } catch (error) {
      if ((error as Error)?.name === "AbortError") throw error;
      throw new ApiError("NETWORK_ERROR", OFFLINE_MESSAGE, 0);
    }
    if (!response.ok) {
      let code = "UPLOAD_FAILED";
      let message = "The upload could not be processed.";
      let details: Record<string, unknown> | null | undefined;
      try {
        const parsed = (await response.json()) as ApiErrorBody;
        if (parsed?.error) {
          code = parsed.error.code ?? code;
          message = parsed.error.message ?? message;
          details = parsed.error.details;
        }
      } catch {
        /* ignore */
      }
      throw new ApiError(code, message, response.status, details);
    }
    return (await response.json()) as UploadResponse;
  },

  /** Create an analysis from the bundled sample study (demonstration only). */
  loadSample: () =>
    request<UploadResponse>("/api/analysis/sample", { method: "POST" }),

  run: (id: string) =>
    request<RunResponse>(`/api/analysis/${id}/run`, { method: "POST" }),

  status: (id: string) => request<StatusResponse>(`/api/analysis/${id}/status`),

  result: (id: string) => request<AnalysisResult>(`/api/analysis/${id}/result`),

  history: (limit = 50) =>
    request<HistoryResponse>(`/api/analysis?limit=${limit}`),

  remove: (id: string) =>
    request<{ analysis_id: string; deleted: boolean }>(`/api/analysis/${id}`, {
      method: "DELETE",
    }),

  /**
   * URL for one rendered slice.
   *
   * Returned as a URL rather than fetched bytes so the browser's image cache and
   * the backend's immutable Cache-Control header do the work; the volume itself
   * never crosses the network.
   */
  sliceUrl(
    id: string,
    sliceIndex: number,
    options: {
      mode?: SliceMode;
      opacity?: number;
      classes?: number[];
      scale?: number;
      highlightDisc?: number | null;
      /**
       * `findings` asks the server to tint finding-associated disc regions. The
       * server decides which discs qualify; this only requests the rendering.
       */
      highlight?: HighlightMode;
      findingOpacity?: number;
    } = {},
  ): string {
    const {
      mode = "overlay",
      opacity = 0.45,
      classes,
      scale,
      highlightDisc,
      highlight,
      findingOpacity,
    } = options;
    const params = new URLSearchParams({ mode, opacity: opacity.toFixed(2) });
    if (classes && classes.length > 0) params.set("classes", classes.join(","));
    if (scale && scale > 1) params.set("scale", String(scale));
    if (highlightDisc != null) params.set("highlight_disc", String(highlightDisc));
    // Omitted when `disc`, which is the endpoint's default, so existing URLs and
    // their cache entries are unchanged while the overlay is off.
    if (highlight && highlight !== "disc") params.set("highlight", highlight);
    if (highlight === "findings" && findingOpacity != null) {
      params.set("finding_opacity", findingOpacity.toFixed(2));
    }
    return `${API_BASE}/api/analysis/${id}/slice/${sliceIndex}?${params.toString()}`;
  },

  downloadUrl: (id: string, fmt: "md" | "json" = "md") =>
    `${API_BASE}/api/analysis/${id}/download?fmt=${fmt}`,
};

/** Human-readable byte size. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Short, locale-stable timestamp. */
export function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "unknown";
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatSeconds(value: number): string {
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}m ${seconds}s`;
}
