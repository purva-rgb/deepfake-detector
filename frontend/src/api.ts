// Typed client for the FastAPI backend (api/main.py). Field names mirror api/serializer.py exactly.
// `null` always means "N/A / not available" and must never be rendered as 0.

export const API_BASE: string = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

export type Band = "LOW" | "MEDIUM" | "HIGH" | "N/A";

export type Risk = { score: number | null; band: Band; reason: string | null };
export type MediaRisk = Risk & { windows_flagged: number; windows_scored: number };
export type ModalityBlock = {
  available: boolean;
  score: number | null;
  reliability: number | null;
  reason: string | null;
  low_confidence: boolean;
};
export type ScamCategory = {
  id: string;
  label: string;
  found: boolean;
  matches: { text: string; at: number }[];
};
export type SemanticBlock = {
  available: boolean;
  score: number | null;
  reliability: number | null;
  reason: string | null;
  categories: ScamCategory[];
  categories_available: boolean;
};
export type SupportBlock = {
  available: boolean;
  windows_available: number;
  windows_total: number;
  aggregation: string;
  features: Record<string, number | null>;
  reason: string | null;
  supporting_only: true;
};
export type SyncBlock = {
  available: boolean;
  score: number | null;
  best_lag_ms: number | null;
  reliability: number | null;
  status: string | null;
  reason: string | null;
  windows_used: number;
  supporting_only: true;
};
export type EvidenceItem = {
  rank: number;
  start: number;
  end: number;
  modality: "visual" | "audio" | "scam";
  score: number;
  reliability: number;
  reason: string;
  text: string | null;
  thumbnail: string | null;
};
export type TranscriptMatch = {
  start: number;
  end: number;
  text: string;
  category: string;
  label: string;
  suppressed: boolean;
  note: string | null;
};
export type TranscriptSegment = {
  start: number;
  end: number;
  text: string;
  confidence: number;
  matches: TranscriptMatch[];
};
export type WindowResult = {
  start: number;
  end: number;
  media: { score: number | null; band: Band };
  scam: { score: number | null; band: Band };
  visual: { score: number | null; reliability: number | null };
  audio: { score: number | null; reliability: number | null };
  semantic: { score: number | null; reliability: number | null; categories: string[] };
  faces: { frames_with_face: number; frames: number };
  acoustic: { available: boolean; reason: string | null; features: Record<string, number | null> };
  facial_temporal: { available: boolean; reason: string | null; features: Record<string, number | null> };
  av_sync: { available: boolean; score: number | null; best_lag_ms: number | null; reliability: number | null; reason: string | null };
};

export type AnalysisResult = {
  status: "ok";
  analysis_id: string;
  video: { duration_s: number | null; analysed_s: number; has_audio: boolean; truncated: boolean; windows: number; window_s: number; hop_s: number };
  insufficient_evidence: boolean;
  media_risk: MediaRisk;
  scam_risk: Risk;
  visual: ModalityBlock;
  audio: ModalityBlock;
  semantic: SemanticBlock;
  acoustic: SupportBlock;
  facial_temporal: SupportBlock;
  av_sync: SyncBlock;
  reliability: { visual: number | null; audio: number | null; transcript: number | null };
  reliability_gate: number;
  warnings: string[];
  technical_notes: string[];
  evidence: EvidenceItem[];
  timeline: { start: number; end: number; media: { score: number | null; band: Band }; scam: { score: number | null; band: Band } }[];
  windows: WindowResult[];
  transcript: string;
  transcript_segments: TranscriptSegment[];
  transcript_available: boolean;
  scam_analysis_available: boolean;
};

export class ApiError extends Error {
  constructor(message: string, public code: string) {
    super(message);
  }
}

async function failure(res: Response): Promise<ApiError> {
  try {
    const body = await res.json();
    const d = body?.detail;
    if (d && typeof d === "object" && d.message) return new ApiError(String(d.message), String(d.code ?? "error"));
    if (typeof d === "string") return new ApiError(d, "error");
  } catch {
    /* fall through */
  }
  return new ApiError(`The analysis service returned an error (${res.status}).`, "http_error");
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError("Could not reach the analysis service. Make sure the backend is running and try again.", "network");
  }
  if (!res.ok) throw await failure(res);
  return (await res.json()) as T;
}

export async function analyzeVideo(file: File): Promise<AnalysisResult> {
  const body = new FormData();
  body.append("video", file);
  const result = await request<AnalysisResult>("/api/analyze", { method: "POST", body });
  console.log("REAL API RESPONSE:", result);
  return result;
}

export function fetchProgress(): Promise<{ running: boolean; stage: string | null; fraction: number }> {
  return request("/api/progress");
}
