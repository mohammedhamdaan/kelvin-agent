/**
 * Client for the Kelvin6K Python API.
 *
 * `ask()` is THE shared query service: the text box and the voice pipeline both
 * call it, so a typed question and a spoken one follow exactly the same path.
 */

import type { AskResponse, HealthResponse, HistoryResponse, StatusResponse } from "./types";

export const API_BASE =
  import.meta.env.VITE_API_BASE?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { signal });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return getJson<HealthResponse>("/api/health", signal);
}

export function getStatus(signal?: AbortSignal): Promise<StatusResponse> {
  return getJson<StatusResponse>("/api/status", signal);
}

const SESSION_STORAGE_KEY = "kelvin6k.session_id";

export function getOrCreateSessionId(): string {
  try {
    let id = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (!id) {
      id = "session_" + Math.random().toString(36).substring(2, 10);
      window.sessionStorage.setItem(SESSION_STORAGE_KEY, id);
    }
    return id;
  } catch {
    return "default_terminal";
  }
}

/** Shared by text input AND voice. Sends query to POST /api/ask preserving session_id. */
export async function ask(
  question: string,
  speak = false,
  sessionId?: string,
): Promise<AskResponse> {
  const session_id = sessionId || getOrCreateSessionId();
  const response = await fetch(`${API_BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, speak, session_id }),
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => response.statusText);
    throw new Error(`Ask failed: ${detail}`);
  }
  return (await response.json()) as AskResponse;
}

/** Server-side speech-to-text (faster-whisper) -- fallback when the browser has no Web Speech API. */
/** Maps a MediaRecorder MIME type to the matching file extension. */
function audioExtension(mime: string): string {
  if (mime.includes("webm")) return "webm";
  if (mime.includes("ogg")) return "ogg";
  if (mime.includes("mp4") || mime.includes("m4a")) return "mp4";
  if (mime.includes("wav")) return "wav";
  return "webm";
}

export async function transcribeAudio(blob: Blob): Promise<string> {
  // The filename must match the ACTUAL format. It was hard-coded to
  // "question.wav" while the browser produces WebM/Opus, so the server wrote
  // WebM bytes into a file named .wav and the decoder had to guess.
  const filename = `question.${audioExtension(blob.type)}`;
  const url = `${API_BASE}/api/stt`;

  console.info("[STT] POST", url);
  console.info(`[STT] Blob: ${blob.size} bytes, type "${blob.type}", sent as ${filename}`);

  const form = new FormData();
  form.append("audio", blob, filename);

  const response = await fetch(url, { method: "POST", body: form });
  console.info("[STT] HTTP", response.status, response.statusText);

  if (!response.ok) {
    const raw = await response.text().catch(() => response.statusText);
    console.error("[STT] Error body:", raw);
    // FastAPI wraps errors as {"detail": "..."}; show the sentence, not JSON.
    let message = raw;
    try {
      const parsed = JSON.parse(raw) as { detail?: unknown };
      if (typeof parsed.detail === "string") message = parsed.detail;
    } catch {
      /* not JSON -- use the raw text */
    }
    throw new Error(message);
  }

  const data = (await response.json()) as { transcript: string };
  console.info("[STT] Transcript:", data.transcript);
  return data.transcript;
}

export const cameraStreamUrl = () => `${API_BASE}/api/camera/stream`;
export const cameraSnapshotUrl = () => `${API_BASE}/api/camera/snapshot?t=${Date.now()}`;
export const camera2StreamUrl = () => `${API_BASE}/api/camera2/stream`;
export const camera2SnapshotUrl = () => `${API_BASE}/api/camera2/snapshot?t=${Date.now()}`;

export function getHistory(signal?: AbortSignal): Promise<HistoryResponse> {
  return getJson<HistoryResponse>("/api/history", signal);
}

/** Speak text through the SERVER's speaker (existing pyttsx3/ElevenLabs TTS). */
export async function speakOnServer(text: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) {
    throw new Error(`Server TTS failed: ${await response.text().catch(() => response.statusText)}`);
  }
}

/**
 * Fetches synthesised speech from the backend as an audio Blob.
 * The BROWSER plays it, so the sound comes out of the operator's speakers
 * rather than the server's.
 */
export async function fetchSpeechAudio(text: string): Promise<{ blob: Blob; engine: string }> {
  const response = await fetch(`${API_BASE}/api/tts/audio`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) {
    throw new Error(`TTS failed: ${await response.text().catch(() => response.statusText)}`);
  }
  return {
    blob: await response.blob(),
    engine: response.headers.get("X-TTS-Engine") ?? "unknown",
  };
}

export interface TtsDiagnostics {
  provider: string;
  elevenlabs_key_configured: boolean;
  voice_count: number;
  pyttsx3_voices: string[];
  volume?: number;
  current_voice?: string | null;
  engine_error?: string;
}

export function getTtsDiagnostics(): Promise<TtsDiagnostics> {
  return getJson<TtsDiagnostics>("/api/tts/diagnostics");
}
