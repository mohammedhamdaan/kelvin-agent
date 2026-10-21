/** Dashboard header: Voice Assistant, LLM Brain, STT, and RAG status. */

import "./Header.css";
import type { HealthResponse } from "../types";

interface Props {
  health: HealthResponse | null;
  connected: boolean;
}

function StatusTile({
  label,
  value,
  tone,
  detail,
}: {
  label: string;
  value: string;
  tone: "ok" | "warn" | "bad" | "muted";
  detail?: string;
}) {
  return (
    <div className={`status-tile status-tile-${tone}`} title={detail}>
      <span className="status-tile-label">{label}</span>
      <span className="status-tile-value">
        <span className={`status-dot status-${tone === "ok" ? "live" : tone === "bad" ? "bad" : "ok"}`} />
        {value}
      </span>
    </div>
  );
}

export function Header({ health, connected }: Props) {
  const llmOnline = Boolean(connected && health?.llm && health.llm !== "offline-router");
  const modelName = health?.llm_model || health?.llm || "Gemini";

  return (
    <header className="app-header">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true">
          <span className="brand-mark-ring" />
          K6
        </div>
        <div>
          <h1>KELVIN6K VOICE ASSISTANT</h1>
          <span className="brand-sub">Autonomous Agentic Brain · SQLite DB · RAG Knowledge</span>
        </div>
      </div>

      <div className="status-tiles">
        <StatusTile
          label="API"
          value={connected ? "CONNECTED" : "OFFLINE"}
          tone={connected ? "ok" : "bad"}
          detail={connected ? "Kelvin6K API server is reachable" : "Backend unreachable"}
        />
        <StatusTile
          label="Brain"
          value={llmOnline ? "ONLINE" : "OFFLINE ROUTER"}
          tone={llmOnline ? "ok" : "warn"}
          detail={`Model: ${modelName}`}
        />
        <StatusTile
          label="STT"
          value={connected ? "WHISPER" : "OFFLINE"}
          tone={connected ? "ok" : "bad"}
          detail={health?.stt || "faster-whisper"}
        />
        <StatusTile
          label="RAG"
          value={health ? `${health.rag_chunks} CHUNKS` : "IDLE"}
          tone={health?.rag_chunks ? "ok" : "muted"}
          detail={`Mode: ${health?.rag_mode || "hybrid"}`}
        />
      </div>

      <div className="header-chips">
        {health ? (
          <>
            <span className="header-chip">LLM: {modelName}</span>
            <span className="header-chip">TTS: {health.tts || "ElevenLabs / pyttsx3"}</span>
            <span className="header-chip">RAG: {health.rag_mode}</span>
          </>
        ) : null}
        <span className="header-chip header-chip-readonly">READ-ONLY</span>
      </div>
    </header>
  );
}
