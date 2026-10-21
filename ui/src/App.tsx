import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import { getHealth } from "./api";
import { Header } from "./components/Header";
import { VoiceAssistant } from "./components/VoiceAssistant";
import type { HealthResponse } from "./types";

const HEALTH_POLL_MS = 5000;

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const timerRef = useRef<number | null>(null);

  const pollHealth = useCallback(async () => {
    try {
      const data = await getHealth();
      setHealth(data);
      setConnectionError(null);
    } catch (error) {
      setConnectionError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    void pollHealth();
    timerRef.current = window.setInterval(() => void pollHealth(), HEALTH_POLL_MS);

    return () => {
      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
      }
    };
  }, [pollHealth]);

  return (
    <div className="app">
      <Header health={health} connected={!connectionError} />

      {connectionError ? (
        <div className="banner banner-error">
          Cannot reach the Kelvin6K API server. Start it with <code>python run_api.py</code>.
          <span className="banner-detail">({connectionError})</span>
        </div>
      ) : null}

      <main className="voice-main-container">
        <VoiceAssistant />
      </main>
    </div>
  );
}
