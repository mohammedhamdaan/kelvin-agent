/**
 * Kelvin6K Voice Assistant -- Manual Mic ON/OFF Toggle.
 *
 * Operates with a strict manual push-to-talk / toggle workflow:
 *
 *   State 1: MIC OFF (IDLE)
 *     -> Button: "Turn Mic ON (Start Speaking)"
 *     -> Mic stream is completely stopped/released. KelvinCore3D is IDLE.
 *
 *   State 2: User Clicks Button -> MIC ON (LISTENING / HUMAN_SPEAKING)
 *     -> Starts capturing audio via useRecorder.
 *     -> Displays live audio levels in VoiceLevel.tsx.
 *     -> KelvinCore3D state is LISTENING / HUMAN_SPEAKING.
 *     -> Button: Pulsing red/amber "Turn Mic OFF (Stop & Send)".
 *
 *   State 3: User Clicks Button Again -> MIC OFF & PROCESS (THINKING -> KELVIN_SPEAKING -> IDLE)
 *     -> Immediately stops audio recording and releases/mutes mic stream.
 *     -> Transitions KelvinCore3D to THINKING.
 *     -> Sends recorded audio blob to POST /api/stt.
 *     -> Sends transcript to POST /api/ask (preserving session_id across turns).
 *     -> Plays returned TTS audio (KELVIN_SPEAKING).
 *     -> Once audio playback finishes, transitions cleanly back to IDLE (MIC OFF).
 *
 * READ-ONLY: this panel only asks questions; it never commands the machine.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE, ask, getHealth, getOrCreateSessionId, speakOnServer, transcribeAudio } from "../api";
import type { HealthResponse, MetrologyAnnouncementResponse } from "../types";
import { KelvinCore3D, type CoreState } from "./KelvinCore3D";
import { MicDiagnostic } from "./MicDiagnostic";
import { useAudioPlayer } from "../hooks/useAudioPlayer";
import { useRecorder } from "../hooks/useRecorder";
import { VoiceLevel } from "./VoiceLevel";
import { useVoice } from "../hooks/useVoice";
import "./VoiceAssistant.css";

// Minimal typings: SpeechRecognition is not in the standard TS DOM lib.
interface SpeechRecognitionAlternativeLike {
  transcript: string;
  confidence: number;
}
interface SpeechRecognitionResultLike {
  isFinal: boolean;
  0: SpeechRecognitionAlternativeLike;
  length: number;
}
interface SpeechRecognitionEventLike {
  resultIndex: number;
  results: { length: number; [index: number]: SpeechRecognitionResultLike };
}
interface SpeechRecognitionLike {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start: () => void;
  stop: () => void;
  abort: () => void;
  onstart: (() => void) | null;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
}

function getRecognitionCtor(): (new () => SpeechRecognitionLike) | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: new () => SpeechRecognitionLike;
    webkitSpeechRecognition?: new () => SpeechRecognitionLike;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

/** Speaks text instantly using browser window.speechSynthesis with natural English voice and rate = 1.05. */
function speakBrowserText(text: string): Promise<void> {
  return new Promise((resolve) => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      resolve();
      return;
    }
    const synth = window.speechSynthesis;
    try {
      synth.cancel();
    } catch {
      /* ignore */
    }

    const cleanText = text
      .replace(/[*#_`]/g, "")
      .replace(/\bmm\b/g, "millimeters")
      .replace(/%/g, " percent")
      .trim();

    if (!cleanText) {
      resolve();
      return;
    }

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;
    utterance.lang = "en-US";

    const voices = synth.getVoices();
    const enVoice =
      voices.find(
        (v) =>
          (v.lang === "en-US" || v.lang.startsWith("en")) &&
          (v.name.includes("Google") ||
            v.name.includes("Natural") ||
            v.name.includes("Samantha") ||
            v.name.includes("Daniel") ||
            !v.localService),
      ) ||
      voices.find((v) => v.lang.startsWith("en")) ||
      null;

    if (enVoice) {
      utterance.voice = enVoice;
    }

    let finished = false;
    const done = () => {
      if (!finished) {
        finished = true;
        resolve();
      }
    };

    utterance.onend = done;
    utterance.onerror = done;
    // Safety timeout so audio playback never blocks the assistant permanently
    window.setTimeout(done, 15000);

    synth.speak(utterance);
  });
}

interface Exchange {
  question: string;
  answer: string;
  intent: string | null;
  dataAvailable: boolean;
  at: number;
}

/** Fine-grained voice pipeline display states. */
export type VoiceDisplayState =
  | "WAITING_FOR_SPEECH"
  | "SPEECH_DETECTED"
  | "SILENCE_WHILE_RECORDING"
  | "PROCESSING"
  | "SPEAKING"
  | "READY";

interface VoiceDisplayInfo {
  title: string;
  subtitle: string;
}

const VOICE_DISPLAY_MAP: Record<VoiceDisplayState, VoiceDisplayInfo> = {
  WAITING_FOR_SPEECH: {
    title: "Microphone Active — Speak Now",
    subtitle: "Click 'Turn Mic OFF' when finished speaking",
  },
  SPEECH_DETECTED: {
    title: "Hearing You Speak…",
    subtitle: "Click 'Turn Mic OFF' when finished speaking",
  },
  SILENCE_WHILE_RECORDING: {
    title: "Microphone Active",
    subtitle: "Click 'Turn Mic OFF' when finished speaking",
  },
  PROCESSING: {
    title: "Processing…",
    subtitle: "Querying machine data and AI…",
  },
  SPEAKING: {
    title: "Speaking…",
    subtitle: "Playing audio response…",
  },
  READY: {
    title: "Ready — Mic is OFF",
    subtitle: "Click 'Turn Mic ON' to begin speaking",
  },
};

/** Headline shown under the 3D core, one per visual state. */
const CORE_TITLE: Record<CoreState, string> = {
  IDLE: "Ready — Mic is OFF",
  LISTENING: "Listening… Speak now",
  HUMAN_SPEAKING: "Hearing you speak…",
  THINKING: "Processing question…",
  KELVIN_SPEAKING: "Kelvin is speaking…",
  DONE: "Done",
  ERROR: "Something went wrong",
};

/** How long the transient DONE / ERROR looks hold before returning to IDLE. */
const DONE_HOLD_MS = 1400;
const ERROR_HOLD_MS = 1800;

const CONFIRMATION_INTENTS: ReadonlySet<string> = new Set([
  "SHIFT_START",
  "SHIFT_END",
  "LOG_WALL_PRODUCTION",
  "LOG_SHIFT_ISSUE",
  "LOG_SHIFT_HANDOVER",
]);

/** Upper bound on waiting for an answer to finish speaking. */
const SPEAK_TIMEOUT_MS = 30000;

/** Developer controls are hidden unless the page is opened with ?dev=1. */
const DEV_MODE =
  typeof window !== "undefined" && new URLSearchParams(window.location.search).has("dev");

/** Conversation history survives a page reload. */
const HISTORY_KEY = "kelvin6k.conversation";
const HISTORY_LIMIT = 100;
const METROLOGY_POLL_MS = 1500;

function loadHistory(): Exchange[] {
  try {
    const raw = window.localStorage.getItem(HISTORY_KEY);
    const parsed = raw ? (JSON.parse(raw) as Exchange[]) : [];
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item) =>
        item &&
        typeof item.question === "string" &&
        typeof item.answer === "string" &&
        typeof item.at === "number",
    );
  } catch {
    return [];
  }
}

function saveHistory(history: Exchange[]): void {
  try {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-HISTORY_LIMIT)));
  } catch {
    /* not fatal */
  }
}

export function VoiceAssistant({ autoSpeak: initialAutoSpeak = true }: { autoSpeak?: boolean }) {
  const [history, setHistory] = useState<Exchange[]>(loadHistory);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [autoSpeak, setAutoSpeak] = useState(initialAutoSpeak);
  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [serverTts, setServerTts] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [ttsFellBack, setTtsFellBack] = useState(false);
  const [speakingViaServer, setSpeakingViaServer] = useState(false);
  const [synthSpeaking, setSynthSpeaking] = useState(false);

  // Stable session_id preserved across manual clicks
  const sessionIdRef = useRef<string>(getOrCreateSessionId());
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const liveTranscriptRef = useRef<string>("");

  const autoSpeakRef = useRef(autoSpeak);
  useEffect(() => {
    autoSpeakRef.current = autoSpeak;
  }, [autoSpeak]);

  const serverTtsRef = useRef(serverTts);
  useEffect(() => {
    serverTtsRef.current = serverTts;
  }, [serverTts]);

  const lastSpokenAnnouncementRef = useRef(0);

  const recorder = useRecorder({ autoStopOnSilence: false });
  const player = useAudioPlayer();
  const voice = useVoice({
    onQuestion: () => {},
  });

  const isRecording = recorder.status === "recording";
  const speaking = synthSpeaking || player.speaking || voice.speaking || speakingViaServer;

  // Cleanup browser recognition and speech on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort();
        } catch {
          /* ignore */
        }
      }
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        try {
          window.speechSynthesis.cancel();
        } catch {
          /* ignore */
        }
      }
    };
  }, []);

  /** Speaks text and resolves when speech finishes. */
  const speakAnswerText = useCallback(
    async (text: string): Promise<void> => {
      if (serverTtsRef.current) {
        setSpeakingViaServer(true);
        try {
          await speakOnServer(text);
        } catch (err) {
          setApiError(
            `Backend TTS failed: ${err instanceof Error ? err.message : String(err)} — trying browser audio.`,
          );
          setTtsFellBack(true);
          await player.speak(text);
        } finally {
          setSpeakingViaServer(false);
        }
        return;
      }
      setSynthSpeaking(true);
      try {
        await speakBrowserText(text);
      } catch (err) {
        console.warn("[TTS] Browser SpeechSynthesis error:", err);
      } finally {
        setSynthSpeaking(false);
      }
    },
    [player],
  );

  const speakRef = useRef(speakAnswerText);
  useEffect(() => {
    speakRef.current = speakAnswerText;
  }, [speakAnswerText]);

  /** Sends a question to the backend -- identical path for text and voice. */
  const submit = useCallback(async (question: string, via: "text" | "voice") => {
    const trimmed = question.trim();
    if (!trimmed) {
      setBusy(false);
      return;
    }

    setBusy(true);
    setApiError(null);
    try {
      const askedAt = Date.now();
      const response = await ask(trimmed, false, sessionIdRef.current);
      const answer = response.answer;

      const isConfirmation =
        via === "voice" && CONFIRMATION_INTENTS.has(response.intent ?? "");
      setConfirming(isConfirmation);

      console.info(`[TIMING] Query + answer : ${((Date.now() - askedAt) / 1000).toFixed(2)}s`);
      console.info("[ASSISTANT] Answer:", answer);

      // 1. Immediately update conversation history so answer appears in chat box right away
      try {
        if (response.intent !== "WAKE_WORD") {
          setHistory((prev) =>
            [
              ...prev,
              {
                question: trimmed,
                answer,
                intent: response.intent,
                dataAvailable: response.data_available,
                at: Date.now(),
              },
            ].slice(-HISTORY_LIMIT),
          );
        }
      } catch (historyError) {
        console.warn("[ASSISTANT] Could not record the exchange:", historyError);
      }

      // 2. Play speech immediately without blocking UI rendering
      if (autoSpeakRef.current && answer) {
        await Promise.race([
          speakRef.current?.(answer),
          new Promise((resolve) => window.setTimeout(resolve, SPEAK_TIMEOUT_MS)),
        ]);
      }
    } catch (error) {
      setApiError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
      // Microphone is NEVER automatically re-opened! Stays strictly OFF (IDLE).
    }
  }, []);

  /** Main manual toggle button handler implementing the 3-state state machine. */
  const handleMicToggle = useCallback(async () => {
    if (isRecording) {
      // State 3: User Clicks Button Again -> Turn Mic OFF (Stop & Send)
      setIsProcessing(true);
      setApiError(null);

      // Stop browser recognition immediately and capture recognized text
      const rec = recognitionRef.current;
      if (rec) {
        try {
          rec.stop();
        } catch {
          /* already stopped */
        }
      }
      const capturedText = liveTranscriptRef.current.trim();
      setInterimTranscript("");

      // Immediately stop audio recording and release/mute the microphone input
      const blobPromise = recorder.stopAndGetBlob();

      try {
        if (capturedText) {
          // INSTANT 0-SECOND PATH: browser SpeechRecognition captured text!
          console.info("[VOICE] Instant STT via browser SpeechRecognition:", capturedText);
          setTranscript(capturedText);
          await submit(capturedText, "voice");
        } else {
          // FALLBACK PATH: browser SpeechRecognition empty or unsupported -> use /api/stt
          console.info("[VOICE] Browser STT empty, falling back to /api/stt blob upload...");
          const blob = await blobPromise;
          if (!blob || blob.size < 400) {
            setApiError("Recording was too short. Turn the mic ON and speak your question.");
            setIsProcessing(false);
            return;
          }

          const text = await transcribeAudio(blob);
          const trimmed = (text || "").trim();
          setTranscript(trimmed);

          if (!trimmed) {
            setApiError("Nothing was recognised as speech. Turn the mic ON and try again.");
            setIsProcessing(false);
            return;
          }

          await submit(trimmed, "voice");
        }
      } catch (err) {
        console.error("[VOICE] Error processing audio recording:", err);
        setApiError(err instanceof Error ? err.message : String(err));
      } finally {
        setIsProcessing(false);
      }
    } else {
      // State 2: User Clicks Button -> MIC ON (LISTENING / HUMAN_SPEAKING)
      if (isProcessing || speaking || busy) return;
      setApiError(null);
      setTranscript("");
      setInterimTranscript("");
      liveTranscriptRef.current = "";

      // Start browser SpeechRecognition simultaneously with the microphone
      const Ctor = getRecognitionCtor();
      if (Ctor) {
        try {
          if (recognitionRef.current) {
            try {
              recognitionRef.current.abort();
            } catch {
              /* ignore */
            }
          }
          const recognition = new Ctor();
          recognition.continuous = true;
          recognition.interimResults = true;
          recognition.lang = "en-US";

          recognition.onresult = (event: SpeechRecognitionEventLike) => {
            let finalText = "";
            let interimText = "";
            for (let i = 0; i < event.results.length; i += 1) {
              const res = event.results[i];
              if (res.isFinal) finalText += res[0].transcript + " ";
              else interimText += res[0].transcript;
            }
            const combined = `${finalText} ${interimText}`.trim();
            liveTranscriptRef.current = combined;
            setInterimTranscript(interimText || finalText || combined);
          };

          recognition.onerror = (e) => {
            console.warn("[BROWSER STT] Error:", e);
          };

          recognitionRef.current = recognition;
          recognition.start();
        } catch (err) {
          console.warn("[BROWSER STT] Could not start:", err);
        }
      }

      await recorder.start({ autoStopOnSilence: false });
    }
  }, [isRecording, isProcessing, speaking, busy, recorder, submit]);

  // Metrology layer announcements
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      if (cancelled || busy || isRecording || isProcessing) return;
      if (!autoSpeakRef.current) return;
      try {
        const response = await fetch(`${API_BASE}/api/metrology/announcement`);
        if (!response.ok) return;
        const data = (await response.json()) as MetrologyAnnouncementResponse;
        const announcement = data.announcement;
        if (!announcement || cancelled) return;
        if (announcement.id <= lastSpokenAnnouncementRef.current) return;
        lastSpokenAnnouncementRef.current = announcement.id;
        console.info("[METROLOGY] speaking:", announcement.text);
        await speakRef.current?.(announcement.text);
      } catch {
        /* backend unreachable, retry next tick */
      }
    };
    const timer = window.setInterval(() => void poll(), METROLOGY_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [busy, isRecording, isProcessing]);

  useEffect(() => {
    saveHistory(history);
  }, [history]);

  useEffect(() => {
    void getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  // 3D core: transient DONE / ERROR looks
  const [coreFlash, setCoreFlash] = useState<"DONE" | "ERROR" | null>(null);
  const coreFlashTimerRef = useRef<number | null>(null);
  const turnErroredRef = useRef(false);
  const wasSpeakingRef = useRef(false);

  useEffect(() => {
    const flash = (kind: "DONE" | "ERROR", holdMs: number) => {
      if (coreFlashTimerRef.current) window.clearTimeout(coreFlashTimerRef.current);
      setCoreFlash(kind);
      coreFlashTimerRef.current = window.setTimeout(() => {
        setCoreFlash(null);
        coreFlashTimerRef.current = null;
      }, holdMs);
    };

    if (recorder.status === "error" || apiError) {
      turnErroredRef.current = true;
      flash("ERROR", ERROR_HOLD_MS);
    }
    if (wasSpeakingRef.current && !speaking) {
      if (!turnErroredRef.current) flash("DONE", DONE_HOLD_MS);
      turnErroredRef.current = false;
    }
    wasSpeakingRef.current = speaking;
  }, [speaking, recorder.status, apiError]);

  useEffect(
    () => () => {
      if (coreFlashTimerRef.current) window.clearTimeout(coreFlashTimerRef.current);
    },
    [],
  );

  // Derive visual states
  const isHumanSpeaking =
    isRecording && (recorder.phase === "speaking" || recorder.level >= recorder.threshold);

  const coreState: CoreState =
    coreFlash === "ERROR"
      ? "ERROR"
      : speaking
        ? "KELVIN_SPEAKING"
        : isProcessing || busy || recorder.status === "transcribing"
          ? "THINKING"
          : isHumanSpeaking
            ? "HUMAN_SPEAKING"
            : isRecording
              ? "LISTENING"
              : coreFlash === "DONE"
                ? "DONE"
                : "IDLE";

  const coreLevel = isRecording
    ? Math.min(1, recorder.level / Math.max(0.02, recorder.threshold * 3))
    : 0;

  const deriveDisplayState = (): VoiceDisplayState => {
    if (speaking) return "SPEAKING";
    if (isProcessing || busy || recorder.status === "transcribing") return "PROCESSING";
    if (isRecording) {
      if (isHumanSpeaking) return "SPEECH_DETECTED";
      if (recorder.phase === "trailing") return "SILENCE_WHILE_RECORDING";
      return "WAITING_FOR_SPEECH";
    }
    return "READY";
  };

  const displayState = deriveDisplayState();
  const displayInfo = VOICE_DISPLAY_MAP[displayState];

  const coreTitle = confirming
    ? isHumanSpeaking
      ? "Hearing Confirmation…"
      : "Listening for Confirmation…"
    : CORE_TITLE[coreState];

  const clock = (ms: number) => {
    const total = Math.floor(ms / 1000);
    return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
  };

  const clearHistory = useCallback(() => setHistory([]), []);

  const micOk = recorder.supported && recorder.status !== "denied";
  const micBroken = !recorder.supported || recorder.status === "denied";
  const showDeveloper = DEV_MODE || micBroken;

  const components: Array<[string, string, boolean]> = [
    [
      "Microphone Mode",
      "Manual ON / OFF Toggle",
      true,
    ],
    [
      "Microphone Hardware",
      micOk
        ? recorder.deviceLabel || (isRecording ? "Recording active" : "Standby (released)")
        : "permission denied",
      micOk,
    ],
    [
      "Mic Stream Status",
      isRecording
        ? `CAPTURING (${clock(recorder.durationMs)})`
        : isProcessing
          ? "RELEASED (PROCESSING)"
          : speaking
            ? "RELEASED (SPEAKING)"
            : "STOPPED / RELEASED (IDLE)",
      true,
    ],
    ["Speech-to-Text", health?.stt ?? "faster-whisper", !!health],
    [
      "Answer Engine",
      health?.llm === "gemini" ? (health.llm_model ?? "gemini") : "offline router",
      !!health,
    ],
    ["Text-to-Speech", health?.tts ?? "pyttsx3 / browser", !!health],
  ];

  return (
    <section className="panel va">
      <header className="va-header">
        <h2>KELVIN6K VOICE ASSISTANT</h2>
        <span className="pill pill-readonly" title="This assistant can only read machine data.">
          READ-ONLY
        </span>
      </header>

      {/* --- Central stage with Kelvin 3D Character and Manual Toggle Button --- */}
      <div className={`va-stage va-stage-${displayState.toLowerCase()}`}>
        <KelvinCore3D state={coreState} level={coreLevel} />

        <p className="va-stage-text" aria-live="polite">
          {coreTitle}
        </p>

        <p className={`va-stage-sub va-stage-sub-${displayState.toLowerCase()}`}>
          <span className="va-dot" />
          {confirming
            ? "Turn Mic ON to say 'Yes' or 'No' to confirm"
            : displayInfo.subtitle}
        </p>

        {/* Real microphone waveform bars, driven purely by existing recorder.level */}
        {isRecording ? (
          <div className="va-level-wrapper" aria-hidden="true">
            <VoiceLevel
              level={recorder.level}
              threshold={recorder.threshold}
              active={isRecording}
            />
          </div>
        ) : null}

        {/* Live speech-to-text transcript while user is speaking */}
        {isRecording && (interimTranscript || liveTranscriptRef.current) ? (
          <div className="va-interim-box">
            <span className="va-interim-tag">Hearing:</span>
            <span className="va-interim-text">
              “{interimTranscript || liveTranscriptRef.current}”
            </span>
          </div>
        ) : null}

        {/* PROMINENT MANUAL ON / OFF TOGGLE BUTTON */}
        <div className="va-mic-toggle-wrapper">
          <button
            type="button"
            className={`va-mic-toggle-btn ${
              isRecording
                ? "va-mic-toggle-recording"
                : isProcessing || busy
                  ? "va-mic-toggle-processing"
                  : speaking
                    ? "va-mic-toggle-speaking"
                    : "va-mic-toggle-ready"
            }`}
            onClick={() => void handleMicToggle()}
            disabled={isProcessing || busy || speaking}
            aria-label={
              isRecording
                ? "Turn Mic OFF (Stop & Send)"
                : isProcessing || busy
                  ? "Processing question..."
                  : speaking
                    ? "Kelvin speaking..."
                    : "Turn Mic ON (Start Speaking)"
            }
          >
            {isRecording ? (
              <>
                <span className="va-mic-icon-pulse" aria-hidden="true">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" />
                  </svg>
                </span>
                <span className="va-mic-label">Turn Mic OFF (Stop & Send)</span>
              </>
            ) : isProcessing || busy ? (
              <>
                <span className="va-mic-spinner" aria-hidden="true" />
                <span className="va-mic-label">Processing Question…</span>
              </>
            ) : speaking ? (
              <>
                <span className="va-mic-speaker-icon" aria-hidden="true">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                    <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
                    <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
                  </svg>
                </span>
                <span className="va-mic-label">Kelvin Speaking…</span>
              </>
            ) : (
              <>
                <span className="va-mic-icon-ready" aria-hidden="true">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="1" y1="1" x2="23" y2="23" />
                    <path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6" />
                    <path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23" />
                    <line x1="12" y1="19" x2="12" y2="23" />
                    <line x1="8" y1="23" x2="16" y2="23" />
                  </svg>
                </span>
                <span className="va-mic-label">Turn Mic ON (Start Speaking)</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* --- Conversation History --- */}
      {history.length > 0 ? (
        <div className="va-convo">
          <div className="va-convo-head">
            <h3>CONVERSATION</h3>
            <span className="va-convo-count">
              {history.length} {history.length === 1 ? "question" : "questions"}
            </span>
            <button type="button" className="va-convo-clear" onClick={clearHistory}>
              CLEAR
            </button>
          </div>

          <div className="va-convo-list">
            {history
              .slice()
              .reverse()
              .map((item, index) => (
                <article
                  key={item.at}
                  className={`va-turn${index === 0 ? " va-turn-latest" : ""}`}
                >
                  <p className="va-msg">
                    <span className="va-tag va-tag-human">👤 HUMAN</span>
                    <span className="va-msg-text">{item.question}</span>
                  </p>
                  <p className="va-msg">
                    <span className="va-tag va-tag-robot">🤖 KELVIN AGENT</span>
                    <span className="va-msg-text va-msg-robot">{item.answer}</span>
                  </p>
                  <p className="va-turn-meta">
                    {new Date(item.at).toLocaleTimeString()}
                    {DEV_MODE ? ` · ${item.intent}` : ""}
                    {item.dataAvailable === false ? " · live data not available" : ""}
                  </p>
                </article>
              ))}
          </div>
        </div>
      ) : (
        <p className="va-empty">
          {transcript
            ? `Heard: “${transcript}”`
            : "No questions yet. Click “Turn Mic ON (Start Speaking)” to begin."}
        </p>
      )}

      {apiError ? <p className="va-warn">{apiError}</p> : null}
      {recorder.error ? <p className="va-warn">{recorder.error}</p> : null}

      {/* --- System Status --- */}
      <dl className="va-system">
        {components.map(([name, value, ok]) => (
          <div key={name} className="va-system-row">
            <dt>{name}</dt>
            <dd>
              <span className={`va-chip ${ok ? "va-chip-ok" : "va-chip-bad"}`}>
                {ok ? "READY" : "ERROR"}
              </span>
              <span className="va-system-detail">{value}</span>
            </dd>
          </div>
        ))}
      </dl>

      {/* --- Developer Section --- */}
      {showDeveloper ? (
        <details className="va-dev" open={micBroken}>
          <summary>Developer / Diagnostics</summary>

          <div className="va-dev-body">
            <div className="va-dev-controls">
              <button
                className={`btn ${isRecording ? "btn-danger" : "btn-primary"}`}
                onClick={() => void handleMicToggle()}
                disabled={!recorder.supported || isProcessing || busy}
              >
                {isRecording ? `STOP & SEND (${clock(recorder.durationMs)})` : "START SPEAKING"}
              </button>
              <button
                className="btn"
                onClick={() => {
                  const latest = history[history.length - 1];
                  if (latest) void speakAnswerText(latest.answer);
                }}
                disabled={history.length === 0}
              >
                SPEAK ANSWER
              </button>
            </div>

            <div className="va-dev-toggles">
              <label>
                <input
                  type="checkbox"
                  checked={autoSpeak}
                  onChange={(event) => setAutoSpeak(event.target.checked)}
                />
                <span>Speak answers</span>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={serverTts}
                  onChange={(event) => setServerTts(event.target.checked)}
                />
                <span>Use backend speaker</span>
              </label>
              {ttsFellBack ? (
                <span className="va-note">Backend audio failed; using browser speech.</span>
              ) : null}
            </div>

            <form
              className="va-dev-ask"
              onSubmit={(event) => {
                event.preventDefault();
                void submit(input, "text");
                setInput("");
              }}
            >
              <input
                type="text"
                value={input}
                placeholder="Type a question (no microphone needed)"
                onChange={(event) => setInput(event.target.value)}
              />
              <button className="btn" type="submit" disabled={busy || isProcessing || !input.trim()}>
                ASK
              </button>
            </form>

            {isRecording || recorder.audioBytes > 0 ? (
              <pre className="va-debug">
{`MIC DEVICE    : ${recorder.debug.device}
DEVICE ID     : ${recorder.debug.deviceId}
PERMISSION    : ${recorder.debug.permission}
TRACK STATE   : ${recorder.debug.trackState}
TRACK MUTED   : ${recorder.debug.trackMuted}
AUDIO CONTEXT : ${recorder.debug.audioContext}
LIVE RMS      : ${recorder.level.toFixed(4)}
LIVE PEAK     : ${recorder.peakLevel.toFixed(4)}
THRESHOLD     : ${recorder.threshold.toFixed(4)}
MEDIA RECORDER: ${recorder.debug.recorderState}
CHUNKS        : ${recorder.chunkCount}
BLOB SIZE     : ${recorder.audioBytes}
DURATION      : ${(recorder.durationMs / 1000).toFixed(2)}s
CONSTRAINTS   : ${recorder.debug.constraints}`}
              </pre>
            ) : null}

            {recorder.playbackUrl ? (
              <div className="va-playback">
                <span>Play back the last recording</span>
                <audio controls src={recorder.playbackUrl} />
              </div>
            ) : null}

            <MicDiagnostic />

            <button className="btn" onClick={clearHistory} disabled={history.length === 0}>
              CLEAR CONVERSATION
            </button>
          </div>
        </details>
      ) : (
        <p className="va-devhint">
          Manual mic mode active. Open this page with <code>?dev=1</code> for advanced diagnostics.
        </p>
      )}
    </section>
  );
}
