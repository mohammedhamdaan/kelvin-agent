/**
 * Alexa-style microphone capture: the operator speaks a complete sentence and
 * the recorder decides for itself when they have finished.
 *
 * There is deliberately NO fixed recording length. The flow is:
 *
 *   CALIBRATING -- measure the room's background noise for a moment
 *   WAITING     -- armed, waiting for the operator to start speaking
 *   SPEAKING    -- recording; stays here for as long as speech continues
 *   TRAILING    -- speech stopped; if the silence lasts, finish
 *
 * Two details make this behave naturally rather than cutting people off:
 *
 *  * The speech threshold is measured, not hard-coded. A noisy print floor and
 *    a quiet office need very different values, so the background level is
 *    sampled first and the threshold is set above it.
 *  * Hysteresis: it takes a fairly loud sound to START speech, but a much
 *    quieter one to CONTINUE it. Without this, the natural pauses between
 *    words in a sentence end the recording halfway through.
 *
 * The level is read from an AnalyserNode on every animation frame, so the UI
 * meter reflects the real microphone signal rather than an animation.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { transcribeAudio } from "../api";
import { constraintsFor, loadMicChoice } from "../audio/micConfigs";
import { claimMicrophone } from "../audio/micLock";

export type RecorderStatus =
  | "idle"
  | "requesting"
  | "recording"
  | "transcribing"
  | "denied"
  | "error";

/** Where we are inside a single recording. Drives the UI wording. */
export type RecorderPhase = "idle" | "calibrating" | "waiting" | "speaking" | "trailing";

export interface MicDevice {
  deviceId: string;
  label: string;
  isCommunications: boolean;
}

/** Chunk interval: guarantees several ondataavailable events per recording. */
const TIMESLICE_MS = 250;
/** How long the background noise is sampled before listening begins. */
const CALIBRATION_MS = 400;
/** Never spend longer than this calibrating, even without usable samples. */
const CALIBRATION_MAX_MS = 1200;
/** Silence must be this far below the measured room noise, as a multiplier. */
const NOISE_HEADROOM = 1.8;
/** The silence bar never drops below this, however quiet the room seems. */
const MIN_SILENCE_LEVEL = 0.003;
/** ...and never rises above this fraction of the speech bar, so a drifting
 *  noise estimate can never end up treating the operator's voice as silence. */
const SILENCE_CEILING_RATIO = 0.8;
/** Give up if the operator never says anything. */
const WAIT_FOR_SPEECH_MS = 12000;
/**
 * Silence AFTER speech that means "they have finished".
 *
 * Reduced to ~650ms to deliver fast Alexa-like response times while the hysteresis
 * below (speech continues at half the start threshold) prevents mid-sentence pauses
 * from cutting speech prematurely.
 */
const SILENCE_HOLD_MS = 650;
/** Safety cap only -- not a recording limit. A sentence never runs this long. */
const MAX_RECORDING_MS = 30000;
/** Guards against an accidental instant stop producing a header-only file. */
export const MIN_RECORDING_MS = 800;
/** The threshold never drops below this, however quiet the room is (tuned for far-field). */
const MIN_SPEECH_LEVEL = 0.005;
/**
 * The threshold never rises above this, however loud the room is.
 *
 * THIS CAP IS WHY THE ASSISTANT CAN STILL HEAR YOU. Calibration measures the
 * first 400 ms as "background noise", but an operator naturally starts talking
 * the moment they click START LISTENING -- so their own voice gets measured as
 * the noise floor. Threshold was then noiseFloor x 3, which lands far above
 * normal speech, and the recorder waited forever for a sound that could never
 * come. Normal speech peaks around 0.05-0.30, so the bar must never exceed a
 * quiet speaking voice.
 */
const MAX_SPEECH_LEVEL = 0.045;
/** Speech must be this many times louder than the measured background noise. */
const NOISE_MARGIN = 2.2;
/** Continuing speech only needs this fraction of the start threshold, so the
 *  pauses between words do not end the sentence. */
const CONTINUE_RATIO = 0.5;
/** Below this many bytes there is no usable audio. */
const MIN_USABLE_BYTES = 8000;
/** Below this many bytes there is no usable confirmation audio (e.g. short "Yes", "Haan", "No", "Nahi"). */
const MIN_CONFIRMATION_BYTES = 1800;

function pickMimeType(): string | undefined {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported?.(type));
}

/**
 * Picks the device most likely to carry audio.
 *
 * Chrome exposes two pseudo-devices alongside the real ones: "default" follows
 * Windows Settings > Sound > Input, and "communications" is the phone-call
 * endpoint, which is frequently silent. "default" is the right choice because
 * it follows whatever the operator selected in Windows.
 */
function preferredDevice(devices: MicDevice[]): string | undefined {
  const usable = devices.filter((d) => !d.isCommunications);
  return (usable.find((d) => d.deviceId === "default") ?? usable[0] ?? devices[0])?.deviceId;
}

export function useRecorder(options?: {
  autoStopOnSilence?: boolean;
  /**
   * Called with the transcript however the recording ended.
   *
   * This matters because silence detection stops the recording by itself: with
   * only a promise returned from stopAndTranscribe(), a hands-free question
   * would be transcribed and then silently dropped, because nobody was waiting
   * on that promise.
   */
  onTranscript?: (text: string) => void;
}) {
  // Manual by default. Silence detection is an extra on top of a flow that
  // must work on its own first: press START, speak, press STOP.
  const autoStop = options?.autoStopOnSilence ?? false;
  /**
   * Whether THIS recording stops itself on silence.
   *
   * The manual button and the hands-free wake-word flow need opposite
   * behaviour, so start() takes a per-recording override: pressing
   * START LISTENING keeps recording until STOP, while "Okay Kelvin" must end the
   * recording by itself because nobody is going to press anything.
   */
  const autoStopRef = useRef(autoStop);
  const onTranscriptRef = useRef(options?.onTranscript);
  onTranscriptRef.current = options?.onTranscript;

  const supported =
    typeof navigator !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof MediaRecorder !== "undefined";

  const [status, setStatus] = useState<RecorderStatus>("idle");
  const [phase, setPhase] = useState<RecorderPhase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [audioBytes, setAudioBytes] = useState(0);
  const [chunkCount, setChunkCount] = useState(0);
  const [durationMs, setDurationMs] = useState(0);
  const [speechMs, setSpeechMs] = useState(0);
  const [mimeType, setMimeType] = useState("");
  const [level, setLevel] = useState(0);
  const [peakLevel, setPeakLevel] = useState(0);
  const [threshold, setThreshold] = useState(MIN_SPEECH_LEVEL);
  const [deviceLabel, setDeviceLabel] = useState("");
  const [devices, setDevices] = useState<MicDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [processingEnabled, setProcessingEnabled] = useState(false);
  const [playbackUrl, setPlaybackUrl] = useState<string | null>(null);
  /** Live diagnostics for the on-screen readout. */
  const [debug, setDebug] = useState({
    device: "-", deviceId: "-", permission: "-", trackState: "-", trackMuted: "-",
    audioContext: "-", recorderState: "-", constraints: "-",
  });

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const resolveRef = useRef<((text: string) => void) | null>(null);
  const startedAtRef = useRef(0);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number | null>(null);
  const stoppingRef = useRef(false);
  const peakRef = useRef(0);
  const heardSpeechRef = useRef(false);
  const speechStartedAtRef = useRef(0);
  const playbackUrlRef = useRef<string | null>(null);
  /** Set by start(); read by the animation-frame loop. */
  const finishRef = useRef<() => void>(() => {});
  /** Resolves stopAndGetBlob() when recording stops. */
  const resolveBlobRef = useRef<((blob: Blob) => void) | null>(null);
  /** Rolling ambient-noise estimate so wake turns can start immediately without a 400ms delay. */
  const rollingNoiseRef = useRef(0.003);
  /** Whether the active recording turn is expecting a short confirmation reply. */
  const isConfirmationRef = useRef(false);

  const refreshDevices = useCallback(async (): Promise<MicDevice[]> => {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      const mics: MicDevice[] = all
        .filter((d) => d.kind === "audioinput")
        .map((d) => ({
          deviceId: d.deviceId,
          label: d.label || "Microphone (allow access to see its name)",
          isCommunications: d.deviceId === "communications" || /communications/i.test(d.label),
        }));
      setDevices(mics);
      return mics;
    } catch (err) {
      console.warn("[MIC] Could not list devices:", err);
      return [];
    }
  }, []);

  useEffect(() => {
    void refreshDevices();
  }, [refreshDevices]);

  /** Ends the active recording session and completely releases the microphone hardware. */
  const stopSession = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    recorderRef.current = null;
    setLevel(0);
    setPhase("idle");
    if (streamRef.current) {
      try {
        streamRef.current.getTracks().forEach((track) => track.stop());
      } catch {
        /* ignore */
      }
      streamRef.current = null;
    }
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      try {
        void audioCtxRef.current.close();
      } catch {
        /* ignore */
      }
      audioCtxRef.current = null;
    }
  }, []);

  /** Full hardware release on unmount or device switch. */
  const teardown = useCallback(() => {
    stopSession();
  }, [stopSession]);

  /** Stops the recorder, refusing to stop before MIN_RECORDING_MS. */
  const finish = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive" || stoppingRef.current) return;

    const elapsed = Date.now() - startedAtRef.current;
    if (elapsed < MIN_RECORDING_MS) {
      window.setTimeout(finish, MIN_RECORDING_MS - elapsed + 50);
      return;
    }

    stoppingRef.current = true;
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    try {
      recorder.requestData?.();
      recorder.stop();
    } catch (err) {
      console.warn("[MIC] stop() failed:", err);
    }
  }, []);
  finishRef.current = finish;

  const start = useCallback(
    async (opts?: { autoStopOnSilence?: boolean; isConfirmation?: boolean }): Promise<boolean> => {
      autoStopRef.current = opts?.autoStopOnSilence ?? autoStop;
      isConfirmationRef.current = opts?.isConfirmation ?? false;
    if (!supported) {
      setStatus("error");
      setError("This browser cannot record audio (MediaRecorder is unavailable).");
      return false;
    }
    if (recorderRef.current && recorderRef.current.state === "recording") return true;
    if (!window.isSecureContext) {
      setStatus("error");
      setError(
        "The microphone is blocked because this page is not on a secure origin. " +
          "Open it as http://localhost:5173 (not the machine's IP address).",
      );
      return false;
    }

    setError(null);
    setStatus("requesting");
    setPhase("calibrating");
    setAudioBytes(0);
    setChunkCount(0);
    setDurationMs(0);
    setSpeechMs(0);
    setPeakLevel(0);
    peakRef.current = 0;
    heardSpeechRef.current = false;
    speechStartedAtRef.current = 0;
    stoppingRef.current = false;
    chunksRef.current = [];

    if (playbackUrlRef.current) {
      URL.revokeObjectURL(playbackUrlRef.current);
      playbackUrlRef.current = null;
      setPlaybackUrl(null);
    }

    let wantedId = selectedDeviceId;
    if (!wantedId) {
      const mics = devices.length ? devices : await refreshDevices();
      wantedId = preferredDevice(mics) ?? "";
      if (wantedId) setSelectedDeviceId(wantedId);
    }

    // Prefer the configuration the diagnostic proved works on this machine.
    const saved = loadMicChoice();
    let audioConstraints: MediaTrackConstraints | true;
    if (saved) {
      // `wantedId` (the dropdown) takes priority over the remembered device --
      // this is the SAME device the diagnostic's live meter opens, so both
      // paths now always listen to the same microphone.
      audioConstraints = constraintsFor(saved.configId, wantedId || saved.deviceId || undefined);
      console.info(`[MIC] Using the configuration saved by the diagnostic: ${saved.configId}`);
    } else {
      const built: MediaTrackConstraints = {
        echoCancellation: processingEnabled,
        noiseSuppression: processingEnabled,
        autoGainControl: true,
      };
      if (wantedId) built.deviceId = { exact: wantedId };
      audioConstraints = built;
    }
    console.info("[MIC] Constraints:", JSON.stringify(audioConstraints));

    // Make sure nothing else (the diagnostic's live meter) still holds the
    // microphone -- a second stream on the same device can come back muted.
    claimMicrophone();

    // Report the browser's own permission state, not an assumption about it.
    try {
      const status = await navigator.permissions?.query({ name: "microphone" as PermissionName });
      setDebug((d) => ({ ...d, permission: status?.state ?? "unknown" }));
    } catch {
      setDebug((d) => ({ ...d, permission: "not reported by this browser" }));
    }

    let stream: MediaStream;
    const isStreamAlive =
      streamRef.current &&
      streamRef.current.active &&
      streamRef.current.getAudioTracks().some((t) => t.readyState === "live");

    if (isStreamAlive) {
      stream = streamRef.current!;
      console.info("[MIC] Reusing active microphone stream (0ms acquisition delay)");
    } else {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
      } catch (err) {
        const name = err instanceof Error ? err.name : "";
        setPhase("idle");
        if (name === "NotAllowedError" || name === "SecurityError") {
          setStatus("denied");
          setError(
            "Microphone permission was denied. Click the padlock icon in the address bar, " +
              "set Microphone to Allow, then reload the page.",
          );
        } else if (name === "NotReadableError") {
          setStatus("error");
          setError(
            "Windows would not open the microphone — another program is using it. " +
              "Close Zoom, Teams, or any Python script that records audio, then try again.",
          );
        } else if (name === "NotFoundError" || name === "OverconstrainedError") {
          setStatus("error");
          setError("That microphone is not available. Run the microphone diagnostic below.");
        } else {
          setStatus("error");
          setError(`Could not open the microphone: ${err instanceof Error ? err.message : String(err)}`);
        }
        return false;
      }
      streamRef.current = stream;
    }

    const track = stream.getAudioTracks()[0];
    const settings = track?.getSettings?.() ?? {};
    setDeviceLabel(track?.label || "default microphone");
    setDebug((d) => ({
      ...d,
      device: track?.label || "(unknown)",
      deviceId: String(settings.deviceId ?? "(not reported)"),
      trackState: track?.readyState ?? "-",
      trackMuted: String(!!track?.muted),
      constraints: JSON.stringify(audioConstraints),
    }));
    console.info("[MIC] Device:", track?.label, "| muted:", track?.muted, "|", JSON.stringify(settings));

    // --- audio graph -----------------------------------------------------
    const AudioCtor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    let ctx = audioCtxRef.current;
    if (!ctx || ctx.state === "closed") {
      ctx = new AudioCtor();
      audioCtxRef.current = ctx;
    }
    // A suspended context's AnalyserNode returns nothing but zeros forever.
    if (ctx.state === "suspended") await ctx.resume();
    setDebug((d) => ({ ...d, audioContext: ctx.state }));

    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024; // smaller window => a more responsive meter
    source.connect(analyser);
    // Keep the graph pulled, without playing the microphone back into the room.
    const silence = ctx.createGain();
    silence.gain.value = 0;
    analyser.connect(silence);
    silence.connect(ctx.destination);

    const buffer = new Float32Array(analyser.fftSize);

    // --- recorder --------------------------------------------------------
    const chosenMime = pickMimeType();
    setMimeType(chosenMime ?? "browser default");

    const recorder = new MediaRecorder(stream, chosenMime ? { mimeType: chosenMime } : undefined);
    recorderRef.current = recorder;

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data);
        setChunkCount(chunksRef.current.length);
      }
    };

    recorder.onstop = async () => {
      const duration = Date.now() - startedAtRef.current;
      const blob = new Blob(chunksRef.current, { type: chosenMime ?? "audio/webm" });
      const peak = peakRef.current;
      const spokeFor = speechStartedAtRef.current ? Date.now() - speechStartedAtRef.current : 0;

      setDurationMs(duration);
      setAudioBytes(blob.size);

      const url = URL.createObjectURL(blob);
      playbackUrlRef.current = url;
      setPlaybackUrl(url);

      stopSession();

      if (resolveBlobRef.current) {
        resolveBlobRef.current(blob);
        resolveBlobRef.current = null;
      }

      console.info(
        `[TIMING] Recording     : ${(duration / 1000).toFixed(2)}s`,
      );
      console.info(
        `[MIC] Stopped — ${(duration / 1000).toFixed(2)}s, ${chunksRef.current.length} chunks, ` +
          `${blob.size} bytes, peak ${peak.toFixed(4)}, speech ${(spokeFor / 1000).toFixed(1)}s`,
      );

      // In MANUAL mode the operator decided the recording was finished, so it
      // is sent even if the level meter never crossed the speech threshold.
      // The backend measures the real peak amplitude and reports silence
      // precisely -- far better evidence than refusing to send anything.
      const minBytes = isConfirmationRef.current ? MIN_CONFIRMATION_BYTES : (!autoStopRef.current ? 400 : MIN_USABLE_BYTES);
      const rejectAsSilent = autoStopRef.current && !heardSpeechRef.current;
      if (rejectAsSilent || blob.size < minBytes) {
        setStatus("error");
        setError(
          peak < MIN_SPEECH_LEVEL
            ? `No sound reached the browser (peak ${(peak * 100).toFixed(1)}%). ` +
              `Run the microphone diagnostic below to find a working configuration.`
            : `Nothing was recognised as speech (peak ${(peak * 100).toFixed(1)}%, ${blob.size} bytes). ` +
              `Speak a little louder or move closer to the microphone.`,
        );
        resolveRef.current?.("");
        resolveRef.current = null;
        stoppingRef.current = false;
        isConfirmationRef.current = false;
        return;
      }

      setStatus("transcribing");
      try {
        const text = await transcribeAudio(blob);
        console.info("[STT] Transcript:", text);
        const trimmed = text ? text.trim() : "";
        if (autoStopRef.current && !trimmed) {
          setStatus("error");
          setError("Nothing was recognised as speech. Speak a little louder or move closer to the microphone.");
          resolveRef.current?.("");
        } else {
          setStatus("idle");
          resolveRef.current?.(text);
          if (!resolveRef.current && trimmed) onTranscriptRef.current?.(text);
        }
      } catch (err) {
        setStatus("error");
        setError(err instanceof Error ? err.message : String(err));
        resolveRef.current?.("");
      } finally {
        resolveRef.current = null;
        stoppingRef.current = false;
        isConfirmationRef.current = false;
      }
    };

    // --- voice activity detection, on every animation frame ---------------
    startedAtRef.current = Date.now();
    let noiseSum = 0;
    let noiseCount = 0;
    let measuredNoise = rollingNoiseRef.current || 0.003;
    let speechThreshold = Math.min(
      MAX_SPEECH_LEVEL,
      Math.max(MIN_SPEECH_LEVEL, measuredNoise * NOISE_MARGIN),
    );
    let silenceThreshold = Math.min(
      speechThreshold * SILENCE_CEILING_RATIO,
      Math.max(
        speechThreshold * CONTINUE_RATIO,
        measuredNoise * NOISE_HEADROOM,
        MIN_SILENCE_LEVEL,
      ),
    );
    setThreshold(speechThreshold);
    let silenceSince: number | null = null;
    let currentPhase: RecorderPhase = autoStopRef.current ? "waiting" : "calibrating";
    setPhase(currentPhase);

    const tick = () => {
      const ctxRef = audioCtxRef.current;
      if (!ctxRef || stoppingRef.current) return;

      analyser.getFloatTimeDomainData(buffer);
      let sum = 0;
      for (let i = 0; i < buffer.length; i += 1) sum += buffer[i] * buffer[i];
      const rms = Math.sqrt(sum / buffer.length);
      const now = Date.now();
      const elapsed = now - startedAtRef.current;

      setLevel(rms);
      if (rms > peakRef.current) {
        peakRef.current = rms;
        setPeakLevel(rms);
      }
      setDurationMs(elapsed);
      if (speechStartedAtRef.current) setSpeechMs(now - speechStartedAtRef.current);

      // Guard against the chime (150ms) or residual "Okay Kelvin" room echo triggering speech start
      const pastWakeGrace = elapsed >= 120;

      if (currentPhase === "calibrating") {
        // Measure the room, then set the bar just above it.
        if (rms > 0 && rms < MAX_SPEECH_LEVEL) {
          noiseSum += rms;
          noiseCount += 1;
        }
        // Wait for real samples rather than trusting a fixed window.
        const enoughSamples = noiseCount >= 8;
        if (elapsed >= CALIBRATION_MS && (enoughSamples || elapsed >= CALIBRATION_MAX_MS)) {
          const noiseFloor = noiseCount ? noiseSum / noiseCount : 0;
          measuredNoise = noiseFloor;
          rollingNoiseRef.current = noiseFloor;
          speechThreshold = Math.min(
            MAX_SPEECH_LEVEL,
            Math.max(MIN_SPEECH_LEVEL, noiseFloor * NOISE_MARGIN),
          );
          setThreshold(speechThreshold);
          currentPhase = "waiting";
          setPhase("waiting");
          silenceThreshold = Math.min(
            speechThreshold * SILENCE_CEILING_RATIO,
            Math.max(
              speechThreshold * CONTINUE_RATIO,
              noiseFloor * NOISE_HEADROOM,
              MIN_SILENCE_LEVEL,
            ),
          );
          console.info(
            `[MIC] Background noise ${noiseFloor.toFixed(4)} (${noiseCount} samples) → ` +
              `speech starts at ${speechThreshold.toFixed(4)}, ` +
              `silence below ${silenceThreshold.toFixed(4)}`,
          );
        }
      } else if (currentPhase === "waiting") {
        // While waiting for speech, continuously adapt ambient noise floor
        if (rms > 0 && rms < speechThreshold) {
          measuredNoise = measuredNoise * 0.98 + rms * 0.02;
          rollingNoiseRef.current = measuredNoise;
          speechThreshold = Math.min(
            MAX_SPEECH_LEVEL,
            Math.max(MIN_SPEECH_LEVEL, measuredNoise * NOISE_MARGIN),
          );
          silenceThreshold = Math.min(
            speechThreshold * SILENCE_CEILING_RATIO,
            Math.max(
              speechThreshold * CONTINUE_RATIO,
              measuredNoise * NOISE_HEADROOM,
              MIN_SILENCE_LEVEL,
            ),
          );
          setThreshold(speechThreshold);
        }

        // Only count speech after wake grace period (120ms) to avoid chime reflection
        if (pastWakeGrace && rms >= speechThreshold) {
          heardSpeechRef.current = true;
          speechStartedAtRef.current = now;
          silenceSince = null;
          currentPhase = "speaking";
          setPhase("speaking");
          console.info(`[MIC] Speech started (level ${rms.toFixed(4)} >= threshold ${speechThreshold.toFixed(4)})`);
        } else if (autoStopRef.current && elapsed > WAIT_FOR_SPEECH_MS) {
          console.info("[MIC] No speech within the waiting window — stopping");
          finishRef.current();
          return;
        }
      } else {
        // Speaking or trailing. Continuing speech is judged against a LOWER bar
        // so that the gaps between words do not end the sentence.
        const continueThreshold = silenceThreshold;
        // Hysteresis on the silence decision too: once quiet, it takes a
        // clearly louder sound to count as speech again. Without this the
        // level hovers either side of one number and the silence timer keeps
        // restarting, so the recording never ends.
        const resumeThreshold = continueThreshold * 1.3;
        const stillSpeaking = silenceSince === null ? rms >= continueThreshold : rms >= resumeThreshold;
        if (stillSpeaking) {
          if (silenceSince !== null) {
            console.info(`[MIC] Speech resumed (level ${rms.toFixed(4)})`);
          }
          silenceSince = null;
          if (currentPhase !== "speaking") {
            currentPhase = "speaking";
            setPhase("speaking");
          }
        } else {
          // Keep learning the room, but SLOWLY and within a ceiling. An
          // unbounded estimate climbed 0.031 -> 0.052 in three seconds during
          // one recording, and left unchecked it would eventually rise above
          // the operator's voice and treat speech itself as silence.
          measuredNoise = measuredNoise * 0.99 + rms * 0.01;
          silenceThreshold = Math.min(
            speechThreshold * SILENCE_CEILING_RATIO,
            Math.max(
              speechThreshold * CONTINUE_RATIO,
              measuredNoise * NOISE_HEADROOM,
              MIN_SILENCE_LEVEL,
            ),
          );
          if (silenceSince === null) {
            silenceSince = now;
            currentPhase = "trailing";
            setPhase("trailing");
            console.info(
              `[MIC] Silence began (level ${rms.toFixed(4)} < ${continueThreshold.toFixed(4)}) — ` +
                `stopping in ${SILENCE_HOLD_MS}ms unless speech resumes`,
            );
          } else if (autoStopRef.current && now - silenceSince >= SILENCE_HOLD_MS) {
            console.info(`[MIC] ${SILENCE_HOLD_MS}ms of silence after speech — finished`);
            finishRef.current();
            return;
          }
        }
      }

      if (elapsed >= MAX_RECORDING_MS) {
        console.info("[MIC] Safety cap reached — stopping");
        finishRef.current();
        return;
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    recorder.start(TIMESLICE_MS);
    setStatus("recording");
    setDebug((d) => ({ ...d, recorderState: recorder.state }));
    // `muted` changes over time, so keep reporting it rather than reading once.
    track?.addEventListener("mute", () => setDebug((d) => ({ ...d, trackMuted: "true" })));
    track?.addEventListener("unmute", () => setDebug((d) => ({ ...d, trackMuted: "false" })));
    rafRef.current = requestAnimationFrame(tick);
    console.info("[MIC] Recording started — speak your full question, no time limit");

    void refreshDevices(); // labels are revealed once permission is granted
    return true;
  }, [supported, autoStop, devices, selectedDeviceId, processingEnabled, refreshDevices, stopSession]);

  /** Pre-warming is disabled in manual mode so the mic remains completely closed until user clicks. */
  const prewarm = useCallback(async (): Promise<void> => {
    // Disabled in manual mic toggle mode
  }, []);

  /** Stops recording and returns the raw recorded audio Blob. */
  const stopAndGetBlob = useCallback((): Promise<Blob> => {
    return new Promise((resolve) => {
      const recorder = recorderRef.current;
      if (!recorder || recorder.state === "inactive") {
        resolve(new Blob([], { type: "audio/webm" }));
        return;
      }
      resolveBlobRef.current = resolve;
      finish();
    });
  }, [finish]);

  /** Stops recording and resolves with the transcript. */
  const stopAndTranscribe = useCallback((): Promise<string> => {
    return new Promise((resolve) => {
      const recorder = recorderRef.current;
      if (!recorder || recorder.state === "inactive") {
        resolve("");
        return;
      }
      resolveRef.current = resolve;
      finish();
    });
  }, [finish]);

  /** Stops recording; the transcript arrives via the onTranscript callback. */
  const stop = useCallback(() => {
    finish();
  }, [finish]);

  const cancel = useCallback(() => {
    stoppingRef.current = true;
    try {
      recorderRef.current?.stop();
    } catch {
      /* ignore */
    }
    stopSession();
    setStatus("idle");
  }, [stopSession]);

  useEffect(
    () => () => {
      teardown();
      if (playbackUrlRef.current) URL.revokeObjectURL(playbackUrlRef.current);
    },
    [teardown],
  );

  return {
    supported,
    status,
    phase,
    error,
    audioBytes,
    chunkCount,
    durationMs,
    speechMs,
    mimeType,
    level,
    peakLevel,
    threshold,
    debug,
    deviceLabel,
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
    refreshDevices,
    processingEnabled,
    setProcessingEnabled,
    playbackUrl,
    start,
    stop,
    stopAndGetBlob,
    stopAndTranscribe,
    cancel,
    prewarm,
  };
}
