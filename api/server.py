"""
HTTP API for the Kelvin6K React dashboard.

A thin wrapper over the EXISTING Python modules -- no machine logic is
reimplemented here:

    /api/status           -> agent.services.machine_state (+ agent.sensors, agent.db)
    /api/assistant/query  -> agent.assistant / agent.query_router (tools + RAG)
    /api/stt              -> agent.listener   (faster-whisper)
    /api/tts*             -> agent.voice      (ElevenLabs / pyttsx3)
    /api/camera/*         -> api.camera_service (CameraStream + VisionPipeline)

Every value carries a provenance tag so the UI can never present a simulated or
missing number as if it were measured:

    REAL         measured from the camera/AI pipeline, or entered for the job
    MOCK         produced by the built-in simulator (SimulatedSensors)
    UNAVAILABLE  not connected / not measured -- shown as such, never guessed

READ-ONLY: there are no endpoints that command the robot, pump or accelerator.
"""

import asyncio
import datetime
import io
import json
import os
import sqlite3
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from agent import config, db, metrics, shift_store
from agent.assistant import Assistant
from agent.logging_setup import configure_logging, get_logger
from agent.rag.retriever import index_summary
from agent.rules import RuleEngine
from agent.sensors import NotConnectedMachine, SimulatedSensors
from agent.services import machine_state as ms
from agent.state import PrinterState

configure_logging()
log = get_logger("API")

REAL = "REAL"
MOCK = "MOCK"
UNAVAILABLE = "UNAVAILABLE"

app = FastAPI(title="Kelvin6K API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dashboard; tighten for production deployment
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-TTS-Engine"],
)

class UnavailableCameraService:
    status = "unavailable"
    source = "none"
    fps = 0.0
    resolution = (0, 0)
    error = "Camera functionality is unavailable in this version."
    metrology = None
    metrology_announcement = None
    def start(self): pass
    def stop(self): pass
    def latest_jpeg(self): return None
    def latest_people_count(self): return 0
    def latest_fire_count(self): return 0
    def latest_detections(self): return []

# Shared singletons -- the same objects the console app uses.
state = PrinterState()
rules = RuleEngine()
camera_service = UnavailableCameraService()
site_camera_service = UnavailableCameraService()
ms.bind_site_camera_service(site_camera_service)
assistant = Assistant(state=state)
_sensors = SimulatedSensors()
_machine_link = NotConnectedMachine()


def field(value: Any, provenance: str, unit: str = "", reason: str = "") -> Dict[str, Any]:
    """Uniform value envelope so the UI always knows where a number came from."""
    return {"value": value, "provenance": provenance, "unit": unit, "reason": reason}


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    ms.bind_live_state(state)
    ms.bind_site_camera_service(site_camera_service)

    # Load speech-to-text in the background so the FIRST question does not pay
    # the model load. Threaded, so startup is not blocked either.
    import threading

    from agent import listener

    threading.Thread(target=listener.preload, name="whisper-preload", daemon=True).start()

    def _warm_answer_engine() -> None:
        try:
            assistant.ask("what is the machine status")
        except Exception:
            pass

    threading.Thread(target=_warm_answer_engine, name="answer-warmup", daemon=True).start()
    log.info("API started (camera disabled); speech-to-text loading in background")


@app.on_event("shutdown")
def _shutdown() -> None:
    pass


# --- health -------------------------------------------------------------

@app.get("/api/health")
def health() -> Dict[str, Any]:
    from agent.voice import provider_name

    rag = index_summary()
    return {
        "ok": True,
        "camera_source": config.CAMERA_SOURCE,
        "camera_status": camera_service.status,
        "llm": "gemini" if config.GEMINI_API_KEY else "offline-router",
        "llm_model": config.GEMINI_MODEL if config.GEMINI_API_KEY else None,
        "stt": f"faster-whisper ({config.WHISPER_MODEL_SIZE}) - {listener.get_status()}",
        "tts": provider_name(),
        "rag_chunks": rag["chunks"],
        "rag_mode": "semantic" if rag["semantic"] else "lexical",
        "calibrated": config.PIXELS_PER_MM > 0,
        "read_only": True,
    }


# --- status -------------------------------------------------------------

@app.get("/api/status")
def status() -> Dict[str, Any]:
    """Everything the dashboard shows, each field tagged with its provenance."""
    progress = ms.get_print_progress()
    layer = ms.get_current_layer()
    width = ms.get_current_layer_width()
    print_status = ms.get_current_print_status()
    machine = ms.get_machine_status()

    camera = {
        "status": camera_service.status,
        "source": camera_service.source,
        "fps": round(camera_service.fps, 1),
        "resolution": list(camera_service.resolution),
        "error": camera_service.error,
        "provenance": REAL if camera_service.status == "live" else UNAVAILABLE,
    }

    camera2 = {
        "status": site_camera_service.status,
        "source": site_camera_service.source,
        "fps": round(site_camera_service.fps, 1),
        "resolution": list(site_camera_service.resolution),
        "error": site_camera_service.error,
        "provenance": REAL if site_camera_service.status == "live" else UNAVAILABLE,
        "people_count": site_camera_service.latest_people_count(),
        "fire_count": site_camera_service.latest_fire_count(),
        "detections": site_camera_service.latest_detections(),
    }

    # --- production (from the tracked print job) ---
    if progress.get("available"):
        production = {
            "job_name": field(progress["job_name"], REAL),
            "total_mm": field(progress["total_mm"], REAL, "mm"),
            "completed_mm": field(progress["completed_mm"], REAL, "mm"),
            "remaining_mm": field(progress["remaining_mm"], REAL, "mm"),
            "percentage": field(progress["percentage"], REAL, "%"),
            "current_layer": field(progress.get("current_layer"), REAL),
            "total_layers": field(progress.get("total_layers"), REAL),
        }
    else:
        reason = progress.get("reason", "No active print job.")
        production = {
            key: field(None, UNAVAILABLE, unit, reason)
            for key, unit in (
                ("job_name", ""), ("total_mm", "mm"), ("completed_mm", "mm"),
                ("remaining_mm", "mm"), ("percentage", "%"),
                ("current_layer", ""), ("total_layers", ""),
            )
        }

    # --- vision-derived ---
    vision = {
        "printing": field(
            print_status.get("printing") if print_status.get("available") else None,
            REAL if print_status.get("available") else UNAVAILABLE,
            "", print_status.get("reason", ""),
        ),
        "print_status": field(
            print_status.get("status") if print_status.get("available") else None,
            REAL if print_status.get("available") else UNAVAILABLE,
            "", print_status.get("reason", ""),
        ),
        "layer_number": field(
            layer.get("current_layer") if layer.get("available") else None,
            REAL if layer.get("available") else UNAVAILABLE,
            "", layer.get("reason", ""),
        ),
        "layer_width_mm": field(
            width.get("layer_width_mm") if width.get("available") else None,
            REAL if width.get("available") else UNAVAILABLE,
            "mm", width.get("reason", ""),
        ),
        "layer_width_px": field(
            width.get("width_px") or width.get("layer_width_px"),
            REAL if (width.get("width_px") or width.get("layer_width_px")) is not None else UNAVAILABLE,
            "px", "",
        ),
        "confidence": field(
            machine.get("confidence") if machine.get("available") else None,
            REAL if machine.get("available") else UNAVAILABLE, "", "",
        ),
        "motion_score": field(
            machine.get("motion_score") if machine.get("available") else None,
            REAL if machine.get("available") else UNAVAILABLE, "", "",
        ),
        "note": width.get("note") or print_status.get("note") or machine.get("note"),
    }

    # --- process sensors: SIMULATED until a machine link exists ---
    reading = _sensors.read()
    sensor_provenance = MOCK if reading.source == "SIMULATED" else REAL
    material = {
        "material_flow_percent": field(reading.material_flow_percent, sensor_provenance, "%"),
        "nozzle_temp_c": field(reading.nozzle_temp_c, sensor_provenance, "C"),
        "bed_temp_c": field(reading.bed_temp_c, sensor_provenance, "C"),
        "machine_state": field(reading.machine_state, sensor_provenance),
        "source_note": (
            "These process values come from the built-in SIMULATOR, not the real "
            "machine. Connect a MachineInterface to replace them."
            if sensor_provenance == MOCK else ""
        ),
    }

    # --- machine telemetry: genuinely not connected ---
    link = _machine_link.read()
    link_reason = link.unavailable_reason
    machine_telemetry = {
        "pump_running": field(link.pump_running, UNAVAILABLE, "", link_reason),
        "pump_speed_rpm": field(link.pump_speed_rpm, UNAVAILABLE, "RPM", link_reason),
        "accelerator_running": field(link.accelerator_running, UNAVAILABLE, "", link_reason),
        "accelerator_speed_rpm": field(link.accelerator_speed_rpm, UNAVAILABLE, "RPM", link_reason),
        "print_speed_mm_s": field(link.print_speed_mm_s, UNAVAILABLE, "mm/s", link_reason),
        "pressure_bar": field(link.pressure_bar, UNAVAILABLE, "bar", link_reason),
    }

    snapshot = state.snapshot()
    alerts = [
        {"level": a["level"], "message": a["message"], "seconds_ago": a["seconds_ago"]}
        for a in snapshot.get("recent_alerts", [])
    ]

    # --- derived timing / geometry / material ---
    job = db.get_active_job()
    timing = metrics.production_metrics(job)
    elapsed_seconds = timing.get("elapsed_seconds", {}).get("value")
    geometry = metrics.geometry_metrics(job, snapshot.get("vision", {}))
    material_detail = metrics.material_metrics(job, elapsed_seconds)

    # --- robot / communication ---
    machine_link_connected = link.source != "NOT_CONNECTED"
    camera_live = camera_service.status == "live"
    robot_online = camera_live or machine_link_connected

    if print_status.get("available"):
        operation_field = metrics.field(
            "PRINTING" if print_status.get("printing") else "IDLE", REAL
        )
    else:
        operation_field = metrics.unavailable(
            print_status.get("reason", "No current machine status.")
        )

    robot = {
        "robot_online": metrics.field(robot_online, REAL),
        "robot_status": metrics.field("ONLINE" if robot_online else "OFFLINE", REAL),
        "machine_state": material["machine_state"],
        "current_operation": operation_field,
        "current_shift": metrics.current_shift(),
        "current_operator": metrics.current_operator(job),
        "communication": {
            "api": metrics.field("CONNECTED", REAL),
            "camera": metrics.field(
                camera_service.status.upper(),
                REAL if camera_live else UNAVAILABLE, "", camera_service.error or "",
            ),
            "machine_link": metrics.field(
                "CONNECTED" if machine_link_connected else "NOT CONNECTED",
                REAL if machine_link_connected else UNAVAILABLE,
                "", "" if machine_link_connected else link_reason,
            ),
        },
    }

    # --- problems ---
    warnings = [a for a in alerts if a["level"] == "warning"]
    criticals = [a for a in alerts if a["level"] == "critical"]
    machine_problems = [
        a for a in alerts
        if any(w in a["message"].lower() for w in ("flow", "temp", "pump", "accelerator", "machine"))
    ]
    camera_problems = []
    if camera_service.status != "live":
        camera_problems.append({
            "level": "warning",
            "message": camera_service.error or f"Camera status: {camera_service.status}",
            "seconds_ago": 0.0,
        })

    problems = {
        "warnings": warnings,
        "critical": criticals,
        "machine": machine_problems,
        "camera": camera_problems,
        "counts": {
            "warnings": len(warnings), "critical": len(criticals),
            "machine": len(machine_problems), "camera": len(camera_problems),
        },
    }

    return {
        "timestamp": time.time(),
        "camera": camera,
        "camera2": camera2,
        "people_count": site_camera_service.latest_people_count(),
        "production": {**production, **timing},
        "vision": vision,
        "geometry": geometry,
        "material": {**material, **material_detail},
        "machine_telemetry": machine_telemetry,
        "robot": robot,
        "problems": problems,
        "alerts": alerts,
        "calibrated": config.PIXELS_PER_MM > 0,
        "llm_mode": "gemini_agentic" if config.GEMINI_API_KEY else "missing_api_key",
        # Additive: existing keys above are unchanged. None until the camera
        # service has produced a measurement, or when metrology is disabled.
        "metrology": (
            camera_service.metrology.to_dict()
            if getattr(camera_service, "metrology", None) is not None
            else None
        ),
    }


@app.get("/api/metrology/announcement")
def metrology_announcement() -> Dict[str, Any]:
    """
    The most recent layer-width announcement worth speaking, if any.

    The dashboard polls this and hands the text to the EXISTING Voice
    Assistant's speak path, so there is one voice system rather than a second
    one talking over the first. `id` increments per announcement; the UI speaks
    only when it sees an id it has not spoken yet.

    The text is composed from the same MetrologyResult published above, so the
    spoken width and the width drawn on the camera can never disagree.
    """
    announcement = getattr(camera_service, "metrology_announcement", None)
    if announcement is None:
        return {"announcement": None, "enabled": config.METROLOGY_ENABLED}
    return {"announcement": announcement.to_dict(), "enabled": config.METROLOGY_ENABLED}


@app.get("/api/history")
def history() -> Dict[str, Any]:
    """Production over the last 10 minutes / 1 hour / 8 hours / 24 hours."""
    return metrics.production_history()


# --- assistant (shared by text AND voice) -------------------------------

class AskRequest(BaseModel):
    question: str
    speak: bool = False  # server-side TTS; the UI plays audio in the browser
    session_id: str = "default_terminal"
    line_id: str = "line_1"
    operator_id: Optional[str] = None
    stream: bool = False


class AskResponse(BaseModel):
    question: str
    answer: str
    tool: Optional[str] = None
    intent: Optional[str] = None
    facts: Optional[Dict[str, Any]] = None
    knowledge_sources: Optional[list] = None
    data_available: bool = True
    llm: str
    timestamp: float


def _stream_ask_response(request: AskRequest):
    question = (request.question or "").strip()
    full_chunks = []
    _t0 = time.time()
    try:
        for chunk in assistant.ask_stream(
            question,
            session_id=request.session_id,
            line_id=request.line_id,
            operator_id=request.operator_id,
        ):
            full_chunks.append(chunk)
            payload = json.dumps({"chunk": chunk, "done": False})
            yield f"data: {payload}\n\n"
    except Exception as exc:
        log.warning("Stream error: %s", exc)
        yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"
        return

    full_answer = "".join(full_chunks)
    log.info("[TIMING] Stream completed in %.3fs for query: %s", time.time() - _t0, question)

    # Server-side TTS run in a background thread so the HTTP response is never blocked
    if request.speak and full_answer:
        def _bg_speak_stream(text: str):
            try:
                from agent.voice import speak as server_speak
                server_speak(text)
            except Exception as exc:
                log.warning("Server-side TTS stream failed in background: %s", exc)

        threading.Thread(target=_bg_speak_stream, args=(full_answer,), daemon=True).start()

    final_payload = json.dumps({
        "chunk": "",
        "done": True,
        "answer": full_answer,
        "tool": Assistant.route(question),
        "llm": "gemini" if assistant.use_llm else "offline-router",
        "timestamp": time.time(),
    })
    yield f"data: {final_payload}\n\n"


@app.post("/api/assistant/query", response_model=AskResponse)
@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    THE shared query service. Both the text box and the voice pipeline call
    this, so a typed and a spoken question follow exactly the same path.
    Optimized for instantaneous time-to-first-token and non-blocking background tasks.
    """
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    if request.stream:
        return StreamingResponse(
            _stream_ask_response(request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    from agent import query_router

    log.info("[QUERY] Question: %s", question)
    _t_start = time.time()

    if assistant.use_llm:
        answer = assistant.ask(
            question,
            session_id=request.session_id,
            line_id=request.line_id,
            operator_id=request.operator_id,
        )
        tool_name = "gemini_agent"
        intent_val = "CONVERSATIONAL"
        facts_data = None
        knowledge_sources = None
        data_available = True
    else:
        # Fast path for simple greetings and casual conversational turns when offline
        pending = assistant._actions.get(request.session_id, request.line_id)
        casual_reply = Assistant._is_casual_or_greeting(question) if not pending else None

        if casual_reply:
            answer = casual_reply
            tool_name = "casual_greeting"
            intent_val = "WAKE_WORD"
            facts_data = None
            knowledge_sources = None
            data_available = True
        else:
            # Route deterministically when offline without LLM
            routed = query_router.route(
                question,
                session_id=request.session_id,
                use_agentic=False,
                context={
                    "session_id": request.session_id,
                    "line_id": request.line_id,
                    "operator_id": request.operator_id,
                },
            )
            answer = assistant.ask(
                question,
                session_id=request.session_id,
                line_id=request.line_id,
                operator_id=request.operator_id,
                routed=routed,
            )
            tool_name = Assistant.route(question)
            intent_val = routed.intent.value
            facts_data = routed.facts or None
            knowledge_sources = (
                [r["source"] for r in routed.knowledge["results"][:3]]
                if routed.knowledge and routed.knowledge.get("results")
                else None
            )
            data_available = routed.available

    log.info("[TIMING] Total ask response time: %.3fs", time.time() - _t_start)
    log.info("[ASSISTANT] Answer: %s", answer)

    # Server-side TTS executed asynchronously in a background thread to prevent HTTP response lag
    if request.speak:
        def _bg_speak(text_to_speak: str):
            try:
                from agent.voice import speak as server_speak
                server_speak(text_to_speak)
            except Exception as exc:
                log.warning("Server-side TTS failed in background: %s", exc)

        threading.Thread(target=_bg_speak, args=(answer,), daemon=True).start()

    return AskResponse(
        question=question,
        answer=answer,
        tool=tool_name,
        intent=intent_val,
        facts=facts_data,
        knowledge_sources=knowledge_sources,
        data_available=data_available,
        llm="gemini" if assistant.use_llm else "offline-router",
        timestamp=time.time(),
    )


@app.post("/api/assistant/stream")
def assistant_stream(request: AskRequest):
    """Dedicated endpoint to stream conversational chunks with minimal time-to-first-token."""
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")
    return StreamingResponse(
        _stream_ask_response(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/assistant/stream")
def assistant_stream_get(
    question: str,
    session_id: str = "default_terminal",
    line_id: str = "line_1",
    operator_id: Optional[str] = None,
    speak: bool = False,
):
    """GET endpoint for Server-Sent Events (SSE) streaming directly from browser EventSource."""
    req = AskRequest(
        question=question,
        session_id=session_id,
        line_id=line_id,
        operator_id=operator_id,
        speak=speak,
        stream=True,
    )
    return assistant_stream(req)


# --- speech to text (reuses the existing faster-whisper pipeline) -------

def _probe_audio(path: str):
    """
    Measures an audio file by DECODING it: (duration_seconds, peak, codec).

    Why not the container header: the WebM that a browser's MediaRecorder
    produces is written as a live stream, so its header carries no Duration
    element at all. `container.duration` is therefore None for every browser
    recording, however long it actually is -- and treating that as 0 rejected
    perfectly good audio as "too short". Counting decoded samples is the only
    reliable measure, and it yields the peak amplitude for free, which
    separates "no audio" from "silent audio".
    """
    import av
    import numpy as np

    total_samples = 0
    sample_rate = 0
    peak = 0.0
    codec = "unknown"

    with av.open(path) as container:
        streams = [st for st in container.streams if st.type == "audio"]
        if not streams:
            raise ValueError("no audio stream in container")
        stream = streams[0]
        codec = stream.codec_context.name or "unknown"
        sample_rate = stream.codec_context.sample_rate or getattr(stream, "rate", 0) or 0

        for frame in container.decode(audio=0):
            total_samples += frame.samples
            if not sample_rate:
                sample_rate = frame.sample_rate or 0
            array = frame.to_ndarray()
            if array.size:
                magnitude = float(np.abs(array).max())
                # Integer formats are full-scale; normalise so `peak` is 0..1
                # regardless of whether the decoder gave int16 or float32.
                if array.dtype.kind in "iu":
                    magnitude /= float(np.iinfo(array.dtype).max)
                peak = max(peak, magnitude)

    duration = (total_samples / float(sample_rate)) if sample_rate else 0.0
    return duration, peak, codec


@app.post("/api/stt")
async def speech_to_text(audio: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Transcribes uploaded audio with the project's existing faster-whisper model.

    Accepts whatever the browser's MediaRecorder produces (webm/opus, ogg, mp4)
    plus wav/mp3: the audio is written to a temp file and decoded by PyAV inside
    faster-whisper, so voice input works in every browser.
    """
    payload = await audio.read()
    # Full request trace: every one of these lines has, at some point, been the
    # one that identified a failure, so they all stay.
    log.info("[STT] --- request received ---")
    log.info("[STT] filename     : %s", audio.filename or "(none)")
    log.info("[STT] content type : %s", audio.content_type or "(none)")
    log.info("[STT] upload size  : %d bytes", len(payload))
    if not payload:
        log.error("[STT ERROR] Microphone capture failed -- 0 bytes received")
        raise HTTPException(status_code=400, detail="MICROPHONE CAPTURE FAILED: 0 audio bytes received")

    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(payload)
            temp_path = handle.name
        log.info("[STT] temp file    : %s", temp_path)
        log.info(
            "[STT] file on disk : exists=%s, %d bytes",
            os.path.exists(temp_path),
            os.path.getsize(temp_path) if os.path.exists(temp_path) else -1,
        )

        from agent.listener import transcribe
        # Decode-level validation: a container header with no audio frames
        # (the classic "recording stopped instantly" case) has ~0 duration and
        # must be reported as such rather than silently returning nothing.
        try:
            probe_duration, peak, codec = _probe_audio(temp_path)
            log.info(
                "[STT] Audio format: %s, duration: %.2fs, peak amplitude: %.4f",
                codec, probe_duration, peak,
            )

            if probe_duration < 0.3:
                log.error("[STT ERROR] Only %.2fs of audio decoded -- no usable speech", probe_duration)
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Only {probe_duration:.2f} seconds of audio arrived. "
                        "Click START LISTENING, wait for LISTENING, then speak your question."
                    ),
                )
            # Real duration but a flat waveform means the microphone delivered
            # digital silence -- a different problem with a different fix, so it
            # must not be reported as "too short".
            if peak <= 0.0005:
                log.error("[STT ERROR] %.2fs of audio but peak amplitude %.5f -- silence", probe_duration, peak)
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"The recording is {probe_duration:.1f} seconds of silence. "
                        "The microphone is open but Windows is sending no sound. "
                        "Run the microphone diagnostic in the assistant panel."
                    ),
                )
        except HTTPException:
            raise
        except Exception as exc:
            log.warning("[STT] Could not probe audio (%s) -- letting Whisper try anyway", exc)

        log.info("[STT] Transcribing with faster-whisper (%s)...", config.WHISPER_MODEL_SIZE)
        started = time.time()
        text, info = transcribe(temp_path, return_info=True)
        stt_seconds = time.time() - started
        log.info(
            "[TIMING] STT              : %.2fs  (language=%s)",
            stt_seconds, getattr(info, "language", "?"),
        )
        log.info("[STT] Transcript   : %r", text)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("[STT ERROR] Transcription raised %s", type(exc).__name__)
        # Most common cause by far: a container header with no audio frames,
        # i.e. the recording was stopped before any speech was captured.
        message = str(exc)
        if "Invalid data" in message or "moov atom" in message or "EBML" in message:
            log.error("[STT ERROR] Invalid/unsupported audio format -- likely an empty recording")
            raise HTTPException(
                status_code=422,
                detail=(
                    "The recording contained no usable audio (only a file header). "
                    "Click START LISTENING, speak for 2-5 seconds, then stop."
                ),
            )
        log.error("[STT ERROR] %s", message)
        raise HTTPException(status_code=400, detail=f"could not transcribe audio: {message}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    # Whisper hallucinates stock phrases when handed silence or noise ("You",
    # "Thank you.", "Thanks for watching!"). Those must never reach the query
    # router, or a silent recording yields a confident answer to a question the
    # operator never asked.
    _NOISE_TRANSCRIPTS = {
        "you", "thank you", "thanks for watching", "thanks for watching!",
        "bye", "bye.", "silence", ".", "...", "um", "uh",
    }
    normalised = text.lower().strip(" .!?,")
    valid_short = {"no", "na", "ha", "ok", "hi", "ya"}

    if not text or normalised in _NOISE_TRANSCRIPTS or (len(normalised) < 3 and normalised not in valid_short):
        log.error("[STT ERROR] Empty or noise-only transcript: %r", text)
        raise HTTPException(
            status_code=422,
            detail="I couldn't hear a clear question. Please try again.",
        )

    log.info("[STT] Transcript: %s", text)

    return {
        "text": text,
        "transcript": text,
        "engine": f"faster-whisper ({config.WHISPER_MODEL_SIZE})",
        "language": getattr(info, "language", None),
        "confidence": None,
        "bytes_received": len(payload),
    }


# --- text to speech -----------------------------------------------------

class SpeakRequest(BaseModel):
    text: str


@app.post("/api/tts")
def text_to_speech(request: SpeakRequest) -> Dict[str, Any]:
    """
    Speaks through the SERVER's speaker (agent/voice.py).

    Only useful when browser and backend are the same machine. The dashboard
    prefers /api/tts/audio, which returns the audio so the BROWSER plays it on
    the operator's own speakers.
    """
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    from agent.voice import TTSError, provider_name, speak as server_speak

    log.info("[TTS] Sending answer to speak(): %s", text)
    _t_tts = time.time()
    try:
        ok = server_speak(text, raise_on_error=True)
        log.info("[TIMING] TTS (spoken)     : %.2fs", time.time() - _t_tts)
    except TTSError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"spoken": bool(ok), "text": text, "engine": provider_name()}


@app.post("/api/tts/audio")
def text_to_speech_audio(request: SpeakRequest):
    """
    Returns synthesised speech AUDIO for the browser to play.

    This is the architecture the dashboard uses:
        Python TTS -> audio bytes -> HTTP -> <audio> -> operator's speaker

    It does not depend on the server having a usable output device, and avoids
    the browser's speechSynthesis, which fails silently in several situations.
    """
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    from agent.voice import TTSError, synthesize

    try:
        path, mime = synthesize(text)
    except TTSError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log.warning("TTS synthesis failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"TTS failed: {exc}")

    try:
        with open(path, "rb") as handle:
            audio_bytes = handle.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    return Response(
        content=audio_bytes,
        media_type=mime,
        headers={
            "Cache-Control": "no-store",
            "X-TTS-Engine": "elevenlabs" if config.ELEVENLABS_API_KEY else "pyttsx3",
        },
    )


@app.get("/api/tts/diagnostics")
def tts_diagnostics() -> Dict[str, Any]:
    """TTS health: provider, installed voices, volume. No secrets."""
    from agent.voice import diagnostics

    return diagnostics()


# --- camera (disabled in this version) ----------------------------------

@app.get("/api/camera/stream")
def camera_stream(request: Request):
    raise HTTPException(
        status_code=503,
        detail="Camera functionality is unavailable in this version.",
    )


@app.get("/api/camera/snapshot")
def camera_snapshot():
    raise HTTPException(
        status_code=503,
        detail="Camera functionality is unavailable in this version.",
    )


@app.get("/api/camera/status")
def camera_status() -> Dict[str, Any]:
    return {
        "status": "unavailable",
        "source": "none",
        "fps": 0.0,
        "resolution": [0, 0],
        "error": "Camera functionality is unavailable in this version.",
        "has_frame": False,
    }


# --- site camera 2 preview (disabled in this version) -------------------

@app.get("/api/camera2/stream")
async def camera2_stream(request: Request):
    raise HTTPException(
        status_code=503,
        detail="Camera functionality is unavailable in this version.",
    )


@app.get("/api/camera2/snapshot")
def camera2_snapshot():
    raise HTTPException(
        status_code=503,
        detail="Camera functionality is unavailable in this version.",
    )


@app.get("/api/camera2/detections")
def camera2_detections() -> Dict[str, Any]:
    """Returns unavailable status since camera processing is removed."""
    return ms.get_site_detections()


# --- Phase 2C: Shift & Production Data API -----------------------------

class CreateOperatorRequest(BaseModel):
    operator_id: str
    name: str
    employee_code: Optional[str] = None
    role: str = "operator"
    active: int = 1


class StartShiftRequest(BaseModel):
    operator_id: str
    shift_type: str  # "Morning", "Afternoon", "Night"
    shift_date: Optional[str] = None
    line_id: str = "line_1"
    target_walls: int = 0
    summary_notes: Optional[str] = None


class EndShiftRequest(BaseModel):
    status: str = "closed"  # "closed", "flagged", "abandoned"
    summary_notes: Optional[str] = None


class RecordProductionRequest(BaseModel):
    shift_id: int
    wall_number: int
    job_id: Optional[int] = None
    print_file: Optional[str] = None
    wall_height: float = 0.0
    segment_layers: int = 0
    layers_completed: int = 0
    status: str = "in_progress"  # in_progress, completed, aborted, defect
    is_continuation_of: Optional[int] = None
    notes: Optional[str] = None


class LogIssueRequest(BaseModel):
    shift_id: int
    issue_type: str
    description: str
    wall_number: Optional[int] = None
    cause: Optional[str] = None
    action_taken: Optional[str] = None
    result: Optional[str] = None
    severity: str = "warning"
    downtime_minutes: float = 0.0


class InitiateHandoverRequest(BaseModel):
    from_shift_id: int
    line_id: str = "line_1"
    completed_work: Optional[str] = None
    pending_work: Optional[str] = None
    open_issues: Optional[str] = None
    notes: Optional[str] = None


class CompleteHandoverRequest(BaseModel):
    to_shift_id: int
    notes: Optional[str] = None
    close_from_shift: bool = True


@app.get("/api/operators")
def get_operators(active_only: bool = True) -> Dict[str, Any]:
    """Lists operators, optionally filtering by active."""
    return {"operators": shift_store.list_operators(active_only=active_only)}


@app.post("/api/operators")
def create_operator(req: CreateOperatorRequest) -> Dict[str, Any]:
    """Creates a new operator."""
    try:
        op_id = shift_store.create_operator(
            operator_id=req.operator_id,
            name=req.name,
            employee_code=req.employee_code,
            role=req.role,
            active=req.active,
        )
        return {"success": True, "operator_id": op_id}
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail=f"Operator already exists or employee code conflict: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/shifts/active")
def get_active_shift(line_id: str = "line_1") -> Dict[str, Any]:
    """Gets the currently active shift on the specified line."""
    active = shift_store.get_active_shift_for_line(line_id=line_id)
    if not active:
        return {"active": False, "shift": None}
    op = shift_store.get_operator(active["operator_id"])
    return {
        "active": True,
        "shift": active,
        "operator": op,
    }


@app.post("/api/shifts/start")
def start_shift(req: StartShiftRequest) -> Dict[str, Any]:
    """Starts a new shift on a line. Blocks if line already has an active shift."""
    existing = shift_store.get_active_shift_for_line(req.line_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Line '{req.line_id}' already has an active shift (ID {existing['shift_id']}). Close it first.",
        )
    op = shift_store.get_operator(req.operator_id)
    if not op or not op.get("active"):
        raise HTTPException(status_code=400, detail=f"Operator '{req.operator_id}' not found or is inactive.")
    shift_date = req.shift_date or datetime.date.today().isoformat()
    try:
        shift_id = shift_store.start_shift(
            operator_id=req.operator_id,
            shift_date=shift_date,
            shift_type=req.shift_type,
            line_id=req.line_id,
            target_walls=req.target_walls,
            summary_notes=req.summary_notes,
        )
        return {
            "success": True,
            "shift_id": shift_id,
            "line_id": req.line_id,
            "operator_id": req.operator_id,
            "shift_type": req.shift_type,
            "shift_date": shift_date,
        }
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail=f"Unique active shift constraint violated: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/shifts/{shift_id}/end")
def end_shift(shift_id: int, req: EndShiftRequest) -> Dict[str, Any]:
    """Ends a shift and updates its status."""
    shift = shift_store.get_shift(shift_id)
    if not shift:
        raise HTTPException(status_code=404, detail=f"Shift {shift_id} not found.")
    try:
        ok = shift_store.end_shift(
            shift_id=shift_id,
            status=req.status,
            summary_notes=req.summary_notes,
        )
        return {"success": ok, "shift_id": shift_id, "status": req.status}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/shifts")
def list_shifts(
    shift_date: Optional[str] = None,
    shift_type: Optional[str] = None,
    operator_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Lists shifts matching criteria."""
    shifts = shift_store.list_shifts(
        shift_date=shift_date,
        shift_type=shift_type,
        operator_id=operator_id,
        status=status,
        limit=limit,
    )
    return {"shifts": shifts, "count": len(shifts)}


@app.post("/api/production")
def record_production(req: RecordProductionRequest) -> Dict[str, Any]:
    """Records production progress for a wall within a shift."""
    shift = shift_store.get_shift(req.shift_id)
    if not shift:
        raise HTTPException(status_code=404, detail=f"Shift {req.shift_id} not found.")
    try:
        prod_id = shift_store.record_production(
            shift_id=req.shift_id,
            wall_number=req.wall_number,
            job_id=req.job_id,
            print_file=req.print_file,
            wall_height=req.wall_height,
            segment_layers=req.segment_layers,
            layers_completed=req.layers_completed,
            status=req.status,
            is_continuation_of=req.is_continuation_of,
            notes=req.notes,
        )
        return {"success": True, "production_id": prod_id}
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail=f"Unique constraint (shift_id, wall_number) violated: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/production/shift/{shift_id}")
def get_shift_production(shift_id: int) -> Dict[str, Any]:
    """Lists all production records for a shift."""
    records = shift_store.get_shift_production(shift_id)
    return {"shift_id": shift_id, "production_records": records, "count": len(records)}


@app.post("/api/issues")
def log_issue(req: LogIssueRequest) -> Dict[str, Any]:
    """Logs an issue against a shift."""
    shift = shift_store.get_shift(req.shift_id)
    if not shift:
        raise HTTPException(status_code=404, detail=f"Shift {req.shift_id} not found.")
    try:
        issue_id = shift_store.log_issue(
            shift_id=req.shift_id,
            issue_type=req.issue_type,
            description=req.description,
            wall_number=req.wall_number,
            cause=req.cause,
            action_taken=req.action_taken,
            result=req.result,
            severity=req.severity,
            downtime_minutes=req.downtime_minutes,
        )
        return {"success": True, "issue_id": issue_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/issues/shift/{shift_id}")
def get_shift_issues(shift_id: int) -> Dict[str, Any]:
    """Lists all issues reported for a shift."""
    issues = shift_store.list_shift_issues(shift_id=shift_id)
    return {"shift_id": shift_id, "issues": issues, "count": len(issues)}


@app.post("/api/handovers/initiate")
def initiate_handover(req: InitiateHandoverRequest) -> Dict[str, Any]:
    """Initiates an outgoing shift handover."""
    from_shift = shift_store.get_shift(req.from_shift_id)
    if not from_shift:
        raise HTTPException(status_code=404, detail=f"From-shift {req.from_shift_id} not found.")
    try:
        handover_id = shift_store.create_handover(
            from_shift_id=req.from_shift_id,
            line_id=req.line_id,
            completed_work=req.completed_work,
            pending_work=req.pending_work,
            open_issues=req.open_issues,
            notes=req.notes,
        )
        return {"success": True, "handover_id": handover_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/handovers/{handover_id}/complete")
def complete_handover(handover_id: int, req: CompleteHandoverRequest) -> Dict[str, Any]:
    """Completes an initiated handover by linking to the incoming shift."""
    h = shift_store.get_handover(handover_id)
    if not h:
        raise HTTPException(status_code=404, detail=f"Handover {handover_id} not found.")
    to_shift = shift_store.get_shift(req.to_shift_id)
    if not to_shift:
        raise HTTPException(status_code=404, detail=f"To-shift {req.to_shift_id} not found.")
    try:
        ok = shift_store.complete_handover(
            handover_id=handover_id,
            to_shift_id=req.to_shift_id,
            notes=req.notes,
            close_from_shift=req.close_from_shift,
        )
        return {"success": ok, "handover_id": handover_id, "to_shift_id": req.to_shift_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
