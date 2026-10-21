"""
HTTP API for the Kelvin6K Voice Assistant.

Endpoints:
    /api/health           -> Server & Voice Agent health status
    /api/ask              -> Assistant query (Voice / Text)
    /api/assistant/query  -> Alias for /api/ask
    /api/assistant/stream -> SSE streaming assistant response
    /api/stt              -> faster-whisper speech-to-text
    /api/tts*             -> ElevenLabs / pyttsx3 speech synthesis
    /api/history          -> Production & query event history
    /api/operators        -> Operator management
    /api/shifts*          -> Shift & attendance tracking
    /api/production*      -> Wall production records
    /api/issues*          -> Shop-floor issue logging & resolution
    /api/handovers*       -> Shift handover orchestration
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

from agent import config, db, listener, metrics, shift_store
from agent.assistant import Assistant
from agent.logging_setup import configure_logging, get_logger
from agent.rag.retriever import index_summary

configure_logging()
log = get_logger("API")

app = FastAPI(title="Kelvin6K Voice Assistant API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dashboard; tighten for production deployment
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-TTS-Engine"],
)

# Shared singleton voice assistant instance
assistant = Assistant()


@app.on_event("startup")
def _startup() -> None:
    db.init_db()

    # Preload whisper in background so first query has zero model load latency
    threading.Thread(target=listener.preload, name="whisper-preload", daemon=True).start()

    def _warm_answer_engine() -> None:
        try:
            assistant.ask("what is the machine status")
        except Exception:
            pass

    threading.Thread(target=_warm_answer_engine, name="answer-warmup", daemon=True).start()
    log.info("Kelvin6K Voice Assistant API ready; speech-to-text loading in background")


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
        "camera_source": "none",
        "camera_status": "disabled",
        "llm": "gemini" if config.GEMINI_API_KEY else "offline-router",
        "llm_model": config.GEMINI_MODEL if config.GEMINI_API_KEY else None,
        "stt": f"faster-whisper ({config.WHISPER_MODEL_SIZE}) - {listener.get_status()}",
        "tts": provider_name(),
        "rag_chunks": rag["chunks"],
        "rag_mode": "semantic" if rag["semantic"] else "lexical",
        "calibrated": False,
        "read_only": True,
    }


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
    except (sqlite3.IntegrityError, ValueError) as exc:
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
    except (sqlite3.IntegrityError, ValueError) as exc:
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
    except (sqlite3.IntegrityError, ValueError) as exc:
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
    except (sqlite3.IntegrityError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
