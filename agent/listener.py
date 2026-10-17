import os
import sys
import threading
import time
import re
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from agent import config

_SAMPLE_RATE = 16000
_model = None
_model_lock = threading.Lock()
_load_status = "loading"

INITIAL_PROMPT = (
    "Vaniyambadi, Ambur, Tirupattur, Gudiyatham, Chennai, Coimbatore, Krishnagiri, "
    "Thiruvananthapuram, Bengaluru, Kanchipuram, Kelvin6K, Faster-Whisper, CTranslate2, "
    "PostgreSQL, Gemini, FastAPI, concrete extrusion, layer width, wall continuity, "
    "under-extrusion, delamination. "
    "उत्तरदायित्व, आत्मनिर्भरता, प्रौद्योगिकी, परिस्थितियों, अंतरराष्ट्रीय, कार्यप्रणाली, "
    "दूरसंचार, अभियांत्रिकी, कृत्रिम बुद्धिमत्ता, मशीनी अधिगम. "
    "Print chal raha hai kya? Current layer width kitna hai? Vaniyambadi se material aa raha hai."
)

COMMON_HALLUCINATIONS = [
    r"subtitles by",
    r"amara\.org",
    r"thank you for watching",
    r"thanks for watching",
    r"like and subscribe",
    r"subscribe to my channel",
    r"watch next video"
]

def _best_device():
    """
    Returns (device, compute_type) for this machine according to config or auto-detection.
    """
    configured_device = getattr(config, "WHISPER_DEVICE", "cpu")
    configured_compute = getattr(config, "WHISPER_COMPUTE_TYPE", "int8")

    if configured_device != "auto" and configured_device:
        return configured_device, configured_compute

    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _get_model() -> WhisperModel:
    """The shared Faster-Whisper model loaded once and reused for all requests."""
    global _model, _load_status
    if _model is None:
        with _model_lock:
            if _model is None:
                device, compute = _best_device()
                started = time.time()
                print(
                    f"Loading speech-to-text model ({config.WHISPER_MODEL_SIZE}, "
                    f"{device}/{compute})...", file=sys.stderr,
                )
                try:
                    _model = WhisperModel(
                        config.WHISPER_MODEL_SIZE,
                        device=device,
                        compute_type=compute,
                        cpu_threads=os.cpu_count() or 4,
                    )
                    _load_status = "OK"
                    print(f"Speech-to-text model ready in {time.time() - started:.1f}s",
                          file=sys.stderr)
                except Exception as exc:
                    _load_status = "unavailable"
                    print(f"[STT ERROR] Failed to load model: {exc}", file=sys.stderr)
                    raise
    return _model


def is_loaded() -> bool:
    """Check if the Faster-Whisper model is already initialized."""
    return _model is not None


def get_status() -> str:
    """Returns the current loading status ('OK', 'loading', or 'unavailable')."""
    if _model is not None:
        return "OK"
    return _load_status


def preload() -> None:
    """Loads the model ahead of time off the request path."""
    try:
        _get_model()
    except Exception as exc:
        print(f"[STT] Preload failed ({type(exc).__name__}); "
              f"the model will load on first use.", file=sys.stderr)


def decode_options(language: str = None) -> dict:
    """Decoding settings shared by every caller tuned for maximum speed and fallback reliability."""
    if language is None:
        language = None if config.WHISPER_LANGUAGE in ("", "auto") else config.WHISPER_LANGUAGE
    return {
        "language": language,
        "beam_size": 1,
        "best_of": 1,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "initial_prompt": INITIAL_PROMPT,
    }


def clean_transcript(text: str) -> str:
    """Filter out common Whisper silence artifacts and repetition loops."""
    if not text:
        return ""
    cleaned = text
    for pattern in COMMON_HALLUCINATIONS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    
    words = cleaned.split()
    if len(words) > 8:
        for n in range(1, 4):
            pattern = r'(\b' + r'\s+'.join([r'\w+'] * n) + r'\b)(\s+\1){3,}'
            cleaned = re.sub(pattern, r'\1', cleaned, flags=re.IGNORECASE)
            
    return cleaned.strip()


def record_question() -> np.ndarray:
    """
    Captures audio from default microphone using VAD silence detection.
    Stops dynamically when the user finishes speaking or reaches MAX_RECORD_SECONDS.
    """
    print(f"Listening for speech (up to {config.MAX_RECORD_SECONDS:.0f}s)... speak now.", file=sys.stderr)
    recorded_chunks = []
    silence_threshold = 0.005
    speech_detected = False
    silence_start_time = None
    max_duration = config.MAX_RECORD_SECONDS
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[MIC STATUS] {status}", file=sys.stderr)
        recorded_chunks.append(indata.copy())

    try:
        with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
            while time.time() - start_time < max_duration:
                time.sleep(0.05)
                if not recorded_chunks:
                    continue
                
                recent_audio = np.concatenate(recorded_chunks[-5:], axis=0) if len(recorded_chunks) >= 5 else np.concatenate(recorded_chunks, axis=0)
                rms = float(np.sqrt(np.mean(recent_audio ** 2)))
                
                if not speech_detected:
                    if rms > silence_threshold * 1.5:
                        speech_detected = True
                        print("[STT] Speech started...", file=sys.stderr)
                else:
                    if rms < silence_threshold:
                        if silence_start_time is None:
                            silence_start_time = time.time()
                        elif time.time() - silence_start_time >= 1.5:
                            print("[STT] End of speech detected.", file=sys.stderr)
                            break
                    else:
                        silence_start_time = None
    except Exception as exc:
        print(f"[MIC ERROR] {exc}", file=sys.stderr)
        return np.array([], dtype="float32")

    if not recorded_chunks:
        return np.array([], dtype="float32")
    return np.concatenate(recorded_chunks, axis=0).flatten()


def transcribe(audio, return_info: bool = False):
    """Converts audio into text using Faster-Whisper."""
    if audio is None:
        return ("", None) if return_info else ""
    if isinstance(audio, np.ndarray) and audio.size == 0:
        return ("", None) if return_info else ""
    if isinstance(audio, str):
        if not os.path.exists(audio) or os.path.getsize(audio) == 0:
            return ("", None) if return_info else ""
    try:
        if len(audio) == 0:
            return ("", None) if return_info else ""
    except TypeError:
        pass

    model = _get_model()
    opts = decode_options()

    try:
        segments, info = model.transcribe(audio, **opts)
    except (ValueError, IndexError):
        return ("", None) if return_info else ""
    except Exception as exc:
        print(f"[STT] Transcription error: {exc}", file=sys.stderr)
        return ("", None) if return_info else ""

    if getattr(info, "duration_after_vad", None) is not None and info.duration_after_vad <= 0:
        return ("", info) if return_info else ""

    detected_lang = getattr(info, "language", None)
    if config.WHISPER_LANGUAGE in ("", "auto") and detected_lang not in ("en", "hi"):
        if detected_lang == "ur":
            fallback_lang = "hi"
        else:
            probs_raw = getattr(info, "all_language_probs", None)
            probs = {}
            if isinstance(probs_raw, dict):
                probs = probs_raw
            elif isinstance(probs_raw, (list, tuple)):
                try:
                    probs = dict(probs_raw)
                except (ValueError, TypeError):
                    probs = {}

            hi_prob = probs.get("hi", 0.0)
            en_prob = probs.get("en", 0.0)
            fallback_lang = "hi" if hi_prob >= en_prob else "en"

        fallback_opts = decode_options(language=fallback_lang)
        try:
            fallback_segments, fallback_info = model.transcribe(audio, **fallback_opts)
            raw_text = " ".join(segment.text.strip() for segment in fallback_segments).strip()
            clean_text = clean_transcript(raw_text)
            return (clean_text, fallback_info) if return_info else clean_text
        except (ValueError, IndexError):
            return ("", info) if return_info else ""
        except Exception as exc:
            print(f"[STT] Fallback transcription error: {exc}", file=sys.stderr)
            return ("", info) if return_info else ""

    try:
        raw_text = " ".join(segment.text.strip() for segment in segments).strip()
        clean_text = clean_transcript(raw_text)
        return (clean_text, info) if return_info else clean_text
    except (ValueError, IndexError):
        return ("", info) if return_info else ""
    except Exception as exc:
        print(f"[STT] Segment processing error: {exc}", file=sys.stderr)
        return ("", info) if return_info else ""


def listen_for_question() -> str:
    """Records and transcribes one question using Faster-Whisper. Returns '' if silence."""
    return transcribe(record_question())
