"""
Adapter between the Kelvin6K database / machine state and the voice assistant.

This provides machine facts sourcing them from the SQLite database (the
print_jobs and machine_state tables).

Every function returns a dict with an explicit `available` flag. Nothing is ever
fabricated: unavailable means unavailable, and low confidence is flagged rather
than smoothed over.
"""

import time
from typing import Any, Dict, Optional

from agent import config, db
from agent.logging_setup import get_logger

log = get_logger("MACHINE")

_live_state = None


def bind_live_state(state) -> None:
    """Safe no-op kept for backwards compatibility."""
    global _live_state
    _live_state = state


_site_camera_service = None


def bind_site_camera_service(service) -> None:
    """Safe no-op kept for backwards compatibility."""
    global _site_camera_service
    _site_camera_service = service


def _fresh(timestamp: Optional[float]) -> bool:
    if timestamp is None:
        return False
    return (time.time() - timestamp) <= config.MACHINE_STATE_MAX_AGE_SECONDS


def _snapshot() -> Optional[Dict[str, Any]]:
    """Newest trustworthy machine reading from the database."""
    row = db.get_latest_machine_state()
    if row and _fresh(row.get("timestamp")):
        return {
            "printing": bool(row.get("printing")),
            "status": row.get("status"),
            "layer_detected": row.get("status") in ("LAYER_DETECTED", "PRINTING"),
            "current_layer": row.get("current_layer"),
            "layer_width": row.get("layer_width_mm"),
            "layer_width_px": row.get("layer_width_px"),
            "width_unit": "mm",
            "confidence": row.get("confidence") or 0.0,
            "motion_score": row.get("motion_score"),
            "camera_connected": bool(row.get("camera_connected")),
            "completed_length_mm": row.get("completed_length_mm"),
            "timestamp": row.get("timestamp"),
            "source": "database",
        }
    return None


def _unavailable(reason: str = "I don't have a current machine status.") -> Dict[str, Any]:
    return {"available": False, "reason": reason}


def _confidence_note(confidence: Optional[float]) -> Optional[str]:
    if confidence is None:
        return None
    if confidence < config.MIN_REPORT_CONFIDENCE:
        return (
            f"The current measurement is uncertain (confidence "
            f"{confidence * 100:.0f} percent)."
        )
    return None


# --- public API used as LLM tools / assistant methods --------------------

def get_current_print_status() -> Dict[str, Any]:
    """Whether printing is currently running, per the latest machine state."""
    snap = _snapshot()
    if snap is None:
        return _unavailable()
    return {
        "available": True,
        "printing": snap.get("printing"),
        "status": snap.get("status"),
        "camera_connected": snap.get("camera_connected"),
        "source": snap.get("source"),
        "age_seconds": round(time.time() - (snap.get("timestamp") or 0), 1),
        "note": _confidence_note(snap.get("confidence")),
    }


def get_current_layer() -> Dict[str, Any]:
    """The current layer number from the active job or latest state."""
    job = db.get_active_job()
    if job and job.get("current_layer") is not None:
        return {
            "available": True,
            "current_layer": job["current_layer"],
            "total_layers": job.get("total_layers"),
            "source": "print_job",
        }

    snap = _snapshot()
    if snap and snap.get("current_layer") is not None:
        return {"available": True, "current_layer": snap["current_layer"], "source": snap["source"]}

    return _unavailable(
        "No current layer number is being tracked. It needs an active print job "
        "or a machine link."
    )


def get_current_layer_width() -> Dict[str, Any]:
    """Measured width of the layer being printed, in millimetres when calibrated."""
    snap = _snapshot()
    if snap is None:
        return _unavailable()

    width_mm = snap.get("layer_width")
    width_px = snap.get("layer_width_px")
    confidence = snap.get("confidence")

    if width_mm is None and width_px is not None:
        return {
            "available": False,
            "width_px": width_px,
            "confidence": confidence,
            "reason": (
                "Layer width calibration is not configured, so only a pixel "
                "measurement exists. Set PIXELS_PER_MM to get millimetres."
            ),
        }
    if width_mm is None:
        return _unavailable("No layer width has been measured.")

    return {
        "available": True,
        "layer_width_mm": round(float(width_mm), 2),
        "layer_width_px": width_px,
        "confidence": confidence,
        "source": snap.get("source"),
        "note": _confidence_note(confidence),
    }


def get_print_progress() -> Dict[str, Any]:
    """How much of the wall is printed and how much remains."""
    job = db.get_active_job()
    if job is None:
        return _unavailable(
            "No active print job is being tracked, so progress cannot be reported. "
            "Start one with start_print_job()."
        )

    total = float(job.get("total_length_mm") or 0.0)
    completed = float(job.get("completed_length_mm") or 0.0)
    remaining = max(total - completed, 0.0)
    percentage = round((completed / total) * 100.0, 1) if total > 0 else None

    return {
        "available": True,
        "job_name": job.get("job_name"),
        "total_mm": round(total, 1),
        "completed_mm": round(completed, 1),
        "remaining_mm": round(remaining, 1),
        "percentage": percentage,
        "current_layer": job.get("current_layer"),
        "total_layers": job.get("total_layers"),
        "status": job.get("status"),
    }


def get_machine_status() -> Dict[str, Any]:
    """Machine and process status from the database."""
    snap = _snapshot()
    if snap is None:
        return _unavailable("No active machine state reading available.")

    return {
        "available": True,
        "status": snap.get("status") or "idle",
        "printing": snap.get("printing") or 0,
        "confidence": snap.get("confidence"),
        "motion_score": snap.get("motion_score"),
        "source": snap.get("source") or "system",
        "note": "",
    }


def get_machine_summary() -> Dict[str, Any]:
    """Summary of print status, layer, width, and progress."""
    return {
        "print_status": get_current_print_status(),
        "layer": get_current_layer(),
        "layer_width": get_current_layer_width(),
        "progress": get_print_progress(),
        "machine": get_machine_status(),
    }


# --- job management ------------------------------------------------------

def start_print_job(
    job_name: str,
    total_length_mm: float,
    material: Optional[str] = None,
    total_layers: Optional[int] = None,
) -> Dict[str, Any]:
    """Starts tracking a print job so progress questions can be answered."""
    db.init_db()
    job_id = db.create_print_job(job_name, total_length_mm, material, total_layers)
    db.record_event("job_started", f"Started job '{job_name}' ({total_length_mm} mm)", job_id)
    log.info("Started print job %s (id=%s, %.1f mm)", job_name, job_id, total_length_mm)
    return {"available": True, "job_id": job_id, "job_name": job_name}


def update_progress(
    completed_length_mm: Optional[float] = None,
    current_layer: Optional[int] = None,
) -> Dict[str, Any]:
    """Updates the active job's progress. remaining is recomputed automatically."""
    job = db.get_active_job()
    if job is None:
        return _unavailable("No active print job to update.")
    db.update_job_progress(job["id"], completed_length_mm, current_layer)
    return get_print_progress()


def get_site_people_count(force_refresh: bool = False) -> Dict[str, Any]:
    return {
        "available": False,
        "people_count": 0,
        "confidence": 0.0,
        "provenance": "UNAVAILABLE",
        "reason": "Camera functionality is unavailable in this version.",
        "note": "",
    }


def get_site_detections(force_refresh: bool = False) -> Dict[str, Any]:
    return {
        "available": False,
        "provenance": "UNAVAILABLE",
        "reason": "Camera functionality is unavailable in this version.",
        "entities": [],
        "people_count": 0,
        "supported_classes": [],
    }
