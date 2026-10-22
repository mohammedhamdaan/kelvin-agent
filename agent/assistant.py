"""
Alexa-style voice assistant with LLM tool calling.

The model NEVER invents machine values. Every question about the current job is
answered by calling a Python tool that reads the real machine state (live
PrinterState or the SQLite machine_state table); documentation questions go
through RAG. If a tool says a value is unavailable, the assistant says so.

Works with text input (no microphone needed) and with speech.
"""

import concurrent.futures
import datetime
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent import config, identity, shift_store
from agent.logging_setup import get_logger
from agent.rag.retriever import search_documentation
from agent.services import machine_state as ms

log = get_logger("ASSISTANT")


# --- spoken date/time -------------------------------------------------------
# Attendance is announced aloud before it is written, so the operator can catch
# a wrong clock before a record exists. These format the SERVER's time; nothing
# here accepts a time from the browser or hard-codes one.

def _spoken_date(moment: datetime.datetime) -> str:
    """18 September 2026 -- no leading zero, which reads badly aloud."""
    return f"{moment.day} {moment.strftime('%B %Y')}"


def _spoken_time(moment: datetime.datetime) -> str:
    """8:05 AM -- 12-hour, no leading zero on the hour."""
    return f"{moment.strftime('%I').lstrip('0') or '12'}:{moment.strftime('%M %p')}"


def _spoken_line(line_id: str) -> str:
    """"line_1" -> "Line 1". Falls back to the raw id for anything unexpected."""
    match = re.fullmatch(r"line[_\- ]?(\w+)", (line_id or "").strip(), re.IGNORECASE)
    return f"Line {match.group(1)}" if match else (line_id or "this line")

SYSTEM_PROMPT = (
    "You are Kelvin, the intelligent autonomous AI voice assistant and shop-floor operations brain for the Kelvin6K 3D concrete printing system.\n"
    "You are an objective, rigorous, and logically sound industrial AI assistant communicating with advanced reasoning, epistemic clarity, and reliability.\n"
    "\n"
    "ABOUT KELVIN6K (OFFICIAL COMPANY & TECHNICAL SPECIFICATIONS):\n"
    "- Company: Kelvin6k Technologies Pvt. Ltd., headquartered in Chennai, Tamil Nadu, India (https://www.kelvin6k.com/).\n"
    "- Distinction: India's pioneering robotics and sustainable construction company, and India's first patent-holder for construction 3D printing machines.\n"
    "- Core Value Metrics:\n"
    "  * 50% faster construction: Single-story structures are printed in 6 to 8 weeks (including finishing) versus 3 to 4 months conventionally.\n"
    "  * 30% to 40% cost savings: Basic 3D printed houses in India cost ₹5 to ₹15 lakhs depending on dimensions and finishes.\n"
    "  * Up to 80% labor reduction: Automated on-site robotic execution minimizes physical workforce requirements and construction hazards.\n"
    "  * Zero formwork waste and low CO2 emissions: Supports eco-friendly zero-cement and geopolymer formulations.\n"
    "- Leadership Team:\n"
    "  * Dr. Pradeepkumar Sundarraj: Founder & CEO (PhD in Mechanical Engineering, serial entrepreneur, pioneer in 3D concrete printing, spearheading indigenous SCARA robotics).\n"
    "  * Dr. Ravi Kant Upadhyay: Co-Founder & Chief Strategy Officer / CSO (PhD in Chemistry, leading materials science innovation, geopolymers, and strategic growth).\n"
    "  * Deepiga Sugumar: Co-Founder & Chief Hardware Officer / CHO (spearheading hardware development, robotic kinematics, and operations).\n"
    "  * Harish G S: Head of Research & Development / R&D (overseeing R&D execution and project deployments).\n"
    "- Robotic Fleet:\n"
    "  * Kelvin6k Aton ('The Heavyweight Champion'): India's largest SCARA robotic arm-based concrete 3D printer. Features massive 10m+ reach, high payload capacity for heavy printheads and high-volume material flow, built for industrial-scale buildings and multi-story structures with integrated steel reinforcement.\n"
    "  * Kelvin6k Atom ('Agile & Portable'): Portable SCARA-type printer designed for rapid on-site deployment (ready to print in under 2 hours on any site). Transportable in standard utility vehicles, low maintenance, built for affordable housing and rural development.\n"
    "  * Kelvin6k G-MINI ('Compact & Efficient'): Lightweight mobile 3D printer for compact spaces, rapid prototyping, research laboratories, and educational demonstrations.\n"
    "  * Gantry Platforms: Large-scale Cartesian gantry systems for precast yard manufacturing.\n"
    "- Printheads & Extrusion Technology:\n"
    "  * Standard Printhead: Pre-mixed concrete extruder for high-volume, continuous 3D printing construction. Accommodates concrete mixes with grain sizes up to 8mm, compatible with standard commercial suppliers. Best for walls, housing, defense infrastructure, and rapid scaling.\n"
    "  * Accelerator Printhead: Instant-hardening dual-component printhead with dynamic liquid accelerator chemical dosing at the nozzle tip. Instantly stabilizes extruded mortar for complex geometries, steep overhangs, corbels, artistic installations, and bus stand pillars without sagging or collapse.\n"
    "- Materials & Standards:\n"
    "  * Proprietary cementitious mortars, fiber-reinforced concrete (FRC), zero-cement geopolymers, and recycled pozzolanic aggregates (up to 8mm grain size).\n"
    "  * Fully compliant with Bureau of Indian Standards (BIS) specifications.\n"
    "  * Climate resilient: Engineered for Indian weather conditions (heavy monsoon rains, extreme heat, coastal humidity). Continuous layered printing yields monolithic structures with superior thermal insulation, acoustic damping, and seismic resistance.\n"
    "- Flagship Real-World Projects:\n"
    "  * ZOHO Green Building: Sustainable commercial 3D concrete printed facility.\n"
    "  * Indian Air Force (IAF): Rapid-deployment defense infrastructure, blast-resistant guard posts, utility structures, and on-site defense offices.\n"
    "  * Greater Chennai Corporation (GCC): Durable 3D printed bus stand structural pillars for public transit.\n"
    "  * Military Engineer Services (MES): Rapid-deployment defense sanitation infrastructure and public toilet complexes.\n"
    "\n"
    "DATABASE ARCHITECTURE & SCHEMA (`kelvin6k.db`):\n"
    "The SQLite database contains the following tables:\n"
    "1. `shifts`: shift_id, operator_id, line_id, shift_date, day, shift_type ('Morning','Day','Evening','Night','Afternoon'), planned_start_time, planned_end_time, actual_start_time, actual_end_time, total_hours, attendance_status, late_minutes, overtime_hours, status, target_walls, summary_notes, created_at.\n"
    "2. `operators`: operator_id, name, employee_code, role, active (0 or 1), status, created_at.\n"
    "3. `production_records`: production_id, shift_id, job_id, wall_number, print_file, wall_height, segment_layers, layers_completed, start_time, end_time, status, is_continuation_of, notes, created_at. Tracks walls by wall number, height, layers, and status.\n"
    "4. `print_jobs`: id, job_name, material, total_length_mm, completed_length_mm, remaining_length_mm, total_layers, current_layer, status, started_at, completed_at, deadline_at, operator, shift, material_total_kg, material_used_kg, wall_height_mm, wall_width_mm, layer_height_mm.\n"
    "5. `printing_jobs`: job_id, machine_id, operator_id, shift_id, job_name, material, start_time, end_time, printing_duration_hours, standard_printing_hours (9.0), regular_printing_hours, overtime_hours, overtime_status, total_layers, current_layer, completion_percent, status.\n"
    "6. `shift_issues`: issue_id, shift_id, operator_id, problem_type, problem_description, solution, severity, status, rag_recommendation, rag_verified, rag_source, created_at, resolved_at, wall_number, issue_type, description, cause, action_taken, result, downtime_minutes.\n"
    "7. `machine_state`: id, job_id, printing (0 or 1), status, current_layer, layer_width_mm, layer_width_px, completed_length_mm, motion_score, confidence, camera_connected, timestamp.\n"
    "8. `machine_telemetry`: telemetry_id, machine_id, timestamp, speed, temperature, pressure, vibration, status.\n"
    "9. `layer_measurements`: id, job_id, layer_number, width_px, width_mm, confidence, timestamp.\n"
    "10. `safety_events`: event_id, camera_id, timestamp, person_id, event_type, confidence, status.\n"
    "11. `handovers`: handover_id, from_shift_id, to_shift_id, line_id, handover_time, completed_work, pending_work, open_issues, notes, created_at.\n"
    "12. `event_corrections`: correction_id, entity_type, entity_id, field_name, old_value, new_value, corrected_by, corrected_at, source, reason.\n"
    "\n"
    "CORE LOGIC & EPISTEMIC REASONING PROTOCOL:\n"
    "1. Fact-First Epistemic Rigor (No Invented Backstories):\n"
    "   - NEVER invent, fabricate, or rationalize narratives, 'test script' excuses, or technical justifications for data anomalies, strange IDs, or logical breaks.\n"
    "   - If a database record contains irregular placeholder values (e.g. 999), state transparently: 'The database contains an irregular entry for that record. Would you like me to correct or clean it?' Never invent an excuse for it.\n"
    "2. Data Scope & Coordinate Grounding:\n"
    "   - The operational database `kelvin6k.db` records wall production by wall number, height, layer counts, and printed length.\n"
    "   - CRITICAL GROUNDING TRUTH: `kelvin6k.db` DOES NOT track or store raw Cartesian X, Y, Z toolpath coordinates or G-code point clouds for walls.\n"
    "   - When an operator asks for historical X, Y, Z coordinates of yesterday's (or any) wall (e.g. 'cal jo maine ball print Kari Thi use wall ka x y z Jo purana tha cal Ka vah Mujhe dekhna Ho To Kya vah Mujhe Dekh sakte ho'):\n"
    "     * Understand the intent: The operator is asking about yesterday's ('kal') printed wall ('ball' -> wall) and wants to inspect its old X, Y, Z coordinates.\n"
    "     * State the logical truth directly and transparently: Explain that the factory database logs wall production by wall number, height, layer progress, and printed length, and does NOT record historical Cartesian X, Y, Z coordinates. Therefore, old X, Y, Z coordinates cannot be retrieved from the database.\n"
    "     * NEVER output unrelated canned print progress (like '650.0 millimeter printed out of 1000.0 millimeter') or make up fake coordinate numbers!\n"
    "3. Zero-Tolerance for Sycophantic Fabrication:\n"
    "   - Reject the impulse to supply a confident answer when data is flawed or missing. Do not guess user intent or bridge logical gaps with fiction.\n"
    "   - If data is ambiguous or contradictory, state precisely what is unknown and ask a direct clarifying question.\n"
    "4. Tool Access & Grounding:\n"
    "   - Always ground your answers in verified database records or retrieved RAG documentation.\n"
    "   - Use your tools (`execute_sql_query`, `execute_sql_write`, `search_rag_brain`, `get_machine_status`, `get_print_progress`, `query_shifts_and_attendance`, `query_operator_wall_production`, `record_operator_wall_production`, `start_or_update_printing_job`, etc.) whenever checking or modifying facts.\n"
    "\n"
    "SPEECH-TO-TEXT (STT) PHONETIC & HINGLISH NORMALIZATION:\n"
    "Operator voice inputs come from factory-floor speech-to-text microphones, which frequently introduce phonetic misrecognitions in English, Hindi, and Hinglish. Automatically normalize these based on shop-floor context:\n"
    "- 'ball' / 'bol' / 'baal' -> 'wall' (the concrete 3D printed wall).\n"
    "- 'cal' / 'call' / 'kal' -> 'kal' (Hindi for yesterday when used with past tense verbs like 'tha', 'thi', 'print kari thi', or tomorrow when with future verbs).\n"
    "- 'fifth' / 'sift' -> 'shift' (e.g. morning/evening shift).\n"
    "- 'Anand Vihar' / 'Anand yaha' -> Operator name 'Anand' or operator location context.\n"
    "- 'to walls' / 'too walls' -> '2 walls'.\n"
    "- 'for walls' -> '4 walls'.\n"
    "- 'one wall' / 'won wall' -> '1 wall'.\n"
    "- 'hamdan' / 'hamdaan' -> Operator 'Hamdan'.\n"
    "- Always interpret these phonetic slips naturally and address the operator's actual question directly.\n"
    "\n"
    "YOUR ROLE & OBJECTIVES:\n"
    "1. You assist shop-floor operators and supervisors with full autonomy: monitoring printing, checking machine health, managing employee shifts and attendance, tracking 3D printing jobs and overtime, recording completed wall production, troubleshooting issues, and learning new knowledge.\n"
    "2. You have direct access to system tools to inspect (READ) and modify (WRITE) the SQLite database (`kelvin6k.db`), check machine and print progress status, and query or learn into the RAG knowledge brain.\n"
    "3. NEVER invent machine numbers or shift data when facts are required. Call the appropriate tool (e.g. `execute_sql_query`, `execute_sql_write`, `get_current_print_status`, `get_machine_status`, `get_current_layer`, `get_print_progress`, `query_shifts_and_attendance`, `query_operator_wall_production`, `search_rag_brain`, etc.) to read or modify facts in the database.\n"
    "4. When the user asks you to perform an action (e.g. record completed walls, start a shift, log an issue, remember a fix, update a job), execute the corresponding tool immediately.\n"
    "   - When an operator says 'Hamdan completed printing two walls' or 'I printed to walls', call record_operator_wall_production(operator_name='Hamdan', walls_completed=2) or start_or_update_printing_job / learn_and_store_in_rag_brain, and confirm naturally (e.g. 'Got it, I have logged that Hamdan completed printing 2 walls today.').\n"
    "   - When asked 'How many walls did Hamdan complete?', call query_operator_wall_production(operator_name='Hamdan') to look up his completed walls and report the exact count naturally.\n"
    "   - When asked 'Can you tell me about this project?' or 'Which material is used in printing walls?', use search_rag_brain or search_docs if needed, along with your system knowledge of Kelvin6K, to provide a smart, natural, conversational answer. Never return raw copy-pasted markdown chunks or say 'Sorry, I didn't understand that'.\n"
    "5. If a request is open-ended or missing details (e.g., 'I am facing a problem', 'Start my shift' without an employee name), naturally converse in friendly, conversational English (e.g., 'I am here to help. What problem are you facing?').\n"
    "6. Standard printing time rule: Standard printing hours is 9.0 hours. Any printing duration beyond 9.0 hours is overtime (overtime_hours = max(duration - 9.0, 0.0)). The printing tools handle this math automatically.\n"
    "7. When an operator describes a fix or new procedure they want you to remember (e.g. 'Remember this fix', 'Learn this', 'Save this SOP'), call learn_and_store_in_rag_brain to persistently store it in the RAG brain.\n"
    "\n"
    "HARDWARE MONITORING PRINCIPLE:\n"
    "You are a monitoring, logging, and advisory system only. Never claim to physically drive motors, actuate pumps, or alter hardware settings on the physical machine. If telemetry or sensor metrics are unavailable, state clearly that they are not connected or currently unavailable.\n"
    "\n"
    "LANGUAGE & SPOKEN CONVERSATION RULES:\n"
    "1. Strict English Only: Always speak and answer in natural, fluent spoken English, even if the user speaks in Hindi or Hinglish.\n"
    "2. Natural spoken delivery: Keep answers concise (1-3 sentences), professional, clear, and direct, speaking like a helpful shop-floor engineering colleague. Do NOT use markdown syntax (no asterisks, no hashtags, no bullet points, no code blocks, no backticks). Spell units in words (e.g., '650 millimeters', '18.5 millimeters', '8.5 hours')."
)


# --- tools exposed to the model ----------------------------------------

def list_or_search_employees(name_query: str = "") -> list:
    """Search or list employee records in the database. Use to find active or registered operators."""
    return shift_store.list_or_search_employees(name_query or None)


def create_or_update_employee(name: str, role: str = "Operator", status: str = "Active") -> dict:
    """Register a new employee/operator or update their role or status in the database."""
    return shift_store.create_or_update_employee(name, role, status)


def start_employee_shift(operator_name: str, shift_type: str = "", actual_start_time: str = "") -> dict:
    """Start an employee shift. If operator does not exist, prompts user first. Calculates start time and attendance."""
    return shift_store.start_employee_shift(operator_name, shift_type or None, actual_start_time or None)


def end_employee_shift(operator_name: str = "") -> dict:
    """End an active employee shift and calculate total hours worked."""
    return shift_store.end_employee_shift(operator_name or None)


def query_shifts_and_attendance(operator_name: str = "", shift_type: str = "", date: str = "") -> list:
    """Query shift history, Came In and Came Out times, late arrivals, and total working days count."""
    return shift_store.query_shifts_and_attendance(operator_name or None, shift_type or None, date or None)


def start_or_update_printing_job(
    job_name: str,
    operator_name: str = "",
    duration_hours: float = 0.0,
    status: str = "",
    current_layer: int = 0,
    total_layers: int = 0,
) -> dict:
    """Start or update a 3D printing job. Automatically calculates 9-hour regular time and overtime hours."""
    dur = duration_hours if duration_hours > 0 else None
    return shift_store.start_or_update_printing_job(
        job_name,
        operator_name or None,
        dur,
        status or None,
        current_layer or None,
        total_layers or None,
    )


def query_printing_jobs_and_overtime(
    operator_name: str = "", job_name: str = "", only_overtime: bool = False
) -> list:
    """Query 3D printing jobs, actual duration, and 9-hour overtime breakdown."""
    return shift_store.query_printing_jobs_and_overtime(
        operator_name or None, job_name or None, only_overtime
    )


def record_operator_wall_production(
    operator_name: str = "",
    walls_completed: int = 1,
    wall_number: int = 0,
    notes: str = "",
) -> dict:
    """Record completed wall production for an operator. Use when an operator reports completing wall(s) (e.g. 'Hamdan completed printing 2 walls', 'I printed to walls', 'Recorded 2 walls for Hamdan'). Handles numbers and STT homophones like 'to walls' = 2 walls."""
    from agent import identity
    name_to_use = (operator_name or "").strip() or "Operator"
    op = shift_store.find_operator_by_name(name_to_use)
    if not op:
        res = identity.resolve_employee(name_to_use, active_only=False)
        if res.status == "MATCHED" and res.operator:
            op = res.operator
        else:
            op = shift_store.create_or_update_employee(name_to_use, "Operator", "Active")
    op_id = str(op["operator_id"])
    actual_name = op.get("name", name_to_use)

    active = shift_store.get_active_shift_for_line("line_1")
    shift_id = None
    today = datetime.date.today().isoformat()
    if active and str(active.get("operator_id")) == op_id:
        shift_id = active["shift_id"]
    else:
        shifts = shift_store.query_shifts_and_attendance(operator_name=actual_name, date=today)
        if shifts:
            shift_id = shifts[-1].get("shift_id")
        elif active:
            shift_id = active["shift_id"]
        else:
            started = shift_store.start_shift(
                operator_id=op_id,
                shift_date=today,
                shift_type="Morning",
                line_id="line_1",
            )
            shift_id = started

    count = max(1, int(walls_completed))
    recorded_walls = []
    existing = shift_store.get_operator_production(operator_id=op_id, shift_id=shift_id)
    max_existing_wall = max([r.get("wall_number", 0) for r in existing], default=0)

    for i in range(1, count + 1):
        wn = wall_number if (wall_number > 0 and count == 1) else (max_existing_wall + i)
        shift_store.record_production(
            shift_id=shift_id,
            wall_number=wn,
            wall_height=3000.0,
            segment_layers=50,
            layers_completed=50,
            status="completed",
            notes=notes or f"Completed by {actual_name}",
        )
        recorded_walls.append(wn)

    shift_store.start_or_update_printing_job(
        job_name=f"Wall_{'_'.join(str(w) for w in recorded_walls)}",
        operator_name=actual_name,
        status="Completed",
    )

    from agent.rag.ingest import learn_and_store_in_rag_brain
    learn_and_store_in_rag_brain(
        title=f"Production Update - {actual_name}",
        category="Production",
        content=f"Operator {actual_name} completed printing {count} wall(s): wall numbers {recorded_walls} on {today}.",
    )

    return {
        "success": True,
        "operator_id": op_id,
        "operator_name": actual_name,
        "walls_completed": count,
        "wall_numbers": recorded_walls,
        "message": f"Successfully recorded {count} completed wall(s) (Wall {', '.join(str(w) for w in recorded_walls)}) for {actual_name}.",
    }


def query_operator_wall_production(operator_name: str = "") -> dict:
    """Query the number of walls completed by an operator. Use when asking how many walls someone completed (e.g. 'How many walls did Hamdan complete?')."""
    name_to_use = (operator_name or "").strip()
    if not name_to_use:
        return {"found": False, "message": "No operator name provided to query wall production."}
    op = shift_store.find_operator_by_name(name_to_use)
    if not op:
        from agent import identity
        res = identity.resolve_employee(name_to_use, active_only=False)
        if res.status == "MATCHED" and res.operator:
            op = res.operator
    if not op:
        return {"found": False, "operator_name": name_to_use, "walls_completed": 0, "message": f"Operator {name_to_use} not found in database."}

    op_id = str(op["operator_id"])
    actual_name = op.get("name", name_to_use)
    records = shift_store.get_operator_production(operator_id=op_id)
    completed = [r for r in records if r.get("status") in ("completed", "Completed", "in_progress")]
    wall_numbers = [r.get("wall_number") for r in completed if r.get("wall_number") is not None]

    return {
        "found": True,
        "operator_name": actual_name,
        "walls_completed": len(completed),
        "wall_numbers": wall_numbers,
        "records_count": len(records),
        "message": f"{actual_name} has completed {len(completed)} wall(s) (Wall {', '.join(str(w) for w in wall_numbers)})." if completed else f"No completed wall records found for {actual_name}.",
    }


def log_or_update_shift_issue(
    problem_type: str,
    problem_description: str,
    operator_name: str = "",
    severity: str = "Medium",
    solution: str = "",
    status: str = "Open",
) -> dict:
    """Log a shop-floor issue or update an existing issue with solution and status."""
    return shift_store.log_or_update_shift_issue(
        problem_type,
        problem_description,
        operator_name or None,
        severity,
        solution or None,
        status,
    )


def execute_sql_query(query: str) -> List[Dict[str, Any]]:
    """Execute a read-only SQL SELECT query on the kelvin6k SQLite database. Returns list of rows as dictionaries."""
    import sqlite3
    db_path = config.DATABASE_PATH
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = [dict(r) for r in cur.fetchall()]
        return rows
    except Exception as exc:
        return [{"error": str(exc)}]
    finally:
        conn.close()


def execute_sql_write(query: str) -> Dict[str, Any]:
    """Execute an INSERT, UPDATE, or DELETE SQL statement on the kelvin6k SQLite database."""
    import sqlite3
    db_path = config.DATABASE_PATH
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        cur = conn.cursor()
        cur.execute(query)
        conn.commit()
        return {"success": True, "rows_affected": cur.rowcount, "lastrowid": cur.lastrowid}
    except Exception as exc:
        conn.rollback()
        return {"success": False, "error": str(exc)}
    finally:
        conn.close()


def learn_and_store_in_rag_brain(
    title: str,
    category: str = "Operations",
    content: str = "",
    problem: str = "",
    solution: str = "",
) -> dict:
    """Persistently store new operational knowledge, SOP, parameter, or issue fix into the RAG brain for immediate retrieval."""
    from agent.rag.ingest import learn_and_store_in_rag_brain as _rag_learn
    return _rag_learn(
        title, category, content, problem or None, solution or None
    )


store_learned_knowledge = learn_and_store_in_rag_brain


def search_rag_brain(query: str) -> dict:
    """Search both RAG documentation manuals/SOPs and historical database shift issues for solutions, procedures, or specifications."""
    q_clean = (query or "").strip().lower()
    if not q_clean or len(q_clean) < 3:
        return {"found": False, "results": [], "message": "No query provided for RAG search."}
    if any(q_clean == g for g in ("hi", "hello", "hey", "thanks", "thank you", "bye", "good morning", "good evening", "ok", "okay")):
        return {"found": False, "results": [], "message": "Conversational greeting does not require documentation lookup."}
    from agent.rag.retriever import search_rag_brain as _rag_search
    return _rag_search(query)


search_rag_brain_knowledge = search_rag_brain


def get_current_print_status() -> dict:
    """Check whether the machine is currently printing. Use for questions like 'is printing running', 'is the machine working'."""
    return ms.get_current_print_status()


def get_current_layer() -> dict:
    """Get the current layer number being printed."""
    return ms.get_current_layer()


def get_current_layer_width() -> dict:
    """Get the measured width of the current printed layer in millimetres."""
    return ms.get_current_layer_width()


def get_print_progress() -> dict:
    """Get print progress: total, completed and remaining wall length in millimetres plus percentage."""
    return ms.get_print_progress()


def get_machine_status() -> dict:
    """Get overall machine and print status."""
    return ms.get_machine_status()


def search_docs(query: str) -> dict:
    """Search the ingested machine manuals, SOPs, maintenance and troubleshooting documentation."""
    q_clean = (query or "").strip().lower()
    if not q_clean or len(q_clean) < 3:
        return {"found": False, "results": [], "message": "No query provided for documentation search."}
    if any(q_clean == g for g in ("hi", "hello", "hey", "thanks", "thank you", "bye", "good morning", "good evening", "ok", "okay")):
        return {"found": False, "results": [], "message": "Conversational turn does not require documentation search."}
    return search_documentation(query)


def get_active_shift() -> dict:
    """Get the active shift and operator currently running on the line."""
    active = shift_store.get_active_shift_for_line("line_1")
    if not active:
        return {"available": True, "active": False, "reason": "No active shift is currently running on Line 1."}
    op = shift_store.get_operator(active["operator_id"])
    return {
        "available": True,
        "active": True,
        "shift_id": active["shift_id"],
        "shift_type": active["shift_type"],
        "shift_date": active["shift_date"],
        "operator_id": active["operator_id"],
        "operator_name": op["name"] if op else active["operator_id"],
        "line_id": active["line_id"],
    }


TOOLS: List[Callable] = [
    execute_sql_query,
    execute_sql_write,
    search_rag_brain,
    learn_and_store_in_rag_brain,
    list_or_search_employees,
    create_or_update_employee,
    start_employee_shift,
    end_employee_shift,
    query_shifts_and_attendance,
    record_operator_wall_production,
    query_operator_wall_production,
    start_or_update_printing_job,
    query_printing_jobs_and_overtime,
    log_or_update_shift_issue,
    get_active_shift,
    get_current_print_status,
    get_current_layer,
    get_current_layer_width,
    get_print_progress,
    get_machine_status,
    search_docs,
]

_TOOLS_BY_NAME = {fn.__name__: fn for fn in TOOLS}
_TOOLS_BY_NAME["store_learned_knowledge"] = store_learned_knowledge
_TOOLS_BY_NAME["search_rag_brain_knowledge"] = search_rag_brain_knowledge


@dataclass
class PendingAction:
    session_id: str
    line_id: str
    intent: str
    params: Dict[str, Any]
    prompt_spoken: str
    created_at: float
    expires_at: float
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_request: str = ""
    missing_fields: List[str] = field(default_factory=list)
    status: str = "pending"  # "pending", "executed", "cancelled", "expired"
    idempotency_key: str = ""


class PendingActionManager:
    """Manages multi-terminal pending voice actions with 30s confirmation TTL."""

    def __init__(self, ttl_seconds: float = 30.0):
        self.ttl = ttl_seconds
        self._pending: Dict[Tuple[str, str], PendingAction] = {}
        self._lock = threading.Lock()

    def set(
        self,
        session_id: str,
        line_id: str,
        intent: str,
        params: Dict[str, Any],
        prompt: str,
        original_request: str = "",
        missing_fields: Optional[List[str]] = None,
        idempotency_key: str = "",
    ) -> PendingAction:
        now = time.time()
        with self._lock:
            action = PendingAction(
                session_id=session_id,
                line_id=line_id,
                intent=intent,
                params=params,
                prompt_spoken=prompt,
                created_at=now,
                expires_at=now + self.ttl,
                action_id=str(uuid.uuid4()),
                original_request=original_request,
                missing_fields=missing_fields or [],
                status="pending",
                idempotency_key=idempotency_key or f"{session_id}:{line_id}:{intent}:{int(now)}",
            )
            self._pending[(line_id, session_id)] = action
            return action

    def get(self, session_id: str, line_id: str) -> Optional[PendingAction]:
        now = time.time()
        with self._lock:
            action = self._pending.get((line_id, session_id))
            if action and now <= action.expires_at and action.status == "pending":
                return action
            if action:
                if now > action.expires_at and action.status == "pending":
                    action.status = "expired"
                self._pending.pop((line_id, session_id), None)
            return None

    def refresh(self, session_id: str, line_id: str) -> None:
        now = time.time()
        with self._lock:
            action = self._pending.get((line_id, session_id))
            if action and action.status == "pending":
                action.expires_at = now + self.ttl

    def clear(self, session_id: str, line_id: str, status: str = "cancelled") -> None:
        with self._lock:
            action = self._pending.get((line_id, session_id))
            if action:
                action.status = status
            self._pending.pop((line_id, session_id), None)


@dataclass
class SessionContext:
    line_id: str
    session_id: str
    last_issue_id: Optional[int] = None
    last_rag_recommendation: Optional[str] = None
    awaiting_disambiguation: Optional[Dict[str, Any]] = None
    awaiting_outcome: Optional[Dict[str, Any]] = None
    awaiting_problem_details: bool = False
    pending_problem_info: Optional[Dict[str, Any]] = None
    awaiting_action_for_issue: bool = False
    last_updated_at: float = field(default_factory=time.time)
    current_operator_id: Optional[str] = None
    current_operator_name: Optional[str] = None
    current_shift_id: Optional[int] = None
    current_line_id: Optional[str] = None
    current_issue_id: Optional[int] = None


class SessionManager:
    """Manages lightweight in-memory session-scoped issue context with sliding TTL."""

    def __init__(self, ttl_seconds: float = 1800.0):
        self.ttl = ttl_seconds
        self._sessions: Dict[Tuple[str, str], SessionContext] = {}
        self._lock = threading.Lock()

    def _get_locked(self, session_id: str, line_id: str, now: float) -> SessionContext:
        key = (line_id, session_id)
        ctx = self._sessions.get(key)
        if ctx and (now - ctx.last_updated_at > self.ttl):
            ctx = SessionContext(line_id=line_id, session_id=session_id, last_updated_at=now)
            self._sessions[key] = ctx
        elif not ctx:
            ctx = SessionContext(line_id=line_id, session_id=session_id, last_updated_at=now)
            self._sessions[key] = ctx
        else:
            ctx.last_updated_at = now

        if not ctx.current_line_id:
            ctx.current_line_id = line_id

        # NOTE: Do NOT infer current_operator_id or current_operator_name from active shift!
        # Operator identity must come strictly from authenticated session or explicit introduction.

        if ctx.current_issue_id is not None and ctx.last_issue_id is None:
            ctx.last_issue_id = ctx.current_issue_id
        elif ctx.last_issue_id is not None and ctx.current_issue_id is None:
            ctx.current_issue_id = ctx.last_issue_id

        return ctx

    def get(self, session_id: str, line_id: str) -> SessionContext:
        now = time.time()
        with self._lock:
            return self._get_locked(session_id, line_id, now)

    def update(self, session_id: str, line_id: str, **kwargs) -> SessionContext:
        now = time.time()
        with self._lock:
            ctx = self._get_locked(session_id, line_id, now)
            for k, v in kwargs.items():
                if hasattr(ctx, k):
                    setattr(ctx, k, v)
            if "last_issue_id" in kwargs and "current_issue_id" not in kwargs:
                ctx.current_issue_id = kwargs["last_issue_id"]
            elif "current_issue_id" in kwargs and "last_issue_id" not in kwargs:
                ctx.last_issue_id = kwargs["current_issue_id"]
            ctx.last_updated_at = now
            return ctx

    def clear(self, session_id: str, line_id: str) -> None:
        with self._lock:
            self._sessions.pop((line_id, session_id), None)


_GLOBAL_ASSISTANT: Optional["Assistant"] = None


def get_global_assistant() -> "Assistant":
    global _GLOBAL_ASSISTANT
    if _GLOBAL_ASSISTANT is None:
        _GLOBAL_ASSISTANT = Assistant()
    return _GLOBAL_ASSISTANT


_shared_gemini_client = None
_shared_client_lock = threading.Lock()
_llm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm_assistant")


def get_shared_gemini_client():
    """Module-level persistent Gemini client reused across all requests and assistant instances."""
    global _shared_gemini_client
    if _shared_gemini_client is None and config.GEMINI_API_KEY:
        with _shared_client_lock:
            if _shared_gemini_client is None and config.GEMINI_API_KEY:
                try:
                    from google import genai

                    _shared_gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
                    log.info("Initialized shared persistent Gemini genai.Client at module level")
                except Exception as exc:
                    log.warning("Could not create shared Gemini client: %s", exc)
    return _shared_gemini_client


class Assistant:
    """
    Tool-calling assistant over Gemini.

    Falls back to a deterministic local router when no API key is configured, so
    the whole stack (tools, database, RAG) is testable offline.
    """

    def __init__(self, state=None, use_llm: Optional[bool] = None):
        if state is not None:
            ms.bind_live_state(state)
        self.use_llm = (bool(config.GEMINI_API_KEY) if use_llm is None else use_llm)
        self._client = None
        self._chats: Dict[str, Any] = {}
        self._chat_history_turns: Dict[str, List[Dict[str, Any]]] = {}
        self._history: List[Any] = []
        # Latched off after a configuration failure so one bad key does not
        # add a failing network round trip to every single question.
        try:
            import anthropic
            self._claude_disabled = not bool(config.ANTHROPIC_API_KEY)
        except ImportError:
            self._claude_disabled = True
        self._actions = PendingActionManager(ttl_seconds=30.0)
        self._session_ctx = SessionManager(ttl_seconds=1800.0)

    @staticmethod
    def _server_now() -> float:
        """
        The authoritative clock for attendance.

        One method so the time SPOKEN in the confirmation and the time WRITTEN
        by start_shift come from the same source, and so a test can freeze it.
        """
        return time.time()

    # --- LLM path ---

    def _client_or_none(self):
        client = get_shared_gemini_client()
        if client is not None:
            self._client = client
        return self._client

    def _get_or_create_chat(self, session_id: str, client):
        if session_id in self._chats:
            return self._chats[session_id]

        from google.genai import types

        # Make gemini-flash-lite-latest the first and primary model with zero sleep delays
        models_to_try = ["gemini-flash-lite-latest"]
        if config.GEMINI_MODEL and config.GEMINI_MODEL not in models_to_try:
            if "2.5" not in config.GEMINI_MODEL and "preview" not in config.GEMINI_MODEL:
                models_to_try.append(config.GEMINI_MODEL)
        for alt in ("gemini-flash-latest",):
            if alt not in models_to_try:
                models_to_try.append(alt)

        for m in models_to_try:
            try:
                chat = client.chats.create(
                    model=m,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=TOOLS,
                        temperature=0.1,
                        max_output_tokens=300,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            maximum_remote_calls=5
                        ),
                    ),
                )
                self._chats[session_id] = chat
                log.info("Initialized Gemini multi-turn agentic chat for session %s with model %s", session_id, m)
                return chat
            except Exception as exc:
                log.debug("Could not create chat with model %s: %s", m, exc)

        log.warning("Could not create Gemini chat session with any candidate model.")
        return None

    @staticmethod
    def _is_casual_or_greeting(question: str) -> Optional[str]:
        q = (question or "").strip().lower()
        clean = re.sub(r"[^\w\s]", "", q).strip()
        if not clean:
            return None

        casual_map = {
            "hi": "Hello! How can I assist you with the Kelvin6K today?",
            "hello": "Hello! How can I assist you with the Kelvin6K today?",
            "hey": "Hello! What can I help you with today?",
            "hey kelvin": "Hello! What can I do for you today?",
            "hello kelvin": "Hello! How can I assist you today?",
            "hi kelvin": "Hi! How can I assist you on the shop floor today?",
            "good morning": "Good morning! Ready to assist with operations. What do you need?",
            "good morning kelvin": "Good morning! Ready to assist with operations. What do you need?",
            "good afternoon": "Good afternoon! How can I assist with your shift or printing today?",
            "good afternoon kelvin": "Good afternoon! How can I assist with your shift or printing today?",
            "good evening": "Good evening! How can I assist you with operations tonight?",
            "good evening kelvin": "Good evening! How can I assist you with operations tonight?",
            "namaste": "Namaste! How can I assist you today?",
            "namaste kelvin": "Namaste! How can I assist you today?",
            "how are you": "I am operating normally and ready to help. How can I assist you today?",
            "how are you doing": "I am operating normally and ready to help. How can I assist you today?",
            "how are you kelvin": "I am operating normally and ready to help. How can I assist you today?",
            "whats up": "All systems operational. How can I assist you today?",
            "what's up": "All systems operational. How can I assist you today?",
            "thank you": "You're welcome! Let me know if you need anything else.",
            "thanks": "You're welcome! Let me know if you need anything else.",
            "thank you kelvin": "You're welcome! Let me know if you need anything else.",
            "thanks kelvin": "You're welcome! Let me know if you need anything else.",
            "who are you": "I am Kelvin, the autonomous voice assistant for the Kelvin6K 3D concrete printing system.",
            "what are you": "I am Kelvin, the autonomous voice assistant for the Kelvin6K 3D concrete printing system.",
            "who is kelvin": "I am Kelvin, the autonomous voice assistant for the Kelvin6K 3D concrete printing system.",
            "what can you do": "I monitor printing, check machine health, manage operator shifts, track jobs and overtime, troubleshoot issues, and search documentation.",
            "help": "You can ask about machine status, print progress, layer metrics, shifts, or troubleshooting.",
            "bye": "Goodbye! Have a safe shift.",
            "goodbye": "Goodbye! Have a safe shift.",
            "bye kelvin": "Goodbye! Have a safe shift.",
            "goodbye kelvin": "Goodbye! Have a safe shift.",
            "ok": "Understood. Let me know what you need next.",
            "okay": "Understood. Let me know what you need next.",
        }

        return casual_map.get(clean)

    @staticmethod
    def _prune_chat_history(chat: Any, max_messages: int = 12) -> None:
        """
        Safely prunes chat history so that:
        1. It never cuts between a model function_call and user function_response.
        2. It only starts at a legitimate user text prompt.
        """
        for attr in ("_curated_history", "_comprehensive_history"):
            history = getattr(chat, attr, None)
            if history and len(history) > max_messages:
                start_idx = max(0, len(history) - max_messages)
                cut_idx = None
                for i in range(start_idx, len(history)):
                    msg = history[i]
                    role = getattr(msg, "role", None)
                    parts = getattr(msg, "parts", []) or []
                    has_fn_resp = any(getattr(p, "function_response", None) is not None for p in parts)
                    if role == "user" and not has_fn_resp:
                        cut_idx = i
                        break
                if cut_idx is not None and cut_idx > 0:
                    setattr(chat, attr, history[cut_idx:])

    def _ask_llm(
        self,
        question: str,
        session_id: str = "default_terminal",
        routed: Optional[Any] = None,
        line_id: str = "line_1",
        operator_id: Optional[str] = None,
    ) -> Optional[str]:
        client = self._client_or_none()
        if client is None:
            return None
        try:
            chat = self._get_or_create_chat(session_id, client)
            if chat is None:
                return None

            max_turns = 10
            self._prune_chat_history(chat, 12)

            now = datetime.datetime.now()
            today_str = now.strftime("%Y-%m-%d")
            yesterday_str = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            current_time_str = now.strftime("%I:%M %p")

            active = shift_store.get_active_shift_for_line(line_id)
            if active:
                op = shift_store.get_operator(active.get("operator_id"))
                op_name = op.get("name") if op else active.get("operator_id")
                shift_info = f"Shift #{active.get('shift_id')} ({active.get('shift_type')} Shift, Operator: {op_name}, Date: {active.get('shift_date')})"
            else:
                shift_info = f"No active shift currently running on {line_id}"

            p_stat = ms.get_current_print_status()
            progress = ms.get_print_progress()
            is_printing = p_stat.get("printing", False)
            print_status_str = f"Printing: {'Running' if is_printing else 'Idle'}"
            if progress.get("available"):
                print_status_str += f", Completed: {progress.get('completed_mm', 0)} mm / {progress.get('total_mm', 0)} mm"

            rag_context = ""
            try:
                rag_res = search_rag_brain(question)
                if rag_res and rag_res.get("found") and rag_res.get("results"):
                    top_excerpts = []
                    for r in rag_res["results"][:3]:
                        src = r.get("source", "knowledge_base")
                        txt = r.get("text", "").strip()
                        if txt:
                            top_excerpts.append(f"[{src}]: {txt[:350]}")
                    if top_excerpts:
                        rag_context = "\n".join(top_excerpts)
            except Exception as rag_err:
                log.debug("RAG search in _ask_llm failed: %s", rag_err)

            full_prompt = (
                f"[LIVE OPERATIONAL CONTEXT]\n"
                f"- Current Date: {today_str} ({now.strftime('%A')}), Current Time: {current_time_str}\n"
                f"- Yesterday's Date: {yesterday_str}\n"
                f"- Machine Line: {line_id}\n"
                f"- Active Shift: {shift_info}\n"
                f"- Machine Telemetry: {print_status_str}\n"
            )
            if rag_context:
                full_prompt += f"\n[RETRIEVED KNOWLEDGE BASE EXCERPTS]\n{rag_context}\n"
            full_prompt += f"\n[OPERATOR VOICE QUERY]\n{question}"

            try:
                response = chat.send_message(full_prompt)
            except Exception as exc:
                if "503" in str(exc) or "UNAVAILABLE" in str(exc):
                    log.warning("Primary Gemini model returned 503; falling back to gemini-flash-latest: %s", exc)
                    from google.genai import types
                    backup_chat = client.chats.create(
                        model="gemini-flash-latest",
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            tools=TOOLS,
                            temperature=0.1,
                            max_output_tokens=300,
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                maximum_remote_calls=5
                            ),
                        ),
                    )
                    self._chats[session_id] = backup_chat
                    try:
                        response = backup_chat.send_message(full_prompt)
                    except Exception as exc2:
                        if "503" in str(exc2) or "UNAVAILABLE" in str(exc2):
                            log.warning("Backup model also returned 503; falling back to gemini-2.5-flash: %s", exc2)
                            backup2_chat = client.chats.create(
                                model="gemini-2.5-flash",
                                config=types.GenerateContentConfig(
                                    system_instruction=SYSTEM_PROMPT,
                                    tools=TOOLS,
                                    temperature=0.1,
                                    max_output_tokens=300,
                                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                        maximum_remote_calls=5
                                    ),
                                ),
                            )
                            self._chats[session_id] = backup2_chat
                            response = backup2_chat.send_message(full_prompt)
                        else:
                            raise exc2
                else:
                    raise exc
            raw_text = (response.text or "").strip()
            if not raw_text:
                return None

            from agent.spoken import format_spoken_response
            spoken_text = format_spoken_response(raw_text)

            history = self._chat_history_turns.setdefault(session_id, [])
            history.append({"role": "user", "text": question, "timestamp": time.time()})
            history.append({"role": "assistant", "text": spoken_text, "timestamp": time.time()})
            if len(history) > max_turns:
                self._chat_history_turns[session_id] = history[-max_turns:]

            return spoken_text
        except Exception as exc:
            log.warning("Agentic Gemini call failed (%s); using conversational fallback.", exc)
            self._chats.pop(session_id, None)
            clean_norm = question.strip().lower().rstrip(".!?")
            open_problem_queries = (
                "i am facing a problem", "i have a problem", "there is a problem",
                "we have a problem", "facing a problem", "facing an issue",
                "i have an issue", "there is an issue", "we have an issue",
                "having a problem", "having an issue", "help me with a problem",
                "help me with an issue", "i am having a problem",
            )
            if any(clean_norm == p or clean_norm.startswith(p + " ") for p in open_problem_queries):
                return "I'm here to help. What problem are you facing?"
            return None

    def _ask_llm_stream(self, question: str, session_id: str = "default_terminal"):
        """Streams chunk tokens from Gemini chat with 250 token limit and 6 turns memory."""
        client = self._client_or_none()
        if client is None:
            return
        chat = self._get_or_create_chat(session_id, client)
        if chat is None:
            return

        max_turns = 6
        self._prune_chat_history(chat, 12)

        try:
            response_stream = chat.send_message_stream(question)
            accumulated = []
            for chunk in response_stream:
                text = chunk.text or ""
                if text:
                    accumulated.append(text)
                    yield text
        except Exception:
            self._chats.pop(session_id, None)
            raise

        full_raw = "".join(accumulated).strip()
        if full_raw:
            from agent.spoken import format_spoken_response
            spoken_text = format_spoken_response(full_raw)
            history = self._chat_history_turns.setdefault(session_id, [])
            history.append({"role": "user", "text": question, "timestamp": time.time()})
            history.append({"role": "assistant", "text": spoken_text, "timestamp": time.time()})
            if len(history) > max_turns:
                self._chat_history_turns[session_id] = history[-max_turns:]

    # --- offline deterministic router ---

    @staticmethod
    def route(question: str) -> str:
        """Maps a question to a tool name using keywords (English + Hinglish)."""
        q = question.lower()

        def has(*words: str) -> bool:
            return any(w in q for w in words)

        if has("width", "chaudai", "chaudayi", "mota"):
            return "get_current_layer_width"
        # Progress questions. "wall"/"printed"/"how many" matter: the machine
        # prints wall length, so "how many walls have been printed?" is a
        # progress question, not a machine-status one.
        if has("baki", "remaining", "bacha", "left", "progress", "kitni wall",
               "how much", "kitna print", "complete", "percent", "%",
               "wall", "how many", "printed", "print ho gay", "ho gayi", "ho gaya",
               "done", "target", "total"):
            return "get_print_progress"
        if has("layer"):
            # "layer width" handled above; plain layer questions -> layer number
            return "get_current_layer"
        if has("camera", "problem", "issue", "dikkat", "kharab", "status"):
            return "get_machine_status"
        if has("printing", "print ho", "chal rah", "chal rha", "running", "working"):
            return "get_current_print_status"
        if has("how do i", "procedure", "manual", "sop", "maintenance", "error code",
               "kaise", "troubleshoot"):
            return "search_docs"
        return "get_machine_status"

    @staticmethod
    def _speak_number(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.1f}".rstrip("0").rstrip(".")
        return str(value)

    # Returned when the question could not be classified at all. This is
    # deliberately NOT a machine-status answer: previously an unrecognised
    # question fell through to a keyword tool whose default was
    # get_machine_status, so a misheard question produced a confident-sounding
    # "Camera is connected. Machine status is PRINTING." That hid the real
    # failure (bad transcript) behind a plausible answer.
    NOT_UNDERSTOOD = (
        "Sorry, I didn't understand that. Please try again. You can ask about "
        "production, layers, material, timing, machine or camera status, "
        "problems, or the documentation."
    )

    # Claude phrases facts; it never originates them. The facts come from the
    # query router, which reads the database and live machine state, and they
    # are passed in as JSON. The prompt forbids inventing or adjusting any
    # value, so the worst a model failure can do is produce a clumsy sentence
    # around correct numbers -- never a wrong number.
    _CLAUDE_SYSTEM = (
        "You are the voice of the Kelvin6K concrete 3D printing machine, speaking "
        "to its operator.\n"
        "You are given the operator\'s question and a JSON object of FACTS "
        "read from the machine\'s database and live state.\n"
        "RULES:\n"
        "1. Use ONLY the numbers in FACTS. Never invent, estimate, or infer a "
        "value that is not there.\n"
        "2. If FACTS says a value is unavailable, say plainly that it is not "
        "available and give the reason from FACTS.\n"
        "3. Reply in ENGLISH ONLY, even if the question is Hindi or Hinglish.\n"
        "4. One or two short spoken sentences. No markdown, no lists. Spell "
        "units in words: 650 millimeters, 65 percent, layer 42.\n"
        "5. You are read-only: never claim to have changed a machine setting."
    )

    def _phrase_with_claude(self, question: str, result) -> Optional[str]:
        return None

    def _answer_locally(self, question: str, routed: Optional[Any] = None) -> str:
        """
        Deterministic answers without an LLM.

        Uses the central query router (agent/query_router.py), which classifies
        the question into one of the documented intents and resolves it against
        real data.

        Three distinct outcomes, never conflated:
          * understood + data present  -> the real answer
          * understood + data missing  -> why it is unavailable
          * not understood             -> say so; never guess an intent
        """
        from agent import query_router

        question = (question or "").strip()
        if not question:
            return "I didn't catch a question. Please try again."

        result = routed if routed is not None else query_router.route(question)
        if result.intent == query_router.Intent.UNKNOWN_QUERY:
            log.info("Unclassified question: %r", question)
            return self.NOT_UNDERSTOOD

        # Deterministic sentence first: it is always correct, and becomes the
        # fallback if Claude is unreachable or misconfigured.
        deterministic = query_router.summarise(result)
        # A wake-word acknowledgement is fixed wording, not a fact to phrase.
        if result.intent == query_router.Intent.WAKE_WORD:
            return deterministic
        spoken = self._phrase_with_claude(question, result)
        if spoken:
            log.info("[CLAUDE] %s", spoken)
            return spoken
        return deterministic

    def _answer_with_tools(self, question: str) -> str:
        """Legacy keyword->tool path, kept as the last-resort fallback."""
        tool = self.route(question)

        if tool == "search_docs":
            result = search_docs(question)
            if not result.get("found"):
                return result.get("message", "That was not found in the available documentation.")
            top = result["results"][0]
            return f"From {top['source']}: {top['text'][:300]}"

        result = _TOOLS_BY_NAME[tool]()

        if tool == "get_print_progress":
            if not result.get("available"):
                return result.get("reason", "I don't have a current machine status.")
            return (
                f"{self._speak_number(result['completed_mm'])} millimeter printed, "
                f"{self._speak_number(result['remaining_mm'])} millimeter remaining "
                f"out of {self._speak_number(result['total_mm'])} millimeter"
                + (f" ({result['percentage']} percent)." if result.get("percentage") is not None else ".")
            )

        if tool == "get_current_layer":
            if not result.get("available"):
                return result.get("reason", "I don't have a current layer number.")
            return f"Current layer is {result['current_layer']}."

        if tool == "get_current_layer_width":
            if not result.get("available"):
                return result.get("reason", "I don't have a current layer width.")
            note = f" {result['note']}" if result.get("note") else ""
            return f"Current layer width is {result['layer_width_mm']} millimeter.{note}"

        if tool == "get_current_print_status":
            if not result.get("available"):
                return result.get("reason", "I don't have a current machine status.")
            note = f" {result['note']}" if result.get("note") else ""
            return ("Yes, printing is currently running." if result.get("printing")
                    else "No, printing is not currently detected.") + note

        if not result.get("available"):
            return result.get("reason", "I don't have a current machine status.")
        camera = result.get("camera") or {}
        connected = "connected" if camera.get("connected") else "not connected"
        note = f" {result['note']}" if result.get("note") else ""
        return f"Camera is {connected}. Machine status is {result.get('status') or 'unknown'}.{note}"

    # --- operator resolution & action execution ---

    def _resolve_operator(
        self,
        operator_id: Optional[str],
        line_id: str,
        spoken_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Resolves the operator this action belongs to, most trustworthy first:

          1. an operator_id supplied by the terminal (a real credential)
          2. a name the operator spoke, matched against the operators table
          3. whoever already holds the active shift on this line

        NOTE ON TRUST: a spoken name is an ASSERTION, not authentication --
        anyone can say "Anand here". It is accepted because voice attendance
        was specified that way, and the confirmation step is the only check.
        If these records ever feed payroll or compliance, step 1 (a badge scan
        or PIN from the terminal) must become mandatory.
        """
        if operator_id:
            op = shift_store.get_operator(operator_id)
            if op and op.get("active"):
                return op

        if spoken_name:
            res = identity.resolve_employee(spoken_name, active_only=True)
            if res.status == "MATCHED" and res.operator:
                return res.operator
            if res.status == "AMBIGUOUS":
                log.warning("_resolve_operator: ambiguous match for %r", spoken_name)
                return None
            op = shift_store.find_operator_by_name(spoken_name)
            if op and op.get("active"):
                return op
            return None

        return None

    def _prepare_confirmation_prompt(self, intent_val: str, params: Dict[str, Any]) -> str:
        line_id = params.get("line_id", "line_1")
        if intent_val == "SHIFT_START":
            # Attendance is a record of WHEN, so the operator hears the date and
            # time they are about to be clocked in for and can refuse if the
            # terminal's clock is wrong. Both come from the server clock -- the
            # same source start_shift() stamps the record with -- never from the
            # browser and never from a fixed string.
            now = datetime.datetime.fromtimestamp(self._server_now())
            name = params.get("operator_name") or "Operator"
            line_label = _spoken_line(line_id)
            if params.get("create_operator"):
                return (
                    f"{name}, today is {_spoken_date(now)} and the current time is "
                    f"{_spoken_time(now)}. I could not find an operator named {name}. "
                    f"Shall I register you and start your shift on {line_label}?"
                )
            return (
                f"{name}, today is {_spoken_date(now)} and the current time is "
                f"{_spoken_time(now)}. Shall I start your shift on {line_label}?"
            )
        if intent_val == "SHIFT_END":
            active = shift_store.get_active_shift_for_line(line_id)
            if not active:
                return f"You are about to end the active shift on {line_id}. Please confirm to proceed."
            op = shift_store.get_operator(active["operator_id"])
            name = op["name"] if op else active["operator_id"]
            shift_date = active["shift_date"]
            st = active.get("start_time")
            now_ts = self._server_now()
            st_str = _spoken_time(datetime.datetime.fromtimestamp(st)) if st else "unknown time"
            et_str = _spoken_time(datetime.datetime.fromtimestamp(now_ts))
            line_label = _spoken_line(line_id)

            prod = shift_store.get_shift_production(active["shift_id"])
            completed_walls = [r for r in prod if r.get("status") == "completed"]
            wall_cnt = len(completed_walls)
            wall_str = f"{wall_cnt} walls" if wall_cnt != 1 else "1 wall"

            issues = shift_store.list_shift_issues(shift_id=active["shift_id"])
            if not issues:
                issue_str = "no issues"
            else:
                resolved_cnt = sum(1 for iss in issues if iss.get("result") == "resolved")
                open_cnt = len(issues) - resolved_cnt
                parts = []
                if resolved_cnt:
                    parts.append(f"{resolved_cnt} resolved")
                if open_cnt:
                    parts.append(f"{open_cnt} unresolved")
                issue_str = f"{len(issues)} issues ({', '.join(parts)})" if len(issues) != 1 else f"1 issue ({', '.join(parts)})"

            return (
                f"{name}, on your {shift_date} shift starting at {st_str}, "
                f"current time is {et_str}. You completed {wall_str} with {issue_str}. "
                f"Shall I end your shift on {line_label}?"
            )
        if intent_val == "REGISTER_EMPLOYEE":
            name = params.get("operator_name") or "Operator"
            return f"I could not find an employee named {name}. Shall I register {name} as a new operator?"
        if intent_val == "LOG_WALL_PRODUCTION":
            wn = params.get("wall_number")
            st = params.get("status", "completed")
            op_name = params.get("operator_name")
            name_prefix = f"{op_name}, " if op_name else ""
            return f"{name_prefix}You are about to log Wall {wn} as {st} on {line_id}. Please confirm to proceed."
        if intent_val == "LOG_SHIFT_ISSUE":
            dt = params.get("downtime_minutes", 0)
            dt_str = f" with {dt:.0f} minutes downtime" if dt else ""
            action_taken = params.get("action_taken")
            desc = params.get("description")
            result = params.get("result")
            op_name = params.get("operator_name")
            if not op_name:
                active = shift_store.get_active_shift_for_line(line_id)
                op = shift_store.get_operator(active["operator_id"]) if active else None
                op_name = op["name"] if op and op.get("name") else None
            name_prefix = f"{op_name}, " if op_name else ""
            if action_taken and desc:
                res_part = f", status '{result}'" if result else ""
                return f"{name_prefix}You are about to log an issue for '{desc}' with action '{action_taken}'{res_part} on {line_id}. Please confirm to proceed."
            return f"{name_prefix}You are about to log an issue on {line_id}{dt_str}. Please confirm to proceed."
        if intent_val == "LOG_SHIFT_HANDOVER":
            return f"You are about to initiate a shift handover on {line_id}. Please confirm to proceed."
        if intent_val == "RESOLVE_SHIFT_ISSUE":
            t_id = params.get("target_issue_id")
            action_taken = params.get("action_taken")
            cause = params.get("cause")
            result = params.get("result")
            id_str = f"Issue #{t_id}" if t_id else "the issue"
            details = []
            if cause:
                details.append(f"cause '{cause}'")
            if action_taken:
                details.append(f"action '{action_taken}'")
            if result:
                details.append(f"status '{result}'")
            det_str = f" with {', '.join(details)}" if details else ""
            active = shift_store.get_active_shift_for_line(line_id)
            op = shift_store.get_operator(active["operator_id"]) if active else None
            name_prefix = f"{op['name']}, " if op and op.get("name") else ""
            return f"{name_prefix}You are about to update {id_str}{det_str} on {line_id}. Please confirm to proceed."
        if intent_val == "START_PRINTING_JOB":
            j_name = params.get("job_name", "Print_Job_1")
            return f"You are about to start printing job {j_name} on {line_id}. Please confirm to proceed."
        if intent_val == "END_PRINTING_JOB":
            j_name = params.get("job_name", "Print_Job_1")
            return f"You are about to finish printing job {j_name} on {line_id}. Please confirm to proceed."
        return "Please confirm this action to proceed."

    def _execute_action(self, action: PendingAction) -> str:
        if action.status == "executed":
            return "This action has already been executed."
        if action.status in ("cancelled", "expired"):
            return "Action is no longer valid."
        action.status = "executed"

        line_id = action.params.get("line_id", "line_1")

        if action.intent == "REGISTER_EMPLOYEE":
            op_name = action.params.get("operator_name", "Operator")
            clean_code = re.sub(r"[^A-Za-z0-9]", "", op_name).upper()
            new_op_id = f"OP_{clean_code}"
            if shift_store.get_operator(new_op_id):
                idx = 1
                while shift_store.get_operator(f"{new_op_id}_{idx}"):
                    idx += 1
                new_op_id = f"{new_op_id}_{idx}"
            try:
                shift_store.create_operator(
                    operator_id=new_op_id,
                    name=op_name,
                    role="operator",
                    active=1,
                )
                # Verify DB write
                created = shift_store.get_operator(new_op_id)
                if not created:
                    return f"Failed to register operator {op_name}."
                self._session_ctx.update(
                    action.session_id,
                    line_id,
                    current_operator_id=new_op_id,
                    current_operator_name=op_name,
                )
                return f"{op_name} has been successfully registered. Shall I start your shift now?"
            except Exception as exc:
                log.error("Failed to register operator: %s", exc)
                return f"Failed to register operator: {exc}"

        if action.intent == "SHIFT_START":
            shift_type = action.params.get("shift_type", "Morning")
            operator_id = action.params.get("operator_id")
            target_walls = action.params.get("target_walls", 0)
            existing = shift_store.get_active_shift_for_line(line_id)
            if existing:
                return "Line 1 already has an active shift. Please close the current shift first."

            if action.params.get("create_operator"):
                op_name = action.params.get("operator_name", "Operator")
                clean_code = re.sub(r"[^A-Za-z0-9]", "", op_name).upper()
                new_op_id = f"OP_{clean_code}"
                if shift_store.get_operator(new_op_id):
                    idx = 1
                    while shift_store.get_operator(f"{new_op_id}_{idx}"):
                        idx += 1
                    new_op_id = f"{new_op_id}_{idx}"
                shift_store.create_operator(
                    operator_id=new_op_id,
                    name=op_name,
                    role="operator",
                    active=1,
                )
                operator_id = new_op_id
                action.params["operator_id"] = new_op_id

            # One timestamp for the whole record: shift_date and start_time
            # must agree, and both come from the server clock -- the same source
            # the operator just heard read back to them.
            started_at = self._server_now()
            moment = datetime.datetime.fromtimestamp(started_at)
            shift_date = moment.date().isoformat()
            try:
                shift_id = shift_store.start_shift(
                    operator_id=operator_id,
                    shift_date=shift_date,
                    shift_type=shift_type,
                    line_id=line_id,
                    target_walls=target_walls,
                    start_time=started_at,
                )
                name = action.params.get("operator_name") or operator_id
                self._session_ctx.update(
                    action.session_id,
                    line_id,
                    current_operator_id=operator_id,
                    current_operator_name=name,
                    current_shift_id=shift_id,
                    current_line_id=line_id,
                    current_issue_id=None,
                    last_issue_id=None,
                )
                if action.params.get("create_operator"):
                    return (
                        f"I have added {name} as a new operator and started their {shift_type} shift on {_spoken_line(line_id)}. "
                        f"{name}, your {shift_type} shift has started at {_spoken_time(moment)}."
                    )
                return (
                    f"{name}, your {shift_type} shift on {_spoken_line(line_id)} "
                    f"has started at {_spoken_time(moment)}."
                )
            except Exception as exc:
                log.error("Failed to start shift: %s", exc)
                return f"Failed to start shift: {exc}"

        if action.intent == "SHIFT_END":
            active = shift_store.get_active_shift_for_line(line_id)
            if not active:
                return f"No active shift found to end on {line_id}."
            try:
                ended_at = self._server_now()
                shift_store.end_shift(active["shift_id"], status="closed", end_time=ended_at)
                st = active.get("start_time")
                dur_formatted = ""
                if st:
                    dur_sec = ended_at - st
                    hours = int(dur_sec // 3600)
                    minutes = int((dur_sec % 3600) // 60)
                    dur_formatted = f"{hours} hours {minutes} minutes" if hours > 0 else f"{minutes} minutes"
                op = shift_store.get_operator(active["operator_id"])
                name = op["name"] if op else active["operator_id"]
                self._session_ctx.clear(action.session_id, line_id)
                dur_str = f" Duration: {dur_formatted}." if dur_formatted else ""
                return f"{name}, your shift on {_spoken_line(line_id)} has ended.{dur_str}"
            except Exception as exc:
                log.error("Failed to end shift: %s", exc)
                return f"Failed to end shift: {exc}"

        if action.intent == "LOG_WALL_PRODUCTION":
            active = shift_store.get_active_shift_for_line(line_id)
            if not active:
                return f"Cannot log wall production because no active shift is running on {line_id}."
            wall_num = action.params.get("wall_number")
            if not wall_num:
                return "No wall number specified to log production."
            try:
                op_name = action.params.get("operator_name")
                op = shift_store.get_operator(active["operator_id"]) if active else None
                if not op_name and op and op.get("name"):
                    op_name = op["name"]
                name_prefix = f"{op_name}, " if op_name else ""

                shift_store.record_production(
                    shift_id=active["shift_id"],
                    wall_number=wall_num,
                    print_file=action.params.get("print_file") or "current_job.gcode",
                    wall_height=float(action.params.get("wall_height", 3000.0)),
                    segment_layers=int(action.params.get("segment_layers", 50)),
                    layers_completed=int(action.params.get("layers_completed", 50)),
                    status=action.params.get("status", "completed"),
                )
                return f"{name_prefix}Wall {wall_num} production recorded successfully."
            except Exception as exc:
                log.error("Failed to record production: %s", exc)
                return f"Failed to record production: {exc}"

        if action.intent == "LOG_SHIFT_ISSUE":
            active = shift_store.get_active_shift_for_line(line_id)
            if not active:
                return f"Cannot log issue because no active shift is running on {line_id}."
            try:
                new_id = shift_store.log_issue(
                    shift_id=active["shift_id"],
                    issue_type=action.params.get("issue_type", "operational_issue"),
                    description=action.params.get("description", "Operator reported issue"),
                    wall_number=action.params.get("wall_number"),
                    severity=action.params.get("severity", "warning"),
                    downtime_minutes=float(action.params.get("downtime_minutes", 0.0)),
                    rag_recommendation=action.params.get("rag_recommendation"),
                    rag_verified=int(action.params.get("rag_verified", 0)),
                    rag_source=action.params.get("rag_source"),
                    action_taken=action.params.get("action_taken"),
                    result=action.params.get("result"),
                    cause=action.params.get("cause"),
                )
                self._session_ctx.update(action.session_id, line_id, last_issue_id=new_id, current_issue_id=new_id)
                op = shift_store.get_operator(active["operator_id"])
                name_prefix = f"{op['name']}, " if op and op.get("name") else ""

                rag_rec = action.params.get("rag_recommendation")
                rag_ver = action.params.get("rag_verified", 0)
                guidance = ""
                if rag_rec:
                    if rag_ver == 1:
                        guidance = f" Recommendation: {rag_rec}"
                    else:
                        guidance = " Note: No verified solution is available for this issue."

                action_taken = action.params.get("action_taken")
                result = action.params.get("result")
                details = []
                if action_taken:
                    details.append(f"action '{action_taken}'")
                if result:
                    details.append(f"status '{result}'")
                det_str = f" with {', '.join(details)}" if details else ""
                return f"{name_prefix}Issue #{new_id} logged successfully on {line_id}{det_str}.{guidance}"
            except Exception as exc:
                log.error("Failed to log issue: %s", exc)
                return f"Failed to log issue: {exc}"

        if action.intent == "LOG_SHIFT_HANDOVER":
            active = shift_store.get_active_shift_for_line(line_id)
            if not active:
                return f"Cannot initiate handover because no active shift is running on {line_id}."
            try:
                shift_store.create_handover(
                    from_shift_id=active["shift_id"],
                    line_id=line_id,
                    completed_work=action.params.get("completed_work", "Shift completed work"),
                    pending_work=action.params.get("pending_work", "Shift pending work"),
                    open_issues=action.params.get("open_issues", "None"),
                )
                return f"Shift handover initiated successfully for {line_id}."
            except Exception as exc:
                log.error("Failed to create handover: %s", exc)
                return f"Failed to initiate handover: {exc}"

        if action.intent == "RESOLVE_SHIFT_ISSUE":
            issue_id = action.params.get("target_issue_id")
            if not issue_id:
                return "No target issue ID specified to resolve."
            cause = action.params.get("cause")
            action_taken = action.params.get("action_taken")
            result = action.params.get("result")
            try:
                ok = shift_store.update_issue(
                    issue_id=issue_id,
                    cause=cause,
                    action_taken=action_taken,
                    result=result,
                )
                if not ok:
                    return f"Failed to update Issue #{issue_id}."
                if result == "resolved":
                    sess = self._session_ctx.get(action.session_id, line_id)
                    if sess.last_issue_id == issue_id or sess.current_issue_id == issue_id:
                        self._session_ctx.update(action.session_id, line_id, last_issue_id=None, current_issue_id=None)
                res_str = f" as {result}" if result else ""
                active = shift_store.get_active_shift_for_line(line_id)
                op = shift_store.get_operator(active["operator_id"]) if active else None
                name_prefix = f"{op['name']}, " if op and op.get("name") else ""
                return f"{name_prefix}Issue #{issue_id} updated successfully{res_str}."
            except Exception as exc:
                log.error("Failed to update issue: %s", exc)
                return f"Failed to update issue: {exc}"

        if action.intent == "START_PRINTING_JOB":
            j_name = action.params.get("job_name", "Print_Job_1")
            active = shift_store.get_active_shift_for_line(line_id)
            shift_id = active["shift_id"] if active else None
            op_id = active["operator_id"] if active else None
            try:
                job_id = shift_store.create_printing_job(
                    job_name=j_name,
                    operator_id=op_id,
                    shift_id=shift_id,
                    status="Running",
                )
                return f"Started printing job {j_name} (Job #{job_id})."
            except Exception as exc:
                log.error("Failed to start printing job: %s", exc)
                return f"Failed to start printing job: {exc}"

        if action.intent == "END_PRINTING_JOB":
            j_name = action.params.get("job_name", "Print_Job_1")
            try:
                jobs = shift_store.list_printing_jobs(status="Running")
                matching = next((j for j in jobs if j["job_name"].lower() == j_name.lower()), None)
                if not matching and jobs:
                    matching = jobs[0]
                if not matching:
                    all_jobs = shift_store.list_printing_jobs()
                    matching = next((j for j in all_jobs if j["job_name"].lower() == j_name.lower()), None)

                if not matching:
                    return f"Could not find printing job {j_name} to finish."

                shift_store.end_printing_job(matching["job_id"], status="Completed")
                updated = shift_store.get_printing_job(matching["job_id"])
                dur = updated.get("printing_duration_hours", 0.0)
                ot = updated.get("overtime_hours", 0.0)
                ot_str = f" with {ot} hours overtime" if ot > 0 else ""
                return f"Finished printing job {j_name}. Total duration was {dur} hours{ot_str}."
            except Exception as exc:
                log.error("Failed to finish printing job: %s", exc)
                return f"Failed to finish printing job: {exc}"

        return "Unknown action."

    # --- public ---

    def ask(
        self,
        question: str,
        session_id: str = "default_terminal",
        line_id: str = "line_1",
        operator_id: Optional[str] = None,
        routed: Optional[Any] = None,
    ) -> str:
        """Answers one question or handles two-step confirmed actions."""
        from agent import query_router

        question = (question or "").strip()
        if not question:
            return "I didn't catch a question."

        # 0. Check for pending action confirmations or awaiting interactive states
        pending = self._actions.get(session_id, line_id)
        sess = self._session_ctx.get(session_id, line_id)
        has_awaiting_state = bool(
            sess.awaiting_disambiguation
            or sess.awaiting_outcome
            or sess.awaiting_problem_details
            or sess.awaiting_action_for_issue
        )

        # 1. Primary autonomous Gemini Agentic Brain
        # When GEMINI_API_KEY is configured (self.use_llm is True) and there is no pending confirmation,
        # EVERY single user message flows directly through Gemini Agentic Brain with full DB/RAG tools.
        if self.use_llm and not pending and not has_awaiting_state:
            try:
                fut = _llm_executor.submit(
                    self._ask_llm,
                    question,
                    session_id=session_id,
                    routed=routed,
                    line_id=line_id,
                    operator_id=operator_id,
                )
                llm_reply = fut.result(timeout=30.0)
                if llm_reply:
                    return llm_reply
            except concurrent.futures.TimeoutError:
                log.warning("Agentic Gemini call exceeded 30.0s timeout; falling back to offline router.")
            except Exception as exc:
                log.warning("Agentic Gemini call failed (%s); falling back to offline router.", exc)

        # 2. Offline fallback path (used when self.use_llm is False or Gemini timed out / failed)

        # Snappy conversational reply for greetings/casual turns when offline
        if not pending and not has_awaiting_state:
            casual_reply = self._is_casual_or_greeting(question)
            if casual_reply:
                return casual_reply

        # Fast routing check for offline router
        if not pending and not has_awaiting_state and routed is None:
            context = {
                "line_id": line_id,
                "session_id": session_id,
                "operator_id": operator_id or sess.current_operator_id,
                "operator_name": sess.current_operator_name,
                "shift_id": sess.current_shift_id,
                "last_issue_id": sess.last_issue_id or sess.current_issue_id,
            }
            routed = query_router.route(question, session_id=session_id, context=context)

        # Open-ended conversational problem/issue queries:
        problem_starters = (
            "i have a problem", "there is a problem", "we have a problem",
            "i am facing a problem", "facing a problem", "facing an issue",
            "i have an issue", "there is an issue", "we have an issue",
            "having a problem", "having an issue", "help me with a problem",
            "help me with an issue", "i am having a problem", "problem aa rahi hai",
        )
        clean_norm = query_router.strip_wake_words(question).strip()
        clean_lower = clean_norm.lower().rstrip(".!?")
        if any(clean_lower == p or clean_lower.startswith(p + " ") for p in problem_starters):
            active = shift_store.get_active_shift_for_line(line_id)
            op = shift_store.get_operator(active["operator_id"]) if active else None
            name = op["name"] if op else None
            self._session_ctx.update(
                session_id,
                line_id,
                awaiting_problem_details=True,
                pending_problem_info=None,
                awaiting_action_for_issue=False,
            )
            prefix = f"{name}, " if name else "I'm here to help. "
            return f"{prefix}what problem are you facing?"

        # Pure self-introduction check: "I am Hamdaan", "Hamdaan here", "Main Hamdaan"
        intro_cand = identity.extract_name_candidate(clean_norm)
        # Check if question is purely introducing oneself without an action verb
        lower_q = clean_norm.lower()
        action_verbs = {"start", "end", "complete", "completed", "print", "printed", "wall", "issue", "problem", "record", "logged", "log", "chalu", "khatam", "band", "roko"}
        q_words = set(re.findall(r"\b\w+\b", lower_q))
        if intro_cand and not (q_words & action_verbs):
            res = identity.resolve_employee(intro_cand, active_only=True)
            if res.status == "MATCHED" and res.operator:
                op = res.operator
                self._session_ctx.update(
                    session_id,
                    line_id,
                    current_operator_id=op["operator_id"],
                    current_operator_name=op["name"],
                )
                return f"Hi {op['name']}. What would you like to do?"
            elif res.status == "AMBIGUOUS":
                return res.message

        # 1. Check for pending action confirmation
        pending = self._actions.get(session_id, line_id)
        if pending:
            clean_q = query_router.strip_wake_words(question).strip().lower()
            q_lower = clean_q or question.lower().strip()
            words = set(re.findall(r"\b\w+\b", q_lower))

            # Strict confirmation:
            # Only explicit YES/HAAN/CONFIRM executes
            # Only NO/NAHI/NOT rejects
            # Only CANCEL/STOP cancels
            # Do not treat "Okay Kelvin", "Sure", "Kar do", "Start karo" or unrelated speech as confirmation.
            affirmative_words = {"yes", "haan", "ha", "haanji", "confirm", "confirmed"}
            negative_words = {"no", "nahi", "na", "not", "dont", "don't"}
            cancel_words = {"cancel", "cancelled", "stop", "stopped", "roko"}

            uncertain_phrases = ("not sure", "dont know", "don't know", "no idea", "not clear", "pata nahi", "nahi pata", "samajh nahi")
            uncertain_words = {"unsure", "confused", "maybe", "shyad", "shayad"}

            is_uncertain = any(p in q_lower for p in uncertain_phrases) or bool(words & uncertain_words)

            has_cancel = bool(words & cancel_words)
            has_no = bool(words & negative_words)
            has_yes = bool(words & affirmative_words)

            if not clean_q or is_uncertain:
                is_yes = False
                is_no = False
                is_cancel = False
            elif has_yes and (has_no or has_cancel):
                # Conflicting / contradictory
                is_yes = False
                is_no = False
                is_cancel = False
            elif has_yes:
                is_yes = True
                is_no = False
                is_cancel = False
            elif has_cancel:
                is_yes = False
                is_no = False
                is_cancel = True
            elif has_no:
                is_yes = False
                is_no = True
                is_cancel = False
            else:
                is_yes = False
                is_no = False
                is_cancel = False

            # Check if this utterance provides missing fields (e.g., operator name)
            if pending.missing_fields and "operator_id" in pending.missing_fields:
                name_cand = identity.extract_name_candidate(question) or question.strip()
                res = identity.resolve_employee(name_cand, active_only=True)
                if res.status == "MATCHED" and res.operator:
                    matched_op = res.operator
                    pending.params["operator_id"] = matched_op["operator_id"]
                    pending.params["operator_name"] = matched_op["name"]
                    pending.missing_fields.remove("operator_id")
                    prompt = self._prepare_confirmation_prompt(pending.intent, pending.params)
                    pending.prompt_spoken = prompt
                    self._actions.refresh(session_id, line_id)
                    return prompt
                elif res.status == "AMBIGUOUS":
                    return res.message

            # Check if this utterance is a correction (e.g., "No, it was Anand")
            if any(p in q_lower for p in ("it was", "actually", "not me", "record for", "completed by")):
                name_cand = identity.extract_name_candidate(question)
                if name_cand:
                    res = identity.resolve_employee(name_cand, active_only=True)
                    if res.status == "MATCHED" and res.operator:
                        matched_op = res.operator
                        pending.params["operator_id"] = matched_op["operator_id"]
                        pending.params["operator_name"] = matched_op["name"]
                        prompt = self._prepare_confirmation_prompt(pending.intent, pending.params)
                        pending.prompt_spoken = prompt
                        self._actions.refresh(session_id, line_id)
                        return prompt
                    elif res.status == "AMBIGUOUS":
                        return res.message

            if is_yes and not (is_no or is_cancel):
                res = self._execute_action(pending)
                self._actions.clear(session_id, line_id, status="executed")
                return res
            elif is_cancel or (is_no and len(words) <= 3):
                self._actions.clear(session_id, line_id, status="cancelled")
                return "Action cancelled. No changes were made."
            else:
                test_intent, _ = query_router.classify(question)
                action_intent_names = {
                    query_router.Intent.SHIFT_START,
                    query_router.Intent.SHIFT_END,
                    query_router.Intent.LOG_WALL_PRODUCTION,
                    query_router.Intent.LOG_SHIFT_ISSUE,
                    query_router.Intent.LOG_SHIFT_HANDOVER,
                    query_router.Intent.RESOLVE_SHIFT_ISSUE,
                }
                if clean_q and test_intent not in action_intent_names and test_intent not in (query_router.Intent.UNKNOWN_QUERY, query_router.Intent.WAKE_WORD) and not is_uncertain:
                    log.info("Utterance %r classified as %s; bypassing pending action %s.", question, test_intent, pending.action_id)
                    pass
                else:
                    self._actions.refresh(session_id, line_id)
                    name = pending.params.get("operator_name")
                    name_prefix = f"{name}, " if name else ""
                    prompt = pending.prompt_spoken
                    m_shall = re.search(r"\b(Shall I [^?]+\?)", prompt, re.IGNORECASE)
                    if m_shall:
                        q_part = m_shall.group(1).strip()
                    else:
                        stripped_prompt = re.sub(rf"^{re.escape(name or '')},\s*", "", prompt, flags=re.IGNORECASE).strip()
                        q_part = stripped_prompt

                    please_part = "please answer Yes or No." if name_prefix else "Please answer Yes or No."
                    return f"{name_prefix}{please_part} {q_part}"

        # 1.1 Check awaiting_disambiguation
        sess = self._session_ctx.get(session_id, line_id)
        if sess.awaiting_disambiguation:
            disambig = sess.awaiting_disambiguation
            candidates = disambig["candidate_issues"]
            cand_ids = disambig["candidate_issue_ids"]
            pending_params = disambig["pending_params"]
            q_lower = question.lower().strip()
            words = set(re.findall(r"\b\w+\b", q_lower))

            cancel_terms = {"cancel", "stop", "roko"}
            if (words & cancel_terms) or "mat karo" in q_lower:
                self._session_ctx.update(session_id, line_id, awaiting_disambiguation=None)
                return "Action cancelled. No changes were made."

            if any(w in q_lower for w in ("both", "dono", "all", "sab", "dono ko")):
                return "Please resolve one issue at a time. Which issue number would you like to update?"

            selected_id = None
            m_id = query_router.extract_issue_id(question)
            if m_id and m_id in cand_ids:
                selected_id = m_id
            else:
                m_num = re.search(r"\b(\d+)\b", q_lower)
                if m_num and int(m_num.group(1)) in cand_ids:
                    selected_id = int(m_num.group(1))
                elif any(w in q_lower for w in ("first", "pehla", "pehli", "1st")):
                    selected_id = candidates[0]["issue_id"]
                elif len(candidates) > 1 and any(w in q_lower for w in ("second", "dusra", "dusri", "2nd")):
                    selected_id = candidates[1]["issue_id"]
                else:
                    matched = [c for c in candidates if any(w in c.get("description", "").lower() for w in q_lower.split())]
                    if len(matched) == 1:
                        selected_id = matched[0]["issue_id"]

            if not selected_id:
                cand_list = ", ".join(f"Issue #{c['issue_id']} ({c.get('description') or c.get('issue_type')})" for c in candidates)
                return f"Please specify which issue to update: {cand_list}."

            self._session_ctx.update(session_id, line_id, awaiting_disambiguation=None)
            pending_params["target_issue_id"] = selected_id
            action_taken = pending_params.get("action_taken")
            result = pending_params.get("result")
            cause = pending_params.get("cause")

            if action_taken and not result:
                self._session_ctx.update(
                    session_id,
                    line_id,
                    awaiting_outcome={
                        "target_issue_id": selected_id,
                        "action_taken": action_taken,
                        "cause": cause,
                        "params": pending_params,
                    },
                )
                return f"Recorded action '{action_taken}' for Issue #{selected_id}. Did this resolve the issue and resume printing?"

            if not result and not action_taken:
                return f"What would you like to do with Issue #{selected_id}?"

            prompt = self._prepare_confirmation_prompt("RESOLVE_SHIFT_ISSUE", pending_params)
            self._actions.set(session_id, line_id, "RESOLVE_SHIFT_ISSUE", pending_params, prompt)
            return prompt

        # 1.2 Check awaiting_outcome
        if sess.awaiting_outcome:
            out_info = sess.awaiting_outcome
            q_lower = question.lower().strip()
            words = set(re.findall(r"\b\w+\b", q_lower))

            cancel_terms = {"cancel", "stop", "roko"}
            if (words & cancel_terms) or "mat karo" in q_lower:
                self._session_ctx.update(session_id, line_id, awaiting_outcome=None)
                return "Action cancelled. No changes were made."

            if any(w in q_lower for w in ("agar", "if ", "will", "hoga", "jayega", "dekhte", "check karte", "might")):
                self._session_ctx.update(session_id, line_id, awaiting_outcome=None)
                return "Understood. Let me know once the outcome is verified."

            affirms = {"yes", "haan", "ha", "confirm", "resolved", "solve", "theek", "thik", "resumed", "resume"}
            negatives = {"no", "nahi", "not resolved", "still stuck", "abhi bhi", "fail"}
            res_val = query_router.extract_resolution_result(question)
            if res_val == "resolved":
                is_yes = True
                is_no = False
            elif res_val == "not_resolved":
                is_yes = False
                is_no = True
            else:
                is_yes = bool(words & affirms) or any(w in q_lower for w in ("solve ho gaya", "thik ho gaya", "printing resume"))
                is_no = bool(words & negatives) or any(w in q_lower for w in ("not resolved", "abhi bhi dikkat", "thik nahi"))

            target_id = out_info["target_issue_id"]
            action_taken = out_info["action_taken"]
            cause = out_info.get("cause")
            params = out_info.get("params", {})
            params["target_issue_id"] = target_id
            params["action_taken"] = action_taken
            params["cause"] = cause
            self._session_ctx.update(session_id, line_id, awaiting_outcome=None)

            if is_yes and not is_no:
                params["result"] = "resolved"
            elif is_no:
                params["result"] = "not_resolved"
            else:
                return "Please indicate whether the issue was resolved or is still unresolved."

            prompt = self._prepare_confirmation_prompt("RESOLVE_SHIFT_ISSUE", params)
            self._actions.set(session_id, line_id, "RESOLVE_SHIFT_ISSUE", params, prompt)
            return prompt

        # 1.3 Check awaiting_problem_details
        if sess.awaiting_problem_details:
            clean_q = query_router.strip_wake_words(question).strip().lower()
            q_lower = clean_q or question.lower().strip()
            words = set(re.findall(r"\b\w+\b", q_lower))
            cancel_terms = {"cancel", "stop", "roko"}
            if (words & cancel_terms) or "mat karo" in q_lower:
                self._session_ctx.update(
                    session_id,
                    line_id,
                    awaiting_problem_details=False,
                    pending_problem_info=None,
                    awaiting_action_for_issue=False,
                )
                return "Action cancelled. No changes were made."

            active = shift_store.get_active_shift_for_line(line_id)
            if not active:
                self._session_ctx.update(session_id, line_id, awaiting_problem_details=False)
                return f"Cannot report issue because no active shift is running on {line_id}."

            op = shift_store.get_operator(active["operator_id"]) if active else None
            name = op["name"] if op else (active["operator_id"] if active else "Operator")

            problem_desc = question.strip()
            k = search_documentation(problem_desc)
            rag_rec = None
            rag_src = None
            rag_ver = 0
            if k.get("found") and k.get("results"):
                for res in k["results"]:
                    if query_router._evaluate_rag_verified(res.get("text", "")) == 1:
                        rag_ver = 1
                        rag_src = res.get("source")
                        single_doc = dict(k)
                        single_doc["results"] = [res]
                        spoken = query_router.speak_knowledge(single_doc, problem_desc)
                        rag_rec = spoken or res.get("text", "")
                        break

            issue_type = "operational_issue"
            p_low = problem_desc.lower()
            if "material flow" in p_low or "material slope" in p_low:
                issue_type = "material_flow_issue"
            elif "nozzle" in p_low:
                issue_type = "nozzle_clog"
            elif "pump" in p_low:
                issue_type = "pump_pressure"
            elif "feeder" in p_low:
                issue_type = "feeder_jam"
            elif "sensor" in p_low:
                issue_type = "sensor_fault"

            problem_info = {
                "problem": problem_desc,
                "issue_type": issue_type,
                "rag_recommendation": rag_rec,
                "rag_verified": rag_ver,
                "rag_source": rag_src,
            }
            self._session_ctx.update(
                session_id,
                line_id,
                awaiting_problem_details=False,
                pending_problem_info=problem_info,
                awaiting_action_for_issue=True,
            )
            if rag_ver == 1 and rag_rec:
                return f"{name}, I found a site-verified solution: {rag_rec} Please tell me what action you took."
            else:
                return f"{name}, no site-verified solution was found for this problem. Please tell me what action you took."

        # 1.4 Check awaiting_action_for_issue
        if sess.awaiting_action_for_issue and sess.pending_problem_info:
            clean_q = query_router.strip_wake_words(question).strip().lower()
            q_lower = clean_q or question.lower().strip()
            words = set(re.findall(r"\b\w+\b", q_lower))
            cancel_terms = {"cancel", "stop", "roko"}
            if (words & cancel_terms) or "mat karo" in q_lower:
                self._session_ctx.update(session_id, line_id, awaiting_action_for_issue=False, pending_problem_info=None)
                return "Action cancelled. No changes were made."

            active = shift_store.get_active_shift_for_line(line_id)
            if not active:
                self._session_ctx.update(session_id, line_id, awaiting_action_for_issue=False, pending_problem_info=None)
                return f"Cannot log issue because no active shift is running on {line_id}."

            op = shift_store.get_operator(active["operator_id"]) if active else None
            name = op["name"] if op else (active["operator_id"] if active else "Operator")

            action_val, _ = query_router.extract_operator_action(question)
            if not action_val:
                action_val = question.strip()
            res_val = query_router.extract_resolution_result(question)
            cause_val = query_router.extract_resolution_cause(question)

            info = sess.pending_problem_info
            problem_text = info["problem"]
            rag_ver = info.get("rag_verified", 0)
            rag_rec = info.get("rag_recommendation") if rag_ver == 1 else None
            rag_src = info.get("rag_source") if rag_ver == 1 else None

            params = {
                "action": "LOG_SHIFT_ISSUE",
                "line_id": line_id,
                "operator_id": active["operator_id"],
                "operator_name": name,
                "shift_id": active["shift_id"],
                "issue_type": info.get("issue_type", "operational_issue"),
                "description": problem_text,
                "action_taken": action_val,
                "result": res_val,
                "cause": cause_val,
                "rag_recommendation": rag_rec,
                "rag_verified": rag_ver,
                "rag_source": rag_src,
            }
            self._session_ctx.update(session_id, line_id, awaiting_action_for_issue=False, pending_problem_info=None)
            prompt = self._prepare_confirmation_prompt("LOG_SHIFT_ISSUE", params)
            self._actions.set(session_id, line_id, "LOG_SHIFT_ISSUE", params, prompt)
            return prompt

        # 1.5 Check for explicit issue ID alone (without resolution or action language)
        explicit_id = query_router.extract_issue_id(question)
        action_taken, rag_rec = query_router.extract_operator_action(question)
        cause = query_router.extract_resolution_cause(question)
        result = query_router.extract_resolution_result(question)
        has_res_action = bool(action_taken or rag_rec or result or any(w in question.lower() for w in ("resolve", "resolved", "solve", "theek", "thik", "fix", "close", "khatam", "band", "done")))

        if explicit_id is not None and not has_res_action:
            active = shift_store.get_active_shift_for_line(line_id)
            if not active:
                return f"Cannot access issue because no active shift is running on {line_id}."
            shift_issues = shift_store.list_shift_issues(shift_id=active["shift_id"], limit=50)
            target_issue = next((iss for iss in shift_issues if iss["issue_id"] == explicit_id), None)
            if not target_issue:
                all_issues = shift_store.list_shift_issues(limit=50)
                target_issue = next((iss for iss in all_issues if iss["issue_id"] == explicit_id), None)
                if not target_issue:
                    return f"Issue #{explicit_id} not found."
                if target_issue["shift_id"] != active["shift_id"]:
                    return f"Issue #{explicit_id} belongs to a different shift and cannot be modified."
            if target_issue.get("result") == "resolved":
                return f"Issue #{explicit_id} is already resolved."
            self._session_ctx.update(session_id, line_id, last_issue_id=explicit_id)
            return f"What would you like to do with Issue #{explicit_id}?"

        # 2. Check if the question is an action intent
        if explicit_id is not None and has_res_action:
            classified_intent = query_router.Intent.RESOLVE_SHIFT_ISSUE
        elif routed is not None and hasattr(routed, "intent"):
            classified_intent = routed.intent
        else:
            classified_intent, _ = query_router.classify(question)
        action_intents = {
            query_router.Intent.SHIFT_START,
            query_router.Intent.SHIFT_END,
            query_router.Intent.LOG_WALL_PRODUCTION,
            query_router.Intent.LOG_SHIFT_ISSUE,
            query_router.Intent.LOG_SHIFT_HANDOVER,
            query_router.Intent.RESOLVE_SHIFT_ISSUE,
        }

        # Route action statements and resolution statements to RESOLVE_SHIFT_ISSUE
        # when an active/current unresolved issue exists, without requiring an explicit issue ID
        if classified_intent not in action_intents and (action_taken or result or rag_rec):
            active_shift = shift_store.get_active_shift_for_line(line_id)
            if active_shift:
                unresolved = shift_store.list_shift_issues(shift_id=active_shift["shift_id"], unresolved_only=True)
                if unresolved or (sess.last_issue_id and any(iss["issue_id"] == sess.last_issue_id for iss in shift_store.list_shift_issues(shift_id=active_shift["shift_id"]))):
                    classified_intent = query_router.Intent.RESOLVE_SHIFT_ISSUE

        if classified_intent in action_intents:
            # The router extracts the name; resolving it is the assistant's job,
            # because only the assistant talks to the operators table.
            if routed is None or getattr(routed, "intent", None) != classified_intent:
                routed = query_router.resolve(classified_intent, question)

            if classified_intent == query_router.Intent.LOG_WALL_PRODUCTION:
                wn = routed.facts.get("wall_number") if getattr(routed, "facts", None) else None
                if wn is None:
                    return "Which wall number was completed? Please specify the wall number to record production."

            if classified_intent == query_router.Intent.LOG_SHIFT_ISSUE:
                clean_q = query_router.strip_wake_words(question).strip().lower().rstrip(".!?")
                if clean_q in ("i have a problem", "there is a problem", "we have a problem") or any(clean_q == p or clean_q.startswith(p + " ") for p in problem_starters):
                    active = shift_store.get_active_shift_for_line(line_id)
                    op = shift_store.get_operator(active["operator_id"]) if active else None
                    name = op["name"] if op else None
                    self._session_ctx.update(
                        session_id,
                        line_id,
                        awaiting_problem_details=True,
                        pending_problem_info=None,
                        awaiting_action_for_issue=False,
                    )
                    prefix = f"{name}, " if name else "I'm here to help. "
                    return f"{prefix}what problem are you facing?"

            if classified_intent == query_router.Intent.RESOLVE_SHIFT_ISSUE:
                params = dict(routed.facts)
                target_id = params.get("target_issue_id") or explicit_id
                domain_kws = params.get("domain_keywords", []) or query_router.extract_domain_keywords(question)
                action_taken = params.get("action_taken") or action_taken
                rag_recommended = params.get("rag_recommended", False) or rag_rec
                cause = params.get("cause") or cause
                result = params.get("result") or result

                active = shift_store.get_active_shift_for_line(line_id)
                if not active:
                    return f"Cannot resolve issue because no active shift is running on {line_id}."

                unresolved = shift_store.list_shift_issues(shift_id=active["shift_id"], unresolved_only=True)

                selected_id = None
                # 1. Explicit issue ID from question (must exist and not be already resolved)
                if target_id:
                    shift_issues = shift_store.list_shift_issues(shift_id=active["shift_id"], limit=50)
                    target_issue = next((iss for iss in shift_issues if iss["issue_id"] == target_id), None)
                    if not target_issue:
                        all_issues = shift_store.list_shift_issues(limit=50)
                        target_issue = next((iss for iss in all_issues if iss["issue_id"] == target_id), None)
                        if not target_issue:
                            return f"Issue #{target_id} not found."
                        if target_issue["shift_id"] != active["shift_id"]:
                            return f"Issue #{target_id} belongs to a different shift and cannot be modified."

                    if target_issue.get("result") == "resolved":
                        return f"Issue #{target_id} is already resolved."
                    selected_id = target_id

                # 2. Session last_issue_id (only if on active shift and unresolved)
                if not selected_id and sess.last_issue_id:
                    if any(iss["issue_id"] == sess.last_issue_id for iss in unresolved):
                        selected_id = sess.last_issue_id

                # 3. Unique domain keyword match
                if not selected_id and domain_kws and unresolved:
                    matches = [iss for iss in unresolved if any(kw in iss.get("description", "").lower() for kw in domain_kws)]
                    if len(matches) == 1:
                        selected_id = matches[0]["issue_id"]

                # 4. Exactly one unresolved issue on active shift
                if not selected_id:
                    if len(unresolved) == 1:
                        selected_id = unresolved[0]["issue_id"]
                    elif len(unresolved) > 1:
                        cand_ids = [iss["issue_id"] for iss in unresolved]
                        sess.awaiting_disambiguation = {
                            "candidate_issues": unresolved,
                            "candidate_issue_ids": cand_ids,
                            "pending_params": params,
                        }
                        self._session_ctx.update(session_id, line_id, awaiting_disambiguation=sess.awaiting_disambiguation)
                        cand_strs = ", ".join(f"Issue #{iss['issue_id']} ({iss.get('description') or iss.get('issue_type')})" for iss in unresolved)
                        return f"Multiple unresolved issues found on this shift: {cand_strs}. Which issue would you like to update?"
                    else:
                        return f"No unresolved issues found on the current shift for {line_id}."

                params["target_issue_id"] = selected_id
                if rag_recommended:
                    if sess.last_rag_recommendation:
                        action_taken = f"Applied Kelvin recommendation: {sess.last_rag_recommendation}"
                    else:
                        action_taken = "Applied Kelvin recommendation"
                    params["action_taken"] = action_taken

                if action_taken and not result:
                    self._session_ctx.update(
                        session_id,
                        line_id,
                        awaiting_outcome={
                            "target_issue_id": selected_id,
                            "action_taken": action_taken,
                            "cause": cause,
                            "params": params,
                        },
                    )
                    return f"Recorded action '{action_taken}' for Issue #{selected_id}. Did this resolve the issue and resume printing?"

                if not result and not action_taken:
                    return f"What would you like to do with Issue #{selected_id}?"

                prompt = self._prepare_confirmation_prompt("RESOLVE_SHIFT_ISSUE", params)
                self._actions.set(session_id, line_id, "RESOLVE_SHIFT_ISSUE", params, prompt)
                return prompt
            # Determine operator candidate for this action
            spoken_name = (routed.facts or {}).get("spoken_name") or identity.extract_name_candidate(question)
            op = None
            if operator_id:
                op = shift_store.get_operator(operator_id)
            if not op and spoken_name:
                res = identity.resolve_employee(spoken_name, active_only=True)
                if res.status == "MATCHED":
                    op = res.operator
                elif res.status == "AMBIGUOUS":
                    return res.message
            if not op and sess.current_operator_id:
                op = shift_store.get_operator(sess.current_operator_id)

            # For actions inherently tied to the active shift on the line (issue logging, shift ending, handover),
            # if no operator was explicitly specified, use the operator of the active shift on this line.
            if not op and classified_intent in (
                query_router.Intent.LOG_SHIFT_ISSUE,
                query_router.Intent.SHIFT_END,
                query_router.Intent.LOG_SHIFT_HANDOVER,
            ):
                active_s = shift_store.get_active_shift_for_line(line_id)
                if active_s and active_s.get("operator_id"):
                    op = shift_store.get_operator(active_s["operator_id"])

            if classified_intent == query_router.Intent.LOG_WALL_PRODUCTION:
                if not op:
                    # Do NOT auto-attribute to whoever happens to have an active shift!
                    # Ask the user explicitly.
                    params = dict(routed.facts)
                    params["line_id"] = line_id
                    prompt = "Which employee should I record this work for?"
                    self._actions.set(
                        session_id,
                        line_id,
                        classified_intent.value,
                        params,
                        prompt,
                        missing_fields=["operator_id"],
                        original_request=question,
                    )
                    return prompt

            # Speech-to-text can split a name apart ("Anand here" arrived as
            # "On and Here"), leaving normal extraction with nothing. For a
            # shift start where the operator DID introduce themselves, try to
            # recover them from the name slot -- unique match only.
            name_offered = (
                not op
                and not spoken_name
                and classified_intent == query_router.Intent.SHIFT_START
                and bool(shift_store.name_slot_candidates(question))
            )
            if name_offered:
                op = shift_store.find_operator_in_text(question)
                if not op:
                    return "I didn't catch your name. Please say your name again and start your shift."

            if not op:
                if spoken_name and classified_intent == query_router.Intent.SHIFT_START:
                    inactive_op = shift_store.find_operator_by_name(spoken_name, active_only=False)
                    if inactive_op and not inactive_op.get("active"):
                        return (
                            f"I could not find an operator named {spoken_name.title()}. "
                            "No shift was started. Please check the name or register "
                            "the operator first."
                        )
                    params = dict(routed.facts)
                    params["create_operator"] = True
                    params["operator_name"] = spoken_name.title()
                    params["line_id"] = line_id
                    prompt = self._prepare_confirmation_prompt(classified_intent.value, params)
                    self._actions.set(session_id, line_id, classified_intent.value, params, prompt)
                    return prompt

                if spoken_name:
                    return (
                        f"I could not find an operator named {spoken_name.title()}. "
                        "No shift was started. Please check the name or register "
                        "the operator first."
                    )
                return "No active operator is registered on this line. Please select your operator profile on the terminal."

            params = dict(routed.facts)
            params["operator_id"] = op["operator_id"]
            params["operator_name"] = op.get("name") or op["operator_id"]
            params["line_id"] = line_id
            prompt = self._prepare_confirmation_prompt(classified_intent.value, params)
            self._actions.set(session_id, line_id, classified_intent.value, params, prompt)
            return prompt

        # 3. Deterministic factual intents (resolved directly without Claude/Gemini)
        deterministic_factual_intents = {
            query_router.Intent.GET_COMPLETED_PRODUCTION,
            query_router.Intent.GET_REMAINING_PRODUCTION,
            query_router.Intent.GET_TOTAL_TARGET,
            query_router.Intent.GET_PRINT_RATE,
            query_router.Intent.GET_CURRENT_LAYER,
            query_router.Intent.GET_LAYER_WIDTH,
            query_router.Intent.GET_LAYER_HEIGHT,
            query_router.Intent.GET_WALL_DIMENSIONS,
            query_router.Intent.GET_MATERIAL_USAGE,
            query_router.Intent.GET_MATERIAL_REMAINING,
            query_router.Intent.GET_MATERIAL_RATE,
            query_router.Intent.GET_ELAPSED_TIME,
            query_router.Intent.GET_ETA,
            query_router.Intent.GET_DEADLINE_STATUS,
            query_router.Intent.GET_24_HOUR_PRODUCTION,
            query_router.Intent.GET_SHIFT_PRODUCTION,
            query_router.Intent.GET_OPERATOR,
            query_router.Intent.GET_MACHINE_STATUS,
            query_router.Intent.GET_ROBOT_STATUS,
            query_router.Intent.GET_CAMERA_STATUS,
            query_router.Intent.GET_COMMUNICATION_STATUS,
            query_router.Intent.GET_PRINT_QUALITY,
            query_router.Intent.GET_RECENT_PROBLEMS,
            query_router.Intent.GET_RECENT_EVENTS,
            query_router.Intent.WAKE_WORD,
            query_router.Intent.GET_ACTIVE_SHIFT,
            query_router.Intent.GET_PREVIOUS_SHIFT,
            query_router.Intent.GET_SHIFT_BY_DATE,
            query_router.Intent.GET_SHIFT_BY_OPERATOR,
            query_router.Intent.GET_WALL_HISTORY,
            query_router.Intent.GET_SHIFT_ISSUES,
            query_router.Intent.GET_HANDOVER_STATUS,
            query_router.Intent.GET_SITE_PEOPLE_COUNT,
            query_router.Intent.GET_SITE_DETECTIONS,
        }
        if not self.use_llm and classified_intent in deterministic_factual_intents:
            active = shift_store.get_active_shift_for_line(line_id)
            context = {
                "session_id": session_id,
                "line_id": line_id,
                "operator_id": operator_id or (active.get("operator_id") if active else None) or sess.current_operator_id,
                "shift_id": (active.get("shift_id") if active else None) or sess.current_shift_id,
                "last_issue_id": sess.last_issue_id or sess.current_issue_id,
            }
            if (
                routed is None
                or getattr(routed, "intent", None) != classified_intent
                or (getattr(routed, "facts", None) and routed.facts.get("needs_clarification") and routed.facts.get("clarification_type") == "operator")
            ):
                routed = query_router.route(question, context=context)
            return query_router.summarise(routed)

        # 4. Standard queries (LLM or deterministic router)
        if self.use_llm:
            answer = self._ask_llm(
                question,
                session_id=session_id,
                routed=routed,
                line_id=line_id,
                operator_id=operator_id,
            )
            if answer:
                return answer
            log.info("Falling back to the local router for this question.")
        answer = self._answer_locally(question, routed=routed)
        if routed and getattr(routed, "intent", None) == query_router.Intent.KNOWLEDGE_QUERY:
            if getattr(routed, "knowledge", None) and routed.knowledge.get("found"):
                self._session_ctx.update(session_id, line_id, last_rag_recommendation=query_router.summarise(routed))
        return answer

    def ask_stream(
        self,
        question: str,
        session_id: str = "default_terminal",
        line_id: str = "line_1",
        operator_id: Optional[str] = None,
        routed: Optional[Any] = None,
    ):
        """
        Streaming generator: yields text chunks as they are generated.
        Ensures instantaneous time-to-first-token and snappy conversational replies.
        """
        question = (question or "").strip()
        if not question:
            yield "I didn't catch a question."
            return

        pending = self._actions.get(session_id, line_id)
        sess = self._session_ctx.get(session_id, line_id)
        has_awaiting_state = bool(
            sess.awaiting_disambiguation
            or sess.awaiting_outcome
            or sess.awaiting_problem_details
            or sess.awaiting_action_for_issue
        )

        if pending or has_awaiting_state:
            yield self.ask(question, session_id=session_id, line_id=line_id, operator_id=operator_id, routed=routed)
            return

        casual = self._is_casual_or_greeting(question)
        if casual:
            yield casual
            return

        if self.use_llm:
            try:
                streamed_any = False
                for chunk in self._ask_llm_stream(question, session_id=session_id):
                    streamed_any = True
                    yield chunk
                if streamed_any:
                    return
            except Exception as exc:
                log.warning("Streaming LLM call failed (%s); using fallback.", exc)

        yield self.ask(question, session_id=session_id, line_id=line_id, operator_id=operator_id, routed=routed)

    def ask_and_speak(
        self,
        question: str,
        session_id: str = "default_terminal",
        line_id: str = "line_1",
        operator_id: Optional[str] = None,
        routed: Optional[Any] = None,
    ) -> str:
        """Answers and speaks the reply; speech failure never breaks the answer."""
        answer = self.ask(question, session_id=session_id, line_id=line_id, operator_id=operator_id, routed=routed)
        try:
            from agent.voice import speak

            speak(answer)
        except Exception as exc:
            log.warning("Text-to-speech unavailable: %s", exc)
        return answer


def tool_debug(question: str) -> Dict[str, Any]:
    """Which tool a question routes to, and what it returns. Used by tests/CLI."""
    tool = Assistant.route(question)
    payload = search_docs(question) if tool == "search_docs" else _TOOLS_BY_NAME[tool]()
    return {"question": question, "tool": tool, "result": payload}


if __name__ == "__main__":  # quick manual check
    print(json.dumps(tool_debug("kitni wall print ho gayi hai?"), indent=2, default=str))
