"""
agent/shift_store.py
Data access layer for Phase 2B implementing the FINAL VALIDATED Phase 2A schema.
Preserves exact table names, field names, constraints, foreign keys, and indexes.
"""

import datetime
import difflib
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Union
from agent import db
from agent.logging_setup import get_logger

log = get_logger("ShiftStore")


def _row_to_dict(row: Optional[Any]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None


def _parse_dt(val: Any) -> Optional[datetime.datetime]:
    if not val:
        return None
    if isinstance(val, (int, float)):
        return datetime.datetime.fromtimestamp(val)
    s = str(val).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        return datetime.datetime.fromisoformat(s)
    except Exception:
        return None


def _format_hh_mm(val: Any) -> Optional[str]:
    dt = _parse_dt(val)
    if not dt:
        return None
    return dt.strftime("%H:%M")


# ===========================================================================
# 1. Operators
# ===========================================================================

def create_operator(
    operator_id: Optional[Any] = None,
    name: Optional[str] = None,
    employee_code: Optional[str] = None,
    role: Any = "Operator",
    active: int = 1,
    status: Optional[str] = None,
    db_path: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Creates a new operator record. Primary key: operator_id."""
    # If role was passed as integer (e.g. active flag):
    if isinstance(role, int):
        active = role
        role = "Operator"

    # Support positional name if operator_id is passed as name:
    if operator_id is not None and name is None:
        if isinstance(operator_id, str) and not operator_id.startswith("OP_") and not any(ch.isdigit() for ch in operator_id):
            name = operator_id
            operator_id = None
        else:
            name = str(operator_id)

    if not name or not str(name).strip():
        raise ValueError("name cannot be empty")
    clean_name = str(name).strip()
    clean_emp = employee_code.strip() if employee_code else None
    if active not in (0, 1):
        raise ValueError("active must be 0 or 1")
    op_status = status or ("Active" if active == 1 else "Inactive")
    op_active = active if status is None else (1 if str(op_status).lower() == "active" else 0)
    clean_role = str(role).strip() if role else "Operator"

    now = time.time()
    with db.connect(db_path) as conn:
        if operator_id is None:
            row = conn.execute(
                "SELECT COALESCE(MAX(CAST(operator_id AS INTEGER)), 0) AS max_id FROM operators"
            ).fetchone()
            candidate_id = max(1, int(row["max_id"]) + 1)
            while conn.execute("SELECT 1 FROM operators WHERE operator_id = ? OR CAST(operator_id AS TEXT) = ?", (candidate_id, str(candidate_id))).fetchone():
                candidate_id += 1
            clean_id = candidate_id
        else:
            clean_id = operator_id.strip() if isinstance(operator_id, str) else operator_id

        conn.execute(
            """
            INSERT INTO operators (operator_id, name, employee_code, role, status, active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (clean_id, clean_name, clean_emp, clean_role, op_status, op_active, now),
        )
    log.info("Created operator: %s (%s)", clean_id, clean_name)
    return clean_id


def get_operator(operator_id: Any, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves an operator by operator_id, employee_code, or name."""
    if operator_id is None:
        return None
    with db.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM operators
            WHERE operator_id = ?
               OR CAST(operator_id AS TEXT) = ?
               OR employee_code = ?
               OR name = ? COLLATE NOCASE
            LIMIT 1
            """,
            (operator_id, str(operator_id), str(operator_id), str(operator_id)),
        ).fetchone()
    return _row_to_dict(row)


def find_operator_by_name(
    name: str, active_only: bool = True, db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Resolves a SPOKEN name to an operator record, or None.

    Speech gives a name, not an id, so "Anand here" has to reach the operators
    table by name. Two passes, most specific first:

      1. the whole spoken name against the whole stored name, case-insensitive
      2. the spoken name against the stored FIRST name -- operators are stored
         as "Anand Sharma" but nobody says their surname on the shop floor

    Returns None when the name matches nothing, and also when it matches more
    than one operator: two people called Anand must not be silently collapsed
    into whichever row came back first. The caller reports the failure rather
    than guessing, so an unresolved name can never start a shift.
    """
    clean = (name or "").strip().lower()
    if not clean:
        return None

    with db.connect(db_path) as conn:
        sql = "SELECT * FROM operators"
        if active_only:
            sql += " WHERE (active = 1 OR status = 'Active')"
        rows = [dict(r) for r in conn.execute(sql)]

    exact = [r for r in rows if (r.get("name") or "").strip().lower() == clean]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        log.warning("Spoken name %r matches %d operators; refusing to guess.", name, len(exact))
        return None

    first = [
        r for r in rows
        if (r.get("name") or "").strip().lower().split(" ")[0] == clean
    ]
    if len(first) == 1:
        return first[0]
    if len(first) > 1:
        log.warning("Spoken first name %r matches %d operators; refusing to guess.", name, len(first))
    return None


# --- recovering a name speech-to-text broke apart ---------------------------
#
# Confirmed on the shop floor: "Anand here, start my shift" came back from
# speech-to-text as "On and Here Start My Shift." The name was split into two
# words, one of which ("and") is rightly rejected as not-a-name, so normal
# extraction returned nothing and the operator could not start their shift.
#
# The recovery below looks ONLY at the name slot of a self-introduction -- the
# word or two beside "here", "speaking", "this is" and similar -- never the
# whole sentence, so ordinary words elsewhere cannot be mistaken for a name.

# Words that end a self-introduction: "<name> here", "<name> speaking",
# "<name> bol raha hai", "main <name> hoon".
_NAME_BEFORE = ("here", "speaking", "bol", "hoon", "hu")
# Words that begin one: "this is <name>", "main <name>".
_NAME_AFTER = ("main",)
# Never a name, even when it sits in the slot.
_NOT_OPERATOR = {"okay", "ok", "kelvin", "and", "the", "is", "am", "my", "me", "i"}

# How close a heard fragment must be to an operator's name. Fitted to the
# confirmed case -- "onand" against "anand" scores 0.80 -- while keeping plain
# words out: "and" 0.75, "hand" 0.67, "stand" 0.60 all fall short.
NAME_MATCH_THRESHOLD = 0.80


def name_slot_candidates(text: str) -> List[str]:
    """
    Fragments in the name position of a self-introduction, most likely first.

    Each slot yields the single word beside the anchor and that word joined to
    its neighbour, because speech-to-text most often breaks a name into two:
    "on and here" gives ["and", "onand"]. Empty when the sentence contains no
    self-introduction at all -- which is how the caller tells "no name was
    offered" apart from "a name was offered but not understood".
    """
    words = re.findall(r"[a-z]+", (text or "").lower())
    found: List[str] = []

    def add(fragment: str) -> None:
        if len(fragment) >= 3 and fragment not in _NOT_OPERATOR and fragment not in found:
            found.append(fragment)

    for i, word in enumerate(words):
        if word in _NAME_BEFORE and i >= 1:
            add(words[i - 1])
            if i >= 2:
                add(words[i - 2] + words[i - 1])
        if word in _NAME_AFTER and i + 1 < len(words):
            add(words[i + 1])
            if i + 2 < len(words):
                add(words[i + 1] + words[i + 2])
        if word == "this" and i + 2 < len(words) and words[i + 1] == "is":
            add(words[i + 2])
            if i + 3 < len(words):
                add(words[i + 2] + words[i + 3])
    return found


def find_operator_in_text(
    text: str, active_only: bool = True, db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Recovers the ONE active operator a noisy self-introduction refers to.

    Compares each name-slot fragment against every operator's first name and
    full name (spaces removed). Returns the operator only when exactly one
    operator matches; two or more is ambiguous and returns None, as does no
    match -- an attendance record must never be written for a guessed person.
    The caller still reads the resolved full name back to the operator for
    confirmation, so a wrong recovery is heard and refused before any write.
    """
    candidates = name_slot_candidates(text)
    if not candidates:
        return None

    with db.connect(db_path) as conn:
        sql = "SELECT * FROM operators"
        if active_only:
            sql += " WHERE active = 1"
        rows = [dict(r) for r in conn.execute(sql)]

    matched: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        name = (row.get("name") or "").strip().lower()
        if not name:
            continue
        targets = {name.split(" ")[0], re.sub(r"[^a-z]", "", name)}
        for fragment in candidates:
            if any(
                difflib.SequenceMatcher(None, fragment, target).ratio() >= NAME_MATCH_THRESHOLD
                for target in targets
                if len(target) >= 3
            ):
                matched[row["operator_id"]] = row
                break

    if len(matched) == 1:
        operator = next(iter(matched.values()))
        log.info("Recovered operator %s from noisy transcript %r", operator["operator_id"], text)
        return operator
    if len(matched) > 1:
        log.warning(
            "Transcript %r could be any of %s; refusing to guess.", text, sorted(matched)
        )
    return None


def get_operator_by_employee_code(employee_code: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves an operator by unique employee_code."""
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM operators WHERE employee_code = ?",
            (employee_code.strip(),),
        ).fetchone()
    return _row_to_dict(row)


def list_operators(active_only: bool = True, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists operators, optionally filtering by active."""
    query = "SELECT * FROM operators"
    params: List[Any] = []
    if active_only:
        query += " WHERE active = 1"
    query += " ORDER BY name ASC"

    with db.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def update_operator(
    operator_id: str,
    name: Optional[str] = None,
    employee_code: Optional[str] = None,
    role: Optional[str] = None,
    active: Optional[int] = None,
    db_path: Optional[str] = None,
) -> bool:
    """Updates mutable operator fields."""
    sets, params = [], []
    if name is not None:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("name cannot be empty")
        sets.append("name = ?")
        params.append(clean_name)
    if employee_code is not None:
        sets.append("employee_code = ?")
        params.append(employee_code.strip() if employee_code else None)
    if role is not None:
        sets.append("role = ?")
        params.append(role.strip())
    if active is not None:
        if active not in (0, 1):
            raise ValueError("active must be 0 or 1")
        sets.append("active = ?")
        params.append(active)

    if not sets:
        return False

    params.append(operator_id.strip())
    with db.connect(db_path) as conn:
        cur = conn.execute(
            f"UPDATE operators SET {', '.join(sets)} WHERE operator_id = ?",
            params,
        )
        return cur.rowcount > 0


# ===========================================================================
# 2. Shifts & Overtime Rules
# ===========================================================================

VALID_SHIFT_TYPES = {"Morning", "Day", "Evening", "Night", "Afternoon"}
VALID_SHIFT_STATUSES = {"Scheduled", "Active", "Completed", "Absent", "active", "closed", "flagged", "abandoned"}


def calculate_printing_overtime(actual_duration_hours: float) -> dict:
    """
    Calculates printing overtime based on the 9-hour standard.
    Standard printing hours = 9.0
    Regular printing hours = min(actual_duration_hours, 9.0)
    Overtime hours = max(actual_duration_hours - 9.0, 0.0)
    Overtime status = 'Overtime' if actual_duration_hours > 9.0 else 'Normal'
    """
    standard_printing_hours = 9.0
    regular_printing_hours = round(min(actual_duration_hours, standard_printing_hours), 2)
    overtime_hours = round(max(actual_duration_hours - standard_printing_hours, 0.0), 2)
    overtime_status = "Overtime" if actual_duration_hours > standard_printing_hours else "Normal"
    return {
        "standard_printing_hours": standard_printing_hours,
        "regular_printing_hours": regular_printing_hours,
        "overtime_hours": overtime_hours,
        "overtime_status": overtime_status,
        "printing_duration_hours": round(actual_duration_hours, 2),
    }


def start_shift(
    operator_id: Any,
    shift_date: Optional[str] = None,
    shift_type: str = "Day",
    line_id: str = "line_1",
    target_walls: int = 0,
    summary_notes: Optional[str] = None,
    start_time: Optional[float] = None,
    actual_start_time: Optional[str] = None,
    planned_start_time: Optional[str] = None,
    planned_end_time: Optional[str] = None,
    status: str = "active",
    db_path: Optional[str] = None,
) -> int:
    """Starts a new shift with status 'active' (or custom status). Returns shift_id."""
    clean_op_id = operator_id
    if isinstance(operator_id, str):
        clean_op_id = operator_id.strip()

    # Look up operator if name or code was supplied
    op_record = get_operator(clean_op_id, db_path)
    if op_record:
        clean_op_id = op_record.get("operator_id", clean_op_id)

    if shift_type not in VALID_SHIFT_TYPES:
        raise ValueError(f"Invalid shift_type: {shift_type}. Must be one of {VALID_SHIFT_TYPES}")
    if target_walls < 0:
        raise ValueError("target_walls cannot be negative")

    s_date = shift_date.strip() if shift_date else datetime.date.today().isoformat()
    try:
        day_str = datetime.date.fromisoformat(s_date).strftime("%A")
    except Exception:
        day_str = "Thursday"

    # Default planned times if not supplied
    if not planned_start_time or not planned_end_time:
        if shift_type == "Morning":
            p_start = f"{s_date} 08:00:00"
            p_end = f"{s_date} 17:00:00"
        elif shift_type in ("Day", "Afternoon"):
            p_start = f"{s_date} 09:00:00"
            p_end = f"{s_date} 18:00:00"
        elif shift_type == "Evening":
            p_start = f"{s_date} 14:00:00"
            p_end = f"{s_date} 22:00:00"
        else:  # Night
            try:
                next_d = (datetime.date.fromisoformat(s_date) + datetime.timedelta(days=1)).isoformat()
            except Exception:
                next_d = s_date
            p_start = f"{s_date} 22:00:00"
            p_end = f"{next_d} 06:00:00"
    else:
        p_start = planned_start_time
        p_end = planned_end_time

    ts = start_time if start_time is not None else time.time()
    act_start = actual_start_time or datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    # Late calculation
    p_start_dt = _parse_dt(p_start)
    act_start_dt = _parse_dt(act_start)
    late_minutes = 0
    if p_start_dt and act_start_dt and act_start_dt > p_start_dt:
        late_minutes = max(0, int((act_start_dt - p_start_dt).total_seconds() // 60))

    attendance_status = "Late" if late_minutes > 0 else "Present"

    with db.connect(db_path) as conn:
        if status in ("active", "Active"):
            existing = conn.execute(
                "SELECT shift_id FROM shifts WHERE line_id = ? AND status IN ('active', 'Active') LIMIT 1",
                (line_id.strip(),),
            ).fetchone()
            if existing:
                raise sqlite3.IntegrityError(f"Line '{line_id}' already has an active shift (ID {existing['shift_id']})")

        cur = conn.execute(
            """
            INSERT INTO shifts (
                operator_id, line_id, shift_date, day, shift_type,
                planned_start_time, planned_end_time, actual_start_time,
                attendance_status, late_minutes, overtime_hours,
                start_time, status, target_walls, summary_notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, ?, ?, ?, ?, ?)
            """,
            (
                clean_op_id, line_id.strip(), s_date, day_str, shift_type,
                p_start, p_end, act_start,
                attendance_status, late_minutes,
                ts, status, target_walls, summary_notes, ts
            ),
        )
        shift_id = int(cur.lastrowid)
    log.info("Started shift %d: line=%s, operator=%s, date=%s, type=%s", shift_id, line_id, clean_op_id, s_date, shift_type)
    return shift_id


def get_shift(shift_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves shift details by shift_id."""
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM shifts WHERE shift_id = ?",
            (shift_id,),
        ).fetchone()
    return _row_to_dict(row)


def get_active_shift_for_line(line_id: str = "line_1", db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves the active shift on a specific line utilizing idx_shifts_single_active_line."""
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM shifts WHERE line_id = ? AND status IN ('active', 'Active') LIMIT 1",
            (line_id.strip(),),
        ).fetchone()
    return _row_to_dict(row)


def end_shift(
    shift_id: int,
    end_time: Optional[float] = None,
    status: str = "Completed",
    summary_notes: Optional[str] = None,
    actual_end_time: Optional[str] = None,
    db_path: Optional[str] = None,
) -> bool:
    """Closes a shift with terminal status: 'Completed', 'closed', 'flagged', or 'abandoned'."""
    norm_status = "Completed" if status.lower() in ("completed", "closed") else status
    if status not in VALID_SHIFT_STATUSES:
        raise ValueError(f"Invalid terminal shift status: {status}. Must be one of {VALID_SHIFT_STATUSES}")

    ts = end_time if end_time is not None else time.time()
    act_end = actual_end_time or datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    shift = get_shift(shift_id, db_path)
    total_hours = None
    if shift and shift.get("actual_start_time"):
        start_dt = _parse_dt(shift["actual_start_time"])
        end_dt = _parse_dt(act_end)
        if start_dt and end_dt:
            total_hours = round(max(0.0, (end_dt - start_dt).total_seconds() / 3600.0), 2)
    elif shift and shift.get("start_time"):
        total_hours = round(max(0.0, (ts - float(shift["start_time"])) / 3600.0), 2)

    sets = ["status = ?", "end_time = ?", "actual_end_time = ?"]
    params: List[Any] = [status, ts, act_end]
    if total_hours is not None:
        sets.append("total_hours = ?")
        params.append(total_hours)
    if summary_notes is not None:
        sets.append("summary_notes = ?")
        params.append(summary_notes)
    params.append(shift_id)

    with db.connect(db_path) as conn:
        cur = conn.execute(
            f"UPDATE shifts SET {', '.join(sets)} WHERE shift_id = ?",
            params,
        )
        return cur.rowcount > 0


def list_shifts(
    shift_date: Optional[str] = None,
    shift_type: Optional[str] = None,
    operator_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lists shifts with optional filters on date/type (idx_shifts_date) or operator (idx_shifts_operator)."""
    query = "SELECT * FROM shifts"
    clauses, params = [], []
    if shift_date:
        clauses.append("shift_date = ?")
        params.append(shift_date.strip())
    if shift_type:
        clauses.append("shift_type = ?")
        params.append(shift_type.strip())
    if operator_id:
        clauses.append("operator_id = ?")
        params.append(operator_id.strip())
    if status:
        clauses.append("status = ?")
        params.append(status.strip())
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY start_time DESC LIMIT ?"
    params.append(limit)

    with db.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ===========================================================================
# 3. Production Records
# ===========================================================================

VALID_PRODUCTION_STATUSES = {"in_progress", "completed", "aborted", "defect"}


def record_production(
    shift_id: int,
    wall_number: int,
    job_id: Optional[int] = None,
    print_file: Optional[str] = None,
    wall_height: float = 0.0,
    segment_layers: int = 0,
    layers_completed: int = 0,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    status: str = "in_progress",
    is_continuation_of: Optional[int] = None,
    notes: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """Creates a production record using exact Phase 2A columns. Returns production_id."""
    if wall_height < 0.0:
        raise ValueError("wall_height cannot be negative")
    if segment_layers < 0:
        raise ValueError("segment_layers cannot be negative")
    if layers_completed < 0:
        raise ValueError("layers_completed cannot be negative")
    if status not in VALID_PRODUCTION_STATUSES:
        raise ValueError(f"Invalid production status: {status}. Must be one of {VALID_PRODUCTION_STATUSES}")

    ts = start_time if start_time is not None else time.time()
    now = time.time()
    with db.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO production_records (
                shift_id, job_id, wall_number, print_file, wall_height,
                segment_layers, layers_completed, start_time, end_time,
                status, is_continuation_of, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                shift_id,
                job_id,
                wall_number,
                print_file,
                float(wall_height),
                int(segment_layers),
                int(layers_completed),
                ts,
                end_time,
                status,
                is_continuation_of,
                notes,
                now,
            ),
        )
        return int(cur.lastrowid)


def get_production_record(production_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a production record by production_id."""
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM production_records WHERE production_id = ?",
            (production_id,),
        ).fetchone()
    return _row_to_dict(row)


def update_production_record(
    production_id: int,
    layers_completed: Optional[int] = None,
    end_time: Optional[float] = None,
    status: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Optional[str] = None,
) -> bool:
    """Updates mutable fields on a production record."""
    sets, params = [], []
    if layers_completed is not None:
        if layers_completed < 0:
            raise ValueError("layers_completed cannot be negative")
        sets.append("layers_completed = ?")
        params.append(int(layers_completed))
    if end_time is not None:
        sets.append("end_time = ?")
        params.append(float(end_time))
    if status is not None:
        if status not in VALID_PRODUCTION_STATUSES:
            raise ValueError(f"Invalid status: {status}. Must be one of {VALID_PRODUCTION_STATUSES}")
        sets.append("status = ?")
        params.append(status)
    if notes is not None:
        sets.append("notes = ?")
        params.append(notes)

    if not sets:
        return False

    params.append(production_id)
    with db.connect(db_path) as conn:
        cur = conn.execute(
            f"UPDATE production_records SET {', '.join(sets)} WHERE production_id = ?",
            params,
        )
        return cur.rowcount > 0


def get_shift_production(shift_id: int, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves all production records logged for a shift (idx_production_shift)."""
    with db.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM production_records WHERE shift_id = ? ORDER BY wall_number ASC, start_time ASC",
            (shift_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_production_continuations(parent_production_id: int, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves continuation segments (idx_production_continuation)."""
    with db.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM production_records WHERE is_continuation_of = ? ORDER BY start_time ASC",
            (parent_production_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_operator_production(
    operator_id: str,
    shift_id: Optional[int] = None,
    shift_date: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieves all production records logged for an operator across shifts.

    Joins production_records -> shifts on shift_id, filtering by shifts.operator_id.
    Supports optional shift_id and shift_date filters.
    Returns actual stored production only, including wall, status, timestamps, job info.
    Preserves closed-shift history queryability.
    """
    clean_op = operator_id.strip() if operator_id else ""
    if not clean_op:
        return []
    query = (
        "SELECT pr.*, s.operator_id, s.shift_date, s.shift_type, s.line_id, s.status AS shift_status "
        "FROM production_records pr "
        "JOIN shifts s ON pr.shift_id = s.shift_id "
        "WHERE s.operator_id = ?"
    )
    params: List[Any] = [clean_op]
    if shift_id is not None:
        query += " AND pr.shift_id = ?"
        params.append(shift_id)
    if shift_date is not None:
        query += " AND s.shift_date = ?"
        params.append(shift_date.strip())
    query += " ORDER BY pr.start_time ASC, pr.wall_number ASC"
    with db.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ===========================================================================
# 4. Shift Issues
# ===========================================================================

VALID_ISSUE_SEVERITIES = {
    "info", "warning", "critical",
    "low", "medium", "high",
    "Low", "Medium", "High", "Critical",
    "Warning", "Info",
}


def log_issue(
    shift_id: int,
    issue_type: Optional[str] = None,
    description: Optional[str] = None,
    wall_number: Optional[int] = None,
    timestamp: Optional[float] = None,
    cause: Optional[str] = None,
    action_taken: Optional[str] = None,
    result: Optional[str] = None,
    severity: str = "warning",
    downtime_minutes: float = 0.0,
    rag_recommendation: Optional[str] = None,
    rag_verified: int = 0,
    rag_source: Optional[str] = None,
    problem_type: Optional[str] = None,
    problem_description: Optional[str] = None,
    solution: Optional[str] = None,
    operator_id: Optional[Any] = None,
    status: str = "Open",
    db_path: Optional[str] = None,
) -> int:
    """Logs a shift issue. Returns issue_id."""
    clean_type = (problem_type or issue_type or "operational_issue").strip()
    clean_desc = (problem_description or description or "").strip()
    if not clean_type:
        raise ValueError("issue_type or problem_type cannot be empty")
    if not clean_desc:
        raise ValueError("description or problem_description cannot be empty")
    if severity not in VALID_ISSUE_SEVERITIES:
        raise ValueError(f"Invalid severity: {severity}. Must be one of {VALID_ISSUE_SEVERITIES}")
    if downtime_minutes < 0.0:
        raise ValueError("downtime_minutes cannot be negative")

    clean_sol = solution or action_taken
    ts = timestamp if timestamp is not None else time.time()
    now = time.time()
    created_at_dt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    op_id = operator_id
    if op_id is None:
        with db.connect(db_path) as conn:
            row = conn.execute("SELECT operator_id FROM shifts WHERE shift_id = ?", (shift_id,)).fetchone()
            if row:
                op_id = row["operator_id"]

    with db.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO shift_issues (
                shift_id, operator_id, wall_number, timestamp,
                issue_type, description, problem_type, problem_description,
                cause, action_taken, solution, result, severity, status,
                downtime_minutes, rag_recommendation, rag_verified, rag_source,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                shift_id,
                op_id,
                wall_number,
                ts,
                clean_type,
                clean_desc,
                clean_type,
                clean_desc,
                cause,
                clean_sol,
                clean_sol,
                result,
                severity,
                status,
                float(downtime_minutes),
                rag_recommendation,
                int(rag_verified),
                rag_source,
                created_at_dt,
            ),
        )
        return int(cur.lastrowid)


def get_issue(issue_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves an issue by issue_id."""
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM shift_issues WHERE issue_id = ?",
            (issue_id,),
        ).fetchone()
    return _row_to_dict(row)


def update_issue(
    issue_id: int,
    cause: Optional[str] = None,
    action_taken: Optional[str] = None,
    result: Optional[str] = None,
    severity: Optional[str] = None,
    downtime_minutes: Optional[float] = None,
    rag_recommendation: Optional[str] = None,
    rag_verified: Optional[int] = None,
    rag_source: Optional[str] = None,
    problem_type: Optional[str] = None,
    problem_description: Optional[str] = None,
    solution: Optional[str] = None,
    status: Optional[str] = None,
    resolved_at: Optional[str] = None,
    db_path: Optional[str] = None,
) -> bool:
    """Updates resolution and post-incident investigation details on an issue."""
    sets, params = [], []
    if cause is not None:
        sets.append("cause = ?")
        params.append(cause)
    if action_taken is not None:
        sets.append("action_taken = ?")
        params.append(action_taken)
    if solution is not None:
        sets.append("solution = ?")
        params.append(solution)
        if action_taken is None:
            sets.append("action_taken = ?")
            params.append(solution)
    if problem_type is not None:
        sets.append("problem_type = ?")
        params.append(problem_type)
        sets.append("issue_type = ?")
        params.append(problem_type)
    if problem_description is not None:
        sets.append("problem_description = ?")
        params.append(problem_description)
        sets.append("description = ?")
        params.append(problem_description)
    if status is not None:
        sets.append("status = ?")
        params.append(status)
        if status in ("Resolved", "Closed") and resolved_at is None:
            resolved_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if resolved_at is not None:
        sets.append("resolved_at = ?")
        params.append(resolved_at)
    if result is not None:
        sets.append("result = ?")
        params.append(result)
    if severity is not None:
        if severity not in VALID_ISSUE_SEVERITIES:
            raise ValueError(f"Invalid severity: {severity}. Must be one of {VALID_ISSUE_SEVERITIES}")
        sets.append("severity = ?")
        params.append(severity)
    if downtime_minutes is not None:
        if downtime_minutes < 0.0:
            raise ValueError("downtime_minutes cannot be negative")
        sets.append("downtime_minutes = ?")
        params.append(float(downtime_minutes))
    if rag_recommendation is not None:
        sets.append("rag_recommendation = ?")
        params.append(rag_recommendation)
    if rag_verified is not None:
        sets.append("rag_verified = ?")
        params.append(int(rag_verified))
    if rag_source is not None:
        sets.append("rag_source = ?")
        params.append(rag_source)

    if not sets:
        return False

    params.append(issue_id)
    with db.connect(db_path) as conn:
        cur = conn.execute(
            f"UPDATE shift_issues SET {', '.join(sets)} WHERE issue_id = ?",
            params,
        )
        return cur.rowcount > 0


# ===========================================================================
# 4B. Machines & Printing Jobs
# ===========================================================================

def create_machine(
    machine_name: str,
    machine_type: str,
    status: str = "Off",
    current_speed: float = 0.0,
    db_path: Optional[str] = None,
) -> int:
    """Inserts a machine into machines table. Returns machine_id."""
    with db.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO machines (machine_name, machine_type, status, current_speed)
            VALUES (?, ?, ?, ?)
            """,
            (machine_name.strip(), machine_type.strip(), status, float(current_speed)),
        )
        return int(cur.lastrowid)


def get_machine(machine_id_or_name: Union[int, str], db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a machine by machine_id or machine_name."""
    with db.connect(db_path) as conn:
        if isinstance(machine_id_or_name, int) or (isinstance(machine_id_or_name, str) and machine_id_or_name.isdigit()):
            row = conn.execute("SELECT * FROM machines WHERE machine_id = ?", (int(machine_id_or_name),)).fetchone()
        else:
            row = conn.execute("SELECT * FROM machines WHERE machine_name = ?", (str(machine_id_or_name).strip(),)).fetchone()
    return _row_to_dict(row)


def list_machines(status: Optional[str] = None, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists all machines with optional status filter."""
    query = "SELECT * FROM machines"
    params = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    with db.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def update_machine(
    machine_id: int,
    status: Optional[str] = None,
    current_speed: Optional[float] = None,
    db_path: Optional[str] = None,
) -> bool:
    """Updates machine status or current speed."""
    sets, params = [], []
    if status is not None:
        sets.append("status = ?")
        params.append(status)
    if current_speed is not None:
        sets.append("current_speed = ?")
        params.append(float(current_speed))
    if not sets:
        return False
    sets.append("updated_at = CURRENT_TIMESTAMP")
    params.append(machine_id)
    with db.connect(db_path) as conn:
        cur = conn.execute(f"UPDATE machines SET {', '.join(sets)} WHERE machine_id = ?", params)
        return cur.rowcount > 0


def create_printing_job(
    job_name: str,
    machine_id: Optional[Union[int, str]] = None,
    operator_id: Optional[Union[int, str]] = None,
    shift_id: Optional[int] = None,
    material: str = "Concrete",
    start_time: Optional[Union[str, float, datetime.datetime]] = None,
    total_layers: Optional[int] = None,
    current_layer: int = 0,
    status: str = "Running",
    db_path: Optional[str] = None,
) -> int:
    """Creates a new printing job in printing_jobs. Returns job_id."""
    clean_job_name = job_name.strip()
    if isinstance(start_time, (int, float)):
        st_str = datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(start_time, datetime.datetime):
        st_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(start_time, str) and start_time.strip():
        st_str = start_time.strip()
    else:
        st_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    op_id = operator_id
    if op_id is not None:
        op = get_operator(op_id, db_path)
        if op:
            op_id = op.get("operator_id", op_id)

    mach_id = machine_id
    if mach_id is not None:
        mach = get_machine(mach_id, db_path)
        if mach:
            mach_id = mach["machine_id"]
        else:
            mach_id = None

    with db.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO printing_jobs (
                job_name, machine_id, operator_id, shift_id, material,
                start_time, total_layers, current_layer, status,
                standard_printing_hours, regular_printing_hours, overtime_hours, overtime_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 9.0, 0.0, 0.0, 'Normal')
            """,
            (
                clean_job_name,
                mach_id,
                op_id,
                shift_id,
                material,
                st_str,
                total_layers,
                current_layer,
                status,
            ),
        )
        return int(cur.lastrowid)


def end_printing_job(
    job_id: int,
    end_time: Optional[Union[str, float, datetime.datetime]] = None,
    status: str = "Completed",
    completion_percent: float = 100.0,
    current_layer: Optional[int] = None,
    db_path: Optional[str] = None,
) -> bool:
    """Ends a printing job, calculating 9-hour standard overtime. Returns True if updated."""
    job = get_printing_job(job_id, db_path)
    if not job:
        return False
    st_val = job.get("start_time")
    start_dt = _parse_dt(str(st_val)) if st_val else None

    if isinstance(end_time, (int, float)):
        end_str = datetime.datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
        end_dt = datetime.datetime.fromtimestamp(end_time)
    elif isinstance(end_time, datetime.datetime):
        end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        end_dt = end_time
    elif isinstance(end_time, str) and end_time.strip():
        end_str = end_time.strip()
        end_dt = _parse_dt(end_str)
    else:
        end_dt = datetime.datetime.now()
        end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    duration_hours = 0.0
    if start_dt and end_dt:
        duration_hours = max(0.0, (end_dt - start_dt).total_seconds() / 3600.0)

    ot = calculate_printing_overtime(duration_hours)

    sets = [
        "end_time = ?",
        "printing_duration_hours = ?",
        "standard_printing_hours = 9.0",
        "regular_printing_hours = ?",
        "overtime_hours = ?",
        "overtime_status = ?",
        "status = ?",
        "completion_percent = ?",
    ]
    params: List[Any] = [
        end_str,
        ot["printing_duration_hours"],
        ot["regular_printing_hours"],
        ot["overtime_hours"],
        ot["overtime_status"],
        status,
        float(completion_percent),
    ]
    if current_layer is not None:
        sets.append("current_layer = ?")
        params.append(int(current_layer))

    params.append(job_id)
    with db.connect(db_path) as conn:
        cur = conn.execute(f"UPDATE printing_jobs SET {', '.join(sets)} WHERE job_id = ?", params)
        return cur.rowcount > 0


def get_printing_job(job_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a printing job by job_id."""
    with db.connect(db_path) as conn:
        row = conn.execute("SELECT * FROM printing_jobs WHERE job_id = ?", (job_id,)).fetchone()
    return _row_to_dict(row)


def list_printing_jobs(
    shift_id: Optional[int] = None,
    operator_id: Optional[Any] = None,
    machine_id: Optional[int] = None,
    status: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lists printing jobs matching filters ordered by start_time DESC."""
    query = "SELECT * FROM printing_jobs"
    clauses, params = [], []
    if shift_id is not None:
        clauses.append("shift_id = ?")
        params.append(shift_id)
    if operator_id is not None:
        clauses.append("operator_id = ?")
        params.append(operator_id)
    if machine_id is not None:
        clauses.append("machine_id = ?")
        params.append(machine_id)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY start_time DESC"
    with db.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_agent_reporting_view(
    shift_date: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Returns the logical reporting view joining shifts, operators, printing_jobs,
    and shift_issues with computed overtime, shift duration, arrival delay, and working days count.
    """
    with db.connect(db_path) as conn:
        # Precompute working days count per operator
        counts_rows = conn.execute(
            "SELECT operator_id, COUNT(DISTINCT shift_date) as cnt FROM shifts GROUP BY operator_id"
        ).fetchall()
        working_days_map = {r["operator_id"]: r["cnt"] for r in counts_rows}

        sql = """
        SELECT
            s.shift_id,
            s.shift_date,
            s.shift_type,
            s.day,
            s.planned_start_time,
            s.planned_end_time,
            s.actual_start_time,
            s.actual_end_time,
            s.total_hours,
            s.late_minutes,
            s.attendance_status,
            o.operator_id,
            o.name AS operator_name,
            o.role AS operator_role,
            p.job_id,
            p.job_name,
            p.printing_duration_hours,
            p.standard_printing_hours,
            p.regular_printing_hours,
            p.overtime_hours,
            p.overtime_status,
            i.issue_id,
            i.problem_type,
            i.problem_description,
            i.solution
        FROM shifts s
        LEFT JOIN operators o ON s.operator_id = o.operator_id
        LEFT JOIN printing_jobs p ON p.shift_id = s.shift_id
        LEFT JOIN shift_issues i ON i.shift_id = s.shift_id
        """
        params: List[Any] = []
        if shift_date:
            sql += " WHERE s.shift_date = ?"
            params.append(shift_date.strip())
        sql += " ORDER BY s.shift_id ASC"

        rows = conn.execute(sql, params).fetchall()

    results: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        op_id = d.get("operator_id")
        working_days = working_days_map.get(op_id, 1)

        # Format Came In
        act_start = d.get("actual_start_time")
        came_in = _format_hh_mm(act_start)

        # Format Came Out
        act_end = d.get("actual_end_time")
        came_out = _format_hh_mm(act_end)
        if act_start and act_end:
            s_dt = _parse_dt(act_start)
            e_dt = _parse_dt(act_end)
            if s_dt and e_dt and e_dt.date() > s_dt.date():
                came_out = f"{came_out} (+1 day)"

        late_min = d.get("late_minutes") or 0
        late_by_str = f"+{late_min} mins" if late_min > 0 else "0 mins"
        shift_dur = d.get("total_hours")
        if shift_dur is not None:
            shift_dur = round(float(shift_dur), 2)

        tot_print = d.get("printing_duration_hours")
        if tot_print is not None:
            tot_print = round(float(tot_print), 2)
        reg_print = d.get("regular_printing_hours")
        if reg_print is not None:
            reg_print = round(float(reg_print), 2)
        ot_hours = d.get("overtime_hours")
        if ot_hours is not None:
            ot_hours = round(float(ot_hours), 2)

        record = {
            # Standard snake_case
            "operator_id": op_id,
            "operator_name": d.get("operator_name") or "Unknown",
            "shift_id": d.get("shift_id"),
            "shift_date": d.get("shift_date"),
            "shift_type": d.get("shift_type"),
            "day": d.get("day"),
            "came_in": came_in,
            "came_out": came_out,
            "actual_start_time": act_start,
            "actual_end_time": act_end,
            "shift_duration": shift_dur,
            "total_hours": shift_dur,
            "late_minutes": late_min,
            "late_by": late_by_str,
            "attendance_status": d.get("attendance_status"),
            "job_id": d.get("job_id"),
            "job_name": d.get("job_name"),
            "total_printing_time": tot_print,
            "printing_duration_hours": tot_print,
            "regular_time": reg_print,
            "regular_printing_hours": reg_print,
            "overtime": ot_hours,
            "overtime_hours": ot_hours,
            "overtime_status": d.get("overtime_status"),
            "issue_id": d.get("issue_id"),
            "problem_type": d.get("problem_type"),
            "problem_description": d.get("problem_description"),
            "solution": d.get("solution"),
            "working_days": working_days,
            "working_days_count": working_days,

            # Human Title Case aliases
            "Operator Name": d.get("operator_name") or "Unknown",
            "Shift": d.get("shift_type"),
            "Came In": came_in,
            "Came Out": came_out,
            "Shift Duration": shift_dur,
            "Late By": late_by_str,
            "Job Name": d.get("job_name"),
            "Total Printing Time": tot_print,
            "Regular Time": reg_print,
            "Overtime": ot_hours,
            "Problem Type": d.get("problem_type"),
            "Problem Description": d.get("problem_description"),
            "Solution": d.get("solution"),
            "Working Days": working_days,
        }
        results.append(record)
    return results


def seed_reference_operational_data(db_path: Optional[str] = None) -> None:
    """
    Seeds reference operational data for 08-Oct-2026:
    Aman (Morning), Rahul (Day), Rohit (Evening), Vikas (Night)
    with machines, shifts, printing jobs, issues, and working day histories.
    """
    with db.connect(db_path) as conn:
        # Check if already seeded
        existing = conn.execute(
            "SELECT COUNT(*) as c FROM shifts WHERE shift_date = '2026-10-08'"
        ).fetchone()
        if existing and existing["c"] >= 4:
            return

    # 1. Machine
    mach = get_machine("Gantry_3D_01", db_path)
    if not mach:
        mach_id = create_machine(
            "Gantry_3D_01",
            "Gantry 3D Concrete Printer",
            status="On",
            current_speed=120.0,
            db_path=db_path,
        )
    else:
        mach_id = mach["machine_id"]

    # 2. Operators & Working Day History
    operators_data = [
        ("Aman", "Operator", 18),
        ("Rahul", "Operator", 22),
        ("Rohit", "Operator", 15),
        ("Vikas", "Operator", 12),
    ]

    op_ids: Dict[str, Any] = {}
    for name, role, total_working_days in operators_data:
        op = find_operator_by_name(name, active_only=False, db_path=db_path)
        if not op:
            code = f"EMP_{name.upper()}"
            op_id = create_operator(
                name=name,
                role=role,
                employee_code=code,
                db_path=db_path,
            )
        else:
            op_id = op["operator_id"]
        op_ids[name] = op_id

        # Seed historical shifts to achieve exact working_days count
        with db.connect(db_path) as conn:
            existing_dates = {
                r["shift_date"]
                for r in conn.execute(
                    "SELECT DISTINCT shift_date FROM shifts WHERE operator_id = ?",
                    (op_id,),
                ).fetchall()
            }
            target_past = total_working_days - (0 if "2026-10-08" in existing_dates else 1)
            d = datetime.date(2026, 10, 7)
            while len(existing_dates) < target_past:
                past_date = d.isoformat()
                if past_date not in existing_dates:
                    day_name = d.strftime("%A")
                    past_dt = datetime.datetime.strptime(f"{past_date} 09:00:00", "%Y-%m-%d %H:%M:%S")
                    past_ts = past_dt.timestamp()
                    conn.execute(
                        """
                        INSERT INTO shifts (
                            operator_id, line_id, shift_date, day, shift_type,
                            planned_start_time, planned_end_time, actual_start_time, actual_end_time,
                            total_hours, attendance_status, late_minutes, overtime_hours,
                            status, created_at, start_time, end_time
                        ) VALUES (?, 'line_1', ?, ?, 'Day', ?, ?, ?, ?, 8.0, 'Present', 0, 0.0, 'Completed', ?, ?, ?)
                        """,
                        (
                            op_id,
                            past_date,
                            day_name,
                            f"{past_date} 09:00:00",
                            f"{past_date} 17:00:00",
                            f"{past_date} 09:00:00",
                            f"{past_date} 17:00:00",
                            past_ts,
                            past_ts,
                            past_ts + 28800.0,
                        ),
                    )
                    existing_dates.add(past_date)
                d -= datetime.timedelta(days=1)

    # 3. 2026-10-08 Shifts, Printing Jobs & Issues

    # --- Aman (Morning) ---
    aman_id = op_ids["Aman"]
    s_aman = start_shift(
        operator_id=aman_id,
        shift_date="2026-10-08",
        shift_type="Morning",
        line_id="line_1",
        planned_start_time="2026-10-08 08:00:00",
        planned_end_time="2026-10-08 17:00:00",
        actual_start_time="2026-10-08 07:52:00",
        status="Completed",
        db_path=db_path,
    )
    # End Aman's shift: 07:52 to 17:05 = 9.22h
    end_shift(
        shift_id=s_aman,
        actual_end_time="2026-10-08 17:05:00",
        status="Completed",
        db_path=db_path,
    )
    # Job: Wall_Segment_A (8.5 hours -> 0.0h overtime, Normal)
    job_aman = create_printing_job(
        job_name="Wall_Segment_A",
        machine_id=mach_id,
        operator_id=aman_id,
        shift_id=s_aman,
        start_time="2026-10-08 08:00:00",
        total_layers=60,
        status="Running",
        db_path=db_path,
    )
    end_printing_job(
        job_id=job_aman,
        end_time="2026-10-08 16:30:00",
        status="Completed",
        completion_percent=100.0,
        current_layer=60,
        db_path=db_path,
    )
    # Issue: Slump Loss
    log_issue(
        shift_id=s_aman,
        operator_id=aman_id,
        problem_type="Slump Loss",
        problem_description="Concrete mix setting too fast in pump line",
        solution="Added retarder admixture dosage (0.2%)",
        severity="Medium",
        status="Resolved",
        db_path=db_path,
    )

    # --- Rahul (Day) ---
    rahul_id = op_ids["Rahul"]
    s_rahul = start_shift(
        operator_id=rahul_id,
        shift_date="2026-10-08",
        shift_type="Day",
        line_id="line_2",
        planned_start_time="2026-10-08 08:00:00",
        planned_end_time="2026-10-08 17:00:00",
        actual_start_time="2026-10-08 08:07:00",
        status="Completed",
        db_path=db_path,
    )
    # End Rahul's shift: 08:07 to 17:10 = 9.05h (Late by 7 mins)
    end_shift(
        shift_id=s_rahul,
        actual_end_time="2026-10-08 17:10:00",
        status="Completed",
        db_path=db_path,
    )
    # Job: Wall_Panel_B (11.5 hours -> 2.5h overtime, Overtime)
    job_rahul = create_printing_job(
        job_name="Wall_Panel_B",
        machine_id=mach_id,
        operator_id=rahul_id,
        shift_id=s_rahul,
        start_time="2026-10-08 08:15:00",
        total_layers=50,
        status="Running",
        db_path=db_path,
    )
    end_printing_job(
        job_id=job_rahul,
        end_time="2026-10-08 19:45:00",
        status="Completed",
        completion_percent=100.0,
        current_layer=50,
        db_path=db_path,
    )
    # Issue: Nozzle Choke
    log_issue(
        shift_id=s_rahul,
        operator_id=rahul_id,
        problem_type="Nozzle Choke",
        problem_description="Aggregate clump blocking nozzle tip",
        solution="Cleared tip with scraper and flushed slurry",
        severity="High",
        status="Resolved",
        db_path=db_path,
    )

    # --- Rohit (Evening) ---
    rohit_id = op_ids["Rohit"]
    s_rohit = start_shift(
        operator_id=rohit_id,
        shift_date="2026-10-08",
        shift_type="Evening",
        line_id="line_3",
        planned_start_time="2026-10-08 16:00:00",
        planned_end_time="2026-10-09 00:00:00",
        actual_start_time="2026-10-08 16:02:00",
        status="Completed",
        db_path=db_path,
    )
    # End Rohit's shift: 16:02 to 00:05 (+1 day) = 8.05h (Late by 2 mins)
    end_shift(
        shift_id=s_rohit,
        actual_end_time="2026-10-09 00:05:00",
        status="Completed",
        db_path=db_path,
    )
    # Job: Corner_Column_C (7.5 hours -> 0.0h overtime, Normal)
    job_rohit = create_printing_job(
        job_name="Corner_Column_C",
        machine_id=mach_id,
        operator_id=rohit_id,
        shift_id=s_rohit,
        start_time="2026-10-08 16:15:00",
        total_layers=40,
        status="Running",
        db_path=db_path,
    )
    end_printing_job(
        job_id=job_rohit,
        end_time="2026-10-08 23:45:00",
        status="Completed",
        completion_percent=100.0,
        current_layer=40,
        db_path=db_path,
    )
    # Issue: Gantry Y-axis Vibration
    log_issue(
        shift_id=s_rohit,
        operator_id=rohit_id,
        problem_type="Gantry Y-axis Vibration",
        problem_description="Mechanical chatter during high-speed travel",
        solution="Tightened lead screw brackets and lubricated rails",
        severity="Low",
        status="Resolved",
        db_path=db_path,
    )

    # --- Vikas (Night) ---
    vikas_id = op_ids["Vikas"]
    s_vikas = start_shift(
        operator_id=vikas_id,
        shift_date="2026-10-08",
        shift_type="Night",
        line_id="line_4",
        planned_start_time="2026-10-08 22:00:00",
        planned_end_time="2026-10-09 06:00:00",
        actual_start_time="2026-10-08 21:55:00",
        status="Completed",
        db_path=db_path,
    )
    # End Vikas's shift: 21:55 to 06:08 (+1 day) = 8h 13m = 8.22h
    end_shift(
        shift_id=s_vikas,
        actual_end_time="2026-10-09 06:08:00",
        status="Completed",
        db_path=db_path,
    )
    # Job: Foundation_Block_D (10.0 hours -> 1.0h overtime, Overtime)
    job_vikas = create_printing_job(
        job_name="Foundation_Block_D",
        machine_id=mach_id,
        operator_id=vikas_id,
        shift_id=s_vikas,
        start_time="2026-10-08 22:10:00",
        total_layers=45,
        status="Running",
        db_path=db_path,
    )
    end_printing_job(
        job_id=job_vikas,
        end_time="2026-10-09 08:10:00",
        status="Completed",
        completion_percent=100.0,
        current_layer=45,
        db_path=db_path,
    )
    # Vikas had No issues (Issue: None)


def list_shift_issues(
    shift_id: Optional[int] = None,
    issue_type: Optional[str] = None,
    operator_id: Optional[str] = None,
    unresolved_only: bool = False,
    shift_date: Optional[str] = None,
    limit: int = 50,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lists issues utilizing idx_issues_shift or idx_issues_type_time, with optional operator, shift_date and unresolved filters."""
    clauses, params = [], []
    has_shift_join = operator_id is not None or shift_date is not None
    if has_shift_join:
        query = (
            "SELECT i.*, s.shift_date, s.shift_type, s.operator_id "
            "FROM shift_issues i JOIN shifts s ON i.shift_id = s.shift_id"
        )
        if operator_id is not None:
            clauses.append("s.operator_id = ?")
            params.append(operator_id.strip())
        if shift_date is not None:
            clauses.append("s.shift_date = ?")
            params.append(shift_date.strip())
    else:
        query = "SELECT * FROM shift_issues"

    if shift_id is not None:
        clauses.append("shift_id = ?" if not has_shift_join else "i.shift_id = ?")
        params.append(shift_id)
    if issue_type is not None:
        clauses.append("issue_type = ?" if not has_shift_join else "i.issue_type = ?")
        params.append(issue_type.strip())
    if unresolved_only:
        col = "result" if not has_shift_join else "i.result"
        clauses.append(f"({col} IS NULL OR {col} = '' OR {col} = 'not_resolved')")
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    order_col = "timestamp" if not has_shift_join else "i.timestamp"
    query += f" ORDER BY {order_col} DESC LIMIT ?"
    params.append(limit)

    with db.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ===========================================================================
# 5. Handovers
# ===========================================================================

def create_handover(
    from_shift_id: int,
    to_shift_id: Optional[int] = None,
    line_id: str = "line_1",
    handover_time: Optional[float] = None,
    completed_work: Optional[str] = None,
    pending_work: Optional[str] = None,
    open_issues: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """Initiates a shift handover. Returns handover_id."""
    if to_shift_id is not None and from_shift_id == to_shift_id:
        raise ValueError("from_shift_id and to_shift_id cannot be identical")

    ts = handover_time if handover_time is not None else time.time()
    now = time.time()
    with db.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO handovers (
                from_shift_id, to_shift_id, line_id, handover_time,
                completed_work, pending_work, open_issues, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                from_shift_id,
                to_shift_id,
                line_id.strip(),
                ts,
                completed_work,
                pending_work,
                open_issues,
                notes,
                now,
            ),
        )
        return int(cur.lastrowid)


def get_handover(handover_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves handover details by handover_id."""
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM handovers WHERE handover_id = ?",
            (handover_id,),
        ).fetchone()
    return _row_to_dict(row)


def get_pending_handover_for_line(line_id: str = "line_1", db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves the newest unassigned handover for a line utilizing idx_handovers_line_match."""
    with db.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM handovers
            WHERE line_id = ? AND to_shift_id IS NULL
            ORDER BY handover_time DESC LIMIT 1
            """,
            (line_id.strip(),),
        ).fetchone()
    return _row_to_dict(row)


def complete_handover(
    handover_id: int,
    to_shift_id: int,
    notes: Optional[str] = None,
    close_from_shift: bool = True,
    db_path: Optional[str] = None,
) -> bool:
    """
    Atomically binds to_shift_id to the handover record and optionally
    marks from_shift_id as 'closed'.
    """
    now = time.time()
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT from_shift_id, to_shift_id FROM handovers WHERE handover_id = ?",
            (handover_id,),
        ).fetchone()
        if not row:
            return False
        if row["from_shift_id"] == to_shift_id:
            raise ValueError("to_shift_id cannot match from_shift_id")

        from_shift_id = row["from_shift_id"]
        sets = ["to_shift_id = ?"]
        params: List[Any] = [to_shift_id]
        if notes is not None:
            sets.append("notes = ?")
            params.append(notes)
        params.append(handover_id)

        conn.execute(f"UPDATE handovers SET {', '.join(sets)} WHERE handover_id = ?", params)
        if close_from_shift:
            conn.execute(
                "UPDATE shifts SET status = 'closed', end_time = ? WHERE shift_id = ? AND status = 'active'",
                (now, from_shift_id),
            )
        return True


# ===========================================================================
# 6. Event Corrections
# ===========================================================================

def log_event_correction(
    entity_type: str,
    entity_id: int,
    field_name: str,
    new_value: str,
    corrected_by: str,
    reason: str,
    old_value: Optional[str] = None,
    source: str = "manual",
    corrected_at: Optional[float] = None,
    db_path: Optional[str] = None,
) -> int:
    """Logs an audit correction for an entity field. Returns correction_id."""
    clean_entity_type = entity_type.strip()
    clean_field = field_name.strip()
    clean_new_val = str(new_value).strip()
    clean_by = corrected_by.strip()
    clean_reason = reason.strip()

    if not clean_entity_type:
        raise ValueError("entity_type cannot be empty")
    if not clean_field:
        raise ValueError("field_name cannot be empty")
    if not clean_by:
        raise ValueError("corrected_by cannot be empty")
    if not clean_reason:
        raise ValueError("reason cannot be empty")

    ts = corrected_at if corrected_at is not None else time.time()
    with db.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO event_corrections (
                entity_type, entity_id, field_name, old_value,
                new_value, corrected_by, corrected_at, source, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean_entity_type,
                entity_id,
                clean_field,
                old_value,
                clean_new_val,
                clean_by,
                ts,
                source.strip(),
                clean_reason,
            ),
        )
        return int(cur.lastrowid)


def get_corrections_for_entity(
    entity_type: str,
    entity_id: int,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieves all corrections for an entity utilizing idx_corrections_entity."""
    with db.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM event_corrections
            WHERE entity_type = ? AND entity_id = ?
            ORDER BY corrected_at DESC
            """,
            (entity_type.strip(), entity_id),
        ).fetchall()
    return [dict(r) for r in rows]


def list_event_corrections(limit: int = 50, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists recent event corrections."""
    with db.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM event_corrections ORDER BY corrected_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ===========================================================================
# 7. High-Level Autonomous Agentic Tools (Read & Write)
# ===========================================================================

def list_or_search_employees(name_query: Optional[str] = None, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Checks operators table. If name_query is provided, filters by name/code."""
    with db.connect(db_path) as conn:
        if name_query and str(name_query).strip():
            q = f"%{str(name_query).strip()}%"
            rows = conn.execute(
                "SELECT * FROM operators WHERE name LIKE ? OR employee_code LIKE ? ORDER BY name ASC",
                (q, q),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM operators ORDER BY name ASC").fetchall()
    return [dict(r) for r in rows]


def create_or_update_employee(
    name: str,
    role: str = "Operator",
    status: str = "Active",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Inserts or updates an employee in operators table."""
    clean_name = str(name).strip() if name else ""
    if not clean_name:
        return {"success": False, "error": "Employee name cannot be empty."}

    existing = find_operator_by_name(clean_name, active_only=False, db_path=db_path)
    if existing:
        op_id = existing["operator_id"]
        act_val = 1 if status.lower() == "active" else 0
        update_operator(op_id, role=role, active=act_val, db_path=db_path)
        with db.connect(db_path) as conn:
            conn.execute("UPDATE operators SET status = ? WHERE operator_id = ?", (status, op_id))
        updated = get_operator(op_id, db_path)
        return {"success": True, "action": "updated", "operator": updated}
    else:
        code = f"EMP_{clean_name.upper().replace(' ', '_')}"
        op_id = create_operator(
            name=clean_name,
            role=role,
            employee_code=code,
            status=status,
            active=1 if status.lower() == "active" else 0,
            db_path=db_path,
        )
        created = get_operator(op_id, db_path)
        return {"success": True, "action": "created", "operator": created}


def start_employee_shift(
    operator_name: str,
    shift_type: Optional[str] = None,
    actual_start_time: Optional[str] = None,
    line_id: str = "line_1",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Starts a shift in shifts (or asks user if operator_name does not exist yet)."""
    clean_name = str(operator_name).strip() if operator_name else ""
    if not clean_name:
        return {"success": False, "error": "Operator name must be provided to start a shift."}

    op = find_operator_by_name(clean_name, active_only=False, db_path=db_path)
    if not op:
        return {
            "success": False,
            "error": f"Operator '{clean_name}' does not exist in the database. Please confirm if you would like me to register them as a new employee first.",
            "requires_registration": True,
            "operator_name": clean_name,
        }

    op_id = op["operator_id"]
    now_dt = datetime.datetime.now()
    if actual_start_time:
        parsed_dt = _parse_dt(actual_start_time) or now_dt
    else:
        parsed_dt = now_dt

    if not shift_type or shift_type not in VALID_SHIFT_TYPES:
        hr = parsed_dt.hour
        if 6 <= hr < 14:
            s_type = "Morning"
        elif 14 <= hr < 18:
            s_type = "Day"
        elif 18 <= hr < 22:
            s_type = "Evening"
        else:
            s_type = "Night"
    else:
        s_type = shift_type

    # Close any lingering active shift on this line or for this operator
    with db.connect(db_path) as conn:
        active_row = conn.execute(
            "SELECT shift_id, operator_id FROM shifts WHERE (line_id = ? OR operator_id = ?) AND status IN ('active', 'Active')",
            (line_id, op_id),
        ).fetchone()
        if active_row:
            end_shift(active_row["shift_id"], status="Completed", db_path=db_path)

    try:
        s_id = start_shift(
            operator_id=op_id,
            shift_type=s_type,
            line_id=line_id,
            actual_start_time=actual_start_time or parsed_dt.strftime("%Y-%m-%d %H:%M:%S"),
            status="Active",
            db_path=db_path,
        )
        return {
            "success": True,
            "shift_id": s_id,
            "operator_name": op["name"],
            "operator_id": op_id,
            "shift_type": s_type,
            "line_id": line_id,
            "status": "Active",
            "message": f"Shift started for {op['name']} on {line_id} ({s_type} shift).",
        }
    except Exception as exc:
        log.error("start_employee_shift failed: %s", exc)
        return {"success": False, "error": str(exc)}


def end_employee_shift(
    operator_name: Optional[str] = None,
    line_id: str = "line_1",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Ends the active shift and calculates total_hours."""
    with db.connect(db_path) as conn:
        if operator_name and str(operator_name).strip():
            op = find_operator_by_name(operator_name, active_only=False, db_path=db_path)
            if not op:
                return {"success": False, "error": f"Operator '{operator_name}' not found."}
            row = conn.execute(
                "SELECT * FROM shifts WHERE operator_id = ? AND status IN ('active', 'Active') ORDER BY start_time DESC LIMIT 1",
                (op["operator_id"],),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM shifts WHERE line_id = ? AND status IN ('active', 'Active') ORDER BY start_time DESC LIMIT 1",
                (line_id,),
            ).fetchone()
            if not row:
                row = conn.execute(
                    "SELECT * FROM shifts WHERE status IN ('active', 'Active') ORDER BY start_time DESC LIMIT 1"
                ).fetchone()

    if not row:
        return {"success": False, "error": "No active shift found to end."}

    s_id = row["shift_id"]
    ok = end_shift(s_id, status="Completed", db_path=db_path)
    updated = get_shift(s_id, db_path)
    op_rec = get_operator(row["operator_id"], db_path)
    name = op_rec["name"] if op_rec else str(row["operator_id"])
    return {
        "success": ok,
        "shift_id": s_id,
        "operator_name": name,
        "total_hours": updated.get("total_hours"),
        "actual_start_time": updated.get("actual_start_time"),
        "actual_end_time": updated.get("actual_end_time"),
        "status": "Completed",
        "message": f"Shift ended for {name}. Total hours: {updated.get('total_hours', 0.0)}.",
    }


def query_shifts_and_attendance(
    operator_name: Optional[str] = None,
    shift_type: Optional[str] = None,
    date: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Queries shift history, Came In/Came Out times, late arrivals, and total working days."""
    view_data = get_agent_reporting_view(shift_date=date, db_path=db_path)
    results = []
    for r in view_data:
        if operator_name and operator_name.strip():
            if operator_name.strip().lower() not in (r.get("operator_name") or "").lower():
                continue
        if shift_type and shift_type.strip():
            if shift_type.strip().lower() != (r.get("shift_type") or "").lower():
                continue
        results.append(r)
    return results


def start_or_update_printing_job(
    job_name: str,
    operator_name: Optional[str] = None,
    duration_hours: Optional[float] = None,
    status: Optional[str] = None,
    current_layer: Optional[int] = None,
    total_layers: Optional[int] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Creates or updates a printing job and calculates 9-hour overtime."""
    clean_job = str(job_name).strip()
    if not clean_job:
        return {"success": False, "error": "Job name must be specified."}

    op_id = None
    if operator_name:
        op = find_operator_by_name(operator_name, active_only=False, db_path=db_path)
        if op:
            op_id = op["operator_id"]

    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM printing_jobs WHERE job_name = ? COLLATE NOCASE ORDER BY job_id DESC LIMIT 1",
            (clean_job,),
        ).fetchone()

    if row:
        job_id = row["job_id"]
        if duration_hours is not None:
            ot = calculate_printing_overtime(float(duration_hours))
            job_status = status or "Completed"
            with db.connect(db_path) as conn:
                conn.execute(
                    """
                    UPDATE printing_jobs SET
                        printing_duration_hours = ?,
                        standard_printing_hours = 9.0,
                        regular_printing_hours = ?,
                        overtime_hours = ?,
                        overtime_status = ?,
                        status = ?,
                        completion_percent = 100.0
                    WHERE job_id = ?
                    """,
                    (
                        ot["printing_duration_hours"],
                        ot["regular_printing_hours"],
                        ot["overtime_hours"],
                        ot["overtime_status"],
                        job_status,
                        job_id,
                    ),
                )
        if current_layer is not None or total_layers is not None or status is not None:
            sets, params = [], []
            if current_layer is not None:
                sets.append("current_layer = ?")
                params.append(int(current_layer))
            if total_layers is not None:
                sets.append("total_layers = ?")
                params.append(int(total_layers))
            if status is not None and duration_hours is None:
                sets.append("status = ?")
                params.append(status)
            if sets:
                params.append(job_id)
                with db.connect(db_path) as conn:
                    conn.execute(f"UPDATE printing_jobs SET {', '.join(sets)} WHERE job_id = ?", params)

        updated = get_printing_job(job_id, db_path)
        return {"success": True, "action": "updated", "job": updated}

    # New job
    active_shift = get_active_shift_for_line("line_1", db_path)
    shift_id = active_shift["shift_id"] if active_shift else None
    if not op_id and active_shift:
        op_id = active_shift["operator_id"]

    mach = get_machine("Gantry_3D_01", db_path)
    if not mach:
        try:
            mach_id = create_machine(
                "Gantry_3D_01",
                "Gantry 3D Concrete Printer",
                status="On",
                current_speed=120.0,
                db_path=db_path,
            )
        except Exception:
            mach_id = None
    else:
        mach_id = mach["machine_id"]

    job_id = create_printing_job(
        job_name=clean_job,
        machine_id=mach_id,
        operator_id=op_id,
        shift_id=shift_id,
        total_layers=total_layers,
        current_layer=current_layer or 0,
        status=status or "Running",
        db_path=db_path,
    )
    if duration_hours is not None:
        ot = calculate_printing_overtime(float(duration_hours))
        with db.connect(db_path) as conn:
            conn.execute(
                """
                UPDATE printing_jobs SET
                    printing_duration_hours = ?,
                    regular_printing_hours = ?,
                    overtime_hours = ?,
                    overtime_status = ?,
                    status = ?
                WHERE job_id = ?
                """,
                (
                    ot["printing_duration_hours"],
                    ot["regular_printing_hours"],
                    ot["overtime_hours"],
                    ot["overtime_status"],
                    status or "Completed",
                    job_id,
                ),
            )
    created = get_printing_job(job_id, db_path)
    return {"success": True, "action": "created", "job": created}


def query_printing_jobs_and_overtime(
    operator_name: Optional[str] = None,
    job_name: Optional[str] = None,
    only_overtime: bool = False,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieves job durations, 9-hour overtime calculations, and progress."""
    query = """
    SELECT p.*, o.name AS operator_name
    FROM printing_jobs p
    LEFT JOIN operators o ON p.operator_id = o.operator_id
    """
    clauses, params = [], []
    if operator_name and operator_name.strip():
        clauses.append("o.name LIKE ?")
        params.append(f"%{operator_name.strip()}%")
    if job_name and job_name.strip():
        clauses.append("p.job_name LIKE ?")
        params.append(f"%{job_name.strip()}%")
    if only_overtime:
        clauses.append("(p.overtime_hours > 0 OR p.overtime_status = 'Overtime')")

    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY p.start_time DESC"

    with db.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def log_or_update_shift_issue(
    problem_type: str,
    problem_description: str,
    operator_name: Optional[str] = None,
    severity: str = "Medium",
    solution: Optional[str] = None,
    status: str = "Open",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Logs or updates an issue in shift_issues table."""
    clean_type = str(problem_type).strip() if problem_type else "operational_issue"
    clean_desc = str(problem_description).strip() if problem_description else ""
    if not clean_desc:
        return {"success": False, "error": "Problem description must be provided."}

    op_id = None
    if operator_name:
        op = find_operator_by_name(operator_name, active_only=False, db_path=db_path)
        if op:
            op_id = op["operator_id"]

    active_shift = get_active_shift_for_line("line_1", db_path)
    shift_id = active_shift["shift_id"] if active_shift else 1
    if not op_id and active_shift:
        op_id = active_shift["operator_id"]

    with db.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM shift_issues
            WHERE (problem_description LIKE ? OR problem_type LIKE ?) AND status = 'Open'
            ORDER BY issue_id DESC LIMIT 1
            """,
            (f"%{clean_desc}%", f"%{clean_type}%"),
        ).fetchone()

    if row and (solution or status != "Open"):
        issue_id = row["issue_id"]
        update_issue(
            issue_id=issue_id,
            solution=solution,
            status=status,
            severity=severity,
            db_path=db_path,
        )
        updated = get_issue(issue_id, db_path)
        return {"success": True, "action": "updated", "issue": updated}

    issue_id = log_issue(
        shift_id=shift_id,
        operator_id=op_id,
        problem_type=clean_type,
        problem_description=clean_desc,
        severity=severity,
        solution=solution,
        status=status,
        db_path=db_path,
    )
    created = get_issue(issue_id, db_path)
    return {"success": True, "action": "created", "issue": created}
