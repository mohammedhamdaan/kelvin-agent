

"""
SQLite persistence for machine state, print jobs, layer measurements and events.

Design notes:
  * Derived values are never stored as trusted duplicates. `remaining_length_mm`
    is a SQLite GENERATED column computed as max(total - completed, 0), so it
    cannot drift out of sync with the values it depends on.
  * Writes are small and frequent (the vision loop updates machine_state every
    analysed frame), so WAL journaling is enabled for concurrent read access
    from the assistant while the camera loop writes.
  * Every table carries a timestamp; the assistant only trusts recent state.
"""

import os
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from agent import config
from agent.logging_setup import get_logger

log = get_logger("DB")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS print_jobs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    job_name            TEXT    NOT NULL,
    material            TEXT,
    total_length_mm     REAL    NOT NULL DEFAULT 0,
    completed_length_mm REAL    NOT NULL DEFAULT 0,
    -- Derived, never stored independently: cannot drift from its inputs.
    remaining_length_mm REAL GENERATED ALWAYS AS
        (MAX(total_length_mm - completed_length_mm, 0)) VIRTUAL,
    total_layers        INTEGER,
    current_layer       INTEGER,
    status              TEXT    NOT NULL DEFAULT 'running',
    started_at          REAL    NOT NULL,
    completed_at        REAL
);

CREATE TABLE IF NOT EXISTS machine_state (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id              INTEGER REFERENCES print_jobs(id),
    printing            INTEGER,
    status              TEXT,
    current_layer       INTEGER,
    layer_width_mm      REAL,
    layer_width_px      REAL,
    completed_length_mm REAL,
    motion_score        REAL,
    confidence          REAL,
    camera_connected    INTEGER,
    timestamp           REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_machine_state_ts ON machine_state(timestamp DESC);

CREATE TABLE IF NOT EXISTS layer_measurements (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id       INTEGER REFERENCES print_jobs(id),
    layer_number INTEGER,
    width_px     REAL,
    width_mm     REAL,
    confidence   REAL,
    timestamp    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_layer_ts ON layer_measurements(timestamp DESC);

CREATE TABLE IF NOT EXISTS machine_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     INTEGER REFERENCES print_jobs(id),
    event_type TEXT NOT NULL,
    message    TEXT,
    timestamp  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON machine_events(timestamp DESC);

-- =======================================================================
-- FINAL VALIDATED NORMALIZED INDUSTRIAL AGENT SCHEMA (7 Tables + Legacy)
-- =======================================================================

-- 1. Operators
CREATE TABLE IF NOT EXISTS operators (
    operator_id   INT PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE COLLATE NOCASE,
    role          TEXT NOT NULL DEFAULT 'Operator',
    status        TEXT NOT NULL DEFAULT 'Active',
    employee_code TEXT UNIQUE,
    active        INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    current_work_status TEXT NOT NULL DEFAULT 'Off Shift',
    active_shift_id     INTEGER,
    assigned_line       TEXT NOT NULL DEFAULT 'line_1'
);

-- 2. Shifts (Attendance & Shift Lifecycle)
CREATE TABLE IF NOT EXISTS shifts (
    shift_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    operator_id        INTEGER REFERENCES operators(operator_id),
    line_id            TEXT NOT NULL DEFAULT 'line_1',
    shift_date         DATE NOT NULL,
    day                TEXT NOT NULL DEFAULT 'Thursday',
    shift_type         TEXT NOT NULL CHECK(shift_type IN ('Morning', 'Day', 'Evening', 'Night', 'Afternoon')),
    planned_start_time DATETIME NOT NULL DEFAULT '2026-10-08 08:00:00',
    planned_end_time   DATETIME NOT NULL DEFAULT '2026-10-08 17:00:00',
    actual_start_time  DATETIME,
    actual_end_time    DATETIME,
    total_hours        REAL,
    attendance_status  TEXT NOT NULL DEFAULT 'Scheduled',
    late_minutes       INTEGER DEFAULT 0,
    overtime_hours     REAL DEFAULT 0.0,
    overtime_status    TEXT DEFAULT 'Normal',
    status             TEXT NOT NULL DEFAULT 'Scheduled' CHECK(status IN ('Scheduled', 'Active', 'Completed', 'Absent', 'active', 'closed', 'flagged', 'abandoned')),
    start_time         REAL,
    end_time           REAL,
    target_walls       INTEGER NOT NULL DEFAULT 0,
    summary_notes      TEXT,
    created_at         DATETIME DEFAULT CURRENT_TIMESTAMP,
    regular_hours      REAL DEFAULT 0.0,
    standard_hours     REAL DEFAULT 9.0,
    current_status     TEXT DEFAULT 'Working',
    operator_name      TEXT,
    employee_code      TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_shifts_single_active_line ON shifts(line_id) WHERE status IN ('active', 'Active');
CREATE INDEX IF NOT EXISTS idx_shifts_date ON shifts(shift_date, shift_type);
CREATE INDEX IF NOT EXISTS idx_shifts_operator ON shifts(operator_id, start_time DESC);

-- 3. Machines
CREATE TABLE IF NOT EXISTS machines (
    machine_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_name  TEXT NOT NULL UNIQUE,
    machine_type  TEXT NOT NULL,
    status        TEXT NOT NULL,
    current_speed REAL DEFAULT 0.0,
    updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 4. Printing Jobs (9-Hour Standard Printing Time & Overtime Rule)
CREATE TABLE IF NOT EXISTS printing_jobs (
    job_id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id              INTEGER REFERENCES machines(machine_id),
    operator_id             INTEGER REFERENCES operators(operator_id),
    shift_id                INTEGER REFERENCES shifts(shift_id),
    job_name                TEXT NOT NULL,
    material                TEXT NOT NULL DEFAULT 'Concrete',
    start_time              DATETIME NOT NULL,
    end_time                DATETIME,
    printing_duration_hours REAL DEFAULT 0.0,
    standard_printing_hours REAL NOT NULL DEFAULT 9.0,
    regular_printing_hours  REAL DEFAULT 0.0,
    overtime_hours          REAL DEFAULT 0.0,
    overtime_status         TEXT NOT NULL DEFAULT 'Normal',
    total_layers            INTEGER,
    current_layer           INTEGER DEFAULT 0,
    completion_percent      REAL DEFAULT 0.0,
    status                  TEXT NOT NULL DEFAULT 'Queued'
);
CREATE INDEX IF NOT EXISTS idx_printing_jobs_operator ON printing_jobs(operator_id);
CREATE INDEX IF NOT EXISTS idx_printing_jobs_shift ON printing_jobs(shift_id);

-- 5. Shift Issues
CREATE TABLE IF NOT EXISTS shift_issues (
    issue_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    shift_id            INTEGER REFERENCES shifts(shift_id),
    operator_id         INTEGER REFERENCES operators(operator_id),
    problem_type        TEXT NOT NULL DEFAULT 'operational_issue',
    problem_description TEXT NOT NULL DEFAULT '',
    severity            TEXT NOT NULL DEFAULT 'Medium',
    solution            TEXT,
    status              TEXT NOT NULL DEFAULT 'Open',
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    resolved_at         DATETIME,
    wall_number         INTEGER,
    timestamp           REAL,
    issue_type          TEXT,
    description         TEXT,
    cause               TEXT,
    action_taken        TEXT,
    result              TEXT,
    downtime_minutes    REAL NOT NULL DEFAULT 0.0,
    rag_recommendation  TEXT,
    rag_verified        INTEGER DEFAULT 0,
    rag_source          TEXT
);
CREATE INDEX IF NOT EXISTS idx_issues_type_time ON shift_issues(issue_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_issues_shift ON shift_issues(shift_id);

-- 6. Machine Telemetry
CREATE TABLE IF NOT EXISTS machine_telemetry (
    telemetry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id   INTEGER REFERENCES machines(machine_id),
    timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP,
    speed        REAL,
    temperature  REAL,
    pressure     REAL,
    vibration    REAL,
    status       TEXT NOT NULL
);

-- 7. Safety Events
CREATE TABLE IF NOT EXISTS safety_events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id  TEXT NOT NULL,
    timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP,
    person_id  TEXT,
    event_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    status     TEXT NOT NULL DEFAULT 'New'
);

-- Legacy Support Tables (for backward compatibility)
CREATE TABLE IF NOT EXISTS production_records (
    production_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    shift_id           INTEGER NOT NULL REFERENCES shifts(shift_id),
    job_id             INTEGER REFERENCES print_jobs(id),
    wall_number        INTEGER NOT NULL,
    print_file         TEXT,
    wall_height        REAL NOT NULL DEFAULT 0.0 CHECK (wall_height >= 0.0),
    segment_layers     INTEGER NOT NULL DEFAULT 0 CHECK (segment_layers >= 0),
    layers_completed   INTEGER NOT NULL DEFAULT 0 CHECK (layers_completed >= 0),
    start_time         REAL NOT NULL,
    end_time           REAL,
    status             TEXT NOT NULL DEFAULT 'in_progress' CHECK (status IN ('in_progress', 'completed', 'aborted', 'defect')),
    is_continuation_of INTEGER REFERENCES production_records(production_id),
    notes              TEXT,
    created_at         REAL NOT NULL,
    UNIQUE (shift_id, wall_number)
);
CREATE INDEX IF NOT EXISTS idx_production_shift ON production_records(shift_id, wall_number);
CREATE INDEX IF NOT EXISTS idx_production_start ON production_records(start_time, status);
CREATE INDEX IF NOT EXISTS idx_production_continuation ON production_records(is_continuation_of) WHERE is_continuation_of IS NOT NULL;

CREATE TABLE IF NOT EXISTS handovers (
    handover_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    from_shift_id  INTEGER NOT NULL REFERENCES shifts(shift_id),
    to_shift_id    INTEGER REFERENCES shifts(shift_id),
    line_id        TEXT NOT NULL DEFAULT 'line_1',
    handover_time  REAL NOT NULL,
    completed_work TEXT,
    pending_work   TEXT,
    open_issues    TEXT,
    notes          TEXT,
    created_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_handovers_line_match ON handovers(line_id, handover_time DESC) WHERE to_shift_id IS NULL;

CREATE TABLE IF NOT EXISTS event_corrections (
    correction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type   TEXT NOT NULL,
    entity_id     INTEGER NOT NULL,
    field_name    TEXT NOT NULL,
    old_value     TEXT,
    new_value     TEXT NOT NULL,
    corrected_by  TEXT NOT NULL REFERENCES operators(operator_id),
    corrected_at  REAL NOT NULL,
    source        TEXT NOT NULL DEFAULT 'manual',
    reason        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corrections_entity ON event_corrections(entity_type, entity_id);
"""

_init_lock = threading.Lock()
_initialised: set = set()


@contextmanager
def connect(db_path: Optional[str] = None):
    """Connection with row access by name. Commits on success, rolls back on error."""
    path = db_path or config.DATABASE_PATH
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_MIGRATIONS = [
    ("print_jobs", "deadline_at", "REAL"),
    ("print_jobs", "operator", "TEXT"),
    ("print_jobs", "shift", "TEXT"),
    ("print_jobs", "material_total_kg", "REAL"),
    ("print_jobs", "material_used_kg", "REAL"),
    ("print_jobs", "wall_height_mm", "REAL"),
    ("print_jobs", "wall_width_mm", "REAL"),
    ("print_jobs", "layer_height_mm", "REAL"),
    ("operators", "status", "TEXT NOT NULL DEFAULT 'Active'"),
    ("operators", "current_work_status", "TEXT NOT NULL DEFAULT 'Off Shift'"),
    ("operators", "active_shift_id", "INTEGER"),
    ("operators", "assigned_line", "TEXT DEFAULT 'line_1'"),
    ("shifts", "day", "TEXT DEFAULT 'Thursday'"),
    ("shifts", "planned_start_time", "DATETIME DEFAULT '2026-10-08 08:00:00'"),
    ("shifts", "planned_end_time", "DATETIME DEFAULT '2026-10-08 17:00:00'"),
    ("shifts", "actual_start_time", "DATETIME"),
    ("shifts", "actual_end_time", "DATETIME"),
    ("shifts", "total_hours", "REAL"),
    ("shifts", "attendance_status", "TEXT NOT NULL DEFAULT 'Scheduled'"),
    ("shifts", "late_minutes", "INTEGER DEFAULT 0"),
    ("shifts", "overtime_hours", "REAL DEFAULT 0.0"),
    ("shifts", "overtime_status", "TEXT DEFAULT 'Normal'"),
    ("shifts", "regular_hours", "REAL DEFAULT 0.0"),
    ("shifts", "standard_hours", "REAL DEFAULT 9.0"),
    ("shifts", "current_status", "TEXT DEFAULT 'Working'"),
    ("shifts", "operator_name", "TEXT"),
    ("shifts", "employee_code", "TEXT"),
    ("shift_issues", "problem_type", "TEXT DEFAULT 'operational_issue'"),
    ("shift_issues", "problem_description", "TEXT DEFAULT ''"),
    ("shift_issues", "solution", "TEXT"),
    ("shift_issues", "status", "TEXT NOT NULL DEFAULT 'Open'"),
    ("shift_issues", "resolved_at", "DATETIME"),
    ("shift_issues", "operator_id", "INTEGER"),
    ("shift_issues", "rag_recommendation", "TEXT"),
    ("shift_issues", "rag_verified", "INTEGER DEFAULT 0"),
    ("shift_issues", "rag_source", "TEXT"),
]


def _migrate_shifts_table_if_needed(conn) -> None:
    try:
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='shifts'").fetchone()
        if not row or not row["sql"]:
            return
        sql = row["sql"]
        if "'Evening'" not in sql or "'Day'" not in sql or "'Scheduled'" not in sql:
            log.info("Migrating shifts table schema to support new shift_type and status constraints...")
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute("""
            CREATE TABLE shifts_migrated (
                shift_id           INTEGER PRIMARY KEY AUTOINCREMENT,
                operator_id        TEXT NOT NULL,
                line_id            TEXT NOT NULL DEFAULT 'line_1',
                shift_date         TEXT NOT NULL,
                day                TEXT NOT NULL DEFAULT 'Thursday',
                shift_type         TEXT NOT NULL CHECK(shift_type IN ('Morning', 'Day', 'Evening', 'Night', 'Afternoon')),
                planned_start_time DATETIME NOT NULL DEFAULT '2026-10-08 08:00:00',
                planned_end_time   DATETIME NOT NULL DEFAULT '2026-10-08 17:00:00',
                actual_start_time  DATETIME,
                actual_end_time    DATETIME,
                total_hours        REAL,
                attendance_status  TEXT NOT NULL DEFAULT 'Scheduled',
                late_minutes       INTEGER DEFAULT 0,
                overtime_hours     REAL DEFAULT 0.0,
                overtime_status    TEXT DEFAULT 'Normal',
                status             TEXT NOT NULL DEFAULT 'Scheduled' CHECK(status IN ('Scheduled', 'Active', 'Completed', 'Absent', 'active', 'closed', 'flagged', 'abandoned')),
                start_time         REAL,
                end_time           REAL,
                target_walls       INTEGER NOT NULL DEFAULT 0,
                summary_notes      TEXT,
                created_at         DATETIME DEFAULT CURRENT_TIMESTAMP,
                regular_hours      REAL DEFAULT 0.0,
                standard_hours     REAL DEFAULT 9.0,
                current_status     TEXT DEFAULT 'Working',
                operator_name      TEXT,
                employee_code      TEXT
            );
            """)
            src_cols = [c["name"] for c in conn.execute("PRAGMA table_info(shifts)").fetchall()]
            target_cols = [c["name"] for c in conn.execute("PRAGMA table_info(shifts_migrated)").fetchall()]
            common_cols = [c for c in target_cols if c in src_cols]
            cols_str = ", ".join(common_cols)
            conn.execute(f"INSERT INTO shifts_migrated ({cols_str}) SELECT {cols_str} FROM shifts")
            conn.execute("DROP TABLE shifts")
            conn.execute("ALTER TABLE shifts_migrated RENAME TO shifts")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_shifts_single_active_line ON shifts(line_id) WHERE status IN ('active', 'Active')")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_shifts_date ON shifts(shift_date, shift_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_shifts_operator ON shifts(operator_id, start_time DESC)")
            conn.execute("PRAGMA foreign_keys=ON")
            log.info("Successfully migrated shifts table schema.")
    except Exception as exc:
        log.warning("shifts table migration failed: %s", exc)


def _migrate_shift_issues_table_if_needed(conn) -> None:
    try:
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='shift_issues'").fetchone()
        if not row or not row["sql"]:
            return
        sql = row["sql"]
        if "'Low'" not in sql or "'Medium'" not in sql or "'High'" not in sql:
            log.info("Migrating shift_issues table schema to support new severity and status constraints...")
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute("""
            CREATE TABLE shift_issues_migrated (
                issue_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                shift_id            INTEGER REFERENCES shifts(shift_id),
                operator_id         INTEGER,
                problem_type        TEXT NOT NULL DEFAULT 'operational_issue',
                problem_description TEXT NOT NULL DEFAULT '',
                solution            TEXT,
                severity            TEXT NOT NULL DEFAULT 'Low' CHECK(severity IN ('Low', 'Medium', 'High', 'Critical', 'info', 'warning', 'critical')),
                status              TEXT NOT NULL DEFAULT 'Open' CHECK(status IN ('Open', 'Resolved')),
                rag_recommendation  TEXT,
                rag_verified        INTEGER DEFAULT 0,
                rag_source          TEXT,
                created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
                resolved_at         DATETIME,
                wall_number         INTEGER,
                timestamp           REAL,
                issue_type          TEXT,
                description         TEXT,
                cause               TEXT,
                action_taken        TEXT,
                result              TEXT,
                downtime_minutes    REAL DEFAULT 0.0
            );
            """)
            src_cols = [c["name"] for c in conn.execute("PRAGMA table_info(shift_issues)").fetchall()]
            target_cols = [c["name"] for c in conn.execute("PRAGMA table_info(shift_issues_migrated)").fetchall()]
            common_cols = [c for c in target_cols if c in src_cols]
            cols_str = ", ".join(common_cols)
            conn.execute(f"INSERT INTO shift_issues_migrated ({cols_str}) SELECT {cols_str} FROM shift_issues")
            conn.execute("DROP TABLE shift_issues")
            conn.execute("ALTER TABLE shift_issues_migrated RENAME TO shift_issues")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_shift_issues_shift ON shift_issues(shift_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_shift_issues_operator ON shift_issues(operator_id)")
            conn.execute("PRAGMA foreign_keys=ON")
            log.info("Successfully migrated shift_issues table schema.")
    except Exception as exc:
        log.warning("shift_issues table migration failed: %s", exc)


def _apply_migrations(conn) -> None:
    _migrate_shifts_table_if_needed(conn)
    _migrate_shift_issues_table_if_needed(conn)
    for table, column, coltype in _MIGRATIONS:
        try:
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if existing and column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
                log.info("Migrated: added %s.%s", table, column)
        except Exception as exc:
            log.warning("Migration failed for %s.%s: %s", table, column, exc)


def init_db(db_path: Optional[str] = None) -> None:
    """Creates tables if absent and applies migrations. Safe to call repeatedly."""
    path = db_path or config.DATABASE_PATH
    with _init_lock:
        if path != ":memory:" and os.path.exists(path) and os.path.getsize(path) > 0:
            bak_path = f"{path}.bak.{int(time.time())}"
            try:
                shutil.copy2(path, bak_path)
                log.info("Created database backup: %s", bak_path)
            except Exception as exc:
                log.warning("Could not backup database: %s", exc)
        with connect(path) as conn:
            conn.executescript(_SCHEMA)
            _apply_migrations(conn)
        if path not in _initialised:
            log.info("Database ready at %s", path)
            _initialised.add(path)


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None


# --- print jobs ---------------------------------------------------------

def create_print_job(
    job_name: str,
    total_length_mm: float,
    material: Optional[str] = None,
    total_layers: Optional[int] = None,
    db_path: Optional[str] = None,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO print_jobs (job_name, material, total_length_mm, "
            "completed_length_mm, total_layers, current_layer, status, started_at) "
            "VALUES (?, ?, ?, 0, ?, 0, 'running', ?)",
            (job_name, material, total_length_mm, total_layers, time.time()),
        )
        return int(cur.lastrowid)


def get_active_job(db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Most recently started job that is still running."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM print_jobs WHERE status = 'running' "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    return _row_to_dict(row)


def get_job(job_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM print_jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_dict(row)


def update_job_progress(
    job_id: int,
    completed_length_mm: Optional[float] = None,
    current_layer: Optional[int] = None,
    db_path: Optional[str] = None,
) -> None:
    """Updates only what is supplied. remaining_length_mm recomputes itself."""
    sets, params = [], []
    if completed_length_mm is not None:
        sets.append("completed_length_mm = ?")
        params.append(float(completed_length_mm))
    if current_layer is not None:
        sets.append("current_layer = ?")
        params.append(int(current_layer))
    if not sets:
        return
    params.append(job_id)
    with connect(db_path) as conn:
        conn.execute(f"UPDATE print_jobs SET {', '.join(sets)} WHERE id = ?", params)


_JOB_DETAIL_COLUMNS = {
    "deadline_at", "operator", "shift", "material_total_kg", "material_used_kg",
    "wall_height_mm", "wall_width_mm", "layer_height_mm", "material", "total_layers",
}


def update_job_details(job_id: int, db_path: Optional[str] = None, **fields) -> None:
    """
    Sets operator-supplied job details (deadline, operator, material totals,
    wall geometry). Unknown keys are ignored rather than trusted.
    """
    sets, params = [], []
    for key, value in fields.items():
        if key in _JOB_DETAIL_COLUMNS and value is not None:
            sets.append(f"{key} = ?")
            params.append(value)
    if not sets:
        return
    params.append(job_id)
    with connect(db_path) as conn:
        conn.execute(f"UPDATE print_jobs SET {', '.join(sets)} WHERE id = ?", params)


def get_state_history(
    since_seconds: float,
    limit: int = 500,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Machine-state rows newer than `since_seconds` ago, oldest first."""
    cutoff = time.time() - since_seconds
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT timestamp, printing, completed_length_mm, layer_width_mm, "
            "confidence, motion_score, status FROM machine_state "
            "WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_progress_at_or_before(timestamp: float, db_path: Optional[str] = None) -> Optional[float]:
    """
    completed_length_mm as of a past moment, used to measure production over a
    window. Returns None when no reading that old exists (so the UI can say the
    window isn't covered yet instead of implying zero production).
    """
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT completed_length_mm FROM machine_state "
            "WHERE timestamp <= ? AND completed_length_mm IS NOT NULL "
            "ORDER BY timestamp DESC LIMIT 1",
            (timestamp,),
        ).fetchone()
    return None if row is None else row["completed_length_mm"]


def count_events_since(
    since_seconds: float,
    event_type: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    cutoff = time.time() - since_seconds
    query = "SELECT COUNT(*) AS n FROM machine_events WHERE timestamp >= ?"
    params: List[Any] = [cutoff]
    if event_type:
        query += " AND event_type = ?"
        params.append(event_type)
    with connect(db_path) as conn:
        return int(conn.execute(query, params).fetchone()["n"])


def complete_job(job_id: int, status: str = "completed", db_path: Optional[str] = None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE print_jobs SET status = ?, completed_at = ? WHERE id = ?",
            (status, time.time(), job_id),
        )


# --- machine state ------------------------------------------------------

def record_machine_state(
    printing: Optional[bool] = None,
    status: Optional[str] = None,
    current_layer: Optional[int] = None,
    layer_width_mm: Optional[float] = None,
    layer_width_px: Optional[float] = None,
    completed_length_mm: Optional[float] = None,
    motion_score: Optional[float] = None,
    confidence: Optional[float] = None,
    camera_connected: Optional[bool] = None,
    job_id: Optional[int] = None,
    timestamp: Optional[float] = None,
    db_path: Optional[str] = None,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO machine_state (job_id, printing, status, current_layer, "
            "layer_width_mm, layer_width_px, completed_length_mm, motion_score, "
            "confidence, camera_connected, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                None if printing is None else int(printing),
                status,
                current_layer,
                layer_width_mm,
                layer_width_px,
                completed_length_mm,
                motion_score,
                confidence,
                None if camera_connected is None else int(camera_connected),
                timestamp if timestamp is not None else time.time(),
            ),
        )
        return int(cur.lastrowid)


def get_latest_machine_state(db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM machine_state ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    state = _row_to_dict(row)
    if state is not None:
        for key in ("printing", "camera_connected"):
            if state.get(key) is not None:
                state[key] = bool(state[key])
    return state


# --- layer measurements -------------------------------------------------

def record_layer_measurement(
    layer_number: Optional[int],
    width_px: Optional[float] = None,
    width_mm: Optional[float] = None,
    confidence: Optional[float] = None,
    job_id: Optional[int] = None,
    timestamp: Optional[float] = None,
    db_path: Optional[str] = None,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO layer_measurements (job_id, layer_number, width_px, "
            "width_mm, confidence, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (
                job_id,
                layer_number,
                width_px,
                width_mm,
                confidence,
                timestamp if timestamp is not None else time.time(),
            ),
        )
        return int(cur.lastrowid)


def get_recent_layer_measurements(limit: int = 20, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM layer_measurements ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# --- events -------------------------------------------------------------

def record_event(
    event_type: str,
    message: str = "",
    job_id: Optional[int] = None,
    timestamp: Optional[float] = None,
    db_path: Optional[str] = None,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO machine_events (job_id, event_type, message, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (job_id, event_type, message, timestamp if timestamp is not None else time.time()),
        )
        return int(cur.lastrowid)


def get_recent_events(limit: int = 20, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM machine_events ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
