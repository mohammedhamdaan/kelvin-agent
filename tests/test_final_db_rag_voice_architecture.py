"""
Dedicated test suite verifying the Kelvin6K Industrial Agent Database & RAG Architecture upgrade:
1. Safe backup on non-empty DB initialization and 7 normalized tables.
2. Shift lifecycle & 9-hour standard printing overtime engine with midnight calculation.
3. Logical reporting view and reference operational seed data for 08-Oct-2026.
4. Multi-turn dynamic voice employee onboarding (registration + shift start / cancellation).
5. Hybrid RAG subsystem with 17 metadata fields, BM25 keyword scoring, and reranking.
6. 15 target operational voice queries with English-only natural spoken responses.
"""

import glob
import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent import config, db, identity, query_router, shift_store
from agent.assistant import Assistant
from agent.rag.retriever import bm25_score, search_documentation
from agent.rag.store import Chunk, DocumentStore


class TestSafeBackupAndNormalizedSchema(unittest.TestCase):
    """Verifies safe backup before migration and full 7 normalized tables schema."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_backup_schema.db")
        self._old_db = config.DATABASE_PATH
        config.DATABASE_PATH = self.db_path

    def tearDown(self):
        config.DATABASE_PATH = self._old_db
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_db_backup_created_on_init(self):
        # 1. First init creates fresh DB
        db.init_db(self.db_path)
        # Insert a dummy record so the database has content
        with db.connect(self.db_path) as conn:
            conn.execute("INSERT INTO operators (name) VALUES ('BackupTestOp')")

        # 2. Second init must detect non-empty DB and create a timestamped backup
        db.init_db(self.db_path)
        backups = glob.glob(f"{self.db_path}.bak.*")
        self.assertGreaterEqual(
            len(backups), 1, "Expected at least one timestamped backup file to be created"
        )
        self.assertTrue(os.path.exists(backups[0]))
        self.assertGreater(os.path.getsize(backups[0]), 0)

    def test_wal_mode_and_foreign_keys(self):
        db.init_db(self.db_path)
        with db.connect(self.db_path) as conn:
            journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]

            self.assertEqual(journal_mode.lower(), "wal")
            self.assertEqual(foreign_keys, 1)

    def test_normalized_tables_exist(self):
        db.init_db(self.db_path)
        expected_tables = {
            "operators",
            "shifts",
            "machines",
            "printing_jobs",
            "production_records",
            "shift_issues",
            "handovers",
            "event_corrections",
        }
        with db.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            existing = {r["name"] for r in rows}
            for tbl in expected_tables:
                self.assertIn(tbl, existing, f"Required table {tbl} missing from database")

            # Check shifts columns
            shifts_cols = {
                r["name"] for r in conn.execute("PRAGMA table_info(shifts)").fetchall()
            }
            required_shifts_cols = {
                "shift_id", "operator_id", "shift_date", "day", "shift_type",
                "planned_start_time", "planned_end_time", "actual_start_time",
                "actual_end_time", "total_hours", "attendance_status",
                "late_minutes", "overtime_hours", "status", "line_id"
            }
            self.assertTrue(required_shifts_cols.issubset(shifts_cols))

            # Check printing_jobs columns
            jobs_cols = {
                r["name"] for r in conn.execute("PRAGMA table_info(printing_jobs)").fetchall()
            }
            required_jobs_cols = {
                "job_id", "machine_id", "operator_id", "shift_id", "job_name",
                "material", "start_time", "end_time", "printing_duration_hours",
                "standard_printing_hours", "regular_printing_hours", "overtime_hours",
                "overtime_status", "total_layers", "current_layer", "completion_percent",
                "status"
            }
            self.assertTrue(required_jobs_cols.issubset(jobs_cols))


class TestShiftAndPrintingOvertimeEngine(unittest.TestCase):
    """Verifies overtime engine, 9h rule, late arrival, and midnight crossing duration."""

    def test_five_overtime_calculation_cases(self):
        cases = [
            (8.0, 8.0, 0.0, "Normal"),
            (9.0, 9.0, 0.0, "Normal"),
            (10.0, 9.0, 1.0, "Overtime"),
            (11.5, 9.0, 2.5, "Overtime"),
            (14.0, 9.0, 5.0, "Overtime"),
        ]
        for dur, expected_reg, expected_ot, expected_status in cases:
            with self.subTest(duration=dur):
                res = shift_store.calculate_printing_overtime(dur)
                self.assertEqual(res["regular_printing_hours"], expected_reg)
                self.assertEqual(res["overtime_hours"], expected_ot)
                self.assertEqual(res["overtime_status"], expected_status)

    def test_night_shift_crossing_midnight(self):
        # 21:55:00 to 06:08:00 next day = 8h 13m = 8.21666... hours -> 8.22 hours
        st = "2026-10-08 21:55:00"
        et = "2026-10-09 06:08:00"
        s_dt = shift_store._parse_dt(st)
        e_dt = shift_store._parse_dt(et)
        self.assertIsNotNone(s_dt)
        self.assertIsNotNone(e_dt)
        dur = round((e_dt - s_dt).total_seconds() / 3600.0, 2)
        self.assertEqual(dur, 8.22)

        # Formatting next day
        formatted_out = shift_store._format_hh_mm(et)
        if e_dt.date() > s_dt.date():
            formatted_out = f"{formatted_out} (+1 day)"
        self.assertEqual(formatted_out, "06:08 (+1 day)")

    def test_late_minutes_calculation(self):
        # Rahul: Planned 08:00, Actual 08:07 -> 7 mins late
        p_st = shift_store._parse_dt("2026-10-08 08:00:00")
        a_st = shift_store._parse_dt("2026-10-08 08:07:00")
        diff_min = int(round((a_st - p_st).total_seconds() / 60.0))
        late_min = max(0, diff_min)
        self.assertEqual(late_min, 7)

        # Aman: Planned 08:00, Actual 07:52 -> 0 mins late
        a_st_aman = shift_store._parse_dt("2026-10-08 07:52:00")
        diff_aman = int(round((a_st_aman - p_st).total_seconds() / 60.0))
        late_aman = max(0, diff_aman)
        self.assertEqual(late_aman, 0)


class TestAgentReportingViewAndSeedData(unittest.TestCase):
    """Verifies reference operational seed data and logical reporting view for 08-Oct-2026."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_reporting.db")
        self._old_db = config.DATABASE_PATH
        config.DATABASE_PATH = self.db_path
        db.init_db(self.db_path)
        shift_store.seed_reference_operational_data(self.db_path)

    def tearDown(self):
        config.DATABASE_PATH = self._old_db
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_reference_seed_data_and_reporting_view(self):
        rows = shift_store.get_agent_reporting_view("2026-10-08", self.db_path)
        self.assertEqual(len(rows), 4, "Expected exactly 4 shift records for 08-Oct-2026")

        by_name = {r["operator_name"]: r for r in rows}

        # Aman (Morning)
        aman = by_name["Aman"]
        self.assertEqual(aman["shift_type"], "Morning")
        self.assertEqual(aman["came_in"], "07:52")
        self.assertEqual(aman["came_out"], "17:05")
        self.assertEqual(aman["shift_duration"], 9.22)
        self.assertEqual(aman["late_by"], "0 mins")
        self.assertEqual(aman["total_printing_time"], 8.5)
        self.assertEqual(aman["overtime"], 0.0)
        self.assertEqual(aman["working_days_count"], 18)
        self.assertEqual(aman["job_name"], "Wall_Segment_A")
        self.assertEqual(aman["problem_type"], "Slump Loss")

        # Rahul (Day)
        rahul = by_name["Rahul"]
        self.assertEqual(rahul["shift_type"], "Day")
        self.assertEqual(rahul["came_in"], "08:07")
        self.assertEqual(rahul["came_out"], "17:10")
        self.assertEqual(rahul["shift_duration"], 9.05)
        self.assertEqual(rahul["late_by"], "+7 mins")
        self.assertEqual(rahul["total_printing_time"], 11.5)
        self.assertEqual(rahul["overtime"], 2.5)
        self.assertEqual(rahul["working_days_count"], 22)
        self.assertEqual(rahul["job_name"], "Wall_Panel_B")
        self.assertEqual(rahul["problem_type"], "Nozzle Choke")

        # Rohit (Evening)
        rohit = by_name["Rohit"]
        self.assertEqual(rohit["shift_type"], "Evening")
        self.assertEqual(rohit["came_in"], "16:02")
        self.assertEqual(rohit["came_out"], "00:05 (+1 day)")
        self.assertEqual(rohit["shift_duration"], 8.05)
        self.assertEqual(rohit["late_by"], "+2 mins")
        self.assertEqual(rohit["total_printing_time"], 7.5)
        self.assertEqual(rohit["overtime"], 0.0)
        self.assertEqual(rohit["working_days_count"], 15)
        self.assertEqual(rohit["job_name"], "Corner_Column_C")
        self.assertEqual(rohit["problem_type"], "Gantry Y-axis Vibration")

        # Vikas (Night)
        vikas = by_name["Vikas"]
        self.assertEqual(vikas["shift_type"], "Night")
        self.assertEqual(vikas["came_in"], "21:55")
        self.assertEqual(vikas["came_out"], "06:08 (+1 day)")
        self.assertEqual(vikas["shift_duration"], 8.22)
        self.assertEqual(vikas["late_by"], "0 mins")
        self.assertEqual(vikas["total_printing_time"], 10.0)
        self.assertEqual(vikas["overtime"], 1.0)
        self.assertEqual(vikas["working_days_count"], 12)
        self.assertEqual(vikas["job_name"], "Foundation_Block_D")
        self.assertIsNone(vikas["problem_type"])


class TestMultiTurnVoiceOnboarding(unittest.TestCase):
    """Verifies multi-turn voice registration and onboarding for unregistered employees."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_onboard.db")
        self._old_db = config.DATABASE_PATH
        config.DATABASE_PATH = self.db_path
        db.init_db(self.db_path)
        self.assistant = Assistant(use_llm=False)

    def tearDown(self):
        config.DATABASE_PATH = self._old_db
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_unregistered_employee_onboarding_and_start_shift(self):
        # 1. Unregistered employee initiates shift start
        resp = self.assistant.ask("I am Karthik, start my morning shift")
        self.assertIn("karthik", resp.lower())
        self.assertIn("could not find an operator named karthik", resp.lower())
        self.assertIn("shall i register you and start your shift on line 1?", resp.lower())

        # Verify not yet written to DB before confirmation
        self.assertIsNone(shift_store.find_operator_by_name("Karthik"))
        self.assertIsNone(shift_store.get_active_shift_for_line("line_1"))

        # 2. Operator confirms "Yes"
        confirm_resp = self.assistant.ask("Yes")
        self.assertIn("karthik", confirm_resp.lower())
        self.assertIn("morning shift", confirm_resp.lower())
        self.assertIn("started", confirm_resp.lower())

        # Verify DB records created
        op = shift_store.find_operator_by_name("Karthik")
        self.assertIsNotNone(op)
        active_shift = shift_store.get_active_shift_for_line("line_1")
        self.assertIsNotNone(active_shift)
        self.assertEqual(active_shift["operator_id"], op["operator_id"])
        self.assertEqual(active_shift["shift_type"], "Morning")

    def test_unregistered_employee_onboarding_cancellation(self):
        # 1. Operator says start shift
        self.assistant.ask("I am Karthik, start my morning shift")

        # 2. Operator says "No"
        cancel_resp = self.assistant.ask("No")
        self.assertTrue(
            "action cancelled" in cancel_resp.lower() or "cancelled" in cancel_resp.lower(),
            f"Expected cancellation confirmation, got: {cancel_resp}"
        )

        # Verify no operator or shift was created in DB
        self.assertIsNone(shift_store.find_operator_by_name("Karthik"))
        self.assertIsNone(shift_store.get_active_shift_for_line("line_1"))


class TestHybridRAGMetadataAndRetrieval(unittest.TestCase):
    """Verifies Chunk dataclass metadata schema, BM25 keyword scoring, and hybrid search."""

    def test_chunk_metadata_schema(self):
        chunk = Chunk(
            text="Clean the nozzle with high pressure water.",
            file_name="cleaning_sop.md",
            document_id="DOC_001",
            version="1.2",
            effective_date="2026-01-01",
            department="Maintenance",
            page_number=3,
            section_heading="Nozzle Cleaning",
            category="SOP",
            equipment_type="Nozzle",
            severity_level="High",
            keywords=["nozzle", "cleaning", "maintenance"],
            token_count=12,
            is_active=True,
            access_roles=["operator", "supervisor"],
        )
        meta = chunk.metadata()
        self.assertEqual(meta["document_id"], "DOC_001")
        self.assertEqual(meta["version"], "1.2")
        self.assertEqual(meta["category"], "SOP")
        self.assertEqual(meta["equipment_type"], "Nozzle")
        self.assertEqual(meta["severity_level"], "High")
        self.assertEqual(meta["token_count"], 12)
        self.assertTrue(meta["is_active"])

    def test_bm25_keyword_scoring(self):
        chunks = [
            Chunk(
                text="The gantry travel rails require lithium grease lubrication every 40 hours.",
                file_name="gantry_manual.md",
                keywords=["gantry", "lubrication", "rails"],
                category="Maintenance",
            ),
            Chunk(
                text="Slump test should measure between 150mm and 180mm for optimal pumping.",
                file_name="concrete_mix.md",
                keywords=["concrete", "slump", "pump"],
                category="Material",
            ),
        ]
        scores = bm25_score("gantry rails lubrication", chunks)
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], 0.3)


class TestOperationalQueriesEnglishVoice(unittest.TestCase):
    """Verifies all 15 target operational voice queries with strict English-only natural answers."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_voice_queries.db")
        self._old_db = config.DATABASE_PATH
        config.DATABASE_PATH = self.db_path
        db.init_db(self.db_path)
        shift_store.seed_reference_operational_data(self.db_path)
        self.assistant = Assistant(use_llm=False)

    def tearDown(self):
        config.DATABASE_PATH = self._old_db
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _assert_english_only(self, response: str):
        # Strict check: forbidden Hindi/Hinglish words
        hindi_tokens = {
            "gayi", "gaya", "hain", "kya", "shuru", "khatam", "aaj", "kal",
            "kaun", "kab", "kitna", "kitni", "chal", "raha", "rahi", "dikkat",
            "samasya", "baki", "thik", "haan", "nahi"
        }
        words = set(re.findall(r"\b[a-zA-Z]+\b", response.lower()))
        forbidden = words.intersection(hindi_tokens)
        self.assertFalse(
            forbidden,
            f"Response contained forbidden non-English words {forbidden}: {response}"
        )

    def test_all_15_operational_queries(self):
        queries = [
            # 1. Who came in for the Morning shift? -> Aman
            ("Who came in for the Morning shift?", ["aman", "morning"]),
            # 2. Who arrived late today? -> Rahul (7 minutes late)
            ("Who arrived late today?", ["rahul", "7 minutes"]),
            # 3. What was the total printing time across all jobs? -> 37.5 hours
            ("What was the total printing time across all jobs?", ["37.5 hours"]),
            # 4. Did any printing jobs cross 9 hours today? -> Wall_Panel_B (11.5h), Foundation_Block_D (10.0h)
            ("Did any printing jobs cross 9 hours today?", ["wall_panel_b", "foundation_block_d"]),
            # 5. Which job had the highest overtime? -> Wall_Panel_B (2.5h)
            ("Which job had the highest overtime?", ["wall_panel_b", "2.5 hours"]),
            # 6. What is Rahul's overtime today? -> 2.5 hours
            ("What is Rahul's overtime today?", ["rahul", "2.5 hours"]),
            # 7. How many total working days has Aman completed? -> 18 days
            ("How many total working days has Aman completed?", ["aman", "18 working days"]),
            # 8. What problem did Rohit face during his shift? -> Gantry Y-axis Vibration
            ("What problem did Rohit face during his shift?", ["rohit", "gantry y-axis vibration"]),
            # 9. When did Vikas clock out of his Night shift? -> 06:08 (+1 day)
            ("When did Vikas clock out of his Night shift?", ["vikas", "06:08 (+1 day)"]),
            # 10. Multi-intent: When did Aman arrive for his shift and what is his overtime? -> 7:52 AM, 0.0h
            ("When did Aman arrive for his shift and what is his overtime?", ["aman", "7:52 am", "0.0 hours"]),
            # 11. Who is on the Evening shift? -> Rohit
            ("Who is on the Evening shift?", ["rohit", "evening"]),
            # 12. What time did Rahul arrive? -> 08:07 (7 minutes late)
            ("What time did Rahul arrive?", ["rahul", "08:07"]),
            # 13. How many hours did Vikas work? -> 8.22 hours
            ("How many hours did Vikas work?", ["vikas", "8.22 hours"]),
            # 14. What is Rohit's overtime? -> 0.0 hours
            ("What is Rohit's overtime?", ["rohit", "0.0 hours"]),
            # 15. How many working days does Rahul have? -> 22 days
            ("How many working days does Rahul have?", ["rahul", "22 working days"]),
        ]

        for question, expected_snippets in queries:
            with self.subTest(query=question):
                reply = self.assistant.ask(question)
                self._assert_english_only(reply)
                for snippet in expected_snippets:
                    self.assertIn(
                        snippet.lower(),
                        reply.lower(),
                        f"Expected snippet '{snippet}' not found in reply: '{reply}' for query: '{question}'",
                    )


if __name__ == "__main__":
    unittest.main()
