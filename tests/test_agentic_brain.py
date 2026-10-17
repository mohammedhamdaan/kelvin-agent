"""
Comprehensive test suite for the Gemini-Powered Autonomous Agentic Brain,
9-Hour Printing Overtime Engine, Dynamic Tool Calling, and Live RAG Ingestion.
"""

import datetime
import os
import shutil
import tempfile
import unittest

from agent import config, db, shift_store
from agent.assistant import Assistant, SYSTEM_PROMPT, TOOLS
from agent.rag.ingest import learn_and_store_in_rag_brain
from agent.rag.retriever import get_store, search_rag_brain, search_documentation
from agent.spoken import format_spoken_response, speak_duration, speak_time


class TestAgenticBrainAndOvertime(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self._old_db = config.DATABASE_PATH
        self._old_docs = config.RAG_DOCS_DIR
        self._old_index = config.RAG_INDEX_DIR

        self.test_db = os.path.join(self._tmp_dir, "test_kelvin.db")
        self.test_docs = os.path.join(self._tmp_dir, "docs")
        self.test_index = os.path.join(self._tmp_dir, "rag_index")

        os.makedirs(self.test_docs, exist_ok=True)
        os.makedirs(self.test_index, exist_ok=True)

        config.DATABASE_PATH = self.test_db
        config.RAG_DOCS_DIR = self.test_docs
        config.RAG_INDEX_DIR = self.test_index

        db.init_db(self.test_db)
        get_store(self.test_index, refresh=True)

    def tearDown(self):
        config.DATABASE_PATH = self._old_db
        config.RAG_DOCS_DIR = self._old_docs
        config.RAG_INDEX_DIR = self._old_index
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. Configuration & .env Verification
    # -------------------------------------------------------------------------
    def test_config_and_env_variables(self):
        """Verify clean Gemini configuration variables and defaults."""
        self.assertTrue(hasattr(config, "GEMINI_API_KEY"))
        self.assertTrue(hasattr(config, "GEMINI_MODEL"))
        self.assertTrue(hasattr(config, "GEMINI_EMBEDDING_MODEL"))
        self.assertEqual(config.GEMINI_EMBED_MODEL, config.GEMINI_EMBEDDING_MODEL)

    # -------------------------------------------------------------------------
    # 2. 9-Hour Printing Overtime Rule & Midnight-Crossing Shift Calculations
    # -------------------------------------------------------------------------
    def test_9_hour_printing_overtime_math(self):
        """Verify the strict 9-hour printing overtime calculation."""
        # Case 1: Under standard time (8.5h)
        ot_normal = shift_store.calculate_printing_overtime(8.5)
        self.assertEqual(ot_normal["standard_printing_hours"], 9.0)
        self.assertEqual(ot_normal["regular_printing_hours"], 8.5)
        self.assertEqual(ot_normal["overtime_hours"], 0.0)
        self.assertEqual(ot_normal["overtime_status"], "Normal")

        # Case 2: Exactly 9.0 hours
        ot_exact = shift_store.calculate_printing_overtime(9.0)
        self.assertEqual(ot_exact["regular_printing_hours"], 9.0)
        self.assertEqual(ot_exact["overtime_hours"], 0.0)
        self.assertEqual(ot_exact["overtime_status"], "Normal")

        # Case 3: Overtime (11.5 hours)
        ot_over = shift_store.calculate_printing_overtime(11.5)
        self.assertEqual(ot_over["regular_printing_hours"], 9.0)
        self.assertEqual(ot_over["overtime_hours"], 2.5)
        self.assertEqual(ot_over["overtime_status"], "Overtime")

        # Case 4: Overtime (10.0 hours)
        ot_10 = shift_store.calculate_printing_overtime(10.0)
        self.assertEqual(ot_10["regular_printing_hours"], 9.0)
        self.assertEqual(ot_10["overtime_hours"], 1.0)
        self.assertEqual(ot_10["overtime_status"], "Overtime")

    def test_seed_reference_operational_data_reporting_view(self):
        """Verify reference operational data for 2026-10-08 and SQL join reporting view."""
        shift_store.seed_reference_operational_data(db_path=self.test_db)
        reporting = shift_store.get_agent_reporting_view(shift_date="2026-10-08", db_path=self.test_db)

        self.assertEqual(len(reporting), 4)

        by_operator = {r["operator_name"]: r for r in reporting}

        # Aman: Morning
        aman = by_operator["Aman"]
        self.assertEqual(aman["shift_type"], "Morning")
        self.assertEqual(aman["came_in"], "07:52")
        self.assertEqual(aman["came_out"], "17:05")
        self.assertEqual(aman["job_name"], "Wall_Segment_A")
        self.assertEqual(aman["total_printing_time"], 8.5)
        self.assertEqual(aman["regular_time"], 8.5)
        self.assertEqual(aman["overtime"], 0.0)
        self.assertEqual(aman["overtime_status"], "Normal")
        self.assertEqual(aman["problem_type"], "Slump Loss")
        self.assertEqual(aman["working_days_count"], 18)

        # Rahul: Day (Overtime + Late)
        rahul = by_operator["Rahul"]
        self.assertEqual(rahul["shift_type"], "Day")
        self.assertEqual(rahul["came_in"], "08:07")
        self.assertEqual(rahul["came_out"], "17:10")
        self.assertEqual(rahul["late_minutes"], 7)
        self.assertEqual(rahul["late_by"], "+7 mins")
        self.assertEqual(rahul["job_name"], "Wall_Panel_B")
        self.assertEqual(rahul["total_printing_time"], 11.5)
        self.assertEqual(rahul["regular_time"], 9.0)
        self.assertEqual(rahul["overtime"], 2.5)
        self.assertEqual(rahul["overtime_status"], "Overtime")
        self.assertEqual(rahul["problem_type"], "Nozzle Choke")
        self.assertEqual(rahul["working_days_count"], 22)

        # Rohit: Evening (+1 day crossing midnight)
        rohit = by_operator["Rohit"]
        self.assertEqual(rohit["shift_type"], "Evening")
        self.assertEqual(rohit["came_in"], "16:02")
        self.assertIn("(+1 day)", rohit["came_out"])
        self.assertEqual(rohit["job_name"], "Corner_Column_C")
        self.assertEqual(rohit["total_printing_time"], 7.5)
        self.assertEqual(rohit["regular_time"], 7.5)
        self.assertEqual(rohit["overtime"], 0.0)
        self.assertEqual(rohit["overtime_status"], "Normal")
        self.assertEqual(rohit["working_days_count"], 15)

        # Vikas: Night (+1 day crossing midnight, 10.0h job -> 1.0h overtime)
        vikas = by_operator["Vikas"]
        self.assertEqual(vikas["shift_type"], "Night")
        self.assertEqual(vikas["came_in"], "21:55")
        self.assertIn("(+1 day)", vikas["came_out"])
        self.assertEqual(vikas["job_name"], "Foundation_Block_D")
        self.assertEqual(vikas["total_printing_time"], 10.0)
        self.assertEqual(vikas["regular_time"], 9.0)
        self.assertEqual(vikas["overtime"], 1.0)
        self.assertEqual(vikas["overtime_status"], "Overtime")
        self.assertIsNone(vikas["problem_type"])  # No issues
        self.assertEqual(vikas["working_days_count"], 12)

    # -------------------------------------------------------------------------
    # 3. High-Level Autonomous Agentic Tools (Read & Write)
    # -------------------------------------------------------------------------
    def test_employee_and_shift_tools(self):
        """Verify list_or_search, create_or_update, start_shift, end_shift, and query tools."""
        # 1. Create employee
        created = shift_store.create_or_update_employee(
            name="Hamdaan",
            role="Lead Engineer",
            status="Active",
            db_path=self.test_db,
        )
        self.assertTrue(created["success"])
        self.assertEqual(created["operator"]["name"], "Hamdaan")
        self.assertEqual(created["operator"]["role"], "Lead Engineer")

        # 2. Search employee
        search_res = shift_store.list_or_search_employees(name_query="Hamdaan", db_path=self.test_db)
        self.assertEqual(len(search_res), 1)
        self.assertEqual(search_res[0]["name"], "Hamdaan")

        # 3. Start shift with non-existent operator warns first
        unknown_start = shift_store.start_employee_shift("UnknownPerson", db_path=self.test_db)
        self.assertFalse(unknown_start["success"])
        self.assertTrue(unknown_start.get("requires_registration"))

        # 4. Start shift for registered employee
        shift_start = shift_store.start_employee_shift(
            operator_name="Hamdaan",
            shift_type="Morning",
            actual_start_time="2026-10-15 08:00:00",
            line_id="line_1",
            db_path=self.test_db,
        )
        self.assertTrue(shift_start["success"])
        shift_id = shift_start["shift_id"]
        self.assertGreater(shift_id, 0)

        # 5. End shift
        shift_end = shift_store.end_employee_shift(
            operator_name="Hamdaan",
            line_id="line_1",
            db_path=self.test_db,
        )
        self.assertTrue(shift_end["success"])
        self.assertEqual(shift_end["status"], "Completed")

        # 6. Query shifts
        shifts = shift_store.query_shifts_and_attendance(operator_name="Hamdaan", db_path=self.test_db)
        self.assertEqual(len(shifts), 1)
        self.assertEqual(shifts[0]["operator_name"], "Hamdaan")

    def test_printing_jobs_and_overtime_tools(self):
        """Verify start_or_update_printing_job and query_printing_jobs_and_overtime."""
        # Create an employee and shift first
        shift_store.create_or_update_employee("Karan", role="Operator", db_path=self.test_db)
        shift_store.start_employee_shift("Karan", shift_type="Day", db_path=self.test_db)

        # 1. Start printing job
        job_res = shift_store.start_or_update_printing_job(
            job_name="Bridge_Support_X",
            operator_name="Karan",
            duration_hours=12.5,
            status="Completed",
            total_layers=80,
            current_layer=80,
            db_path=self.test_db,
        )
        self.assertTrue(job_res["success"])
        job = job_res["job"]
        self.assertEqual(job["job_name"], "Bridge_Support_X")
        self.assertEqual(job["printing_duration_hours"], 12.5)
        self.assertEqual(job["regular_printing_hours"], 9.0)
        self.assertEqual(job["overtime_hours"], 3.5)
        self.assertEqual(job["overtime_status"], "Overtime")

        # 2. Query overtime jobs
        ot_jobs = shift_store.query_printing_jobs_and_overtime(only_overtime=True, db_path=self.test_db)
        self.assertEqual(len(ot_jobs), 1)
        self.assertEqual(ot_jobs[0]["job_name"], "Bridge_Support_X")
        self.assertEqual(ot_jobs[0]["overtime_hours"], 3.5)

    def test_issue_logging_tools(self):
        """Verify log_or_update_shift_issue tool."""
        shift_store.create_or_update_employee("Pooja", role="Operator", db_path=self.test_db)
        shift_store.start_employee_shift("Pooja", shift_type="Morning", db_path=self.test_db)

        # 1. Log open issue
        iss = shift_store.log_or_update_shift_issue(
            problem_type="Material Flow",
            problem_description="Viscosity too high at extruder nozzle",
            operator_name="Pooja",
            severity="High",
            db_path=self.test_db,
        )
        self.assertTrue(iss["success"])
        self.assertEqual(iss["issue"]["status"], "Open")

        # 2. Update issue with solution
        iss_updated = shift_store.log_or_update_shift_issue(
            problem_type="Material Flow",
            problem_description="Viscosity too high",
            solution="Added 0.3% superplasticizer",
            status="Resolved",
            db_path=self.test_db,
        )
        self.assertTrue(iss_updated["success"])
        self.assertEqual(iss_updated["issue"]["status"], "Resolved")
        self.assertEqual(iss_updated["issue"]["solution"], "Added 0.3% superplasticizer")

    # -------------------------------------------------------------------------
    # 4. Live RAG Brain Learning Tools
    # -------------------------------------------------------------------------
    def test_live_rag_learning_and_search(self):
        """Verify learn_and_store_in_rag_brain appends to md and indexes in memory immediately."""
        title = "Nozzle Flow Optimization at Low Ambient Temp"
        category = "Troubleshooting"
        content = "When ambient temperature falls below 15C, increase hose heater jacket to 28C."
        sol = "Set heating jacket to 28C and purge with warm slurry."

        result = learn_and_store_in_rag_brain(
            title=title,
            category=category,
            content=content,
            solution=sol,
            index_dir=self.test_index,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["title"], title)

        # Verify disk persistence
        learned_md = os.path.join(self.test_docs, "knowledge", "learned_knowledge.md")
        self.assertTrue(os.path.exists(learned_md))
        with open(learned_md, "r", encoding="utf-8") as f:
            text = f.read()
        self.assertIn("Nozzle Flow Optimization", text)
        self.assertIn("hose heater jacket", text)

        # Verify immediate retrieval without restart
        search_res = search_rag_brain("hose heater jacket temperature below 15C", index_dir=self.test_index)
        self.assertTrue(search_res["found"])
        self.assertTrue(search_res["documentation"]["found"])
        top_match = search_res["documentation"]["results"][0]
        self.assertIn("Nozzle Flow Optimization", top_match["text"])

    # -------------------------------------------------------------------------
    # 5. Natural Spoken English Formatting
    # -------------------------------------------------------------------------
    def test_spoken_english_formatting(self):
        """Verify format_spoken_response eliminates markdown artifacts."""
        raw = "## Status Report\n- **Printing**: Running\n- `Completed`: 450 mm\n```python\nprint('code')\n```"
        spoken = format_spoken_response(raw)
        self.assertNotIn("##", spoken)
        self.assertNotIn("**", spoken)
        self.assertNotIn("```", spoken)
        self.assertNotIn("-", spoken)
        self.assertIn("Printing: Running", spoken)
        self.assertIn("Completed: 450 mm", spoken)

        # Test duration and time formatting
        self.assertEqual(speak_duration(8.5), "8.5 hours")
        self.assertEqual(speak_duration(1.0), "1 hour")
        self.assertIn("AM", speak_time("2026-10-08 07:52:00"))
        self.assertIn("PM", speak_time("17:05:00"))

    # -------------------------------------------------------------------------
    # 6. Assistant Initialization, Tools, and Offline Fallback
    # -------------------------------------------------------------------------
    def test_assistant_tool_declarations_and_offline_fallback(self):
        """Verify all 17 agentic tools are defined and offline fallback works."""
        self.assertEqual(len(TOOLS), 19)  # 10 DB/RAG tools + 9 sensor/camera/doc tools

        bot = Assistant(use_llm=False)
        self.assertFalse(bot.use_llm)

        # Deterministic offline response
        ans = bot.ask("what is the machine status")
        self.assertIsInstance(ans, str)
        self.assertTrue(len(ans) > 0)


if __name__ == "__main__":
    unittest.main()
