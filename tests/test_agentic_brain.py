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

    def test_break_precondition_validation_and_double_action_prevention(self):
        """Verify state precondition checks for breaks and double action prevention."""
        # 1. Register employee Hamdaan (initially Off Shift, active_shift_id is None)
        shift_store.create_or_update_employee(
            name="Hamdaan",
            role="Operator",
            status="Active",
            db_path=self.test_db,
        )
        op = shift_store.find_operator_by_name("Hamdaan", db_path=self.test_db)
        self.assertEqual(op["current_work_status"], "Off Shift")
        self.assertIsNone(op["active_shift_id"])

        # 2. State Precondition Check: Trying to put an Off-Shift employee on break MUST abort
        break_res = shift_store.update_employee_work_status(
            operator_name="Hamdaan",
            work_status="On Break",
            db_path=self.test_db,
        )
        self.assertFalse(break_res["success"])
        expected_error = "Cannot update status to On Break: Operator Hamdaan is currently Off Shift. Please start their shift first."
        self.assertEqual(break_res["error"], expected_error)

        # Verify DB was NOT updated
        op_after = shift_store.find_operator_by_name("Hamdaan", db_path=self.test_db)
        self.assertEqual(op_after["current_work_status"], "Off Shift")
        self.assertIsNone(op_after["active_shift_id"])

        # Also verify Assistant natural response in voice flow
        bot = Assistant(use_llm=False)
        voice_resp = bot.ask("Put Hamdaan on a lunch break")
        self.assertIn("cannot put Hamdaan on a lunch break", voice_resp)
        self.assertTrue("off-shift" in voice_resp.lower() or "shift has already ended" in voice_resp.lower())

        # 3. Double Action Prevention: Prevent ending a shift if already Off Shift
        end_double = shift_store.end_employee_shift("Hamdaan", db_path=self.test_db)
        self.assertFalse(end_double["success"])
        self.assertEqual(end_double["error"], "Cannot end shift: Operator Hamdaan is already Off Shift.")

        off_double = shift_store.update_employee_work_status("Hamdaan", "Off Shift", db_path=self.test_db)
        self.assertFalse(off_double["success"])
        self.assertEqual(off_double["error"], "Cannot end shift: Operator Hamdaan is already Off Shift.")

        # 4. Start Hamdaan's shift (status becomes Working)
        start_res = shift_store.start_employee_shift("Hamdaan", shift_type="Morning", db_path=self.test_db)
        self.assertTrue(start_res["success"])
        op_working = shift_store.find_operator_by_name("Hamdaan", db_path=self.test_db)
        self.assertEqual(op_working["current_work_status"], "Working")
        self.assertIsNotNone(op_working["active_shift_id"])

        # 5. Double Action Prevention / Idempotent Clock-In: Prevent starting a shift if already Working
        start_double = shift_store.start_employee_shift("Hamdaan", shift_type="Morning", db_path=self.test_db)
        self.assertFalse(start_double["success"])
        self.assertEqual(start_double["error"], "Operator Hamdaan is already clocked into an active shift.")

        # 6. Put Hamdaan on Break while Working succeeds
        break_valid = shift_store.update_employee_work_status("Hamdaan", "On Break", db_path=self.test_db)
        self.assertTrue(break_valid["success"])
        self.assertEqual(break_valid["current_work_status"], "On Break")

        # Double Break check
        break_again = shift_store.update_employee_work_status("Hamdaan", "On Break", db_path=self.test_db)
        self.assertFalse(break_again["success"])
        self.assertEqual(break_again["error"], "Operator Hamdaan is already On Break.")

        # Starting shift while on break fails
        start_on_break = shift_store.start_employee_shift("Hamdaan", db_path=self.test_db)
        self.assertFalse(start_on_break["success"])
        self.assertIn("On Break", start_on_break["error"])

        # 7. Resume Working
        resume_res = shift_store.update_employee_work_status("Hamdaan", "Working", db_path=self.test_db)
        self.assertTrue(resume_res["success"])
        self.assertEqual(resume_res["current_work_status"], "Working")

        # 8. End shift properly
        end_ok = shift_store.end_employee_shift("Hamdaan", db_path=self.test_db)
        self.assertTrue(end_ok["success"])
        self.assertEqual(end_ok["status"], "Completed")

        # After ending shift, employee is Off Shift again; putting on break fails again
        break_after_end = shift_store.update_employee_work_status("Hamdaan", "On Break", db_path=self.test_db)
        self.assertFalse(break_after_end["success"])
        self.assertEqual(break_after_end["error"], expected_error)

    def test_gemini_natural_reply_on_break_validation_failure(self):
        """Verify Assistant with LLM returns natural speech when break validation fails."""
        from unittest.mock import MagicMock, patch
        bot = Assistant(use_llm=True)
        mock_chat = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "I cannot put Hamdaan on a lunch break because his shift has already ended / he is off-shift."
        mock_chat.send_message.return_value = mock_resp
        bot._chats["test_session"] = mock_chat
        mock_client = MagicMock()
        with patch.object(bot, "_client_or_none", return_value=mock_client):
            ans = bot.ask("Put Hamdaan on a lunch break", session_id="test_session")
            self.assertIn("cannot put Hamdaan on a lunch break", ans)
            self.assertIn("off-shift", ans.lower())

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
        """Verify agentic voice assistant tools are defined and offline fallback works."""
        tool_names = [t.__name__ for t in TOOLS]
        self.assertIn("execute_sql_query", tool_names)
        self.assertIn("execute_sql_write", tool_names)
        self.assertIn("search_rag_brain", tool_names)
        self.assertIn("learn_and_store_in_rag_brain", tool_names)
        self.assertIn("update_employee_work_status", tool_names)
        self.assertIn("get_active_shifts", tool_names)
        self.assertIn("get_operators_status", tool_names)
        self.assertNotIn("get_site_people_count", tool_names)
        self.assertNotIn("get_site_detections", tool_names)
        self.assertEqual(len(TOOLS), 24)

        bot = Assistant(use_llm=False)
        self.assertFalse(bot.use_llm)

        # Deterministic offline response
        ans = bot.ask("what is the machine status")
        self.assertIsInstance(ans, str)
        self.assertTrue(len(ans) > 0)

    # -------------------------------------------------------------------------
    # 7. Enterprise Safety Guardrails & Validation
    # -------------------------------------------------------------------------
    def test_idempotent_clock_in_prevents_duplicate_shifts(self):
        """Verify starting an active shift twice returns an idempotent error and creates zero duplicate rows."""
        shift_store.create_or_update_employee("Hamdaan", role="Operator", status="Active", db_path=self.test_db)
        first_start = shift_store.start_employee_shift("Hamdaan", shift_type="Morning", db_path=self.test_db)
        self.assertTrue(first_start["success"])
        shift_id = first_start["shift_id"]

        # Attempt duplicate clock-in
        second_start = shift_store.start_employee_shift("Hamdaan", shift_type="Morning", db_path=self.test_db)
        self.assertFalse(second_start["success"])
        expected_err = "Operator Hamdaan is already clocked into an active shift."
        self.assertEqual(second_start["error"], expected_err)

        # Verify only 1 shift row exists in database
        with db.connect(self.test_db) as conn:
            rows = conn.execute("SELECT * FROM shifts WHERE operator_name = 'Hamdaan'").fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["shift_id"], shift_id)

    def test_ghost_shift_auto_closure_16_hour_rule(self):
        """Verify forgotten shifts exceeding 16 hours are automatically closed with 9.0h and auto-closure note."""
        shift_store.create_or_update_employee("Lukhmaan", role="Operator", status="Active", db_path=self.test_db)

        # Start shift 18 hours ago
        start_time_dt = datetime.datetime.now() - datetime.timedelta(hours=18.0)
        start_time_str = start_time_dt.strftime("%Y-%m-%d %H:%M:%S")

        res = shift_store.start_employee_shift(
            "Lukhmaan",
            shift_type="Morning",
            actual_start_time=start_time_str,
            line_id="line_1",
            db_path=self.test_db,
        )
        self.assertTrue(res["success"])
        s_id = res["shift_id"]

        # Verify active shift retrieval automatically sweeps and closes the 18-hour old shift
        active_shift = shift_store.get_active_shift_for_line("line_1", db_path=self.test_db)
        self.assertIsNone(active_shift)

        # Check closed shift properties in shifts table
        closed_shift = shift_store.get_shift(s_id, db_path=self.test_db)
        self.assertEqual(closed_shift["status"], "Completed")
        self.assertEqual(closed_shift["current_status"], "Completed")
        self.assertEqual(closed_shift["total_hours"], 9.0)
        self.assertEqual(closed_shift["regular_hours"], 9.0)
        self.assertEqual(closed_shift["overtime_hours"], 0.0)
        self.assertEqual(closed_shift["overtime_status"], "Normal")
        self.assertIn("Auto-closed due to forgotten clock-out", closed_shift["summary_notes"])

        # Check operator state in operators table
        op = shift_store.find_operator_by_name("Lukhmaan", db_path=self.test_db)
        self.assertEqual(op["current_work_status"], "Off Shift")
        self.assertIsNone(op["active_shift_id"])

    def test_ambiguous_name_collision_resolution(self):
        """Verify spoken name query matching multiple active operators halts mutation and returns clarification."""
        shift_store.create_or_update_employee("Aman", role="Operator", status="Active", db_path=self.test_db)
        shift_store.create_or_update_employee("Anand Sharma", role="Operator", status="Active", db_path=self.test_db)

        # Ambiguous name query on shift start
        amb_start = shift_store.start_employee_shift("Aman", db_path=self.test_db)
        self.assertFalse(amb_start["success"])
        self.assertTrue(amb_start.get("clarification_needed"))
        self.assertEqual(sorted(amb_start.get("matches", [])), ["Aman", "Anand Sharma"])
        self.assertEqual(
            amb_start.get("message"),
            "I found multiple operators matching that name. Did you mean Aman or Anand Sharma?",
        )

        # Verify zero shifts were created for either operator
        with db.connect(self.test_db) as conn:
            rows = conn.execute("SELECT * FROM shifts").fetchall()
            self.assertEqual(len(rows), 0)

        # Ambiguous name on work status update
        amb_status = shift_store.update_employee_work_status("Aman", "On Break", db_path=self.test_db)
        self.assertFalse(amb_status["success"])
        self.assertTrue(amb_status.get("clarification_needed"))
        self.assertEqual(
            amb_status.get("message"),
            "I found multiple operators matching that name. Did you mean Aman or Anand Sharma?",
        )

        # Ambiguous name on shift end
        amb_end = shift_store.end_employee_shift("Aman", db_path=self.test_db)
        self.assertFalse(amb_end["success"])
        self.assertTrue(amb_end.get("clarification_needed"))
        self.assertEqual(
            amb_end.get("message"),
            "I found multiple operators matching that name. Did you mean Aman or Anand Sharma?",
        )

        # Ambiguous name on logging hours
        amb_hours = shift_store.log_operator_hours("Aman", 8.0, db_path=self.test_db)
        self.assertFalse(amb_hours["success"])
        self.assertTrue(amb_hours.get("clarification_needed"))
        self.assertEqual(
            amb_hours.get("message"),
            "I found multiple operators matching that name. Did you mean Aman or Anand Sharma?",
        )

    def test_live_operator_status_queries_and_phonetic_matching(self):
        """Verify queries for working operators query live database and support phonetic matching for Lokmanya."""
        bot = Assistant(use_llm=False)

        # 1. When no shifts are active
        ans_who = bot.ask("who is working right now")
        self.assertEqual(ans_who, "No operators are currently clocked in on an active shift right now.")
        self.assertNotIn(".env", ans_who)
        self.assertNotIn("OPERATOR_NAME", ans_who)

        ans_who_alt = bot.ask("who's working right now")
        self.assertEqual(ans_who_alt, "No operators are currently clocked in on an active shift right now.")

        ans_lokmanya_off = bot.ask("is Lokmanya working right now")
        self.assertIn("Lukhmaan is not working right now", ans_lokmanya_off)
        self.assertIn("Off Shift", ans_lokmanya_off)
        self.assertNotIn(".env", ans_lokmanya_off)

        # 2. Clock in Lukhmaan
        start_res = shift_store.start_employee_shift("Lukhmaan", shift_type="Morning", db_path=self.test_db)
        self.assertTrue(start_res["success"])

        # Query again with phonetic alias 'Lokmanya'
        ans_lokmanya_on = bot.ask("is Lokmanya working right now")
        self.assertIn("Yes, Lukhmaan is currently clocked in and working on shift.", ans_lokmanya_on)

        # Query who is working now
        ans_who_on = bot.ask("who is working right now")
        self.assertIn("Lukhmaan is the operator currently working on shift.", ans_who_on)

        # Direct tool verification
        active_shifts = shift_store.get_active_shifts(db_path=self.test_db)
        self.assertEqual(len(active_shifts), 1)
        self.assertEqual(active_shifts[0]["operator_name"], "Lukhmaan")

        st_alias = shift_store.get_operators_status("Lokmanya", db_path=self.test_db)
        self.assertTrue(st_alias["found"])
        self.assertEqual(st_alias["operator_name"], "Lukhmaan")
        self.assertTrue(st_alias["is_working"])


if __name__ == "__main__":
    unittest.main()
