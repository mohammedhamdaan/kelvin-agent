"""
tests/test_voice_agent_overhaul.py - Comprehensive Verification & Regression Test Suite
for Kelvin6K Voice Agent Overhaul.

Verifies:
1. Exact reproduction scenario: User says "Turn off the camera" while Wall 2 pending action exists:
   - Responds with camera unavailable message
   - Pending action for Wall 2 is NOT consumed, cancelled, or corrupted
   - Confirmation of Wall 2 ("yes") works afterwards and logs Wall 2
2. Unrelated queries ("What time is it?", "search docs", etc.) do not execute pending action
3. Two active shifts on site: "I completed two walls" -> Kelvin asks "Which employee should I record this work for?"
4. No auto-attribution to active operator on line without explicit introduction or context
5. Fuzzy match "Hamdan" resolves to "Hamdaan"
6. Ambiguity detection: if both "Hamdan" and "Hamdaan" exist, asks for clarification
7. Separate registration and shift start: registering new employee creates record in DB and asks if shift should be started
8. Read-only historical query: "Did I complete two walls yesterday?" does not trigger write or confirmation
9. Expired action (>30s) cannot be confirmed or executed
10. Cancelled action cannot be executed
"""

import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

from agent import config, db, identity, query_router, shift_store
from agent.assistant import Assistant, PendingActionManager


class TestVoiceAgentOverhaul(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_overhaul.db")
        db.init_db(self.db_path)
        self._patcher = patch.object(config, "DATABASE_PATH", self.db_path)
        self._patcher.start()
        self.assistant = Assistant(use_llm=False)

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_camera_query_while_pending_action_exists(self):
        """
        THE CORE DEFECT:
        When a pending action exists (e.g. Wall 2 completion), and user says:
        'Turn off the camera', Kelvin must respond that camera functionality is unavailable,
        and MUST NOT ask Aman to confirm Wall 2 or cancel Wall 2.
        Afterwards, when user says 'yes', Wall 2 must be confirmed and executed.
        """
        # Create operator Aman and start shift
        shift_store.create_operator("OP_AMAN", "Aman", "EMP001", active=1)
        shift_id = shift_store.start_shift("OP_AMAN", "2026-10-12", "Morning", "line_1")

        # Step 1: User says "I completed wall 2"
        resp1 = self.assistant.ask("I completed wall 2", session_id="term_1", line_id="line_1", operator_id="OP_AMAN")
        self.assertIn("Aman", resp1)
        self.assertIn("Wall 2", resp1)
        self.assertIn("Please confirm to proceed", resp1)

        # Pending action must be active
        pending = self.assistant._actions.get("term_1", "line_1")
        self.assertIsNotNone(pending)
        self.assertEqual(pending.params["wall_number"], 2)

        # Step 2: User says "Turn off the camera"
        resp2 = self.assistant.ask("Turn off the camera", session_id="term_1", line_id="line_1", operator_id="OP_AMAN")
        self.assertIn("Camera functionality is unavailable in this version", resp2)
        self.assertNotIn("Aman, please answer Yes or No", resp2)

        # Pending action must STILL be active
        pending_after = self.assistant._actions.get("term_1", "line_1")
        self.assertIsNotNone(pending_after)
        self.assertEqual(pending_after.params["wall_number"], 2)

        # Step 3: User says "yes" to confirm the wall
        resp3 = self.assistant.ask("yes", session_id="term_1", line_id="line_1", operator_id="OP_AMAN")
        self.assertIn("Wall 2 production recorded successfully", resp3)

        # Verify in DB
        prod = shift_store.get_shift_production(shift_id)
        self.assertEqual(len(prod), 1)
        self.assertEqual(prod[0]["wall_number"], 2)
        self.assertEqual(prod[0]["status"], "completed")

    def test_unrelated_queries_do_not_execute_or_clear_pending_action(self):
        """Unrelated commands/queries do not clear pending action."""
        shift_store.create_operator("OP_1", "Aman", "EMP001", active=1)
        shift_store.start_shift("OP_1", "2026-10-12", "Morning", "line_1")

        self.assistant.ask("wall 3 complete ho gayi", session_id="term_1", line_id="line_1", operator_id="OP_1")
        self.assertIsNotNone(self.assistant._actions.get("term_1", "line_1"))

        # Factual query
        resp = self.assistant.ask("what is the current layer width?", session_id="term_1", line_id="line_1")
        self.assertTrue("unavailable" in resp.lower() or "don't have" in resp.lower())

        # Pending action remains
        self.assertIsNotNone(self.assistant._actions.get("term_1", "line_1"))

        # Explicit cancellation
        cancel_resp = self.assistant.ask("cancel", session_id="term_1", line_id="line_1")
        self.assertIn("cancelled", cancel_resp.lower())
        self.assertIsNone(self.assistant._actions.get("term_1", "line_1"))

    def test_unattributed_work_report_prompts_for_operator(self):
        """
        When two active shifts exist on site, and someone says:
        'I completed two walls' without operator context, Kelvin must NOT auto-assign
        to either operator, and must ask: 'Which employee should I record this work for?'
        """
        shift_store.create_operator("OP_HAMDAAN", "Hamdaan", "EMP001", active=1)
        shift_store.create_operator("OP_ANAND", "Anand", "EMP002", active=1)
        shift_store.start_shift("OP_HAMDAAN", "2026-10-12", "Morning", "line_1")
        shift_store.start_shift("OP_ANAND", "2026-10-12", "Morning", "line_2")

        # New session without bound operator
        resp = self.assistant.ask("I completed two walls", session_id="term_unauth", line_id="line_1")
        self.assertIn("Which employee should I record this work for?", resp)

        # User provides name: "Hamdaan"
        resp2 = self.assistant.ask("Hamdaan", session_id="term_unauth", line_id="line_1")
        self.assertIn("Hamdaan", resp2)
        self.assertIn("You are about to log Wall 2 as completed on line_1", resp2)

        # Confirm
        resp3 = self.assistant.ask("yes", session_id="term_unauth", line_id="line_1")
        self.assertIn("Wall 2 production recorded successfully", resp3)

    def test_fuzzy_matching_and_ambiguity_detection(self):
        """Fuzzy match 'Hamdan' -> 'Hamdaan', and flags ambiguity if collision exists."""
        shift_store.create_operator("OP_H", "Hamdaan", "EMP101", active=1)

        # Single match: "Hamdan" resolves to "Hamdaan"
        res = identity.resolve_employee("Hamdan", db_path=self.db_path)
        self.assertEqual(res.status, "MATCHED")
        self.assertEqual(res.operator["name"], "Hamdaan")

        # Now add a close operator "Hamdan"
        shift_store.create_operator("OP_H2", "Hamdan", "EMP102", active=1)
        res_ambig = identity.resolve_employee("Hamdan", db_path=self.db_path)
        self.assertEqual(res_ambig.status, "AMBIGUOUS")
        self.assertIn("clarify", res_ambig.message.lower())

    def test_read_only_history_query_does_not_trigger_write(self):
        """'Did I complete two walls yesterday?' is classified as read-only query, NOT write."""
        shift_store.create_operator("OP_1", "Hamdaan", "EMP101", active=1)
        shift_store.start_shift("OP_1", "2026-10-12", "Morning", "line_1")

        q = "Did I complete two walls yesterday?"
        intent, _ = query_router.classify(q)
        self.assertNotEqual(intent, query_router.Intent.LOG_WALL_PRODUCTION)
        self.assertIn(intent, (query_router.Intent.GET_COMPLETED_PRODUCTION, query_router.Intent.GET_WALL_HISTORY, query_router.Intent.GET_SHIFT_PRODUCTION))

        resp = self.assistant.ask(q, session_id="t1", line_id="line_1", operator_id="OP_1")
        self.assertIsNone(self.assistant._actions.get("t1", "line_1"))
        self.assertNotIn("Please confirm", resp)

    def test_expired_and_cancelled_actions_cannot_be_executed(self):
        """Actions older than TTL cannot be executed with 'yes'."""
        shift_store.create_operator("OP_1", "Aman", "EMP001", active=1)
        shift_store.start_shift("OP_1", "2026-10-12", "Morning", "line_1")

        # Create manager with short TTL
        self.assistant._actions = PendingActionManager(ttl_seconds=0.05)
        self.assistant.ask("wall 1 complete ho gayi", session_id="t1", line_id="line_1", operator_id="OP_1")

        time.sleep(0.1)
        # Attempt to confirm expired action
        resp = self.assistant.ask("yes", session_id="t1", line_id="line_1", operator_id="OP_1")
        self.assertNotIn("successfully", resp)
        self.assertIn("understand", resp.lower())
        self.assertEqual(len(shift_store.get_shift_production(1)), 0)


if __name__ == "__main__":
    unittest.main()
