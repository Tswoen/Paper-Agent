import asyncio
import tempfile
import unittest

from src.repositories.sessions.sqlite import SQLiteSessionRepository
from src.repositories.settings.json import SettingsRepository
from src.api import create_app
from src.services.session_runs import SessionRunBroker, SessionRunService


class SessionRunRecoveryTest(unittest.TestCase):
    """验证进程重启后不会留下永久阻塞会话的孤儿运行标记。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.repo = SQLiteSessionRepository(storage_root=self.temp_dir.name)

    def test_recover_marks_orphaned_running_session_failed(self):
        session = self.repo.create("Interrupted run")
        self.repo.set_status(session.key, "running")
        self.repo.set_run_started_at(session.key, "2026-09-18T01:02:03+00:00")

        recovered = SessionRunService(self.repo).recover_interrupted_runs()

        record = self.repo.get(session.key)
        interruption_events = [event for event in record.events if event["event_type"] == "run_interrupted"]
        self.assertEqual(recovered, [session.key])
        self.assertEqual(record.status, "failed")
        self.assertIsNone(record.run_started_at)
        self.assertEqual(len(interruption_events), 1)
        self.assertEqual(interruption_events[0]["metadata"]["recovery_status"], "process_restarted")

        # 对账需要幂等，重复执行不能制造重复的中断事件。
        self.assertEqual(SessionRunService(self.repo).recover_interrupted_runs(), [])
        self.assertEqual(
            len([event for event in self.repo.get(session.key).events if event["event_type"] == "run_interrupted"]),
            1,
        )

    def test_recover_only_clears_stale_marker_for_completed_session(self):
        session = self.repo.create("Completed before cleanup")
        self.repo.set_status(session.key, "completed")
        self.repo.set_run_started_at(session.key, "2026-09-18T01:02:03+00:00")

        SessionRunService(self.repo).recover_interrupted_runs()

        record = self.repo.get(session.key)
        self.assertEqual(record.status, "completed")
        self.assertIsNone(record.run_started_at)
        reconciled_events = [event for event in record.events if event["event_type"] == "run_marker_reconciled"]
        self.assertEqual(len(reconciled_events), 1)
        self.assertEqual(reconciled_events[0]["metadata"]["previous_status"], "completed")

    def test_recover_skips_run_owned_by_current_broker(self):
        session = self.repo.create("Active run")
        self.repo.set_status(session.key, "running")
        self.repo.set_run_started_at(session.key, "2026-09-18T01:02:03+00:00")
        broker = SessionRunBroker()

        async def exercise():
            await broker.open_run(session.key, "run-active", "turn-active", "2026-09-18T01:02:03+00:00")
            task = asyncio.create_task(asyncio.sleep(60))
            broker.attach_task("run-active", task)
            try:
                return SessionRunService(self.repo, broker=broker).recover_interrupted_runs()
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        recovered = asyncio.run(exercise())

        record = self.repo.get(session.key)
        self.assertEqual(recovered, [])
        self.assertEqual(record.status, "running")
        self.assertIsNotNone(record.run_started_at)

    def test_app_startup_reconciles_persisted_orphan(self):
        session = self.repo.create("Restarted app")
        self.repo.set_status(session.key, "running")
        self.repo.set_run_started_at(session.key, "2026-09-18T01:02:03+00:00")

        create_app(settings_repo=SettingsRepository(initial={}), sessions_repo=self.repo)

        record = self.repo.get(session.key)
        self.assertEqual(record.status, "failed")
        self.assertIsNone(record.run_started_at)
        self.assertTrue(any(event["event_type"] == "run_interrupted" for event in record.events))


if __name__ == "__main__":
    unittest.main()
