import tempfile
import unittest
from pathlib import Path

from src.graph.checkpointing import build_checkpoint_bundle
from src.llm.config import CheckpointSettings
from src.repositories.sessions.sqlite import SQLiteSessionRepository
from src.services.session_runs import SessionRunService


NOW = "2026-11-12T00:00:00+00:00"


def _put_checkpoint(bundle, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    bundle.saver.put(
        config,
        {
            "v": 1,
            "id": f"{thread_id}-checkpoint",
            "ts": NOW,
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "updated_channels": None,
        },
        {"source": "input", "step": 0, "parents": {}},
        {},
    )


def _seed(registry, thread_id: str, *, status: str, finished_at: str | None) -> None:
    registry.mark_started(
        thread_id=thread_id,
        session_key="session-a",
        turn_id=thread_id,
        run_id="run-1",
        graph_version=1,
        started_at="2026-01-01T00:00:00+00:00",
    )
    if status != "running":
        registry.mark_finished(thread_id=thread_id, status=status, finished_at=finished_at)


class CheckpointRetentionTest(unittest.TestCase):
    """验证 7/30 天清理只删过期线程，并且同时清掉 saver 快照。"""

    def test_purge_removes_only_expired_finished_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteSessionRepository(storage_root=Path(temp_dir) / "sessions")
            bundle = build_checkpoint_bundle(CheckpointSettings(backend="memory"))
            try:
                service = SessionRunService(repo, checkpoint_bundle=bundle)
                registry = bundle.registry
                _seed(registry, "completed-old", status="completed", finished_at="2026-11-01T00:00:00+00:00")
                _seed(registry, "completed-new", status="completed", finished_at="2026-11-10T00:00:00+00:00")
                _seed(registry, "failed-old", status="failed", finished_at="2026-10-01T00:00:00+00:00")
                _seed(registry, "failed-new", status="failed", finished_at="2026-11-01T00:00:00+00:00")
                _seed(registry, "running-old", status="running", finished_at=None)
                _put_checkpoint(bundle, "completed-old")

                purged = service.purge_expired_checkpoints(now=NOW)

                self.assertEqual(set(purged), {"completed-old", "failed-old"})
                self.assertIsNone(registry.get("completed-old"))
                self.assertIsNone(
                    bundle.saver.get_tuple({"configurable": {"thread_id": "completed-old"}})
                )
                for kept in ("completed-new", "failed-new", "running-old"):
                    self.assertIsNotNone(registry.get(kept), kept)
            finally:
                bundle.close()

    def test_purge_is_disabled_by_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteSessionRepository(storage_root=Path(temp_dir) / "sessions")
            bundle = build_checkpoint_bundle(
                CheckpointSettings(backend="memory", retention_enabled=False)
            )
            try:
                service = SessionRunService(repo, checkpoint_bundle=bundle)
                _seed(service.checkpoint_bundle.registry, "completed-old", status="completed", finished_at="2026-11-01T00:00:00+00:00")

                self.assertEqual(service.purge_expired_checkpoints(now=NOW), [])
                self.assertIsNotNone(service.checkpoint_bundle.registry.get("completed-old"))
            finally:
                bundle.close()

    def test_purge_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteSessionRepository(storage_root=Path(temp_dir) / "sessions")
            bundle = build_checkpoint_bundle(CheckpointSettings(backend="memory"))
            try:
                service = SessionRunService(repo, checkpoint_bundle=bundle)
                _seed(service.checkpoint_bundle.registry, "completed-old", status="completed", finished_at="2026-11-01T00:00:00+00:00")

                self.assertEqual(len(service.purge_expired_checkpoints(now=NOW)), 1)
                self.assertEqual(service.purge_expired_checkpoints(now=NOW), [])
            finally:
                bundle.close()

    def test_purge_works_on_sqlite_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = SQLiteSessionRepository(storage_root=root / "sessions")
            bundle = build_checkpoint_bundle(
                CheckpointSettings(backend="sqlite", path=str(root / "checkpoints.db"))
            )
            try:
                service = SessionRunService(repo, checkpoint_bundle=bundle)
                _seed(service.checkpoint_bundle.registry, "completed-old", status="completed", finished_at="2026-11-01T00:00:00+00:00")
                _seed(service.checkpoint_bundle.registry, "failed-new", status="failed", finished_at="2026-11-01T00:00:00+00:00")

                purged = service.purge_expired_checkpoints(now=NOW)

                self.assertEqual(purged, ["completed-old"])
                self.assertIsNone(service.checkpoint_bundle.registry.get("completed-old"))
                self.assertIsNotNone(service.checkpoint_bundle.registry.get("failed-new"))
            finally:
                bundle.close()


if __name__ == "__main__":
    unittest.main()