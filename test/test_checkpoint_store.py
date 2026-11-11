import sqlite3
import tempfile
import unittest
from contextlib import closing, contextmanager
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.graph.checkpointing import (
    MemoryCheckpointRegistry,
    build_checkpoint_store,
)
from src.llm.config import CheckpointSettings, SystemConfig


class CheckpointStoreTest(unittest.TestCase):
    """验证 checkpoint 工厂、SQLite 建表与注册表在两种后端上的行为一致。"""

    @contextmanager
    def _sqlite_store(self):
        """提供临时目录里的 SQLite store；先关连接再删目录，避免 Windows 文件占用。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "checkpoints.db"
            store = build_checkpoint_store(
                CheckpointSettings(backend="sqlite", path=str(database_path))
            )
            try:
                yield store, database_path
            finally:
                store.close()

    def test_memory_backend_uses_in_memory_saver(self):
        store = build_checkpoint_store(CheckpointSettings(backend="memory"))
        self.addCleanup(store.close)

        self.assertIsInstance(store.saver, InMemorySaver)
        self.assertIsInstance(store.registry, MemoryCheckpointRegistry)
        self.assertFalse(store.is_persistent)

    def test_sqlite_backend_creates_database_and_tables(self):
        with self._sqlite_store() as (store, database_path):
            self.assertIsInstance(store.saver, SqliteSaver)
            self.assertTrue(store.is_persistent)
            self.assertTrue(database_path.exists())

            with closing(sqlite3.connect(database_path)) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            # 官方 saver 的两张表 + 我们自己的 run 账本都必须在同一个文件里。
            self.assertLessEqual({"checkpoints", "writes", "checkpoint_runs"}, tables)

    def test_registry_contract_on_all_backends(self):
        memory_store = build_checkpoint_store(CheckpointSettings(backend="memory"))
        self.addCleanup(memory_store.close)

        with self._sqlite_store() as (sqlite_store, _):
            for store in (memory_store, sqlite_store):
                with self.subTest(backend=store.backend):
                    self._exercise_registry_contract(store)

    def _exercise_registry_contract(self, store) -> None:
        """同一份断言跑两种实现，任何行为不一致都会在这里暴露。"""

        registry = store.registry
        registry.mark_started(
            thread_id="session-a:turn-1",
            session_key="session-a",
            turn_id="turn-1",
            run_id="run-1",
            graph_version=1,
            started_at="2026-11-10T10:00:00+00:00",
        )
        row = registry.get("session-a:turn-1")
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["graph_version"], 1)
        # running 不算可续跑：进程还活着时应由并发保护挡住，而不是续跑。
        self.assertIsNone(registry.latest_resumable("session-a"))

        # 续跑：同一 thread 重新开始，状态回到 running，但版本号保留首次登记值。
        registry.mark_started(
            thread_id="session-a:turn-1",
            session_key="session-a",
            turn_id="turn-1",
            run_id="run-2",
            graph_version=99,
            started_at="2026-11-10T10:01:00+00:00",
        )
        row = registry.get("session-a:turn-1")
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["run_id"], "run-2")
        self.assertEqual(row["graph_version"], 1)

        registry.mark_finished(
            thread_id="session-a:turn-1",
            status="failed",
            finished_at="2026-11-10T10:05:00+00:00",
        )
        resumable = registry.latest_resumable("session-a")
        self.assertIsNotNone(resumable)
        self.assertEqual(resumable["thread_id"], "session-a:turn-1")

        # 完成后的 run 不能再续跑。
        registry.mark_finished(
            thread_id="session-a:turn-1",
            status="completed",
            finished_at="2026-11-10T10:06:00+00:00",
        )
        self.assertIsNone(registry.latest_resumable("session-a"))

        registry.mark_started(
            thread_id="session-a:turn-2",
            session_key="session-a",
            turn_id="turn-2",
            run_id="run-3",
            graph_version=1,
            started_at="2026-11-10T11:00:00+00:00",
        )
        registry.mark_finished(
            thread_id="session-a:turn-2",
            status="cancelled",
            finished_at="2026-11-10T11:01:00+00:00",
        )

        listed = [row["thread_id"] for row in registry.list_threads("session-a")]
        self.assertEqual(listed, ["session-a:turn-2", "session-a:turn-1"])
        self.assertEqual(registry.latest_resumable("session-a")["thread_id"], "session-a:turn-2")

        self.assertTrue(registry.delete_thread("session-a:turn-2"))
        self.assertFalse(registry.delete_thread("session-a:turn-2"))
        self.assertEqual(len(registry.list_threads()), 1)

        # 进程重启对账：running → interrupted 后进入可续跑集合，且不会重复计数。
        registry.mark_started(
            thread_id="session-a:turn-3",
            session_key="session-a",
            turn_id="turn-3",
            run_id="run-4",
            graph_version=1,
            started_at="2026-11-10T12:00:00+00:00",
        )
        self.assertEqual(
            registry.mark_running_interrupted(
                "session-a", finished_at="2026-11-10T12:05:00+00:00"
            ),
            1,
        )
        row = registry.get("session-a:turn-3")
        self.assertEqual(row["status"], "interrupted")
        self.assertEqual(registry.latest_resumable("session-a")["thread_id"], "session-a:turn-3")
        self.assertEqual(registry.mark_running_interrupted("session-a"), 0)

    def test_mark_finished_validates_status_and_existence(self):
        store = build_checkpoint_store(CheckpointSettings(backend="memory"))
        self.addCleanup(store.close)

        with self.assertRaises(ValueError):
            store.registry.mark_finished(thread_id="missing", status="running")
        with self.assertRaises(KeyError):
            store.registry.mark_finished(thread_id="missing", status="failed")

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            SystemConfig.from_dict({"checkpoint": {"backend": "postgres"}})
        with self.assertRaises(ValueError):
            build_checkpoint_store(CheckpointSettings(backend="postgres"))

    def test_system_config_parses_checkpoint_section(self):
        config = SystemConfig.from_dict(
            {
                "checkpoint": {
                    "backend": "memory",
                    "path": "data/test-checkpoints.db",
                    "durability": "async",
                    "retention": {
                        "enabled": False,
                        "completed_days": 3,
                        "failed_days": 9,
                    },
                }
            }
        )

        self.assertEqual(config.checkpoint.backend, "memory")
        self.assertEqual(config.checkpoint.path, "data/test-checkpoints.db")
        self.assertEqual(config.checkpoint.durability, "async")
        self.assertFalse(config.checkpoint.retention_enabled)
        self.assertEqual(config.checkpoint.retention_completed_days, 3)
        self.assertEqual(config.checkpoint.retention_failed_days, 9)

    def test_system_config_loads_nested_checkpoint_yaml(self):
        """手写的二级 YAML 解析要能读到 retention 这一层。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "system.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "checkpoint:",
                        "  backend: memory",
                        "  retention:",
                        "    completed_days: 2",
                        "    failed_days: 4",
                    ]
                ),
                encoding="utf-8",
            )

            config = SystemConfig.load(config_path)

        self.assertEqual(config.checkpoint.backend, "memory")
        self.assertEqual(config.checkpoint.retention_completed_days, 2)
        self.assertEqual(config.checkpoint.retention_failed_days, 4)

    def test_repository_system_yaml_has_checkpoint_defaults(self):
        config = SystemConfig.load("config/system.yaml")

        self.assertEqual(config.checkpoint.backend, "sqlite")
        self.assertEqual(config.checkpoint.path, "data/checkpoints.db")
        self.assertEqual(config.checkpoint.durability, "sync")
        self.assertTrue(config.checkpoint.retention_enabled)


if __name__ == "__main__":
    unittest.main()