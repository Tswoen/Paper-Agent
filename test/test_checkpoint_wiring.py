import asyncio
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from src.graph.checkpointing import build_checkpoint_store
from src.llm.config import CheckpointSettings
from src.repositories.sessions.sqlite import SQLiteSessionRepository
from src.services.paper_workflow_runtime import build_paper_workflow_message_handler
from src.services.session_runs import SessionRunService

try:
    from .graph_test_support import hermetic_graph_llms
    from .test_search_graph import _StubService, _fake_backends, _fake_snapshot, _read_snapshot
except ImportError:
    from graph_test_support import hermetic_graph_llms
    from test_search_graph import _StubService, _fake_backends, _fake_snapshot, _read_snapshot


class CheckpointWiringTest(unittest.TestCase):
    """验证生产服务路径真的把 store/thread 传进图，并把结果记账、落盘。"""

    def test_session_run_passes_store_and_thread_to_runtime(self):
        """普通 run：runtime 拿到 store 和 thread_id，事件带 thread_id，注册表收尾为 completed。"""

        async def exercise():
            temp_dir = tempfile.TemporaryDirectory()
            self.addCleanup(temp_dir.cleanup)
            repo = SQLiteSessionRepository(storage_root=Path(temp_dir.name) / "sessions")
            session = repo.create("Wiring")
            store = build_checkpoint_store(CheckpointSettings(backend="memory"))
            captured = {}

            async def handler(chat_id, content, frame, emit):
                captured.update(frame)

            service = SessionRunService(repo, message_handler=handler, checkpoint_store=store)
            accepted = await service.start_run(session.key, {"content": "wiring", "turn_id": "t1"})
            task = service.broker.get_run(accepted["run_id"]).task
            await asyncio.wait_for(task, timeout=30)
            return {
                "expected_thread": f"{session.key}:t1",
                "runtime_thread": captured["runtime_context"].thread_id,
                "runtime_store": captured["runtime_context"].checkpoint_store,
                "store": store,
                "resume": captured["resume_from_checkpoint"],
                "row": store.registry.get(f"{session.key}:t1"),
                "run_id": accepted["run_id"],
                "events": repo.get(session.key).events,
            }

        result = asyncio.run(exercise())

        self.assertEqual(result["runtime_thread"], result["expected_thread"])
        self.assertIs(result["runtime_store"], result["store"])
        self.assertFalse(result["resume"])
        self.assertEqual(result["row"]["status"], "completed")
        # 中文注释：会话创建事件不属于本次 run，按 run_id 过滤后再断言 thread_id。
        run_events = [
            event
            for event in result["events"]
            if event["metadata"].get("run_id") == result["run_id"]
        ]
        self.assertTrue(run_events)
        self.assertEqual({event["metadata"].get("thread_id") for event in run_events}, {result["expected_thread"]})

    def test_full_run_persists_checkpoints_for_thread(self):
        """真实 handler + SQLite store：每个节点写一份快照，注册表收尾为 completed。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = SQLiteSessionRepository(storage_root=root / "sessions")
            session = repo.create("Checkpoint persistence")
            store = build_checkpoint_store(
                CheckpointSettings(backend="sqlite", path=str(root / "checkpoints.db"))
            )
            try:
                service = SessionRunService(
                    repo,
                    message_handler=build_paper_workflow_message_handler(repo),
                    checkpoint_store=store,
                )
                thread_id = f"{session.key}:t1"
                stub = _StubService()

                with hermetic_graph_llms(), _fake_backends(
                    stub,
                    search_snapshot=_fake_snapshot(),
                    read_snapshot=_read_snapshot(),
                ):

                    async def exercise():
                        accepted = await service.start_run(
                            session.key,
                            {
                                "content": "multi-agent literature review",
                                "turn_id": "t1",
                                "constraints": {"sources": ["openalex"], "max_results": 3},
                            },
                        )
                        task = service.broker.get_run(accepted["run_id"]).task
                        await asyncio.wait_for(task, timeout=60)

                    asyncio.run(exercise())

                row = store.registry.get(thread_id)
                self.assertIsNotNone(row)
                self.assertEqual(row["status"], "completed")
                self.assertIsNotNone(
                    store.saver.get_tuple({"configurable": {"thread_id": thread_id}})
                )
                with closing(sqlite3.connect(root / "checkpoints.db")) as connection:
                    checkpoint_count = connection.execute(
                        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                        (thread_id,),
                    ).fetchone()[0]
                self.assertGreater(checkpoint_count, 1)
            finally:
                store.close()

    def test_service_defaults_to_memory_checkpoint_store(self):
        """直接构造服务（测试/脚本）默认用内存 store，不会在仓库里留下 checkpoint 文件。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteSessionRepository(storage_root=Path(temp_dir) / "sessions")
            service = SessionRunService(repo)

        self.assertFalse(service.checkpoint_store.is_persistent)
        self.assertEqual(service.checkpoint_store.registry.list_threads(), [])


if __name__ == "__main__":
    unittest.main()