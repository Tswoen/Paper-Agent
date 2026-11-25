import asyncio
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from src.api import create_app
from src.graph.checkpointing import build_checkpoint_bundle
from src.graph.runtime import WorkflowRuntimeContext
from src.llm.config import CheckpointSettings
from src.repositories.sessions.sqlite import SQLiteSessionRepository
from src.repositories.settings.json import SettingsRepository
from src.services.paper_workflow_runtime import build_paper_workflow_message_handler
from src.services.session_runs import SessionRunService

try:
    from .graph_test_support import hermetic_graph_llms
    from .test_search_graph import _StubService, _fake_backends, _fake_snapshot, _read_snapshot
except ImportError:
    from graph_test_support import hermetic_graph_llms
    from test_search_graph import _StubService, _fake_backends, _fake_snapshot, _read_snapshot


class CheckpointWiringTest(unittest.TestCase):
    """验证生产服务路径真的把 bundle/thread 传进图，并把结果记账、落盘。"""

    def test_session_run_passes_bundle_and_thread_to_runtime(self):
        """普通 run：runtime 拿到 bundle 和 thread_id，事件带 thread_id，注册表收尾为 completed。"""

        async def exercise():
            temp_dir = tempfile.TemporaryDirectory()
            self.addCleanup(temp_dir.cleanup)
            repo = SQLiteSessionRepository(storage_root=Path(temp_dir.name) / "sessions")
            session = repo.create("Wiring")
            bundle = build_checkpoint_bundle(CheckpointSettings(backend="memory"))
            captured = {}

            async def handler(chat_id, content, frame, emit):
                captured.update(frame)

            service = SessionRunService(repo, message_handler=handler, checkpoint_bundle=bundle)
            accepted = await service.start_run(session.key, {"content": "wiring", "turn_id": "t1"})
            task = service.broker.get_run(accepted["run_id"]).task
            await asyncio.wait_for(task, timeout=30)
            return {
                "expected_thread": f"{session.key}:t1",
                "runtime_thread": captured["runtime_context"].thread_id,
                "runtime_bundle": captured["runtime_context"].checkpoint_bundle,
                "bundle": bundle,
                "resume": captured["resume_from_checkpoint"],
                "row": bundle.registry.get(f"{session.key}:t1"),
                "run_id": accepted["run_id"],
                "events": repo.get(session.key).events,
            }

        result = asyncio.run(exercise())

        self.assertEqual(result["runtime_thread"], result["expected_thread"])
        self.assertIs(result["runtime_bundle"], result["bundle"])
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
        """真实 handler + SQLite bundle：每个节点写一份快照，注册表收尾为 completed。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = SQLiteSessionRepository(storage_root=root / "sessions")
            session = repo.create("Checkpoint persistence")
            bundle = build_checkpoint_bundle(
                CheckpointSettings(backend="sqlite", path=str(root / "checkpoints.db"))
            )
            try:
                service = SessionRunService(
                    repo,
                    message_handler=build_paper_workflow_message_handler(repo),
                    checkpoint_bundle=bundle,
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

                row = bundle.registry.get(thread_id)
                self.assertIsNotNone(row)
                self.assertEqual(row["status"], "completed")
                self.assertIsNotNone(
                    bundle.saver.get_tuple({"configurable": {"thread_id": thread_id}})
                )
                with closing(sqlite3.connect(root / "checkpoints.db")) as connection:
                    checkpoint_count = connection.execute(
                        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                        (thread_id,),
                    ).fetchone()[0]
                self.assertGreater(checkpoint_count, 1)

                events = repo.get(session.key).thread()["events"]
                turn_ends = [
                    event
                    for event in events
                    if event["event_type"] == "turn_end"
                    and event["metadata"].get("turn_id") == "t1"
                ]
                self.assertTrue(turn_ends)
                expected_checkpoint_id = bundle.last_checkpoint_id(thread_id)
                self.assertTrue(expected_checkpoint_id)
                self.assertEqual(
                    turn_ends[-1]["metadata"].get("checkpoint_id"),
                    expected_checkpoint_id,
                )
            finally:
                bundle.close()

    def test_service_defaults_to_memory_checkpoint_bundle(self):
        """直接构造服务（测试/脚本）默认用内存 bundle，不会在仓库里留下 checkpoint 文件。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteSessionRepository(storage_root=Path(temp_dir) / "sessions")
            service = SessionRunService(repo)

        self.assertFalse(service.checkpoint_bundle.is_persistent)
        self.assertEqual(service.checkpoint_bundle.registry.list_threads(), [])

    def test_handler_passes_durability_from_bundle_settings(self):
        """配置里的 durability 要真正传到 run_graph，而不是被默认值吞掉。"""

        async def exercise():
            temp_dir = tempfile.TemporaryDirectory()
            self.addCleanup(temp_dir.cleanup)
            repo = SQLiteSessionRepository(storage_root=Path(temp_dir.name) / "sessions")
            bundle = build_checkpoint_bundle(
                CheckpointSettings(backend="memory", durability="async")
            )
            runtime = WorkflowRuntimeContext(
                session_key="s",
                turn_id="t",
                thread_id="s:t",
                checkpoint_bundle=bundle,
            )
            handler = build_paper_workflow_message_handler(repo)
            with mock.patch(
                "src.services.paper_workflow_runtime.run_graph",
                new=mock.AsyncMock(),
            ) as mocked:
                await handler(
                    "s",
                    "topic",
                    {"turn_id": "t", "run_id": "r", "runtime_context": runtime},
                    lambda event: event,
                )
            return mocked.await_args.kwargs, bundle

        kwargs, bundle = asyncio.run(exercise())

        self.assertEqual(kwargs["durability"], "async")
        self.assertIs(kwargs["checkpointer"], bundle.saver)

    def test_create_app_warns_when_durability_is_not_sync(self):
        """持久化 bundle 选了非 sync 时，启动日志要提醒可能丢失最后一个快照。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = SQLiteSessionRepository(storage_root=root / "sessions")
            bundle = build_checkpoint_bundle(
                CheckpointSettings(
                    backend="sqlite",
                    path=str(root / "checkpoints.db"),
                    durability="async",
                )
            )
            try:
                with mock.patch("src.api.app.logger.warning") as warning:
                    create_app(
                        settings_repo=SettingsRepository(initial={}),
                        sessions_repo=repo,
                        checkpoint_bundle=bundle,
                    )
                messages = [str(call) for call in warning.call_args_list]
                self.assertTrue(any("durability" in message for message in messages))
            finally:
                bundle.close()

    def test_shutdown_cancels_active_run_and_finalizes_state(self):
        """应用关闭时先取消活跃 run，让账本/会话在关库前收口。"""

        async def exercise():
            temp_dir = tempfile.TemporaryDirectory()
            self.addCleanup(temp_dir.cleanup)
            repo = SQLiteSessionRepository(storage_root=Path(temp_dir.name) / "sessions")
            bundle = build_checkpoint_bundle(CheckpointSettings(backend="memory"))
            session = repo.create("Shutdown")
            started = asyncio.Event()

            async def handler(chat_id, content, frame, emit):
                started.set()
                await asyncio.sleep(60)

            service = SessionRunService(
                repo,
                message_handler=handler,
                checkpoint_bundle=bundle,
            )
            accepted = await service.start_run(
                session.key,
                {"content": "slow", "turn_id": "t1"},
            )
            task = service.broker.get_run(accepted["run_id"]).task
            await started.wait()
            await service.shutdown()
            return task, repo.get(session.key), bundle.registry.get(f"{session.key}:t1")

        task, record, row = asyncio.run(exercise())

        self.assertTrue(task.done())
        self.assertEqual(record.status, "failed")
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "failed")


if __name__ == "__main__":
    unittest.main()