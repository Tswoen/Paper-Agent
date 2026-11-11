import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agents.analyseAgent import AnalyseModelResult
from src.graph.checkpointing import build_checkpoint_store
from src.graph.graph import GRAPH_VERSION
from src.llm.config import CheckpointSettings
from src.repositories.sessions.sqlite import SQLiteSessionRepository
from src.services.paper_workflow_runtime import build_paper_workflow_message_handler
from src.services.session_runs import SessionRunService
from src.services.sessions import SessionError

try:
    from .graph_test_support import hermetic_graph_llms
    from .test_search_graph import _StubService, _fake_backends, _fake_snapshot, _read_snapshot
except ImportError:
    from graph_test_support import hermetic_graph_llms
    from test_search_graph import _StubService, _fake_backends, _fake_snapshot, _read_snapshot


class _FlakyAnalyseAgent:
    """第一次综合分析失败，第二次成功，用来模拟失败后的续跑。"""

    def __init__(self):
        self.overall_calls = 0

    async def async_analyse_subtopic(self, *, topic, group, usage_callback=None):
        return AnalyseModelResult(parsed=None, reason="stub: 子主题走兜底")

    async def async_analyse_overall(self, *, topic, subtopic_analyses, usage_callback=None):
        self.overall_calls += 1
        if self.overall_calls == 1:
            raise RuntimeError("stub analysis model unavailable")
        return AnalyseModelResult(parsed={}, raw_model_output="{}")


class CheckpointResumeTest(unittest.TestCase):
    """验证续跑只认 checkpoint、复用旧线程，并拒绝版本不符或缺失存档。"""

    async def _run_to_end(self, service: SessionRunService, session_key: str, body: dict):
        accepted = await service.start_run(session_key, body)
        task = service.broker.get_run(accepted["run_id"]).task
        await asyncio.wait_for(task, timeout=60)
        return accepted

    def test_resume_after_failed_run_reuses_thread_and_skips_search(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = SQLiteSessionRepository(storage_root=root / "sessions")
            session = repo.create("Resume after failure")
            store = build_checkpoint_store(
                CheckpointSettings(backend="sqlite", path=str(root / "checkpoints.db"))
            )
            try:
                service = SessionRunService(
                    repo,
                    message_handler=build_paper_workflow_message_handler(repo),
                    checkpoint_store=store,
                )
                stub = _StubService()
                agent = _FlakyAnalyseAgent()
                thread_id = f"{session.key}:t1"

                with hermetic_graph_llms(), _fake_backends(
                    stub,
                    search_snapshot=_fake_snapshot(),
                    read_snapshot=_read_snapshot(),
                ), mock.patch(
                    "src.graph.analyse_node.build_analyse_agent",
                    lambda *args, **kwargs: agent,
                ):
                    asyncio.run(
                        self._run_to_end(
                            service,
                            session.key,
                            {
                                "content": "multi-agent literature review",
                                "turn_id": "t1",
                                "constraints": {"sources": ["openalex"], "max_results": 3},
                            },
                        )
                    )
                    failed_row = store.registry.get(thread_id)
                    self.assertEqual(failed_row["status"], "failed")
                    search_calls_before_resume = len(stub.calls)

                    asyncio.run(
                        self._run_to_end(
                            service,
                            session.key,
                            {
                                "content": "继续上次失败的位置",
                                "turn_id": "t2",
                                "resume_from_last_checkpoint": True,
                            },
                        )
                    )

                row = store.registry.get(thread_id)
                self.assertEqual(row["status"], "completed")
                # 中文注释：注册表保留首次登记的轮次；新 turn 只用于本次消息和事件。
                self.assertEqual(row["turn_id"], "t1")
                self.assertNotEqual(row["run_id"], failed_row["run_id"])
                self.assertEqual(len(stub.calls), search_calls_before_resume)
                artifact_names = [
                    artifact["name"] for artifact in repo.get(session.key).thread()["artifacts"]
                ]
                self.assertEqual(artifact_names.count("search_manifest.json"), 1)
            finally:
                store.close()

    def test_resume_refuses_graph_version_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = SQLiteSessionRepository(storage_root=root / "sessions")
            session = repo.create("Version mismatch")
            store = build_checkpoint_store(
                CheckpointSettings(backend="sqlite", path=str(root / "checkpoints.db"))
            )
            try:
                service = SessionRunService(repo, checkpoint_store=store)
                thread_id = f"{session.key}:legacy"
                store.registry.mark_started(
                    thread_id=thread_id,
                    session_key=session.key,
                    turn_id="legacy",
                    run_id="run-legacy",
                    graph_version=GRAPH_VERSION - 1,
                )
                store.registry.mark_finished(thread_id=thread_id, status="failed")

                with self.assertRaises(SessionError) as raised:
                    asyncio.run(
                        service.start_run(
                            session.key,
                            {
                                "content": "继续上次失败的位置",
                                "turn_id": "t2",
                                "resume_from_last_checkpoint": True,
                            },
                        )
                    )

                self.assertEqual(raised.exception.status, 409)
                self.assertIn(f"v{GRAPH_VERSION - 1}", str(raised.exception))
                self.assertIn(f"v{GRAPH_VERSION}", str(raised.exception))
            finally:
                store.close()

    def test_resume_without_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteSessionRepository(storage_root=Path(temp_dir) / "sessions")
            session = repo.create("No checkpoint")
            service = SessionRunService(repo)

            with self.assertRaises(ValueError) as raised:
                asyncio.run(
                    service.start_run(
                        session.key,
                        {
                            "content": "继续上次失败的位置",
                            "turn_id": "t1",
                            "resume_from_last_checkpoint": True,
                        },
                    )
                )

            self.assertIn("没有找到可继续执行的恢复现场", str(raised.exception))

    def test_completed_run_is_not_resumable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteSessionRepository(storage_root=Path(temp_dir) / "sessions")
            session = repo.create("Completed run")
            service = SessionRunService(repo)
            thread_id = f"{session.key}:t1"
            service.checkpoint_store.registry.mark_started(
                thread_id=thread_id,
                session_key=session.key,
                turn_id="t1",
                run_id="run-done",
                graph_version=GRAPH_VERSION,
            )
            service.checkpoint_store.registry.mark_finished(
                thread_id=thread_id, status="completed"
            )

            with self.assertRaises(ValueError):
                asyncio.run(
                    service.start_run(
                        session.key,
                        {
                            "content": "继续",
                            "turn_id": "t2",
                            "resume_from_last_checkpoint": True,
                        },
                    )
                )

    def test_restart_marks_orphan_interrupted_then_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = SQLiteSessionRepository(storage_root=root / "sessions")
            session = repo.create("Restart resume")
            store = build_checkpoint_store(
                CheckpointSettings(backend="sqlite", path=str(root / "checkpoints.db"))
            )
            stub = _StubService()
            agent = _FlakyAnalyseAgent()
            thread_id = f"{session.key}:t1"
            try:
                first_service = SessionRunService(
                    repo,
                    message_handler=build_paper_workflow_message_handler(repo),
                    checkpoint_store=store,
                )
                with hermetic_graph_llms(), _fake_backends(
                    stub,
                    search_snapshot=_fake_snapshot(),
                    read_snapshot=_read_snapshot(),
                ), mock.patch(
                    "src.graph.analyse_node.build_analyse_agent",
                    lambda *args, **kwargs: agent,
                ):
                    asyncio.run(
                        self._run_to_end(
                            first_service,
                            session.key,
                            {
                                "content": "multi-agent literature review",
                                "turn_id": "t1",
                                "constraints": {"sources": ["openalex"], "max_results": 3},
                            },
                        )
                    )
                    self.assertEqual(store.registry.get(thread_id)["status"], "failed")

                    # 中文注释：模拟进程在 finalize 之前被 kill——会话和账本都停在 running。
                    store.registry.mark_started(
                        thread_id=thread_id,
                        session_key=session.key,
                        turn_id="t1",
                        run_id="run-killed",
                        graph_version=GRAPH_VERSION,
                    )
                    repo.set_status(session.key, "running")
                    repo.set_run_started_at(session.key, "2026-11-10T12:00:00+00:00")

                store.close()
                restarted_store = build_checkpoint_store(
                    CheckpointSettings(backend="sqlite", path=str(root / "checkpoints.db"))
                )
                try:
                    restarted_service = SessionRunService(
                        repo,
                        message_handler=build_paper_workflow_message_handler(repo),
                        checkpoint_store=restarted_store,
                    )
                    self.assertEqual(restarted_service.recover_interrupted_runs(), [session.key])
                    self.assertEqual(
                        restarted_store.registry.get(thread_id)["status"], "interrupted"
                    )

                    with hermetic_graph_llms(), _fake_backends(
                        stub,
                        search_snapshot=_fake_snapshot(),
                        read_snapshot=_read_snapshot(),
                    ), mock.patch(
                        "src.graph.analyse_node.build_analyse_agent",
                        lambda *args, **kwargs: agent,
                    ):
                        asyncio.run(
                            self._run_to_end(
                                restarted_service,
                                session.key,
                                {
                                    "content": "继续上次失败的位置",
                                    "turn_id": "t2",
                                    "resume_from_last_checkpoint": True,
                                },
                            )
                        )

                    self.assertEqual(
                        restarted_store.registry.get(thread_id)["status"], "completed"
                    )
                finally:
                    restarted_store.close()
            finally:
                # 中文注释：close 是幂等的，第一次分支已经关过也没关系。
                store.close()


if __name__ == "__main__":
    unittest.main()