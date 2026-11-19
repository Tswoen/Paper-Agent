import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from langgraph.checkpoint.memory import InMemorySaver

from graph import build_graph, run_graph
from src.agents import ReviewRequest
from src.agents.analyseAgent import AnalyseModelResult
from src.graph.checkpointing import build_checkpoint_bundle
from src.graph.graph import thread_has_pending_work
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
    """验证续跑只认官方 checkpoint、复用旧线程，并拒绝缺失存档或无待执行内容。"""

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
            bundle = build_checkpoint_bundle(
                CheckpointSettings(backend="sqlite", path=str(root / "checkpoints.db"))
            )
            try:
                service = SessionRunService(
                    repo,
                    message_handler=build_paper_workflow_message_handler(repo),
                    checkpoint_bundle=bundle,
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
                    failed_row = bundle.registry.get(thread_id)
                    self.assertEqual(failed_row["status"], "failed")
                    error_events = [
                        event
                        for event in repo.get(session.key).thread()["events"]
                        if event["event_type"] == "error"
                    ]
                    self.assertTrue(error_events)
                    self.assertTrue(error_events[-1]["metadata"].get("checkpoint_id"))
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

                row = bundle.registry.get(thread_id)
                self.assertEqual(row["status"], "completed")
                # 中文注释：注册表保留首次登记的轮次；新 turn 只用于本次消息和事件。
                self.assertEqual(row["turn_id"], "t1")
                self.assertNotEqual(row["run_id"], failed_row["run_id"])
                self.assertEqual(len(stub.calls), search_calls_before_resume)
                artifact_names = [
                    artifact["name"] for artifact in repo.get(session.key).thread()["artifacts"]
                ]
                self.assertEqual(artifact_names.count("search_manifest.json"), 1)

                # 中文注释：续跑事件用新 turn_id，但 thread_id 仍是旧线程（A 方案）。
                resumed_turn_ends = [
                    event
                    for event in repo.get(session.key).thread()["events"]
                    if event["event_type"] == "turn_end"
                    and event["metadata"].get("turn_id") == "t2"
                ]
                self.assertTrue(resumed_turn_ends)
                self.assertEqual(resumed_turn_ends[-1]["metadata"].get("thread_id"), thread_id)
                self.assertEqual(resumed_turn_ends[-1]["metadata"].get("turn_id"), "t2")
            finally:
                bundle.close()

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
            service.checkpoint_bundle.registry.mark_started(
                thread_id=thread_id,
                session_key=session.key,
                turn_id="t1",
                run_id="run-done",
            )
            service.checkpoint_bundle.registry.mark_finished(
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

    def test_new_run_rejects_existing_thread_without_side_effects(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteSessionRepository(storage_root=Path(temp_dir) / "sessions")
            session = repo.create("Duplicate turn")
            service = SessionRunService(repo)
            thread_id = f"{session.key}:t1"
            service.checkpoint_bundle.registry.mark_started(
                thread_id=thread_id,
                session_key=session.key,
                turn_id="t1",
                run_id="run-original",
            )
            service.checkpoint_bundle.registry.mark_finished(thread_id=thread_id, status="failed")
            message_count = len(repo.get(session.key).messages)

            with self.assertRaises(SessionError) as raised:
                asyncio.run(service.start_run(session.key, {"content": "新运行", "turn_id": "t1"}))

            self.assertEqual(raised.exception.status, 409)
            self.assertEqual(len(repo.get(session.key).messages), message_count)
            self.assertEqual(repo.get(session.key).status, "created")
            self.assertIsNone(repo.get(session.key).run_started_at)
            self.assertEqual(service.checkpoint_bundle.registry.get(thread_id)["run_id"], "run-original")

    def test_restart_marks_orphan_interrupted_then_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = SQLiteSessionRepository(storage_root=root / "sessions")
            session = repo.create("Restart resume")
            bundle = build_checkpoint_bundle(
                CheckpointSettings(backend="sqlite", path=str(root / "checkpoints.db"))
            )
            stub = _StubService()
            agent = _FlakyAnalyseAgent()
            thread_id = f"{session.key}:t1"
            try:
                first_service = SessionRunService(
                    repo,
                    message_handler=build_paper_workflow_message_handler(repo),
                    checkpoint_bundle=bundle,
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
                    self.assertEqual(bundle.registry.get(thread_id)["status"], "failed")

                    # 中文注释：模拟进程在 finalize 之前被 kill——会话和账本都停在 running。
                    bundle.registry.mark_started(
                        thread_id=thread_id,
                        session_key=session.key,
                        turn_id="t1",
                        run_id="run-killed",
                    )
                    repo.set_status(session.key, "running")
                    repo.set_run_started_at(session.key, "2026-11-10T12:00:00+00:00")

                bundle.close()
                restarted_bundle = build_checkpoint_bundle(
                    CheckpointSettings(backend="sqlite", path=str(root / "checkpoints.db"))
                )
                try:
                    restarted_service = SessionRunService(
                        repo,
                        message_handler=build_paper_workflow_message_handler(repo),
                        checkpoint_bundle=restarted_bundle,
                    )
                    self.assertEqual(restarted_service.recover_interrupted_runs(), [session.key])
                    self.assertEqual(
                        restarted_bundle.registry.get(thread_id)["status"], "interrupted"
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
                        restarted_bundle.registry.get(thread_id)["status"], "completed"
                    )
                finally:
                    restarted_bundle.close()
            finally:
                # 中文注释：close 是幂等的，第一次分支已经关过也没关系。
                bundle.close()

    def test_resume_uses_latest_thread_not_older_failure(self):
        """最新一轮已经成功时，不能跳回去续更早的失败线程。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteSessionRepository(storage_root=Path(temp_dir) / "sessions")
            session = repo.create("Latest thread wins")
            service = SessionRunService(repo)
            registry = service.checkpoint_bundle.registry
            old_thread = f"{session.key}:t1"
            new_thread = f"{session.key}:t2"

            registry.mark_started(
                thread_id=old_thread,
                session_key=session.key,
                turn_id="t1",
                run_id="run-1",
            )
            registry.mark_finished(
                thread_id=old_thread,
                status="failed",
                finished_at="2026-11-10T10:01:00+00:00",
            )
            registry.mark_started(
                thread_id=new_thread,
                session_key=session.key,
                turn_id="t2",
                run_id="run-2",
            )
            registry.mark_finished(
                thread_id=new_thread,
                status="completed",
                finished_at="2026-11-10T10:02:00+00:00",
            )

            checked: list[str] = []

            async def fake_pending(checkpointer, thread_id):
                checked.append(thread_id)
                return False

            with mock.patch(
                "src.services.session_runs.thread_has_pending_work",
                new=fake_pending,
            ):
                with self.assertRaises(ValueError) as raised:
                    asyncio.run(
                        service.start_run(
                            session.key,
                            {
                                "content": "继续",
                                "turn_id": "t3",
                                "resume_from_last_checkpoint": True,
                            },
                        )
                    )

            # 中文注释：必须问的是“最近一次线程”，不是更早的失败线程。
            self.assertEqual(checked, [new_thread])
            self.assertIn("没有待执行的内容", str(raised.exception))

    def test_resume_self_heals_ownerless_running_row(self):
        """账本 running 但本进程没有活任务：先改 interrupted，再按 pending 续跑。"""

        async def exercise():
            temp_dir = tempfile.TemporaryDirectory()
            self.addCleanup(temp_dir.cleanup)
            repo = SQLiteSessionRepository(storage_root=Path(temp_dir.name) / "sessions")
            session = repo.create("Self heal")
            bundle = build_checkpoint_bundle(CheckpointSettings(backend="memory"))
            thread_id = f"{session.key}:t1"
            bundle.registry.mark_started(
                thread_id=thread_id,
                session_key=session.key,
                turn_id="t1",
                run_id="run-1",
            )
            captured = {}

            async def handler(chat_id, content, frame, emit):
                captured.update(frame)

            service = SessionRunService(repo, message_handler=handler, checkpoint_bundle=bundle)

            async def fake_pending(checkpointer, tid):
                return True

            with mock.patch(
                "src.services.session_runs.thread_has_pending_work",
                new=fake_pending,
            ), mock.patch.object(
                bundle.registry,
                "mark_running_interrupted",
                wraps=bundle.registry.mark_running_interrupted,
            ) as healed:
                accepted = await service.start_run(
                    session.key,
                    {
                        "content": "继续",
                        "turn_id": "t2",
                        "resume_from_last_checkpoint": True,
                    },
                )
                task = service.broker.get_run(accepted["run_id"]).task
                await asyncio.wait_for(task, timeout=30)

            return {
                "expected_thread": thread_id,
                "runtime_thread": captured["runtime_context"].thread_id,
                "resume": captured["resume_from_checkpoint"],
                "row": bundle.registry.get(thread_id),
                "healed": healed.call_count,
            }

        result = asyncio.run(exercise())

        self.assertEqual(result["runtime_thread"], result["expected_thread"])
        self.assertTrue(result["resume"])
        self.assertEqual(result["healed"], 1)
        self.assertEqual(result["row"]["status"], "completed")
        self.assertEqual(result["row"]["turn_id"], "t1")

    def test_resume_marks_running_row_interrupted_when_nothing_pending(self):
        """没有待执行内容时，也要先把无主的 running 行修成 interrupted。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteSessionRepository(storage_root=Path(temp_dir) / "sessions")
            session = repo.create("Nothing pending")
            service = SessionRunService(repo)
            thread_id = f"{session.key}:t1"
            service.checkpoint_bundle.registry.mark_started(
                thread_id=thread_id,
                session_key=session.key,
                turn_id="t1",
                run_id="run-1",
            )

            async def fake_pending(checkpointer, tid):
                return False

            with mock.patch(
                "src.services.session_runs.thread_has_pending_work",
                new=fake_pending,
            ):
                with self.assertRaises(ValueError) as raised:
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

            self.assertIn("没有待执行的内容", str(raised.exception))
            self.assertEqual(
                service.checkpoint_bundle.registry.get(thread_id)["status"],
                "interrupted",
            )

    def test_thread_has_pending_work_detects_uncommitted_writes(self):
        """next 为空但 pending_writes 非空时，必须判定为“还有活”。"""

        async def exercise():
            saver = InMemorySaver()
            thread_id = "pending-window-thread"
            with hermetic_graph_llms(), _fake_backends(
                _StubService(),
                search_snapshot=_fake_snapshot(),
                read_snapshot=_read_snapshot(),
            ):
                await run_graph(
                    ReviewRequest(
                        topic="multi-agent literature review",
                        constraints={"sources": ["openalex"], "max_results": 3},
                    ),
                    checkpointer=saver,
                    thread_id=thread_id,
                )

            config = {"configurable": {"thread_id": thread_id}}
            graph = build_graph(checkpointer=saver)
            before = await graph.aget_state(config)
            checkpoint_tuple = saver.get_tuple(config)
            self.assertIsNotNone(checkpoint_tuple)
            # 中文注释：模拟“任务写完、checkpoint 未推进”的窗口。
            saver.put_writes(
                checkpoint_tuple.config,
                [("current_step", "init")],
                task_id="pending-window-test",
            )
            after = await graph.aget_state(config)
            return before.next, after.next, await thread_has_pending_work(saver, thread_id)

        before_next, after_next, pending = asyncio.run(exercise())

        self.assertEqual(before_next, ())
        self.assertEqual(after_next, ())
        self.assertTrue(pending)

    def test_start_run_rejects_running_session_before_checkpoint_lookup(self):
        """会话在跑时应直接 409，不能先去做账本/存档查询。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteSessionRepository(storage_root=Path(temp_dir) / "sessions")
            session = repo.create("Running session")
            repo.set_status(session.key, "running")
            repo.set_run_started_at(session.key, "2026-11-11T12:00:00+00:00")
            service = SessionRunService(repo)

            with mock.patch("src.services.session_runs.thread_has_pending_work") as pending:
                with self.assertRaises(SessionError) as raised:
                    asyncio.run(
                        service.start_run(
                            session.key,
                            {
                                "content": "继续",
                                "turn_id": "t1",
                                "resume_from_last_checkpoint": True,
                            },
                        )
                    )

            self.assertEqual(raised.exception.status, 409)
            pending.assert_not_called()


if __name__ == "__main__":
    unittest.main()