import asyncio
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from langgraph.runtime import Runtime

from src.graph.read_node import ReadResourceUnavailableError, _read_one_paper as actual_read_one_paper, _run_read_node_async
from src.graph.runtime import WorkflowRuntimeContext
from src.graph.writing_node import _load_session_read_results
from src.models.read_models import FullTextStatus, PaperReadResult, ReadNote, ReadRelevance
from src.paper_retrieval.models import PaperDocument
from src.repositories.node_persistence.read_persistence import ReadPersistenceSink, read_request_fingerprint
from src.repositories.sessions.sqlite import SQLiteSessionRepository


class ArtifactPersistenceTest(unittest.TestCase):
    """验证 5a-1 的原子文件写和按路径 upsert 不改变默认插入语义。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.repo = SQLiteSessionRepository(storage_root=self.temp_dir.name)
        self.session_key = self.repo.create(title="Read idempotency").key

    def _write(self, content: str, *, replace_existing: bool = False):
        return self.repo.write_artifact(
            self.session_key,
            artifact_type="paper_read_note",
            name="paper_note.json",
            content=content,
            relative_path="artifacts/read/threads/thread-1/papers/paper-1/note.json",
            metadata={"turn_id": "turn-1"},
            replace_existing=replace_existing,
        )

    def test_default_write_keeps_append_behavior(self):
        first = self._write("first")
        second = self._write("second")

        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(len(self.repo.get(self.session_key).artifacts), 2)
        self.assertEqual(Path(second["path"]).read_text(encoding="utf-8"), "second")

    def test_replace_existing_preserves_artifact_id_and_avoids_duplicates(self):
        first = self._write("first", replace_existing=True)
        second = self._write("second", replace_existing=True)

        artifacts = self.repo.get(self.session_key).artifacts
        self.assertEqual(second["id"], first["id"])
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["size"], len("second"))
        self.assertEqual(Path(second["path"]).read_text(encoding="utf-8"), "second")
        self.assertFalse(Path(second["path"]).with_name(".note.json.tmp").exists())

    def test_same_path_concurrent_upsert_keeps_file_and_record_consistent(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            records = list(executor.map(lambda index: self._write(str(index) * (index + 1), replace_existing=True), range(12)))

        artifacts = self.repo.get(self.session_key).artifacts
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(len({record["id"] for record in records}), 1)
        content = Path(artifacts[0]["path"]).read_text(encoding="utf-8")
        self.assertEqual(len(content), artifacts[0]["size"])
        self.assertFalse(Path(artifacts[0]["path"]).with_name(".note.json.tmp").exists())

    def test_failed_temporary_write_preserves_target_and_cleans_up(self):
        first = self._write("stable", replace_existing=True)
        target = Path(first["path"])
        temporary = target.with_name(".note.json.tmp")

        with mock.patch.object(Path, "write_text", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self._write("broken", replace_existing=True)

        self.assertEqual(target.read_text(encoding="utf-8"), "stable")
        self.assertFalse(temporary.exists())
        self.assertEqual(len(self.repo.get(self.session_key).artifacts), 1)


class ReadNotePersistenceTest(unittest.TestCase):
    """验证 note 的 thread 稳定路径、版本/指纹校验和失败进度语义。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.repo = SQLiteSessionRepository(storage_root=self.temp_dir.name)
        self.session_key = self.repo.create(title="Notes").key
        self.request = {
            "topic": "多智能体综述",
            "constraints": {"max_results": 5, "sources": ["arxiv"]},
            "language": "zh",
        }
        self.fingerprint = read_request_fingerprint(self.request)
        self.sink = self._sink()

    def _sink(self, *, turn_id: str = "turn-1", fingerprint: str | None = None):
        return ReadPersistenceSink(
            self.repo,
            session_key=self.session_key,
            turn_id=turn_id,
            thread_id="session:thread-1",
            request_fingerprint=self.fingerprint if fingerprint is None else fingerprint,
        )

    def _result(self, *, failure=None):
        return PaperReadResult(
            paper=PaperDocument(id="paper-1", title="Paper"),
            note=ReadNote(short_summary="可用摘要", evidence_level="metadata"),
            relevance=ReadRelevance(relevant=True, status="selected_for_deep_read"),
            failure=failure,
        )

    def test_note_path_and_artifact_survive_turn_change_without_duplicate(self):
        first = self.sink.persist_paper(self._result(), stage="abstract_complete").artifacts[0]
        second = self._sink(turn_id="turn-2").persist_paper(self._result(), stage="abstract_complete").artifacts[0]

        self.assertEqual(first["artifact_id"], second["artifact_id"])
        self.assertEqual(len(self.repo.get(self.session_key).artifacts), 1)
        record = self.repo.get(self.session_key).artifacts[0]
        self.assertIn("artifacts/read/threads/session_thread-1/papers/paper-1/note.json", record["path"].replace("\\", "/"))
        payload = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
        self.assertEqual(payload["stage"], "abstract_complete")
        self.assertEqual(payload["request_fingerprint"], self.fingerprint)
        self.assertIsNone(payload["last_error"])

    def test_writing_consumer_unwraps_note_and_still_filters_discarded(self):
        result = self._result()
        result.extraction = {"methods": ["agent workflow"]}
        self.sink.persist_paper(result, stage="abstract_complete")
        runtime = Runtime(context=WorkflowRuntimeContext(
            session_key=self.session_key,
            turn_id="turn-1",
            thread_id="session:thread-1",
            session_repo=self.repo,
        ))

        loaded = _load_session_read_results(runtime)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["paper"]["id"], "paper-1")
        self.assertEqual(loaded[0]["extraction"], {"methods": ["agent workflow"]})

        # 中文注释：扁平形状的笔记不被消费——当前 schema 只认 envelope。
        self.repo.write_artifact(
            self.session_key,
            artifact_type="paper_read_note",
            name="flat_note.json",
            content=json.dumps(self._result().to_dict()),
            relative_path="artifacts/read/threads/session_thread-1/papers/paper-flat/note.json",
            metadata={"turn_id": "turn-1"},
        )
        self.assertEqual(len(_load_session_read_results(runtime)), 1)

        result.relevance.discarded = True
        result.relevance.status = "discarded"
        self.sink.persist_paper(result, stage="discarded")
        self.assertEqual(_load_session_read_results(runtime), [])

    def test_load_valid_note_and_reject_corruption_flat_or_fingerprint(self):
        persisted = self.sink.persist_paper(self._result(), stage="abstract_complete").artifacts[0]
        path = self.repo.read_artifact_path(self.session_key, persisted["artifact_id"])
        self.assertIsNotNone(path)

        loaded = self.sink.load_paper_note("paper-1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.stage, "abstract_complete")
        self.assertEqual(loaded.result["note"]["short_summary"], "可用摘要")
        self.assertEqual(loaded.artifact["artifact_id"], persisted["artifact_id"])

        path.write_text("{broken", encoding="utf-8")
        self.assertIsNone(self.sink.load_paper_note("paper-1"))

        # 中文注释：当前 schema 只认 envelope；缺少归属字段的扁平结果不能恢复。
        path.write_text(json.dumps(self._result().to_dict()), encoding="utf-8")
        self.assertIsNone(self.sink.load_paper_note("paper-1"))

        self.sink.persist_paper(self._result(), stage="abstract_complete")
        changed = self._sink(fingerprint=read_request_fingerprint({**self.request, "topic": "另一个主题"}))
        self.assertIsNone(changed.load_paper_note("paper-1"))

    def test_failure_does_not_advance_stage_and_is_serializable(self):
        failure = {"stage": "abstract", "error_kind": "RuntimeError", "message": "boom", "attempts": 1}
        persisted = self.sink.persist_paper(self._result(failure=failure)).artifacts[0]
        path = self.repo.read_artifact_path(self.session_key, persisted["artifact_id"])
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertIsNone(payload["stage"])
        self.assertEqual(payload["last_error"], failure)
        self.assertEqual(payload["result"]["failure"], failure)
        loaded = self.sink.load_paper_note("paper-1")
        self.assertIsNone(loaded.stage)

    def test_note_from_other_thread_is_not_loaded(self):
        """note 的 thread 归属校验：别的线程写下的笔记不能被当前线程恢复。"""

        self.sink.persist_paper(self._result(), stage="abstract_complete")
        other = ReadPersistenceSink(
            self.repo,
            session_key=self.session_key,
            turn_id="turn-1",
            thread_id="session:other-thread",
            request_fingerprint=self.fingerprint,
        )
        self.assertIsNone(other.load_paper_note("paper-1"))

    def test_fingerprint_is_stable_for_constraint_key_order(self):
        reordered = {
            "language": "zh",
            "constraints": {"sources": ["arxiv"], "max_results": 5},
            "topic": "多智能体综述",
        }
        self.assertEqual(read_request_fingerprint(reordered), self.fingerprint)


class ReadNodeIdempotencyTest(unittest.TestCase):
    """用真实临时仓储验证 read 节点的跨 turn 磁盘幂等和失败边界。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.repo = SQLiteSessionRepository(storage_root=self.temp_dir.name)
        self.session_key = self.repo.create(title="Read node").key
        self.thread_id = f"{self.session_key}:thread-1"
        self.request = {
            "topic": "agent review",
            "constraints": {"deep_read_limit": 1},
            "language": "zh",
        }

    def _state(self, papers):
        return {
            "request": self.request,
            "search_results": [paper.to_dict() for paper in papers],
            "read_results": [],
            "read_artifact_refs": [],
            "diagnostics": {},
        }

    def _runtime(self, turn_id: str):
        return Runtime(context=WorkflowRuntimeContext(
            session_key=self.session_key,
            turn_id=turn_id,
            thread_id=self.thread_id,
            session_repo=self.repo,
        ))

    def _run(self, papers, turn_id: str):
        return asyncio.run(_run_read_node_async(self._state(papers), self._runtime(turn_id)))

    def _persist_abstract(self, paper):
        sink = ReadPersistenceSink(
            self.repo,
            session_key=self.session_key,
            turn_id="turn-0",
            thread_id=self.thread_id,
            request_fingerprint=read_request_fingerprint(self.request),
        )
        result = PaperReadResult(
            paper=paper,
            note=ReadNote(short_summary="已保存摘要"),
            relevance=ReadRelevance(relevant=True, status="selected_for_deep_read"),
        )
        return sink.persist_paper(result, stage="abstract_complete").artifacts[0]

    def test_completed_paper_skips_models_and_duplicate_artifacts_on_rerun(self):
        paper = PaperDocument(id="paper-1", title="Agent paper", abstract="abstract")
        abstract_calls = []
        fulltext_calls = []

        async def fake_abstract(paper, **kwargs):
            abstract_calls.append(paper.id)
            return ReadNote(short_summary="摘要"), ReadRelevance(relevant=True), []

        async def fake_fulltext(item, **kwargs):
            if not kwargs.get("process_full_text"):
                return await actual_read_one_paper(item, **kwargs)
            fulltext_calls.append(item.paper.id)
            result = item.restored_result
            result.note.evidence_level = "full_text"
            result.full_text = FullTextStatus(status="chunks_saved")
            return result

        with (
            mock.patch("src.graph.read_node.load_read_agent_llm", return_value=None),
            mock.patch("src.graph.read_node._build_abstract_note", new=fake_abstract),
            mock.patch("src.graph.read_node._read_one_paper", new=fake_fulltext),
        ):
            first = self._run([paper], "turn-1")
            with mock.patch.object(self.repo, "write_artifact", wraps=self.repo.write_artifact) as write_spy:
                second = self._run([paper], "turn-2")

        note_writes = [
            call for call in write_spy.call_args_list if call.kwargs.get("artifact_type") == "paper_read_note"
        ]
        self.assertEqual(note_writes, [])
        self.assertEqual(abstract_calls, ["paper-1"])
        self.assertEqual(fulltext_calls, ["paper-1"])
        self.assertEqual(len(first["read_results"]), 1)
        self.assertEqual(len(second["read_results"]), 1)
        artifacts = self.repo.get(self.session_key).artifacts
        self.assertEqual(len(artifacts), 2)
        self.assertEqual(len(second["read_artifact_refs"]), 2)
        self.assertEqual(
            {item["artifact_id"] for item in first["read_artifact_refs"]},
            {item["artifact_id"] for item in second["read_artifact_refs"]},
        )

    def test_saved_abstract_resumes_only_fulltext(self):
        paper = PaperDocument(id="paper-1", title="Agent paper", abstract="abstract")
        note_artifact = self._persist_abstract(paper)
        abstract_calls = []
        fulltext_calls = []

        async def fake_abstract(paper, **kwargs):
            abstract_calls.append(paper.id)
            raise AssertionError("已有摘要时不应重跑摘要")

        async def fake_fulltext(item, **kwargs):
            if not kwargs.get("process_full_text"):
                return await actual_read_one_paper(item, **kwargs)
            fulltext_calls.append(item.paper.id)
            result = item.restored_result
            result.note.evidence_level = "full_text"
            result.full_text = FullTextStatus(status="chunks_saved")
            return result

        with (
            mock.patch("src.graph.read_node.load_read_agent_llm", return_value=None),
            mock.patch("src.graph.read_node._build_abstract_note", new=fake_abstract),
            mock.patch("src.graph.read_node._read_one_paper", new=fake_fulltext),
        ):
            result = self._run([paper], "turn-1")

        self.assertEqual(abstract_calls, [])
        self.assertEqual(fulltext_calls, ["paper-1"])
        self.assertIn(note_artifact["artifact_id"], [item["artifact_id"] for item in result["read_artifact_refs"]])

    def test_state_read_results_is_not_a_recovery_source(self):
        """state.read_results 只写回供下游消费；没有 note 时论文必须重读。"""

        paper = PaperDocument(id="paper-1", title="Agent paper", abstract="abstract")
        fabricated = PaperReadResult(
            paper=paper,
            note=ReadNote(short_summary="伪造完成", evidence_level="full_text"),
            relevance=ReadRelevance(relevant=True, status="selected_for_deep_read"),
            full_text=FullTextStatus(status="chunks_saved"),
        )
        abstract_calls = []

        async def fake_abstract(paper, **kwargs):
            abstract_calls.append(paper.id)
            return ReadNote(short_summary="重新摘要"), ReadRelevance(relevant=False, reason="无关"), []

        state = self._state([paper])
        state["read_results"] = [fabricated.to_dict()]
        with (
            mock.patch("src.graph.read_node.MIN_USABLE_PAPERS", 0),
            mock.patch("src.graph.read_node.load_read_agent_llm", return_value=None),
            mock.patch("src.graph.read_node._build_abstract_note", new=fake_abstract),
        ):
            result = asyncio.run(_run_read_node_async(state, self._runtime("turn-1")))

        self.assertEqual(abstract_calls, ["paper-1"])
        self.assertEqual(result["read_summary"]["discarded_paper_count"], 1)

    def test_fulltext_error_preserves_usable_abstract_as_partial_failure(self):
        paper = PaperDocument(id="paper-1", title="Agent paper", abstract="abstract")
        note_artifact = self._persist_abstract(paper)

        async def fake_fulltext(item, **kwargs):
            if not kwargs.get("process_full_text"):
                return await actual_read_one_paper(item, **kwargs)
            raise RuntimeError("fulltext boom")

        with (
            mock.patch("src.graph.read_node.load_read_agent_llm", return_value=None),
            mock.patch("src.graph.read_node._read_one_paper", new=fake_fulltext),
        ):
            result = self._run([paper], "turn-1")

        self.assertEqual(len(result["read_results"]), 1)
        self.assertIsNone(result["read_results"][0]["failure"])
        self.assertEqual(result["read_results"][0]["full_text"]["status"], "parse_failed")
        self.assertEqual(result["read_summary"]["failed_paper_count"], 0)
        path = self.repo.read_artifact_path(self.session_key, note_artifact["artifact_id"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["stage"], "abstract_complete")
        self.assertEqual(payload["result"]["note"]["short_summary"], "已保存摘要")

    def test_completed_abstract_is_saved_before_another_paper_halts_the_batch(self):
        papers = [
            PaperDocument(id="done", title="Completed"),
            PaperDocument(id="blocked", title="Blocked"),
        ]

        async def fake_abstract(paper, **kwargs):
            if paper.id == "blocked":
                raise ReadResourceUnavailableError("model unavailable")
            return ReadNote(short_summary="已完成摘要"), ReadRelevance(relevant=True), []

        with (
            mock.patch("src.graph.read_node.load_read_agent_llm", return_value=None),
            mock.patch("src.graph.read_node._build_abstract_note", new=fake_abstract),
        ):
            with self.assertRaises(ReadResourceUnavailableError):
                self._run(papers, "turn-1")

        notes = [item for item in self.repo.get(self.session_key).artifacts if item["artifact_type"] == "paper_read_note"]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["metadata"]["paper_id"], "done")
        payload = json.loads(Path(notes[0]["path"]).read_text(encoding="utf-8"))
        self.assertEqual(payload["stage"], "abstract_complete")

    def test_pending_note_selection_is_written_back_and_then_stable(self):
        paper = PaperDocument(id="paper-1", title="Pending paper")
        sink = ReadPersistenceSink(
            self.repo,
            session_key=self.session_key,
            turn_id="turn-0",
            thread_id=self.thread_id,
            request_fingerprint=read_request_fingerprint(self.request),
        )
        pending = PaperReadResult(
            paper=paper,
            note=ReadNote(short_summary="待选择"),
            relevance=ReadRelevance(relevant=False, status="pending"),
        )
        note_artifact = sink.persist_paper(pending, stage="abstract_complete").artifacts[0]

        async def unexpected_abstract(paper, **kwargs):
            raise AssertionError("有效 note 不应重跑摘要")

        with (
            mock.patch("src.graph.read_node.MIN_USABLE_PAPERS", 0),
            mock.patch("src.graph.read_node.load_read_agent_llm", return_value=None),
            mock.patch("src.graph.read_node._build_abstract_note", new=unexpected_abstract),
        ):
            first = self._run([paper], "turn-1")
            with mock.patch.object(self.repo, "write_artifact", wraps=self.repo.write_artifact) as write_spy:
                second = self._run([paper], "turn-2")

        self.assertEqual(first["read_summary"]["discarded_paper_count"], 1)
        self.assertEqual(second["read_summary"]["discarded_paper_count"], 1)
        path = self.repo.read_artifact_path(self.session_key, note_artifact["artifact_id"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["stage"], "discarded")
        self.assertTrue(payload["result"]["relevance"]["discarded"])
        self.assertEqual(
            [call for call in write_spy.call_args_list if call.kwargs.get("artifact_type") == "paper_read_note"],
            [],
        )

    def test_failure_is_not_selected_or_returned_but_is_summarized(self):
        papers = [
            PaperDocument(id="bad", title="Bad paper"),
            PaperDocument(id="discarded", title="Unrelated paper"),
        ]

        async def fake_abstract(paper, **kwargs):
            if paper.id == "bad":
                raise RuntimeError("abstract boom")
            return ReadNote(short_summary="无关"), ReadRelevance(relevant=False, reason="无关"), []

        with (
            mock.patch("src.graph.read_node.MIN_USABLE_PAPERS", 0),
            mock.patch("src.graph.read_node.load_read_agent_llm", return_value=None),
            mock.patch("src.graph.read_node._build_abstract_note", new=fake_abstract),
        ):
            result = self._run(papers, "turn-1")

        self.assertEqual(result["read_results"], [])
        self.assertEqual(result["read_summary"]["failed_paper_count"], 1)
        self.assertEqual(result["read_summary"]["failed_papers"][0]["paperId"], "bad")
        self.assertEqual(result["read_summary"]["discarded_paper_count"], 1)
        self.assertEqual(result["read_summary"]["deep_read_candidate_count"], 0)

    def test_failed_note_is_retried_and_replaced_on_next_run(self):
        paper = PaperDocument(id="paper-1", title="Retry paper")
        abstract_calls = []

        async def flaky_abstract(paper, **kwargs):
            abstract_calls.append(paper.id)
            if len(abstract_calls) == 1:
                raise RuntimeError("temporary abstract failure")
            return ReadNote(short_summary="恢复后完成"), ReadRelevance(relevant=False, reason="无关"), []

        with (
            mock.patch("src.graph.read_node.MIN_USABLE_PAPERS", 0),
            mock.patch("src.graph.read_node.load_read_agent_llm", return_value=None),
            mock.patch("src.graph.read_node._build_abstract_note", new=flaky_abstract),
        ):
            with self.assertRaisesRegex(RuntimeError, "所有论文"):
                self._run([paper], "turn-1")
            second = self._run([paper], "turn-2")

        self.assertEqual(abstract_calls, ["paper-1", "paper-1"])
        self.assertEqual(second["read_summary"]["failed_paper_count"], 0)
        self.assertEqual(second["read_summary"]["discarded_paper_count"], 1)
        note = next(item for item in self.repo.get(self.session_key).artifacts if item["artifact_type"] == "paper_read_note")
        payload = json.loads(Path(note["path"]).read_text(encoding="utf-8"))
        self.assertEqual(payload["stage"], "discarded")
        self.assertIsNone(payload["last_error"])

    def test_all_failures_fail_read_stage(self):
        paper = PaperDocument(id="bad", title="Bad paper")

        async def fake_abstract(paper, **kwargs):
            raise RuntimeError("abstract boom")

        with (
            mock.patch("src.graph.read_node.load_read_agent_llm", return_value=None),
            mock.patch("src.graph.read_node._build_abstract_note", new=fake_abstract),
        ):
            with self.assertRaisesRegex(RuntimeError, "所有论文"):
                self._run([paper], "turn-1")

    def test_all_discarded_is_a_normal_completion(self):
        paper = PaperDocument(id="irrelevant", title="Unrelated paper")

        async def fake_abstract(paper, **kwargs):
            return ReadNote(short_summary="无关"), ReadRelevance(relevant=False, reason="无关"), []

        with (
            mock.patch("src.graph.read_node.MIN_USABLE_PAPERS", 0),
            mock.patch("src.graph.read_node.load_read_agent_llm", return_value=None),
            mock.patch("src.graph.read_node._build_abstract_note", new=fake_abstract),
        ):
            result = self._run([paper], "turn-1")

        self.assertEqual(result["read_results"], [])
        self.assertEqual(result["read_summary"]["failed_paper_count"], 0)
        self.assertEqual(result["read_summary"]["discarded_paper_count"], 1)
        self.assertEqual(result["current_step"], "read")


if __name__ == "__main__":
    unittest.main()
