import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from langgraph.runtime import Runtime

from src.graph.runtime import WorkflowRuntimeContext
from src.graph.writing_node import run_writing_node
from src.repositories.node_persistence.read_persistence import read_request_fingerprint
from src.repositories.node_persistence.writing_persistence import WritingPersistenceSink
from src.repositories.sessions.sqlite import SQLiteSessionRepository


SECTION_1 = "Chapter1.section1"
SECTION_2 = "Chapter1.section2"

OUTLINE = {
    "Chapter1": {
        "title": "第一章",
        "description": "测试章节",
        "Sections": {
            "section1": {
                "title": "第一节",
                "task": "写第一节",
                "evidence-map": [],
                "ref-sections": [],
                "word-count": 500,
            },
            "section2": {
                "title": "第二节",
                "task": "写第二节",
                "evidence-map": [],
                "ref-sections": ["Chapter1.section1"],
                "word-count": 500,
            },
        },
    },
}


class _StubWritingAgent:
    """记录小节调用与前文参考，供节点级幂等测试断言。"""

    def __init__(self, fail_on: str | None = None):
        self.section_calls: list[str] = []
        self.previous_by_section: dict[str, list] = {}
        self.fail_on = fail_on

    async def async_write_section(self, *, section_id, task, previous_sections, **kwargs):
        self.section_calls.append(section_id)
        self.previous_by_section[section_id] = list(previous_sections)
        if self.fail_on == section_id:
            raise RuntimeError("writing model unavailable")
        return {
            "section_id": section_id,
            "task": task,
            "word_count": 500,
            "content": f"{section_id} 的正文",
            "cited_paper_ids": [],
            "tool_results": [],
            "review": {},
            "revision_count": 0,
            "completed": True,
            "warnings": [],
        }

    async def async_write_abstract(self, **kwargs):
        return "摘要", "ok"


class WritingSectionPersistenceTest(unittest.TestCase):
    """验证小节稳定点的 thread 路径、归属校验和报告 upsert。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.repo = SQLiteSessionRepository(storage_root=self.temp_dir.name)
        self.session_key = self.repo.create(title="Writing idempotency").key
        self.thread_id = f"{self.session_key}:thread-1"
        self.request = {"topic": "多智能体综述", "constraints": {}, "language": "zh"}
        self.fingerprint = read_request_fingerprint(self.request)
        self.sink = self._sink()

    def _sink(self, *, turn_id="turn-1", thread_id=None, fingerprint=None):
        return WritingPersistenceSink(
            self.repo,
            session_key=self.session_key,
            turn_id=turn_id,
            thread_id=self.thread_id if thread_id is None else thread_id,
            request_fingerprint=self.fingerprint if fingerprint is None else fingerprint,
        )

    def _section(self, section_id=SECTION_1, content="小节正文"):
        return {
            "section_id": section_id,
            "content": content,
            "cited_paper_ids": [],
            "completed": True,
            "chapter_key": "Chapter1",
            "section_key": section_id.split(".", 1)[1],
            "chapter_title": "第一章",
            "section_title": "小节标题",
            "warnings": [],
        }

    def test_section_path_and_artifact_survive_turn_change_without_duplicate(self):
        first = self.sink.persist_section(self._section()).artifacts[0]
        second = self._sink(turn_id="turn-2").persist_section(self._section(content="更新正文")).artifacts[0]

        self.assertEqual(first["artifact_id"], second["artifact_id"])
        artifacts = self.repo.get(self.session_key).artifacts
        self.assertEqual(len(artifacts), 1)
        self.assertIn(
            "artifacts/writing/threads/",
            artifacts[0]["path"].replace("\\", "/"),
        )
        loaded = self.sink.load_section(SECTION_1)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.section["content"], "更新正文")

    def test_load_valid_section_and_reject_corrupt_foreign_or_empty(self):
        artifact = self.sink.persist_section(self._section()).artifacts[0]
        path = self.repo.read_artifact_path(self.session_key, artifact["artifact_id"])
        self.assertIsNotNone(path)

        loaded = self.sink.load_section(SECTION_1)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.section_id, SECTION_1)
        self.assertEqual(loaded.artifact["artifact_id"], artifact["artifact_id"])

        path.write_text("{broken", encoding="utf-8")
        self.assertIsNone(self.sink.load_section(SECTION_1))

        self.sink.persist_section(self._section())
        foreign_thread = self._sink(thread_id=f"{self.session_key}:other-thread")
        self.assertIsNone(foreign_thread.load_section(SECTION_1))
        foreign_request = self._sink(fingerprint=read_request_fingerprint({**self.request, "topic": "另一个主题"}))
        self.assertIsNone(foreign_request.load_section(SECTION_1))

        self.sink.persist_section(self._section(content="   "))
        self.assertIsNone(self.sink.load_section(SECTION_1))

    def test_report_upsert_keeps_single_artifact_on_thread_path(self):
        first = self.sink.persist_report({"topic": "t", "sections": []}).artifacts[0]
        second = self._sink(turn_id="turn-2").persist_report({"topic": "t", "sections": [1]}).artifacts[0]

        self.assertEqual(first["artifact_id"], second["artifact_id"])
        artifacts = self.repo.get(self.session_key).artifacts
        self.assertEqual(len(artifacts), 1)
        self.assertTrue(artifacts[0]["path"].replace("\\", "/").endswith("/writing.json"))
        payload = json.loads(Path(artifacts[0]["path"]).read_text(encoding="utf-8"))
        self.assertEqual(payload["sections"], [1])


class WritingNodeIdempotencyTest(unittest.TestCase):
    """用真实临时仓储验证写作节点的小节级幂等与失败边界。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.repo = SQLiteSessionRepository(storage_root=self.temp_dir.name)
        self.session_key = self.repo.create(title="Writing node").key
        self.thread_id = f"{self.session_key}:thread-1"
        self.request = {"topic": "agent review", "constraints": {}, "language": "zh"}

    def _state(self, topic="agent review"):
        return {
            "request": {**self.request, "topic": topic},
            "writing_outline": OUTLINE,
            "analysis_report": {"overall_analysis": {}},
            "search_results": [],
            "read_results": [],
            "writing_artifact_refs": [],
            "diagnostics": {},
        }

    def _runtime(self, turn_id):
        return Runtime(context=WorkflowRuntimeContext(
            session_key=self.session_key,
            turn_id=turn_id,
            thread_id=self.thread_id,
            session_repo=self.repo,
        ))

    def _run(self, agent, turn_id="turn-1", topic="agent review"):
        with (
            mock.patch("src.graph.writing_node.load_writing_agent_llm", return_value=None),
            mock.patch("src.graph.writing_node.build_writing_agent", return_value=agent),
        ):
            return asyncio.run(run_writing_node()(self._state(topic), self._runtime(turn_id)))

    def _section_artifacts(self):
        return [
            artifact
            for artifact in self.repo.get(self.session_key).artifacts
            if artifact["artifact_type"] == "writing_section"
        ]

    def test_completed_sections_skip_model_and_artifacts_on_rerun(self):
        first_agent = _StubWritingAgent()
        first = self._run(first_agent, "turn-1")
        self.assertEqual(first_agent.section_calls, [SECTION_1, SECTION_2])

        second_agent = _StubWritingAgent()
        second = self._run(second_agent, "turn-2")

        self.assertEqual(second_agent.section_calls, [])
        self.assertEqual(
            [item["content"] for item in first["writing_report"]["sections"]],
            [item["content"] for item in second["writing_report"]["sections"]],
        )
        artifacts = self.repo.get(self.session_key).artifacts
        self.assertEqual(len(self._section_artifacts()), 2)
        self.assertEqual(len([a for a in artifacts if a["artifact_type"] == "writing"]), 1)
        self.assertEqual(
            {item["artifact_id"] for item in first["writing_artifact_refs"]},
            {item["artifact_id"] for item in second["writing_artifact_refs"]},
        )

    def test_saved_section_resumes_only_missing_and_feeds_previous(self):
        sink = WritingPersistenceSink(
            self.repo,
            session_key=self.session_key,
            turn_id="turn-0",
            thread_id=self.thread_id,
            request_fingerprint=read_request_fingerprint(self.request),
        )
        saved = {
            "section_id": SECTION_1,
            "content": "崩溃前已完成的正文",
            "cited_paper_ids": [],
            "completed": True,
            "chapter_key": "Chapter1",
            "section_key": "section1",
            "chapter_title": "第一章",
            "section_title": "第一节",
            "warnings": [],
        }
        sink.persist_section(saved)

        agent = _StubWritingAgent()
        result = self._run(agent)

        self.assertEqual(agent.section_calls, [SECTION_2])
        previous = agent.previous_by_section[SECTION_2]
        self.assertEqual(len(previous), 1)
        self.assertEqual(previous[0]["section_id"], SECTION_1)
        self.assertEqual(previous[0]["content"], "崩溃前已完成的正文")
        contents = [item["content"] for item in result["writing_report"]["sections"]]
        self.assertEqual(contents[0], "崩溃前已完成的正文")

    def test_corrupt_stable_point_is_rewritten(self):
        self._run(_StubWritingAgent(), "turn-1")
        artifact = next(a for a in self._section_artifacts() if a["metadata"].get("section_id") == SECTION_1)
        path = self.repo.read_artifact_path(self.session_key, artifact["id"])
        path.write_text("{broken", encoding="utf-8")

        agent = _StubWritingAgent()
        result = self._run(agent, "turn-2")
        # 中文注释：只有损坏的小节重写；完好的第二节仍从稳定点复用。
        self.assertEqual(agent.section_calls, [SECTION_1])
        contents = [item["content"] for item in result["writing_report"]["sections"]]
        self.assertEqual(contents[1], f"{SECTION_2} 的正文")

    def test_changed_request_fingerprint_rewrites_all(self):
        self._run(_StubWritingAgent(), "turn-1", topic="主题 A")
        agent = _StubWritingAgent()
        self._run(agent, "turn-2", topic="主题 B")
        self.assertEqual(agent.section_calls, [SECTION_1, SECTION_2])

    def test_section_failure_keeps_previous_stable_point(self):
        agent = _StubWritingAgent(fail_on=SECTION_2)
        with self.assertRaises(RuntimeError):
            self._run(agent)

        artifacts = self._section_artifacts()
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["metadata"].get("section_id"), SECTION_1)
        path = self.repo.read_artifact_path(self.session_key, artifacts[0]["id"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["section"]["content"], f"{SECTION_1} 的正文")


if __name__ == "__main__":
    unittest.main()
