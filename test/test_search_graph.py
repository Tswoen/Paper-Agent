import asyncio
import tempfile
import unittest

from graph import build_graph, run_graph
from src.agents import ReviewRequest
from src.llm.base import LLMResponse
from src.llm.factory import ProviderSnapshot
from src.paper_retrieval.models import PaperDocument, SearchResponse
from src.repositories.sessions.sqlite import SQLiteSessionRepository


class _FakeProvider:
    """为图测试提供稳定关键词输出，避免依赖真实模型配置。"""

    def __init__(self, response_text: str | None = None, response: LLMResponse | None = None):
        """允许不同节点复用同一个可配置假模型。"""

        self.response_text = response_text or '{"keywords":["paper search","literature review"]}'
        self.response = response

    def chat_with_retry(self, messages, **kwargs):
        """返回固定 JSON，确保搜索节点不会因为缺少 LLM 而终止。"""

        if self.response is not None:
            return self.response
        return LLMResponse(content=self.response_text, finish_reason="stop")

    async def chat(self, messages, **kwargs):
        """给阅读节点提供 async 接口，和现在的图执行主链路保持一致。"""

        return self.chat_with_retry(messages, **kwargs)


def _fake_snapshot(response_text: str | None = None, response: LLMResponse | None = None) -> ProviderSnapshot:
    """构造可通过运行时类型检查的假 LLM 快照。"""

    return ProviderSnapshot(
        provider=_FakeProvider(response_text=response_text, response=response),
        model="fake-model",
        context_window_tokens=4096,
        signature="fake-signature",
    )


def _read_snapshot() -> ProviderSnapshot:
    """返回阅读节点可解析的固定 JSON 响应。"""

    return _fake_snapshot(
        response_text=(
            '{"main_question":"agent search","methods":[],"datasets":[],"contributions":[],"limitations":[],'
            '"main_results":[],"short_summary":"agent search summary","match_levels":'
            '{"research_question":"not_match","research_object_or_scene":"not_match",'
            '"method_or_technical_route":"not_match"}}'
        )
    )


class _StubService:
    """用于测试论文处理图流程的桩检索服务。"""

    def __init__(self):
        """记录搜索调用参数，便于验证节点确实执行了检索。"""

        self.calls = []

    def search(self, **kwargs):
        """模拟返回一条稳定论文结果，避免测试依赖外网。"""

        self.calls.append(kwargs)
        source = kwargs.get("source") or "openalex"
        return SearchResponse(
            query=kwargs["query"],
            sources_used=[source],
            source_results={source: 1},
            papers=[
                PaperDocument(
                    id=f"graph-paper-{source}",
                    title=f"LangGraph Powered Paper Search {source}",
                    authors=["Graph Tester"],
                    abstract="LangGraph powered paper search uses agents to review literature.",
                    year=2026,
                    source=source,
                )
            ],
        )

    async def async_search(self, **kwargs):
        """异步检索入口支持单源和多源两种调用，方便验证统一异步接口。"""

        self.calls.append(kwargs)
        sources = list(kwargs.get("sources") or [])
        source = kwargs.get("source")
        if source:
            sources = [source]
        if not sources:
            sources = ["openalex"]
        papers = [
            PaperDocument(
                id=f"graph-paper-{item}",
                title=f"LangGraph Powered Paper Search {item}",
                authors=["Graph Tester"],
                abstract="LangGraph powered paper search uses agents to review literature.",
                year=2026,
                source=item,
            )
            for item in sources
        ]
        return SearchResponse(
            query=kwargs["query"],
            sources_used=list(sources),
            source_results={item: 1 for item in sources},
            papers=papers,
        )


class GraphTest(unittest.TestCase):
    """验证通用图入口在当前搜索场景下的行为稳定性。"""

    def test_build_graph_returns_compiled_graph(self):
        """验证图可以成功编译。"""

        graph = build_graph()

        self.assertTrue(hasattr(graph, "invoke"))
        self.assertTrue(hasattr(graph, "ainvoke"))

    def test_run_graph_returns_ranked_papers(self):
        """验证图执行后会把论文结果写入共享状态与稳定返回值。"""

        stub = _StubService()
        result = asyncio.run(
            run_graph(
            ReviewRequest(
                topic="multi-agent literature review",
                constraints={"sources": ["openalex"], "max_results": 5},
            ),
            state_overrides={
                "search_node_service": stub,
                "search_node_llm": _fake_snapshot(),
                "read_node_llm": _read_snapshot(),
            },
            )
        )

        self.assertGreaterEqual(len(stub.calls), 1)
        self.assertEqual(stub.calls[0]["limit"], 5)
        self.assertGreaterEqual(len(result.papers), 1)
        self.assertEqual(result.state["current_step"], "reply")
        self.assertIn("agent", result.diagnostics)
        self.assertIn("search_scores", result.state)
        self.assertGreaterEqual(len(result.state["search_scores"]), 1)

    def test_run_graph_executes_each_requested_source_with_full_limit(self):
        """验证搜索节点会把多来源请求收口到一次异步服务调用里。"""

        stub = _StubService()
        result = asyncio.run(
            run_graph(
            ReviewRequest(
                topic="multi-agent literature review",
                constraints={"sources": ["openalex", "arxiv"], "max_results": 5},
            ),
            state_overrides={
                "search_node_service": stub,
                "search_node_llm": _fake_snapshot(),
                "read_node_llm": _read_snapshot(),
            },
            )
        )

        self.assertEqual(len(stub.calls), 1)
        self.assertEqual(stub.calls[0]["sources"], ["openalex", "arxiv"])
        self.assertEqual(stub.calls[0]["limit"], 5)
        self.assertLessEqual(len(result.papers), 5)

    def test_run_graph_persists_artifacts_when_session_context_is_provided(self):
        """验证提供会话上下文后，图会把结果落入产物存储。"""

        stub = _StubService()
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteSessionRepository(storage_root=temp_dir)
            session = repo.create("Paper search")
            result = asyncio.run(
                run_graph(
                ReviewRequest(
                    topic="multi-agent literature review",
                    constraints={"sources": ["openalex"], "max_results": 3},
                ),
                session_repo=repo,
                session_key=session.key,
                turn_id="turn-search-1",
                state_overrides={
                    "search_node_service": stub,
                    "search_node_llm": _fake_snapshot(),
                    "read_node_llm": _read_snapshot(),
                },
                )
            )

            thread = repo.get(session.key).thread()
            artifact_names = [artifact["name"] for artifact in thread["artifacts"]]
            completed_search_events = [
                event
                for event in thread["events"]
                if event["event_type"] == "runtime_event"
                and event["metadata"].get("id") == "turn-search-1:search"
                and event["metadata"].get("status") == "completed"
            ]

            self.assertIn("search_manifest.json", artifact_names)
            self.assertTrue(completed_search_events)
            self.assertGreaterEqual(len(result.state["search_artifact_refs"]), 1)

    def test_read_node_saves_checkpoint_when_model_is_unavailable(self):
        """验证阅读模型不可用时不会走保守摘要，而是保存现场并中断。"""

        stub = _StubService()
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteSessionRepository(storage_root=temp_dir)
            session = repo.create("Paper read checkpoint")

            with self.assertRaises(RuntimeError):
                asyncio.run(
                    run_graph(
                    ReviewRequest(
                        topic="multi-agent literature review",
                        constraints={"sources": ["openalex"], "max_results": 3},
                    ),
                    session_repo=repo,
                    session_key=session.key,
                    turn_id="turn-read-checkpoint",
                    state_overrides={
                        "search_node_service": stub,
                        "search_node_llm": _fake_snapshot(),
                        "read_node_llm": _fake_snapshot(
                            response=LLMResponse(
                                content="model unavailable",
                                finish_reason="error",
                                error_kind="server_error",
                                error_status_code=503,
                            )
                        ),
                    },
                    )
                )

            thread = repo.get(session.key).thread()
            checkpoint_artifacts = [
                artifact for artifact in thread["artifacts"] if artifact["artifact_type"] == "paper_read_checkpoint"
            ]
            failed_events = [
                event
                for event in thread["events"]
                if event["event_type"] == "runtime_event"
                and event["metadata"].get("status") == "failed"
                and event["metadata"].get("metadata", {}).get("recovery_status") == "waiting_model"
            ]

            self.assertEqual(len(checkpoint_artifacts), 1)
            self.assertEqual(checkpoint_artifacts[0]["name"], "read_checkpoint.json")
            self.assertTrue(failed_events)
            self.assertEqual(failed_events[-1]["metadata"]["metadata"]["recovery_status"], "waiting_model")

    def test_read_node_can_resume_from_checkpoint(self):
        """验证模型恢复可用后可跳过检索并从 checkpoint 继续阅读。"""

        checkpoint = {
            "request": {
                "topic": "multi-agent literature review",
                "constraints": {"sources": ["openalex"], "max_results": 3},
                "language": "zh",
            },
            "search_results": [
                {
                    "id": "resume-paper-1",
                    "title": "Resume Paper One",
                    "authors": ["Tester"],
                    "abstract": "multi-agent literature review",
                    "year": 2026,
                    "source": "openalex",
                    "metadata": {},
                }
            ],
            "read_results": [],
            "read_artifact_refs": [],
            "next_position": 1,
        }
        stub = _StubService()

        result = asyncio.run(
            run_graph(
            ReviewRequest(topic="placeholder"),
            state_overrides={
                "read_resume_checkpoint": checkpoint,
                "search_node_service": stub,
                "read_node_llm": _read_snapshot(),
            },
            )
        )

        self.assertEqual(stub.calls, [])
        self.assertEqual(result.state["current_step"], "reply")
        self.assertEqual(result.state["search_results"][0].id, "resume-paper-1")
        self.assertEqual(result.state["read_results"][0]["note"]["short_summary"], "agent search summary")


if __name__ == "__main__":
    unittest.main()
