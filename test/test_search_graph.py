import unittest

from graph import build_search_graph, run_search_graph
from src.agents import ReviewRequest
from src.llm.base import LLMResponse
from src.paper_retrieval.models import PaperDocument, SearchResponse


class _FakeProvider:
    """为图测试提供稳定的关键词输出，避免依赖真实模型配置。"""

    def chat_with_retry(self, messages, **kwargs):
        """返回固定 JSON，保证搜索节点不会因为缺少 LLM 而终止。"""

        return LLMResponse(content='{"keywords":["paper search","literature review"]}', finish_reason="stop")


class _FakeSnapshot:
    """最小 LLM 快照，只暴露 SearchAgent 需要的 provider。"""

    def __init__(self):
        """初始化测试用 provider。"""

        self.provider = _FakeProvider()


class _StubService:
    """用于测试搜索图流程的桩检索服务。"""

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
                    id="graph-paper-1",
                    title="LangGraph Powered Paper Search",
                    authors=["Graph Tester"],
                    year=2026,
                    source=source,
                )
            ],
        )


class SearchGraphTest(unittest.TestCase):
    def test_build_search_graph_returns_compiled_graph(self):
        """验证图可以成功编译。"""

        graph = build_search_graph(service=_StubService(), llm=_FakeSnapshot())

        self.assertTrue(hasattr(graph, "invoke"))

    def test_run_search_graph_returns_ranked_papers(self):
        """验证搜索图执行后会把论文结果写入共享状态与稳定返回值。"""

        stub = _StubService()
        result = run_search_graph(
            ReviewRequest(
                topic="multi-agent literature review",
                constraints={"sources": ["openalex"], "max_results": 5},
            ),
            service=stub,
            llm=_FakeSnapshot(),
        )

        self.assertGreaterEqual(len(stub.calls), 1)
        self.assertEqual(stub.calls[0]["limit"], 5)
        self.assertGreaterEqual(len(result.papers), 1)
        self.assertEqual(result.papers[0].title, "LangGraph Powered Paper Search")
        self.assertEqual(result.state["current_step"], "search")
        self.assertIn("agent", result.diagnostics)
        self.assertIn("search_scores", result.state)

    def test_run_search_graph_executes_each_requested_source_with_full_limit(self):
        """验证多个来源都会执行，且每个来源都使用完整 max_results。"""

        stub = _StubService()
        result = run_search_graph(
            ReviewRequest(
                topic="multi-agent literature review",
                constraints={"sources": ["openalex", "arxiv"], "max_results": 5},
            ),
            service=stub,
            llm=_FakeSnapshot(),
        )

        self.assertEqual([call["source"] for call in stub.calls], ["openalex", "arxiv"])
        self.assertEqual([call["limit"] for call in stub.calls], [5, 5])
        self.assertLessEqual(len(result.papers), 5)


if __name__ == "__main__":
    unittest.main()
