import asyncio
import json
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from unittest import mock

from langgraph.checkpoint.memory import InMemorySaver

from graph import build_graph, run_graph
from src.agents import ReviewRequest
from src.agents.analyseAgent import AnalyseModelResult
from src.graph.graph import _build_initial_state
from src.graph.runtime import WorkflowRuntimeContext
from src.llm.base import LLMResponse
from src.llm.factory import ProviderSnapshot
from src.paper_retrieval.models import PaperDocument, SearchResponse
from src.repositories.sessions.sqlite import SQLiteSessionRepository

try:
    from .graph_test_support import hermetic_graph_llms
except ImportError:
    from graph_test_support import hermetic_graph_llms


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


@contextmanager
def _fake_backends(
    stub: "_StubService | None" = None,
    *,
    search_snapshot: ProviderSnapshot | None = None,
    read_snapshot: ProviderSnapshot | None = None,
):
    """把节点内部真实的检索服务和模型装配换成测试替身。

    中文说明：依赖不再经由 state 注入，测试改为替换模块级工厂/加载函数。
    这样测试和生产走同一个 context 通道，不再存在"只有测试会写"的后门字段。
    """

    patches = []
    if stub is not None:
        patches.append(mock.patch("src.graph.search_node.PaperSearchService", lambda: stub))
    if search_snapshot is not None:
        patches.append(mock.patch("src.graph.search_node.load_search_agent_llm", lambda *args, **kwargs: search_snapshot))
    if read_snapshot is not None:
        patches.append(mock.patch("src.graph.read_node.load_read_agent_llm", lambda *args, **kwargs: read_snapshot))
    with ExitStack() as stack:
        for patcher in patches:
            stack.enter_context(patcher)
        yield


class GraphTest(unittest.TestCase):
    """验证通用图入口在当前搜索场景下的行为稳定性。"""

    def setUp(self):
        """把图测试固定在没有本地模型配置的状态，避免结果随 model.json 漂移。"""

        ctx = hermetic_graph_llms()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)

    def test_build_graph_returns_compiled_graph(self):
        """验证图可以成功编译。"""

        graph = build_graph()

        self.assertTrue(hasattr(graph, "invoke"))
        self.assertTrue(hasattr(graph, "ainvoke"))

    def _run_full_graph(self):
        """跑一次完整链路，供多条验收断言复用。"""

        stub = _StubService()
        with _fake_backends(stub, search_snapshot=_fake_snapshot(), read_snapshot=_read_snapshot()):
            result = asyncio.run(
                run_graph(
                    ReviewRequest(
                        topic="multi-agent literature review",
                        constraints={"sources": ["openalex"], "max_results": 3},
                    )
                )
            )
        return stub, result

    def test_final_state_contains_core_fields(self):
        """最小增量返回不能漏掉下游要用的状态字段。"""

        _, result = self._run_full_graph()
        core_fields = {
            "request",
            "search_results",
            "read_results",
            "analysis_report",
            "writing_outline",
            "writing_report",
        }

        self.assertTrue(core_fields <= set(result.state), core_fields - set(result.state))
        # 中文注释：writing_report["sections"] 是正文唯一权威，State 里不能再有平铺副本。
        self.assertNotIn("writing_sections", result.state)
        self.assertTrue(result.state["writing_report"].get("sections"))

    def test_final_state_is_json_serializable(self):
        """最终 state 必须是纯 JSON 数据，这是接 Checkpointer 的硬前提。"""

        _, result = self._run_full_graph()

        payload = json.dumps(result.state, ensure_ascii=False)

        self.assertGreater(len(payload), 0)

    def test_state_survives_failure_and_resumes_from_checkpoint(self):
        """分析节点失败后能从 checkpoint 继续跑完，且不会重跑前面的节点。"""

        class _FlakyAnalyseAgent:
            """第一次综合分析失败，第二次成功，用来模拟"服务崩溃后恢复"。"""

            def __init__(self):
                self.overall_calls = 0

            async def async_analyse_subtopic(self, *, topic, group, usage_callback=None):
                return AnalyseModelResult(parsed=None, reason="stub: 子主题走兜底")

            async def async_analyse_overall(self, *, topic, subtopic_analyses, usage_callback=None):
                self.overall_calls += 1
                if self.overall_calls == 1:
                    raise RuntimeError("stub analysis model unavailable")
                return AnalyseModelResult(parsed={}, raw_model_output="{}")

        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteSessionRepository(storage_root=temp_dir)
            session = repo.create("Resume after failure")
            stub = _StubService()
            agent = _FlakyAnalyseAgent()
            thread_id = "turn-resume-after-failure"
            config = {"configurable": {"thread_id": thread_id}}
            context = WorkflowRuntimeContext(
                session_key=session.key,
                turn_id=thread_id,
                workflow_name="paper_graph",
                session_repo=repo,
            )
            graph = build_graph(checkpointer=InMemorySaver())
            initial_state = _build_initial_state(
                ReviewRequest(
                    topic="multi-agent literature review",
                    constraints={"sources": ["openalex"], "max_results": 3},
                ),
                session_key=session.key,
                turn_id=thread_id,
                state_overrides=None,
            )

            with _fake_backends(stub, search_snapshot=_fake_snapshot(), read_snapshot=_read_snapshot()), mock.patch(
                "src.graph.analyse_node.build_analyse_agent",
                lambda *args, **kwargs: agent,
            ):
                with self.assertRaises(RuntimeError):
                    asyncio.run(graph.ainvoke(initial_state, config=config, context=context))
                search_calls_before_resume = len(stub.calls)
                final_state = asyncio.run(graph.ainvoke(None, config=config, context=context))

            self.assertEqual(final_state["current_step"], "reply")
            # 中文注释：恢复时从 run_analyse 继续，不能再撞一次检索。
            self.assertEqual(len(stub.calls), search_calls_before_resume)
            artifact_names = [artifact["name"] for artifact in repo.get(session.key).thread()["artifacts"]]
            self.assertEqual(artifact_names.count("search_manifest.json"), 1)

    def test_run_graph_returns_ranked_papers(self):
        """验证图执行后会把论文结果写入共享状态与稳定返回值。"""

        stub = _StubService()
        with _fake_backends(stub, search_snapshot=_fake_snapshot(), read_snapshot=_read_snapshot()):
            result = asyncio.run(
                run_graph(
                    ReviewRequest(
                        topic="multi-agent literature review",
                        constraints={"sources": ["openalex"], "max_results": 5},
                    )
                )
            )

        self.assertGreaterEqual(len(stub.calls), 1)
        # 中文注释：limit 是每个来源要拉取的候选池大小，不是最终返回篇数。
        # 当前配置的候选池为 min(60, max(20, 5×max_results))，max_results=5 时为 25。
        self.assertEqual(stub.calls[0]["limit"], 25)
        self.assertGreaterEqual(len(result.papers), 1)
        self.assertEqual(result.state["current_step"], "reply")
        self.assertIn("agent", result.diagnostics)
        # 中文注释：2b 之后 request/search_results 必须是纯 dict，且主题不能在迁移中丢失
        # （`getattr(request, ...)` 在 dict 上会静默返回空串，这条断言专门堵那个坑）。
        self.assertIsInstance(result.state["request"], dict)
        self.assertIsInstance(result.state["search_results"][0], dict)
        self.assertEqual(result.state["request"]["topic"], "multi-agent literature review")
        self.assertEqual(result.state["search_summary"]["topic"], "multi-agent literature review")
        # DEPRECATED: 2026-08-22
        # 原因：关键词打分逻辑已下线，状态里不再有 search_scores 字段。
        # 替代方案：改用 search_results 检查最终论文产出（见上方 assertGreaterEqual(len(result.papers), 1)）。
        # 计划移除：语义过滤稳定运行一段时间后清理。
        # self.assertIn("search_scores", result.state)
        # self.assertGreaterEqual(len(result.state["search_scores"]), 1)

    def test_run_graph_executes_each_requested_source_with_full_limit(self):
        """验证搜索节点会把多来源请求收口到一次异步服务调用里。"""

        stub = _StubService()
        with _fake_backends(stub, search_snapshot=_fake_snapshot(), read_snapshot=_read_snapshot()):
            result = asyncio.run(
                run_graph(
                    ReviewRequest(
                        topic="multi-agent literature review",
                        constraints={"sources": ["openalex", "arxiv"], "max_results": 5},
                    )
                )
            )

        self.assertEqual(len(stub.calls), 1)
        self.assertEqual(stub.calls[0]["sources"], ["openalex", "arxiv"])
        # 中文注释：候选池大小同样按 max_results=5 计算，语义过滤前要多搜一些候选。
        self.assertEqual(stub.calls[0]["limit"], 25)
        self.assertLessEqual(len(result.papers), 5)

    def test_run_graph_persists_artifacts_when_session_context_is_provided(self):
        """验证提供会话上下文后，图会把结果落入产物存储。"""

        stub = _StubService()
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteSessionRepository(storage_root=temp_dir)
            session = repo.create("Paper search")
            with _fake_backends(stub, search_snapshot=_fake_snapshot(), read_snapshot=_read_snapshot()):
                result = asyncio.run(
                    run_graph(
                        ReviewRequest(
                            topic="multi-agent literature review",
                            constraints={"sources": ["openalex"], "max_results": 3},
                        ),
                        session_repo=repo,
                        session_key=session.key,
                        turn_id="turn-search-1",
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
            # 中文注释：事件里的 writing_sections 必须和 report.sections 是同一份数据，
            # 避免删除 State 字段后事件回放拿到空小节。
            assistant_events = [
                event
                for event in thread["events"]
                if event["event_type"] == "message"
                and event["metadata"].get("role") == "assistant"
            ]
            self.assertTrue(assistant_events)
            payload = assistant_events[-1]["metadata"]["metadata"]
            self.assertEqual(payload["writing_sections"], result.state["writing_report"]["sections"])

    def test_read_node_saves_checkpoint_when_model_is_unavailable(self):
        """验证阅读模型不可用时不会走保守摘要，而是保存现场并中断。"""

        stub = _StubService()
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = SQLiteSessionRepository(storage_root=temp_dir)
            session = repo.create("Paper read checkpoint")

            with _fake_backends(
                stub,
                search_snapshot=_fake_snapshot(),
                read_snapshot=_fake_snapshot(
                    response=LLMResponse(
                        content="model unavailable",
                        finish_reason="error",
                        error_kind="server_error",
                        error_status_code=503,
                    )
                ),
            ):
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

        with _fake_backends(stub, read_snapshot=_read_snapshot()):
            result = asyncio.run(
                run_graph(
                    ReviewRequest(topic="placeholder"),
                    state_overrides={"read_resume_checkpoint": checkpoint},
                )
            )

        self.assertEqual(stub.calls, [])
        self.assertEqual(result.state["current_step"], "reply")
        self.assertIsInstance(result.state["search_results"][0], dict)
        self.assertEqual(result.state["search_results"][0]["id"], "resume-paper-1")
        self.assertEqual(result.state["read_results"][0]["note"]["short_summary"], "agent search summary")


if __name__ == "__main__":
    unittest.main()
