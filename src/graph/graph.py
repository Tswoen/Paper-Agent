from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.contracts import ReviewRequest
from src.graph.search_node import run_search_agent_node
from src.graph.search_persistence import SearchPersistenceSink
from src.graph.state_models import State
from src.llm import ProviderSnapshot
from src.paper_retrieval import PaperSearchService
from src.paper_retrieval.models import PaperDocument
from src.repositories.sessions.base import SessionRepository


@dataclass(slots=True)
class GraphRunResult:
    """封装搜索图的最终运行结果。"""

    papers: list[PaperDocument] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


SearchGraphState = State


def build_search_graph(
    service: PaperSearchService | None = None,
    llm: ProviderSnapshot | None | str = "auto",
    sink: SearchPersistenceSink | None = None,
):
    """构建当前最小可运行的论文搜索图。"""

    workflow = StateGraph(State)
    workflow.add_node("run_search_agent", run_search_agent_node(service, llm=llm, sink=sink))
    workflow.add_edge(START, "run_search_agent")
    workflow.add_edge("run_search_agent", END)
    return workflow.compile(name="paper_graph")


def run_search_graph(
    request: ReviewRequest,
    service: PaperSearchService | None = None,
    llm: ProviderSnapshot | None | str = "auto",
    *,
    session_repo: SessionRepository | None = None,
    session_key: str | None = None,
    turn_id: str | None = None,
) -> GraphRunResult:
    """运行搜索图，并把结果转换为稳定输出。"""

    sink = None
    if session_repo is not None and session_key and turn_id:
        # 中文注释：只有当调用方显式提供会话上下文时，才开启检索结果持久化，避免纯单元测试或离线调用被强制依赖存储系统。
        sink = SearchPersistenceSink(session_repo, session_key=session_key, turn_id=turn_id)
    graph = build_search_graph(service=service, llm=llm, sink=sink)
    final_state = graph.invoke(
        State(
            request=request,
            search_results=[],
            search_scores=[],
            search_summary={},
            search_artifact_refs=[],
            diagnostics={},
            current_step="init",
        )
    )
    papers = list(final_state.get("search_results") or [])
    diagnostics = dict(final_state.get("diagnostics") or {})
    return GraphRunResult(
        papers=papers,
        state=dict(final_state),
        diagnostics=diagnostics,
    )
