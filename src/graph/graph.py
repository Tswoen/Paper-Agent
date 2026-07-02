from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.contracts import ReviewRequest
from src.graph.search_node import run_search_agent_node
from src.graph.state_models import State
from src.paper_retrieval.models import PaperDocument
from src.repositories.sessions.base import SessionRepository


@dataclass(slots=True)
class GraphRunResult:
    """封装搜索图的最终运行结果。"""

    papers: list[PaperDocument] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


SearchGraphState = State


def build_search_graph():
    """构建当前最小可运行的论文搜索图。"""

    workflow = StateGraph(State)
    workflow.add_node("run_search_agent", run_search_agent_node())
    workflow.add_edge(START, "run_search_agent")
    workflow.add_edge("run_search_agent", END)
    return workflow.compile(name="paper_graph")


def run_search_graph(
    request: ReviewRequest,
    *,
    session_repo: SessionRepository | None = None,
    session_key: str | None = None,
    turn_id: str | None = None,
    state_overrides: dict[str, Any] | None = None,
) -> GraphRunResult:
    """运行搜索图，并把结果转换为稳定输出。"""

    graph = build_search_graph()
    initial_state = State(
        request=request,
        search_results=[],
        search_scores=[],
        search_summary={},
        search_artifact_refs=[],
        diagnostics={},
        current_step="init",
    )
    # 中文注释：会话上下文直接写入图状态，供检索节点自行决定是否创建持久化 sink。
    if session_repo is not None:
        initial_state["session_repo"] = session_repo
    if session_key:
        initial_state["session_key"] = session_key
    if turn_id:
        initial_state["turn_id"] = turn_id
    # 中文注释：保留一个内部覆盖入口，便于单测注入 stub service 或 fake llm，
    # 同时避免把这些依赖暴露成正式运行接口的一层层透传参数。
    if state_overrides:
        initial_state.update(state_overrides)
    final_state = graph.invoke(initial_state)
    papers = list(final_state.get("search_results") or [])
    diagnostics = dict(final_state.get("diagnostics") or {})
    return GraphRunResult(
        papers=papers,
        state=dict(final_state),
        diagnostics=diagnostics,
    )
