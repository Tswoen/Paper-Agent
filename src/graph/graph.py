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
    """图运行结果。

    该对象用于封装论文处理图执行结束后的稳定输出，避免上层直接依赖
    LangGraph 的原始状态结构，便于后续在图中增加更多节点后继续保持
    对外接口稳定。
    """

    papers: list[PaperDocument] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


GraphState = State


def build_graph():
    """构建论文综述系统当前使用的执行图。

    当前图里只接入了 `search` 节点，但这里保留通用命名，方便后续继续
    扩展筛选、总结、综述生成等节点，而无需再次调整公共 API。
    """

    workflow = StateGraph(State)
    workflow.add_node("run_search_agent", run_search_agent_node())
    workflow.add_edge(START, "run_search_agent")
    workflow.add_edge("run_search_agent", END)
    return workflow.compile(name="paper_graph")


def run_graph(
    request: ReviewRequest,
    *,
    session_repo: SessionRepository | None = None,
    session_key: str | None = None,
    turn_id: str | None = None,
    state_overrides: dict[str, Any] | None = None,
) -> GraphRunResult:
    """运行论文综述系统的执行图并返回稳定结果。

    参数中的会话上下文会被注入到图状态中，供节点在需要时持久化检索产物。
    `state_overrides` 仅作为内部扩展入口，主要用于测试时替换依赖或注入
    stub/fake 实现，避免把底层依赖暴露成正式运行参数。
    """

    # 中文注释：统一通过通用图入口构建工作流，避免对“只有搜索节点”的现状产生 API 绑定。
    graph = build_graph()
    initial_state = State(
        request=request,
        search_results=[],
        search_scores=[],
        search_summary={},
        search_artifact_refs=[],
        diagnostics={},
        current_step="init",
    )

    # 中文注释：会话上下文直接写入图状态，具体是否持久化由图中的节点自行判断。
    if session_repo is not None:
        initial_state["session_repo"] = session_repo
    if session_key:
        initial_state["session_key"] = session_key
    if turn_id:
        initial_state["turn_id"] = turn_id

    # 中文注释：保留一个内部状态覆写入口，方便单测注入 stub service 或 fake llm。
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
