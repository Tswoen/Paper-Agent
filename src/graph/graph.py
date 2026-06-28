from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.contracts import ReviewRequest
from src.graph.search_node import run_search_agent_node
from src.graph.state_models import State
from src.llm import ProviderSnapshot
from src.paper_retrieval import PaperSearchService
from src.paper_retrieval.models import PaperDocument


@dataclass(slots=True)
class GraphRunResult:
    """封装搜索图的最终运行结果。

    当前以共享状态为核心，因此这里只暴露最常用的结果字段，
    同时保留完整 state 方便后续节点开发和调试。
    """

    papers: list[PaperDocument] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)


SearchGraphState = State


def build_search_graph(
    service: PaperSearchService | None = None,
    llm: ProviderSnapshot | None | str = "auto",
):
    """构建当前最小可运行的论文搜索图。"""

    workflow = StateGraph(State)
    workflow.add_node("run_search_agent", run_search_agent_node(service, llm=llm))
    workflow.add_node("finalize_output", _finalize_output_node)
    workflow.add_edge(START, "run_search_agent")
    workflow.add_edge("run_search_agent", "finalize_output")
    workflow.add_edge("finalize_output", END)
    return workflow.compile(name="paper_graph")


def run_search_graph(
    request: ReviewRequest,
    service: PaperSearchService | None = None,
    llm: ProviderSnapshot | None | str = "auto",
) -> GraphRunResult:
    """运行搜索图，并把结果转换为稳定输出。"""

    graph = build_search_graph(service=service, llm=llm)
    final_state = graph.invoke(
        State(
            request=request,
            diagnostics={},
            search_results=[],
            search_scores=[],
            current_step="init",
        )
    )
    papers = list(final_state.get("search_results") or [])
    diagnostics = dict(final_state.get("diagnostics") or {})
    return GraphRunResult(
        papers=papers,
        diagnostics=diagnostics,
        state=dict(final_state),
    )


def _finalize_output_node(state: State) -> State:
    """整理最终诊断信息，并保留统一的出口。"""

    diagnostics = dict(state.get("diagnostics") or {})
    diagnostics["final_state"] = {
        "current_step": state.get("current_step"),
        "paper_count": len(state.get("search_results") or []),
    }
    state["diagnostics"] = diagnostics
    return state
