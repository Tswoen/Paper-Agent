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
    """封装搜索图的最终运行结果。"""

    papers: list[PaperDocument] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)


SearchGraphState = State


def build_search_graph(
    service: PaperSearchService | None = None,
    llm: ProviderSnapshot | None | str = "auto",
):
    """构建当前最小可运行的论文搜索图。"""

    workflow = StateGraph(State)
    workflow.add_node("run_search_agent", run_search_agent_node(service, llm=llm))
    workflow.add_edge(START, "run_search_agent")
    workflow.add_edge("run_search_agent", END)
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
            search_results=[],
            current_step="init",
        )
    )
    papers = list(final_state.get("search_results") or [])
    return GraphRunResult(
        papers=papers,
        state=dict(final_state),
    )
