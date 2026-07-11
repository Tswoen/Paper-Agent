from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.contracts import ReviewRequest
from src.graph.reply_node import run_compose_reply_node
from src.graph.read_node import run_read_node
from src.graph.runtime import InlineWorkflowSyncPort, WorkflowRuntimeContext
from src.graph.search_node import run_search_agent_node
from src.graph.state_models import State
from src.paper_retrieval.models import PaperDocument
from src.repositories.sessions.base import SessionRepository


@dataclass(slots=True)
class GraphRunResult:
    """封装图执行完成后的稳定返回结构。"""

    papers: list[PaperDocument] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


GraphState = State


def build_graph():
    """构建当前论文工作流使用的执行图。"""

    workflow = StateGraph(State)
    workflow.add_node("run_search_agent", run_search_agent_node())
    workflow.add_node("run_read", run_read_node())
    workflow.add_node("compose_reply", run_compose_reply_node())
    workflow.add_edge(START, "run_search_agent")
    workflow.add_edge("run_search_agent", "run_read")
    workflow.add_edge("run_read", "compose_reply")
    workflow.add_edge("compose_reply", END)
    return workflow.compile(name="paper_graph")


def run_graph(
    request: ReviewRequest,
    *,
    runtime: WorkflowRuntimeContext | None = None,
    session_repo: SessionRepository | None = None,
    session_key: str | None = None,
    turn_id: str | None = None,
    state_overrides: dict[str, Any] | None = None,
) -> GraphRunResult:
    """运行执行图，并把运行上下文一并注入共享状态。"""

    # 中文注释：这里仍然保留一个统一的 run_graph 入口，外层不需要感知图里具体有几个节点。
    graph = build_graph()
    initial_state = State(
        request=request,
        search_results=[],
        search_scores=[],
        search_summary={},
        search_artifact_refs=[],
        read_results=[],
        read_summary={},
        read_artifact_refs=[],
        diagnostics={},
        current_step="init",
        assistant_message="",
        assistant_message_metadata={},
    )

    # 中文注释：直接从脚本调用且传入会话时，也建立最小进度上报能力，保证产物和进度记录不会分离。
    if runtime is None and session_repo is not None and session_key and turn_id:
        runtime = WorkflowRuntimeContext(
            session_key=session_key,
            turn_id=turn_id,
            workflow_name="paper_graph",
            sync_port=InlineWorkflowSyncPort(
                _build_repository_emitter(session_repo, session_key),
                session_key=session_key,
                turn_id=turn_id,
                workflow_name="paper_graph",
            ),
        )

    # 中文注释：会话信息、运行信息和节点依赖都放进共享状态，让节点自己决定何时同步中间结果。
    if session_repo is not None:
        initial_state["session_repo"] = session_repo
    if session_key:
        initial_state["session_key"] = session_key
    if turn_id:
        initial_state["turn_id"] = turn_id
    if runtime is not None:
        initial_state["runtime_context"] = runtime

    # 中文注释：这个覆盖入口继续保留，方便测试时注入桩服务，不影响正式运行接口。
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


def _build_repository_emitter(repo: SessionRepository, session_key: str):
    """构造直接写入会话仓储的进度发送函数，供没有 API 外层的调用场景使用。"""

    def _emit(event: dict[str, Any]) -> dict[str, Any]:
        """把工作流事件写入会话记录后原样返回，保持同步端口的调用约定。"""

        repo.append_event(
            session_key,
            str(event.get("event") or "workflow_event"),
            content=str(event.get("message") or event.get("content") or ""),
            metadata=dict(event),
        )
        return event

    return _emit

