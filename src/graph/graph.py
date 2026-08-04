from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.contracts import ReviewRequest
from src.graph.analyse_node import run_analyse_node
from src.graph.reply_node import run_compose_reply_node
from src.graph.read_node import run_read_node
from src.graph.runtime import InlineWorkflowSyncPort, WorkflowRuntimeContext
from src.graph.search_node import run_search_agent_node
from src.graph.state_models import State
from src.graph.writing_outline_node import run_writing_outline_node
from src.graph.writing_node import run_writing_node
from src.paper_retrieval.models import PaperDocument
from src.repositories.sessions.base import SessionRepository


@dataclass(slots=True)
class GraphRunResult:
    """封装图执行完成后的稳定返回结构。"""

    papers: list[PaperDocument] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


GraphState = State


def _entrypoint(state: State) -> str:
    """根据是否带阅读恢复现场决定从检索还是阅读节点开始。"""

    return "run_read" if isinstance(state.get("read_resume_checkpoint"), dict) else "run_search_agent"


def build_graph():
    """构建当前论文工作流使用的执行图。"""

    workflow = StateGraph(State)
    workflow.add_node("run_search_agent", _with_cancellation_boundary("run_search_agent", run_search_agent_node()))
    workflow.add_node("run_read", _with_cancellation_boundary("run_read", run_read_node()))
    workflow.add_node("run_analyse", _with_cancellation_boundary("run_analyse", run_analyse_node()))
    workflow.add_node("run_writing_outline", _with_cancellation_boundary("run_writing_outline", run_writing_outline_node()))
    workflow.add_node("run_writing", _with_cancellation_boundary("run_writing", run_writing_node()))
    workflow.add_node("compose_reply", _with_cancellation_boundary("compose_reply", run_compose_reply_node()))
    workflow.add_conditional_edges(START, _entrypoint, {"run_search_agent": "run_search_agent", "run_read": "run_read"})
    workflow.add_edge("run_search_agent", "run_read")
    workflow.add_edge("run_read", "run_analyse")
    workflow.add_edge("run_analyse", "run_writing_outline")
    workflow.add_edge("run_writing_outline", "run_writing")
    workflow.add_edge("run_writing", "compose_reply")
    workflow.add_edge("compose_reply", END)
    return workflow.compile(name="paper_graph")


def _with_cancellation_boundary(node_name: str, node):
    """给所有图节点加上统一的用户停止检查，避免节点里重复写样板代码。"""

    async def _guarded(state: State) -> State:
        """开始和结束节点时各检查一次，已完成的当前节点不会再启动下一个节点。"""

        runtime = state.get("runtime_context")
        cancellation = getattr(runtime, "cancellation", None)
        if cancellation is not None:
            cancellation.raise_if_requested()

        result = await node(state)

        # 中文注释：如果用户在节点执行期间点了停止，丢弃这个节点尚未提交的结果，
        # 让恢复时从上一个稳定状态重新执行，避免把半成品误认为已经完成。
        if cancellation is not None:
            cancellation.raise_if_requested()
        return result

    _guarded.__name__ = f"{node_name}_with_cancellation"
    return _guarded


async def run_graph(
    request: ReviewRequest,
    *,
    runtime: WorkflowRuntimeContext | None = None,
    session_repo: SessionRepository | None = None,
    session_key: str | None = None,
    turn_id: str | None = None,
    state_overrides: dict[str, Any] | None = None,
) -> GraphRunResult:
    """异步运行执行图，并把运行上下文一并注入共享状态。"""

    graph = build_graph()
    initial_state = _build_initial_state(
        request,
        runtime=runtime,
        session_repo=session_repo,
        session_key=session_key,
        turn_id=turn_id,
        state_overrides=state_overrides,
    )
    final_state = await graph.ainvoke(initial_state)
    papers = list(final_state.get("search_results") or [])
    diagnostics = dict(final_state.get("diagnostics") or {})
    return GraphRunResult(
        papers=papers,
        state=dict(final_state),
        diagnostics=diagnostics,
    )


def run_graph_sync(
    request: ReviewRequest,
    *,
    runtime: WorkflowRuntimeContext | None = None,
    session_repo: SessionRepository | None = None,
    session_key: str | None = None,
    turn_id: str | None = None,
    state_overrides: dict[str, Any] | None = None,
) -> GraphRunResult:
    """给旧同步入口保留一个很薄的兼容壳。"""

    with asyncio.Runner() as runner:
        return runner.run(
            run_graph(
                request,
                runtime=runtime,
                session_repo=session_repo,
                session_key=session_key,
                turn_id=turn_id,
                state_overrides=state_overrides,
            )
        )


def _build_initial_state(
    request: ReviewRequest,
    *,
    runtime: WorkflowRuntimeContext | None,
    session_repo: SessionRepository | None,
    session_key: str | None,
    turn_id: str | None,
    state_overrides: dict[str, Any] | None,
) -> State:
    """整理图执行需要的初始状态，避免同步和异步入口各自拼一遍。"""

    initial_state = State(
        request=request,
        search_results=[],
        search_scores=[],
        search_summary={},
        search_output={},
        search_artifact_refs=[],
        read_results=[],
        read_summary={},
        read_artifact_refs=[],
        analysis_report={},
        analysis_artifact_refs=[],
        writing_outline={},
        writing_outline_report={},
        writing_outline_artifact_refs=[],
        writing_sections=[],
        writing_report={},
        writing_artifact_refs=[],
        final_artifact_refs=[],
        diagnostics={},
        current_step="init",
        assistant_message="",
        assistant_message_metadata={},
    )

    # 中文注释：直接从脚本调用且传入会话时，也建立最小进度上报能力，
    # 这样图里产生的进度和产物仍然会落到同一个会话仓库里。
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

    if session_repo is not None:
        initial_state["session_repo"] = session_repo
    if session_key:
        initial_state["session_key"] = session_key
    if turn_id:
        initial_state["turn_id"] = turn_id
    if runtime is not None:
        initial_state["runtime_context"] = runtime
    if state_overrides:
        initial_state.update(state_overrides)
    return _merge_read_checkpoint(initial_state)


def _merge_read_checkpoint(state: State) -> State:
    """把恢复现场中的请求、检索结果和已读结果合并回初始状态。"""

    checkpoint = state.get("read_resume_checkpoint")
    if not isinstance(checkpoint, dict):
        return state
    request_payload = checkpoint.get("request")
    if isinstance(request_payload, dict) and str(request_payload.get("topic") or "").strip():
        state["request"] = ReviewRequest(
            topic=str(request_payload.get("topic") or ""),
            constraints=dict(request_payload.get("constraints") or {}),
            language=str(request_payload.get("language") or "zh"),
        )
    search_results = _papers_from_checkpoint(checkpoint.get("search_results"))
    if search_results and not state.get("search_results"):
        state["search_results"] = search_results
    if checkpoint.get("read_results") and not state.get("read_results"):
        state["read_results"] = list(checkpoint.get("read_results") or [])
    if checkpoint.get("read_artifact_refs") and not state.get("read_artifact_refs"):
        state["read_artifact_refs"] = list(checkpoint.get("read_artifact_refs") or [])
    # 中文注释：旧 checkpoint 只有阅读模型恢复一种情况，所以这里以前固定写成
    # read_waiting_model。现在 embedding 也可能等待恢复，优先使用 checkpoint 自带步骤。
    state["current_step"] = str(checkpoint.get("current_step") or "read_waiting_model")
    return state


def _papers_from_checkpoint(value: Any) -> list[PaperDocument]:
    """从阅读 checkpoint 恢复检索论文列表。"""

    if not isinstance(value, list):
        return []
    papers: list[PaperDocument] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("id") or "").strip()
        title = str(item.get("title") or "").strip()
        if not paper_id or not title:
            continue
        papers.append(
            PaperDocument(
                id=paper_id,
                title=title,
                authors=[str(author).strip() for author in item.get("authors") or [] if str(author).strip()],
                abstract=str(item.get("abstract")) if item.get("abstract") is not None else None,
                year=_optional_int(item.get("year")),
                venue=str(item.get("venue")) if item.get("venue") is not None else None,
                url=str(item.get("url")) if item.get("url") is not None else None,
                pdf_url=str(item.get("pdf_url")) if item.get("pdf_url") is not None else None,
                doi=str(item.get("doi")) if item.get("doi") is not None else None,
                source=str(item.get("source")) if item.get("source") is not None else None,
                paperId=str(item.get("paperId")) if item.get("paperId") is not None else None,
                publication_date=str(item.get("publication_date") or ""),
                journal_conference=str(item.get("journal_conference") or item.get("journal/conference") or ""),
                volume=str(item.get("volume") or ""),
                issue=str(item.get("issue") or ""),
                language=str(item.get("language") or ""),
                metadata=dict(item.get("metadata") or {}) if isinstance(item.get("metadata"), dict) else {},
            )
        )
    return papers


def _optional_int(value: Any) -> int | None:
    """把 checkpoint 里的可选数字恢复为整数。"""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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

