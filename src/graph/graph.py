from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
from src.repositories.sessions.base import SessionRepository


if TYPE_CHECKING:
    from langgraph.runtime import Runtime


@dataclass(slots=True)
class GraphRunResult:
    """封装图执行完成后的稳定返回结构。"""

    papers: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


GraphState = State

# 中文注释：`TypedDict` 只在类型检查阶段生效，运行时它就是普通 dict。节点返回一个
# 未声明的字段不会报错，它会悄悄进入图状态（曾经出现过的 `read_paper_statuses` 就是
# 这种情况）。这里把声明过的字段名固化成集合，在节点返回值的边界上做断言。
_DECLARED_STATE_KEYS = frozenset(State.__annotations__)


def _check_declared_state_keys(origin: str, payload: Any) -> None:
    """校验一份待合并进共享状态的字典只包含 State 已声明的字段。

    中文注释：只检查"多出来的键"，不检查缺键——`total=False` 的语义就是缺键合法。
    """

    if not isinstance(payload, dict):
        return
    unknown = set(payload) - _DECLARED_STATE_KEYS
    if unknown:
        raise KeyError(
            f"{origin} 返回了未在 State 声明的字段：{sorted(unknown)}；"
            "请先在 src/graph/state_models.py 的 State 中声明，或删除该字段。"
        )


def _entrypoint(state: State) -> str:
    """根据是否带阅读恢复现场决定从检索还是阅读节点开始。

    中文注释：这里用真值判断而不是 isinstance。LangGraph 的 channel 没有删除语义，
    阅读节点成功时会写一个空 dict 表示"现场已清空"；isinstance 会把 {} 也当成
    "有待恢复"，导致下次错误地从阅读节点开始。
    """

    return "run_read" if state.get("read_resume_checkpoint") else "run_search_agent"


def build_graph(*, checkpointer: Any = None):
    """构建当前论文工作流使用的执行图。

    中文注释：checkpointer 目前仅用于测试（2c 的"失败后恢复"验收）；阶段 4 才会
    把它接到服务启动路径上。
    """

    workflow = StateGraph(State, context_schema=WorkflowRuntimeContext)
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
    return workflow.compile(name="paper_graph", checkpointer=checkpointer)


def _with_cancellation_boundary(node_name: str, node):
    """给所有图节点加上统一的停止检查与状态字段校验，避免节点里重复写样板代码。"""

    async def _guarded(state: State, runtime: Runtime[WorkflowRuntimeContext] | None) -> State:
        """开始和结束节点时各检查一次，已完成的当前节点不会再启动下一个节点。"""

        # 中文注释：运行时依赖统一从 graph context 取，不再经由 state 传递。
        context = runtime.context if runtime is not None else None
        cancellation = getattr(context, "cancellation", None)
        if cancellation is not None:
            cancellation.raise_if_requested()

        result = await node(state, runtime)
        _check_declared_state_keys(node_name, result)

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
    """异步运行执行图，并通过 graph context 注入运行时依赖。"""

    graph = build_graph()
    runtime = _ensure_runtime(
        runtime,
        session_repo=session_repo,
        session_key=session_key,
        turn_id=turn_id,
    )
    initial_state = _build_initial_state(
        request,
        session_key=session_key,
        turn_id=turn_id,
        state_overrides=state_overrides,
    )
    final_state = await graph.ainvoke(
        initial_state,
        config=_build_run_config(runtime),
        context=runtime,
    )
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


def _build_run_config(runtime: WorkflowRuntimeContext | None) -> dict[str, Any]:
    """组装本次 run 的追踪元数据，方便按会话/轮次检索轨迹。

    中文注释：这里只放 metadata 和 tags，不写 configurable——thread_id 的粒度
    要等阶段 4 接 Checkpointer 时再定，提前写死会让后续迁移多改一处。
    """

    config: dict[str, Any] = {"tags": ["paper_graph"]}
    if runtime is not None:
        config["metadata"] = {
            "session_key": runtime.session_key,
            "turn_id": runtime.turn_id,
            "run_id": runtime.run_id,
        }
    return config


def _ensure_runtime(
    runtime: WorkflowRuntimeContext | None,
    *,
    session_repo: SessionRepository | None,
    session_key: str | None,
    turn_id: str | None,
) -> WorkflowRuntimeContext | None:
    """把调用方给的会话依赖装进运行时上下文，避免它们再经由 state 传递。

    中文注释：直接从脚本调用且传入会话时，也建立最小进度上报能力，
    这样图里产生的进度和产物仍然会落到同一个会话仓库里。
    """

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
    if runtime is not None and session_repo is not None and runtime.session_repo is None:
        runtime.session_repo = session_repo
    return runtime


def _request_payload(request: ReviewRequest) -> dict[str, Any]:
    """把领域对象转成可持久化的纯 dict。

    中文说明：State 里只放 JSON 数据，`ReviewRequest` 属于领域对象；在入口转换一次，
    节点内部统一按 dict 读取。以后若增减字段，只改这里和 `_merge_read_checkpoint`。
    """

    return {
        "topic": str(request.topic or ""),
        "constraints": dict(request.constraints or {}),
        "language": str(request.language or "zh"),
    }


def _build_initial_state(
    request: ReviewRequest,
    *,
    session_key: str | None,
    turn_id: str | None,
    state_overrides: dict[str, Any] | None,
) -> State:
    """整理图执行需要的初始状态，避免同步和异步入口各自拼一遍。"""

    initial_state = State(
        request=_request_payload(request),
        search_results=[],
        # DEPRECATED: 2026-08-22 随关键词打分下线，初始状态不再准备打分明细字段。
        # search_scores=[],
        search_summary={},
        search_artifact_refs=[],
        read_results=[],
        read_summary={},
        read_artifact_refs=[],
        analysis_report={},
        analysis_artifact_refs=[],
        writing_outline={},
        writing_outline_report={},
        writing_outline_artifact_refs=[],
        writing_report={},
        writing_artifact_refs=[],
        diagnostics={},
        current_step="init",
    )

    if session_key:
        initial_state["session_key"] = session_key
    if turn_id:
        initial_state["turn_id"] = turn_id
    if state_overrides:
        _check_declared_state_keys("state_overrides", state_overrides)
        initial_state.update(state_overrides)
    return _merge_read_checkpoint(initial_state)


def _merge_read_checkpoint(state: State) -> State:
    """把恢复现场中的请求、检索结果和已读结果合并回初始状态。"""

    checkpoint = state.get("read_resume_checkpoint")
    if not isinstance(checkpoint, dict):
        return state
    request_payload = checkpoint.get("request")
    if isinstance(request_payload, dict) and str(request_payload.get("topic") or "").strip():
        state["request"] = {
            "topic": str(request_payload.get("topic") or ""),
            "constraints": dict(request_payload.get("constraints") or {}),
            "language": str(request_payload.get("language") or "zh"),
        }
    search_results = [dict(item) for item in list(checkpoint.get("search_results") or []) if isinstance(item, dict)]
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

