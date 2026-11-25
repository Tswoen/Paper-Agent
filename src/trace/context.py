from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from src.trace.envelope import build_trace_event
from src.utils import get_logger


JsonObject = dict[str, Any]
TraceEmitter = Callable[[JsonObject], JsonObject]

logger = get_logger(__name__)

# 中文注释：trace 上下文走 ContextVar，按 asyncio 任务天然隔离，不进 State、
# 不进 checkpoint。provider 实例是跨 run 共享的（ProviderSnapshot 按签名缓存），
# 所以绝不能把上下文挂在 provider 实例属性上，否则并发 run 会互相串账。
# asyncio.to_thread 与 gather 子任务都会复制当前 context，同步桥接处需要
# 显式把 context 传进 Runner（见 LLMProvider._run_async_compat）。
_TRACE_CONTEXT: ContextVar["TraceContext | None"] = ContextVar(
    "paper_agent_trace_context", default=None
)


@dataclass(slots=True)
class TraceContext:
    """一次运行的 trace 上下文：事件发给谁、当前挂在哪个节点/父事件下。"""

    emit: TraceEmitter
    node_key: str | None = None
    stage: str | None = None
    # 中文注释：parent_id 指向父事件（节点事件或 tool_call 事件），
    # 由绑定方设置；工具埋点引入后可用它把 llm_call 挂到所属工具调用下。
    parent_id: str | None = None


def current_trace_context() -> TraceContext | None:
    """读取当前任务的 trace 上下文；没有绑定时返回 None。"""

    return _TRACE_CONTEXT.get()


@contextmanager
def bind_trace_context(
    emit: TraceEmitter,
    *,
    node_key: str | None = None,
    stage: str | None = None,
    parent_id: str | None = None,
) -> Iterator[TraceContext]:
    """在当前任务作用域内绑定 trace 上下文，退出时精确还原。"""

    context = TraceContext(emit=emit, node_key=node_key, stage=stage, parent_id=parent_id)
    token = _TRACE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _TRACE_CONTEXT.reset(token)


def emit_trace_event(
    event_type: str,
    *,
    status: str,
    started_at: str,
    duration_ms: int | None = None,
    tokens: JsonObject | None = None,
    error: JsonObject | None = None,
    payload: JsonObject | None = None,
    parent_id: str | None = None,
    stage: str | None = None,
    content: str = "",
    event_id: str | None = None,
) -> JsonObject | None:
    """发送一条 trace 事件。

    中文注释：两条铁律——
    1. 没有绑定上下文时静默跳过（设置页连通性测试、裸脚本不受影响）；
    2. 构造或发送的任何异常都只记警告，绝不影响业务运行。
    埋点是旁观者，永远不能变成新的故障源。
    """

    context = _TRACE_CONTEXT.get()
    if context is None:
        return None
    try:
        event = build_trace_event(
            event_type,
            status=status,
            started_at=started_at,
            node_key=context.node_key,
            stage=stage if stage is not None else context.stage,
            duration_ms=duration_ms,
            tokens=tokens,
            error=error,
            payload=payload,
            parent_id=parent_id if parent_id is not None else context.parent_id,
            content=content,
            event_id=event_id,
        )
        return context.emit(event)
    except Exception:
        logger.warning("trace 事件发送失败（不影响运行）", exc_info=True)
        return None
