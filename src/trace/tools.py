from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator
from uuid import uuid4

from src.models.sessions import utc_now
from src.trace.context import current_trace_context, emit_trace_event
from src.utils import get_logger


JsonObject = dict[str, Any]
logger = get_logger(__name__)


@dataclass(slots=True)
class ToolCallHandle:
    """一次工具调用的记账把手：调用方用它登记结果与状态。"""

    call_id: str
    name: str
    result: JsonObject | None = None
    status: str = "completed"
    error: JsonObject | None = None

    def set_result(
        self,
        result: JsonObject,
        *,
        status: str = "completed",
        error: JsonObject | None = None,
    ) -> None:
        """登记工具结果。

        中文注释：下载/转换这类"不抛异常、返回失败状态"的优雅失败，
        调用方显式传 status="failed" 与 error，让账本如实反映成败。
        """

        self.result = result
        self.status = status
        self.error = error


@contextmanager
def trace_tool(
    name: str,
    *,
    args: JsonObject | None = None,
    stage: str | None = None,
) -> Iterator[ToolCallHandle]:
    """给一次工具调用套上"开工单 + 完工单"（tool_call / tool_result）。

    中文注释：三条规则——
    1. 进入时发 tool_call，并把当前 trace 上下文的 parent_id 临时换成本次
       call_id，工具内部的 llm_call 等事件自动挂到工具名下；退出时还原；
    2. 异常先记 failed/cancelled 完工单，然后原样冒泡——不吞、不包装；
    3. 没有上下文时全程 no-op；记账的任何异常只记警告，绝不影响工具本身。
    """

    call_id = uuid4().hex
    handle = ToolCallHandle(call_id=call_id, name=name)
    started_perf = time.perf_counter()
    started_wall = utc_now()
    context = current_trace_context()
    previous_parent = context.parent_id if context is not None else None

    try:
        emit_trace_event(
            "tool_call",
            status="running",
            started_at=started_wall,
            stage=stage,
            payload={"name": name, "args": args if args is not None else {}},
            content=f"{name} 开始",
            # 中文注释：开工单的 id 就是 call_id，完工单以它为 parent_id 配对。
            event_id=call_id,
        )
    except Exception:
        logger.warning("tool_call trace 事件发送失败（不影响运行）", exc_info=True)

    if context is not None:
        context.parent_id = call_id
    try:
        yield handle
    except BaseException as exc:
        # 中文注释：CancelledError 不是 Exception 子类，用 BaseException 才能都接住；
        # 接住只为留张字条，随后立刻原样抛出。
        if isinstance(exc, asyncio.CancelledError):
            handle.status = "cancelled"
        else:
            handle.status = "failed"
            handle.error = {"kind": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if context is not None:
            context.parent_id = previous_parent
        try:
            emit_trace_event(
                "tool_result",
                status=handle.status,
                started_at=started_wall,
                duration_ms=int((time.perf_counter() - started_perf) * 1000),
                stage=stage,
                error=handle.error,
                payload={"name": name, "result": handle.result if handle.result is not None else {}},
                parent_id=call_id,
                content=f"{name} {handle.status}",
            )
        except Exception:
            logger.warning("tool_result trace 事件发送失败（不影响运行）", exc_info=True)
