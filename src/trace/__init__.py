"""Trace V2 公共模块：trace 上下文与统一事件信封。

中文注释：本模块是中立层，llm 与 graph 都可以依赖它；它不依赖任何业务模块。
事件经由现有 sync_port 管道落库与广播，天然按 turn 隔离。
"""

from src.trace.context import (
    TraceContext,
    TraceEmitter,
    bind_trace_context,
    current_trace_context,
    emit_trace_event,
)
from src.trace.envelope import build_trace_event


__all__ = [
    "TraceContext",
    "TraceEmitter",
    "bind_trace_context",
    "build_trace_event",
    "current_trace_context",
    "emit_trace_event",
]
