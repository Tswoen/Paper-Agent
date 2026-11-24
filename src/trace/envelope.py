from __future__ import annotations

from typing import Any
from uuid import uuid4


JsonObject = dict[str, Any]


def build_trace_event(
    event_type: str,
    *,
    status: str,
    started_at: str,
    node_key: str | None = None,
    stage: str | None = None,
    duration_ms: int | None = None,
    tokens: JsonObject | None = None,
    error: JsonObject | None = None,
    payload: JsonObject | None = None,
    parent_id: str | None = None,
    content: str = "",
) -> JsonObject:
    """组装统一的 trace 事件信封。

    中文注释：这里只负责事件自身的固有字段；session_key / run_id / thread_id /
    turn_id / timestamp 由现有事件管道（sync_port → session_runs）统一补齐，
    不在这里重复构造，保证 trace 事件和普通进度事件走同一条落库/广播路径。

    信封字段只增不改：旧事件（runtime_event 等）的结构完全不受影响。
    payload 按全量保真原则记录，不做截断；体积控制由消费端按回合取数解决。
    """

    return {
        # event 是现有管道的事件名路由键；type 是信封规范里的类型字段，两者同值。
        "event": event_type,
        "id": uuid4().hex,
        "parent_id": parent_id,
        "type": event_type,
        "node_key": node_key,
        "stage": stage,
        "status": status,
        "started_at": started_at,
        "duration_ms": duration_ms,
        "tokens": tokens,
        "error": error,
        "payload": payload if payload is not None else {},
        "content": content,
    }
