from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


JsonObject = dict[str, Any]


def utc_now() -> str:
    """返回当前 UTC 时间的 ISO 字符串。"""

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class SessionRecord:
    """单个会话的聚合视图模型。

    中文说明：
    这个对象用于承载会话详情的统一内存表示，让服务层不需要知道底层
    是 SQLite、文件系统还是其他存储实现。只要仓储层最终返回这个模型，
    上层逻辑就能稳定消费。
    """

    key: str
    title: str = "New chat"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    status: str = "created"
    summary_text: str = ""
    messages: list[JsonObject] = field(default_factory=list)
    events: list[JsonObject] = field(default_factory=list)
    artifacts: list[JsonObject] = field(default_factory=list)
    workspace_scope: JsonObject | None = None
    run_started_at: str | None = None
    user_id: str = "local-user"
    last_message_at: str | None = None
    metadata: JsonObject = field(default_factory=dict)

    def summary(self) -> JsonObject:
        """返回适合会话列表展示的摘要信息。"""

        last_message = self.messages[-1]["content"] if self.messages else self.summary_text
        return {
            "key": self.key,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "preview": (last_message or "")[:120],
            "run_started_at": self.run_started_at,
            "workspace_scope": copy.deepcopy(self.workspace_scope),
            "status": self.status,
            "user_id": self.user_id,
            "last_message_at": self.last_message_at,
        }

    def thread(self) -> JsonObject:
        """返回适合前端线程视图使用的完整会话数据。"""

        return {
            "key": self.key,
            "title": self.title,
            "status": self.status,
            "messages": copy.deepcopy(self.messages),
            "events": copy.deepcopy(self.events),
            "artifacts": copy.deepcopy(self.artifacts),
            "workspace_scope": copy.deepcopy(self.workspace_scope),
            "has_pending_tool_calls": bool(self.run_started_at),
            "run_started_at": self.run_started_at,
            "page": {"cursor": None, "has_more": False},
        }
