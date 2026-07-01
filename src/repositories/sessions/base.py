from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.models.sessions import SessionRecord


JsonObject = dict[str, Any]


class SessionRepository(ABC):
    """会话仓储抽象接口。

    中文说明：
    这一层定义“会话数据应该如何被读取与写入”，但不绑定具体实现。
    服务层只依赖这套稳定接口，就可以把 SQLite、内存仓储甚至远端存储
    实现隔离在仓储层下面。
    """

    @abstractmethod
    def create(self, title: str = "New chat", workspace_scope: JsonObject | None = None) -> SessionRecord:
        """创建新的会话记录。"""

    @abstractmethod
    def get(self, key: str) -> SessionRecord:
        """根据会话键获取完整会话记录。"""

    @abstractmethod
    def list(self) -> list[JsonObject]:
        """返回会话摘要列表。"""

    @abstractmethod
    def delete(self, key: str) -> None:
        """删除指定会话。"""

    @abstractmethod
    def append_message(self, key: str, role: str, content: str, **extra: Any) -> JsonObject:
        """向指定会话追加一条消息。"""

    @abstractmethod
    def append_event(
        self,
        key: str,
        event_type: str,
        content: str = "",
        metadata: JsonObject | None = None,
        created_at: str | None = None,
    ) -> JsonObject:
        """向指定会话追加一条结构化事件。"""

    @abstractmethod
    def set_workspace_scope(self, key: str, workspace_scope: JsonObject | None) -> JsonObject:
        """更新会话的工作区范围信息。"""

    @abstractmethod
    def set_run_started_at(self, key: str, started_at: str | None) -> JsonObject:
        """更新当前回合的运行开始时间。"""

    @abstractmethod
    def set_status(self, key: str, status: str) -> JsonObject:
        """更新会话状态。"""
