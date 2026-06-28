from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.utils import get_logger


JsonObject = dict[str, Any]
logger = get_logger(__name__)


class SessionError(Exception):
    """会话业务错误，HTTP 网关层会把它转换成统一 error payload。"""

    def __init__(self, message: str, status: int = 400):
        """初始化会话业务异常。"""

        super().__init__(message)
        self.status = status


def utc_now() -> str:
    """返回当前 UTC 时间的 ISO 字符串。"""

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class SessionRecord:
    """单个聊天会话的持久化模型。

    这里故意只保留与前后端交互协议相关的数据，真正 Agent 的运行逻辑仍然通过
    `message_handler` 注入，避免把会话仓库做成杂糅的“万能类”。
    """

    key: str
    title: str = "New chat"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    messages: list[JsonObject] = field(default_factory=list)
    workspace_scope: JsonObject | None = None
    run_started_at: str | None = None

    def summary(self) -> JsonObject:
        """返回适合会话列表展示的摘要信息。"""

        last_message = self.messages[-1]["content"] if self.messages else ""
        return {
            "key": self.key,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "preview": last_message[:120],
            "run_started_at": self.run_started_at,
            "workspace_scope": copy.deepcopy(self.workspace_scope),
        }

    def thread(self) -> JsonObject:
        """返回适合前端线程视图使用的完整会话数据。"""

        return {
            "key": self.key,
            "messages": copy.deepcopy(self.messages),
            "workspace_scope": copy.deepcopy(self.workspace_scope),
            "has_pending_tool_calls": bool(self.run_started_at),
            "page": {"cursor": None, "has_more": False},
        }


class SessionRepository:
    """内存会话仓库。

    当前实现服务于前后端交互模块的核心协议；若未来需要落盘，只需要替换这一个仓库，
    Gateway/Realtime 层不需要跟着改动。
    """

    def __init__(self, initial: list[JsonObject] | None = None):
        """初始化内存会话仓库。"""

        self._sessions: dict[str, SessionRecord] = {}
        for item in initial or []:
            record = SessionRecord(
                key=str(item.get("key") or uuid.uuid4().hex),
                title=str(item.get("title") or "New chat"),
                created_at=str(item.get("created_at") or utc_now()),
                updated_at=str(item.get("updated_at") or utc_now()),
                messages=copy.deepcopy(item.get("messages") or []),
                workspace_scope=copy.deepcopy(item.get("workspace_scope")),
                run_started_at=item.get("run_started_at"),
            )
            self._sessions[record.key] = record
        logger.info("会话仓库初始化完成", extra={"session_count": len(self._sessions)})

    def create(self, title: str = "New chat", workspace_scope: JsonObject | None = None) -> SessionRecord:
        """创建新的会话记录。"""

        key = uuid.uuid4().hex
        record = SessionRecord(key=key, title=title, workspace_scope=copy.deepcopy(workspace_scope))
        self._sessions[key] = record
        logger.info("创建新会话", extra={"session_key": key, "title": title})
        return record

    def get(self, key: str) -> SessionRecord:
        """根据会话键获取会话记录。"""

        try:
            return self._sessions[key]
        except KeyError as exc:
            logger.warning("会话不存在", extra={"session_key": key})
            raise SessionError(f"session not found: {key}", 404) from exc

    def list(self) -> list[JsonObject]:
        """返回按更新时间倒序排列的会话摘要列表。"""

        # 中文注释：会话列表按最近更新时间倒序返回，符合侧边栏“最近会话优先”的使用习惯。
        records = sorted(self._sessions.values(), key=lambda item: item.updated_at, reverse=True)
        return [record.summary() for record in records]

    def delete(self, key: str) -> None:
        """删除指定会话。"""

        if key not in self._sessions:
            logger.warning("尝试删除不存在的会话", extra={"session_key": key})
            raise SessionError(f"session not found: {key}", 404)
        del self._sessions[key]
        logger.info("删除会话", extra={"session_key": key})

    def append_message(self, key: str, role: str, content: str, **extra: Any) -> JsonObject:
        """向指定会话追加一条消息。"""

        record = self.get(key)
        message = {
            "id": str(extra.pop("id", uuid.uuid4().hex)),
            "role": role,
            "content": content,
            "created_at": utc_now(),
            **extra,
        }
        record.messages.append(message)
        record.updated_at = message["created_at"]
        if role == "user" and (record.title == "New chat" or not record.title.strip()):
            record.title = content[:40] or "New chat"
        logger.debug(
            "追加会话消息",
            extra={"session_key": key, "role": role, "content_length": len(content), "message_id": message["id"]},
        )
        return copy.deepcopy(message)

    def set_workspace_scope(self, key: str, workspace_scope: JsonObject | None) -> JsonObject:
        """更新会话的工作区范围信息。"""

        record = self.get(key)
        record.workspace_scope = copy.deepcopy(workspace_scope)
        record.updated_at = utc_now()
        logger.info("更新会话工作区范围", extra={"session_key": key, "has_workspace_scope": workspace_scope is not None})
        return record.summary()

    def set_run_started_at(self, key: str, started_at: str | None) -> JsonObject:
        """更新会话当前回合的运行状态时间戳。"""

        record = self.get(key)
        record.run_started_at = started_at
        record.updated_at = utc_now()
        logger.debug(
            "更新会话运行状态",
            extra={"session_key": key, "run_started_at": started_at, "is_running": started_at is not None},
        )
        return record.summary()


def list_sessions(repo: SessionRepository) -> JsonObject:
    """返回前端会话列表载荷。"""

    return {"sessions": repo.list()}


def fetch_thread(repo: SessionRepository, key: str) -> JsonObject:
    """返回指定会话的完整线程数据。"""

    return repo.get(key).thread()


def create_session(repo: SessionRepository, body: JsonObject | None = None) -> JsonObject:
    """创建会话并返回标准响应载荷。"""

    body = body or {}
    record = repo.create(title=str(body.get("title") or "New chat"), workspace_scope=body.get("workspace_scope"))
    return {"session": record.summary()}


def delete_session(repo: SessionRepository, key: str) -> JsonObject:
    """删除会话并返回标准响应载荷。"""

    repo.delete(key)
    return {"deleted": True, "key": key}
