from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


JsonObject = dict[str, Any]


class SessionError(Exception):
    """会话业务错误，HTTP 网关层会把它转换成统一 error payload。"""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class SessionRecord:
    """单个聊天会话的持久化模型。

    这里故意只保存与前后端交互协议相关的数据，真正论文 Agent 的运行逻辑可以后续
    通过 message_handler 注入，而不污染会话仓库的职责边界。
    """

    key: str
    title: str = "New chat"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    messages: list[JsonObject] = field(default_factory=list)
    workspace_scope: JsonObject | None = None
    run_started_at: str | None = None

    def summary(self) -> JsonObject:
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
        return {
            "key": self.key,
            "messages": copy.deepcopy(self.messages),
            "workspace_scope": copy.deepcopy(self.workspace_scope),
            "has_pending_tool_calls": bool(self.run_started_at),
            "page": {"cursor": None, "has_more": False},
        }


class SessionRepository:
    """内存会话仓库。

    当前实现服务于前后端交互模块的核心协议；若未来需要落盘，只需要替换这个仓库，
    Gateway/Realtime 层不需要改动。
    """

    def __init__(self, initial: list[JsonObject] | None = None):
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

    def create(self, title: str = "New chat", workspace_scope: JsonObject | None = None) -> SessionRecord:
        key = uuid.uuid4().hex
        record = SessionRecord(key=key, title=title, workspace_scope=copy.deepcopy(workspace_scope))
        self._sessions[key] = record
        return record

    def get(self, key: str) -> SessionRecord:
        try:
            return self._sessions[key]
        except KeyError as exc:
            raise SessionError(f"session not found: {key}", 404) from exc

    def list(self) -> list[JsonObject]:
        # 会话列表按更新时间倒序返回，符合 sidebar 最近会话优先的使用习惯。
        records = sorted(self._sessions.values(), key=lambda item: item.updated_at, reverse=True)
        return [record.summary() for record in records]

    def delete(self, key: str) -> None:
        if key not in self._sessions:
            raise SessionError(f"session not found: {key}", 404)
        del self._sessions[key]

    def append_message(self, key: str, role: str, content: str, **extra: Any) -> JsonObject:
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
        return copy.deepcopy(message)

    def set_workspace_scope(self, key: str, workspace_scope: JsonObject | None) -> JsonObject:
        record = self.get(key)
        record.workspace_scope = copy.deepcopy(workspace_scope)
        record.updated_at = utc_now()
        return record.summary()

    def set_run_started_at(self, key: str, started_at: str | None) -> JsonObject:
        record = self.get(key)
        record.run_started_at = started_at
        record.updated_at = utc_now()
        return record.summary()


def list_sessions(repo: SessionRepository) -> JsonObject:
    return {"sessions": repo.list()}


def fetch_thread(repo: SessionRepository, key: str) -> JsonObject:
    return repo.get(key).thread()


def create_session(repo: SessionRepository, body: JsonObject | None = None) -> JsonObject:
    body = body or {}
    record = repo.create(title=str(body.get("title") or "New chat"), workspace_scope=body.get("workspace_scope"))
    return {"session": record.summary()}


def delete_session(repo: SessionRepository, key: str) -> JsonObject:
    repo.delete(key)
    return {"deleted": True, "key": key}

