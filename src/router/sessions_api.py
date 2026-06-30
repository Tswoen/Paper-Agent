from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from src.utils import get_logger, logging_context


JsonObject = dict[str, Any]
MessageHandler = Callable[[str, str, JsonObject], list[JsonObject]]
logger = get_logger(__name__)


class SessionError(Exception):
    """会话业务错误，HTTP 层会把它转换成统一错误响应。"""

    def __init__(self, message: str, status: int = 400):
        """初始化会话业务异常。"""

        super().__init__(message)
        self.status = status


def utc_now() -> str:
    """返回当前 UTC 时间的 ISO 字符串。"""

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class SessionRecord:
    """单个会话的持久化模型。

    中文说明：
    这里故意只保留会话本身的数据字段，不把“提交消息并触发工作流”的应用逻辑塞进数据对象，
    这样仓库层仍然只负责存取，会更容易在未来替换成数据库实现。
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

    中文说明：
    当前实现主要服务于前后端联调与单机运行场景；如果以后要切到数据库，只需要替换这一层，
    上层的路由与会话应用服务不必跟着改接口。
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


def submit_message(
    repo: SessionRepository,
    session_key: str,
    body: JsonObject | None = None,
    message_handler: MessageHandler | None = None,
) -> JsonObject:
    """提交一条用户消息并触发一次后端处理流程。

    中文说明：
    这里是会话模块的应用服务入口，而不是仓库方法。
    它负责把“校验输入、写入用户消息、标记运行状态、调用工作流处理器、整理事件返回”串起来，
    但不会把 graph/agent 的真实执行细节耦合进 SessionRepository。
    """

    body = body or {}
    content = str(body.get("content") or "")
    if not content.strip() and not body.get("media"):
        raise ValueError("content is required")

    # 中文注释：先确认会话存在，避免把消息写进一个无效的 session。
    repo.get(session_key)

    turn_id = str(body.get("turn_id") or uuid.uuid4().hex)
    media = list(body.get("media") or [])
    started_at = utc_now()
    events: list[JsonObject] = []
    resolved_handler = message_handler or _default_message_handler

    try:
        with logging_context(session_key=session_key, turn_id=turn_id):
            logger.info(
                "收到用户消息提交",
                extra={"content_length": len(content), "media_count": len(media)},
            )

            # 中文注释：先落库用户消息，保证 thread 快照和事件流引用的是同一份输入。
            repo.append_message(session_key, "user", content, media=media, turn_id=turn_id)

            # 中文注释：登记本轮运行开始时间，便于前端判断当前会话是否仍在执行。
            repo.set_run_started_at(session_key, started_at)

            events = [
                {
                    "event": "message",
                    "chat_id": session_key,
                    "role": "user",
                    "content": content,
                    "media": media,
                    "turn_id": turn_id,
                },
                {
                    "event": "goal_status",
                    "chat_id": session_key,
                    "run_started_at": started_at,
                    "turn_id": turn_id,
                },
            ]

            # 中文注释：真正的 agent/graph 执行逻辑依然通过注入完成，方便未来替换实现。
            events.extend(resolved_handler(session_key, content, {"turn_id": turn_id, **body}))

            # 中文注释：保底补齐 turn_end，避免前端因为外部处理器漏收尾而一直显示“执行中”。
            if not any(event.get("event") == "turn_end" for event in events):
                events.append({"event": "turn_end", "chat_id": session_key, "turn_id": turn_id})

            logger.info("消息回合处理完成", extra={"event_count": len(events)})
    finally:
        # 中文注释：无论成功还是失败都清理运行状态，避免前端残留运行中状态。
        repo.set_run_started_at(session_key, None)

    return {
        "session_key": session_key,
        "turn_id": turn_id,
        "events": events,
        "thread": repo.get(session_key).thread(),
    }


def _default_message_handler(chat_id: str, content: str, frame: JsonObject) -> list[JsonObject]:
    """默认消息处理器。

    中文说明：
    这是一个演示型占位实现，用来保证 HTTP 提交接口在没有接入真实工作流时也能形成完整事件闭环。
    当后续接入真正的 graph/agent 时，可以直接通过 submit_message 的 message_handler 参数替换。
    """

    turn_id = frame.get("turn_id")
    return [
        {
            "event": "reasoning_delta",
            "chat_id": chat_id,
            "content": "收到问题，开始整理回答。",
            "turn_id": turn_id,
        },
        {"event": "reasoning_end", "chat_id": chat_id, "turn_id": turn_id},
        {
            "event": "delta",
            "chat_id": chat_id,
            "content": f"已收到：{content}",
            "turn_id": turn_id,
        },
        {"event": "stream_end", "chat_id": chat_id, "turn_id": turn_id},
    ]
