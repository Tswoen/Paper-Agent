from __future__ import annotations

import copy
import uuid
from typing import Any, Callable

from src.models.sessions import utc_now
from src.repositories.sessions.base import SessionRepository
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
    这里是会话模块的应用服务入口，负责串联：
    1. 校验输入。
    2. 落库用户消息。
    3. 标记会话进入运行态。
    4. 调用真正的 agent/graph 处理器。
    5. 持久化运行事件并返回线程快照。
    """

    body = body or {}
    content = str(body.get("content") or "")
    if not content.strip() and not body.get("media"):
        raise ValueError("content is required")

    # 中文注释：先确认会话存在，避免把消息写入一个无效的 session。
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

            # 中文注释：先落库用户消息，保证 thread 快照与事件流引用的是同一份输入。
            repo.append_message(session_key, "user", content, media=media, turn_id=turn_id)

            # 中文注释：记录本轮运行开始时间，便于前端判断会话是否仍在执行。
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

            # 中文注释：真正的 agent/graph 执行逻辑仍通过注入完成，方便未来替换实现。
            events.extend(resolved_handler(session_key, content, {"turn_id": turn_id, **body}))

            # 中文注释：兜底补齐 turn_end，避免前端因处理器漏收尾而一直显示“执行中”。
            if not any(event.get("event") == "turn_end" for event in events):
                events.append({"event": "turn_end", "chat_id": session_key, "turn_id": turn_id})

            _persist_runtime_events(repo, session_key, turn_id, events)
            repo.set_status(session_key, "completed")
            logger.info("消息回合处理完成", extra={"event_count": len(events)})
    except Exception as error:
        repo.append_event(
            session_key,
            "error",
            content=str(error),
            metadata={"turn_id": turn_id},
        )
        repo.set_status(session_key, "failed")
        raise
    finally:
        # 中文注释：无论成功还是失败都清理运行状态，避免前端残留“运行中”。
        repo.set_run_started_at(session_key, None)

    return {
        "session_key": session_key,
        "turn_id": turn_id,
        "events": events,
        "thread": repo.get(session_key).thread(),
    }


def _persist_runtime_events(
    repo: SessionRepository,
    session_key: str,
    turn_id: str,
    events: list[JsonObject],
) -> None:
    """把运行时事件写入事件存储，并补齐 assistant 输出消息。"""

    assistant_chunks: list[str] = []
    assistant_reasoning_chunks: list[str] = []
    assistant_media: list[JsonObject] = []

    for event in events:
        event_name = str(event.get("event") or "unknown")
        content = str(event.get("content") or event.get("delta") or "")
        metadata = {item_key: copy.deepcopy(item_value) for item_key, item_value in event.items() if item_key != "content"}
        repo.append_event(session_key, event_name, content=content, metadata=metadata)
        if event_name == "delta":
            assistant_chunks.append(content)
        elif event_name == "reasoning_delta":
            assistant_reasoning_chunks.append(content)
        elif event_name == "message" and str(event.get("role") or "") == "assistant":
            assistant_chunks = [content]
            assistant_media.extend(copy.deepcopy(event.get("media") or []))

    assistant_content = "".join(assistant_chunks).strip()
    assistant_reasoning = "".join(assistant_reasoning_chunks).strip()
    if assistant_content or assistant_reasoning or assistant_media:
        repo.append_message(
            session_key,
            "assistant",
            assistant_content,
            reasoning=assistant_reasoning,
            media=assistant_media,
            turn_id=turn_id,
        )


def _default_message_handler(chat_id: str, content: str, frame: JsonObject) -> list[JsonObject]:
    """默认消息处理器。

    中文说明：
    这是一个演示型占位实现，用来保证 HTTP 提交接口在没有接入真实工作流时，
    仍然能够形成完整事件闭环。后续接入真实 graph/agent 时，只需要通过
    `submit_message` 的 `message_handler` 参数替换即可。
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
