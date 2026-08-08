from __future__ import annotations

import asyncio
import copy
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from src.models.sessions import utc_now
from src.repositories.sessions.base import SessionRepository
from src.utils import get_logger, logging_context
from src.utils.readable_id import create_readable_id


JsonObject = dict[str, Any]
RuntimeEventEmitter = Callable[[JsonObject], JsonObject]
MessageHandler = Callable[..., Any]
logger = get_logger(__name__)


class SessionError(Exception):
    """表示会话服务层里可预期的业务异常。"""

    def __init__(self, message: str, status: int = 400):
        """保存错误信息和对应的 HTTP 状态码。"""

        super().__init__(message)
        self.status = status


@dataclass(slots=True)
class AssistantMessageBuffer:
    """把一次运行里的助手输出片段聚合成最终消息。"""

    content_chunks: list[str] = field(default_factory=list)
    reasoning_chunks: list[str] = field(default_factory=list)
    media: list[JsonObject] = field(default_factory=list)

    def apply(self, event: JsonObject) -> None:
        """根据单条事件更新当前助手消息缓冲区。"""

        event_name = str(event.get("event") or "")
        content = str(event.get("content") or event.get("delta") or "")
        if event_name == "delta":
            self.content_chunks.append(content)
            return
        if event_name == "reasoning_delta":
            self.reasoning_chunks.append(content)
            return
        if event_name == "message" and str(event.get("role") or "") == "assistant":
            self.content_chunks = [content]
            self.media = list(copy.deepcopy(event.get("media") or []))

    def persist(self, repo: SessionRepository, session_key: str, turn_id: str) -> None:
        """把缓冲区里的内容回写成一条正式的 assistant 消息。"""

        assistant_content = "".join(self.content_chunks).strip()
        assistant_reasoning = "".join(self.reasoning_chunks).strip()
        if not assistant_content and not assistant_reasoning and not self.media:
            return
        repo.append_message(
            session_key,
            "assistant",
            assistant_content,
            reasoning=assistant_reasoning,
            media=copy.deepcopy(self.media),
            turn_id=turn_id,
        )


def list_sessions(repo: SessionRepository) -> JsonObject:
    """返回会话列表给前端。"""

    return {"sessions": repo.list()}


def fetch_thread(repo: SessionRepository, key: str) -> JsonObject:
    """返回指定会话的完整线程快照。"""

    return repo.get(key).thread()


def create_session(repo: SessionRepository, body: JsonObject | None = None) -> JsonObject:
    """创建一个新会话。"""

    body = body or {}
    record = repo.create(title=str(body.get("title") or "New chat"), workspace_scope=body.get("workspace_scope"))
    return {"session": record.summary()}


def delete_session(repo: SessionRepository, key: str) -> JsonObject:
    """删除一个会话，并返回统一响应。"""

    repo.delete(key)
    return {"deleted": True, "key": key}


def submit_message(
    repo: SessionRepository,
    session_key: str,
    body: JsonObject | None = None,
    message_handler: MessageHandler | None = None,
) -> JsonObject:
    """同步提交一条消息，并完整返回这次运行产生的事件。"""

    body = body or {}
    content = str(body.get("content") or "")
    if not content.strip() and not body.get("media"):
        raise ValueError("content is required")

    # 中文注释：先确认会话存在，避免把消息误写到不存在的会话里。
    repo.get(session_key)

    # 调用方没有提供回合编号时，在这里补上带创建时间的编号，方便从日志定位本次提交。
    turn_id = str(body.get("turn_id") or create_readable_id())
    media = list(body.get("media") or [])
    started_at = utc_now()
    events: list[JsonObject] = []
    assistant_buffer = AssistantMessageBuffer()
    resolved_handler = message_handler or _default_message_handler

    def emit(event: JsonObject) -> JsonObject:
        """收集同步运行中的事件，并顺手维护助手消息缓冲区。"""

        event_copy = copy.deepcopy(event)
        events.append(event_copy)
        assistant_buffer.apply(event_copy)
        return event_copy

    try:
        with logging_context(session_key=session_key, turn_id=turn_id):
            logger.info(
                "收到同步消息提交",
                extra={"content_length": len(content), "media_count": len(media)},
            )

            repo.append_message(session_key, "user", content, media=media, turn_id=turn_id)
            repo.set_status(session_key, "running")
            repo.set_run_started_at(session_key, started_at)

            emit(
                {
                    "event": "message",
                    "chat_id": session_key,
                    "session_key": session_key,
                    "role": "user",
                    "content": content,
                    "media": media,
                    "turn_id": turn_id,
                    "timestamp": started_at,
                }
            )
            emit(
                {
                    "event": "status",
                    "chat_id": session_key,
                    "session_key": session_key,
                    "status": "running",
                    "run_started_at": started_at,
                    "turn_id": turn_id,
                    "timestamp": started_at,
                }
            )

            invoke_message_handler(resolved_handler, session_key, content, {"turn_id": turn_id, **body}, emit)

            # 中文注释：无论处理器有没有主动收尾，这里都兜底补一个 turn_end，防止前端一直停在运行中。
            if not any(str(event.get("event") or "") == "turn_end" for event in events):
                emit(
                    {
                        "event": "turn_end",
                        "chat_id": session_key,
                        "session_key": session_key,
                        "turn_id": turn_id,
                        "status": "completed",
                        "timestamp": utc_now(),
                    }
                )

            _persist_runtime_events(repo, session_key, events)
            assistant_buffer.persist(repo, session_key, turn_id)
            repo.set_status(session_key, "completed")
            logger.info("同步消息处理完成", extra={"event_count": len(events)})
    except Exception as error:
        repo.append_event(
            session_key,
            "error",
            content=str(error),
            metadata={"turn_id": turn_id, "message": str(error), "status": "failed"},
        )
        repo.set_status(session_key, "failed")
        raise
    finally:
        # 中文注释：运行结束后必须清掉 run_started_at，否则前端会一直以为当前会话还在执行。
        repo.set_run_started_at(session_key, None)

    return {
        "session_key": session_key,
        "turn_id": turn_id,
        "events": events,
        "thread": repo.get(session_key).thread(),
    }


def invoke_message_handler(
    handler: MessageHandler,
    chat_id: str,
    content: str,
    frame: JsonObject,
    emit: RuntimeEventEmitter,
) -> None:
    """兼容新旧两种消息处理器，统一触发运行逻辑。"""

    result = _call_message_handler(handler, chat_id, content, frame, emit)
    if inspect.isawaitable(result):
        # 中文注释：旧同步入口还在用时，如果消息处理器已经改成 async，
        # 这里临时开一个事件循环把它跑完，避免同步接口直接失效。
        with asyncio.Runner() as runner:
            result = runner.run(result)
    _emit_legacy_result(result, emit)


async def invoke_message_handler_async(
    handler: MessageHandler,
    chat_id: str,
    content: str,
    frame: JsonObject,
    emit: RuntimeEventEmitter,
) -> None:
    """异步触发消息处理器，让后台 run 能直接 await 整条工作流。"""

    result = _call_message_handler(handler, chat_id, content, frame, emit)
    if inspect.isawaitable(result):
        result = await result
    _emit_legacy_result(result, emit)


def _call_message_handler(
    handler: MessageHandler,
    chat_id: str,
    content: str,
    frame: JsonObject,
    emit: RuntimeEventEmitter,
) -> Any:
    """兼容新旧协议的处理器签名，只负责真正调用，不负责等待返回值。"""

    parameter_count = _parameter_count(handler)
    if parameter_count >= 4 or parameter_count < 0:
        return handler(chat_id, content, frame, emit)
    return handler(chat_id, content, frame)


def _emit_legacy_result(result: Any, emit: RuntimeEventEmitter) -> None:
    """兼容旧处理器返回的事件列表，把它逐条重新走一次统一通道。"""

    if result is None:
        return
    if not isinstance(result, list):
        raise TypeError("legacy message handler must return a list of events")
    for item in result:
        if isinstance(item, dict):
            emit(item)


def _parameter_count(handler: MessageHandler) -> int:
    """读取处理器参数个数，拿不到时返回 -1，表示按新协议处理。"""

    try:
        return len(inspect.signature(handler).parameters)
    except (TypeError, ValueError):
        return -1


def _persist_runtime_events(
    repo: SessionRepository,
    session_key: str,
    events: list[JsonObject],
) -> None:
    """把运行期事件逐条写入持久化存储。"""

    for event in events:
        event_name = str(event.get("event") or "unknown")
        content = str(event.get("content") or event.get("delta") or event.get("message") or "")
        metadata = {
            item_key: copy.deepcopy(item_value)
            for item_key, item_value in event.items()
            if item_key != "content"
        }
        repo.append_event(session_key, event_name, content=content, metadata=metadata)


def _default_message_handler(chat_id: str, content: str, frame: JsonObject, emit: RuntimeEventEmitter) -> None:
    """提供一个最小可运行的占位消息处理器。"""

    turn_id = frame.get("turn_id")
    emit(
        {
            "event": "reasoning_delta",
            "chat_id": chat_id,
            "session_key": chat_id,
            "content": "已收到主题，正在整理一份最小演示结果。",
            "turn_id": turn_id,
            "timestamp": utc_now(),
        }
    )
    emit(
        {
            "event": "reasoning_end",
            "chat_id": chat_id,
            "session_key": chat_id,
            "turn_id": turn_id,
            "timestamp": utc_now(),
        }
    )
    emit(
        {
            "event": "delta",
            "chat_id": chat_id,
            "session_key": chat_id,
            "content": f"已收到：{content}",
            "turn_id": turn_id,
            "timestamp": utc_now(),
        }
    )
    emit(
        {
            "event": "turn_end",
            "chat_id": chat_id,
            "session_key": chat_id,
            "turn_id": turn_id,
            "status": "completed",
            "timestamp": utc_now(),
        }
    )

