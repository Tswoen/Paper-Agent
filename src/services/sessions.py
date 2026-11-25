from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from src.repositories.sessions.base import SessionRepository


JsonObject = dict[str, Any]
RuntimeEventEmitter = Callable[[JsonObject], JsonObject]
MessageHandler = Callable[..., Any]


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

