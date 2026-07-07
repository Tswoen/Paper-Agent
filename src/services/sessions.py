from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from src.models.sessions import utc_now
from src.repositories.sessions.base import SessionRepository
from src.utils import get_logger, logging_context


JsonObject = dict[str, Any]
RuntimeEventEmitter = Callable[[JsonObject], None]
MessageHandler = Callable[[str, str, JsonObject, RuntimeEventEmitter], None]
logger = get_logger(__name__)


class SessionError(Exception):
    """会话业务异常。

    中文说明：
    该异常用于表达服务层里的可预期失败，例如会话不存在、请求参数非法、
    或当前会话仍在运行中不能再次发起新任务。HTTP 层会统一把它转换为
    标准 JSON 错误响应。
    """

    def __init__(self, message: str, status: int = 400):
        """初始化业务异常并记录对应的 HTTP 状态码。"""

        super().__init__(message)
        self.status = status


@dataclass(slots=True)
class AssistantMessageBuffer:
    """聚合一次运行中的助手输出片段。

    中文说明：
    运行期间的正文增量、reasoning 增量和媒体信息会先以事件形式逐条落库，
    最后再汇总为一条 assistant message，便于历史线程直接按消息维度渲染。
    """

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
        """把缓冲区中的助手输出写回会话消息表。"""

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
    """返回前端会话列表载荷。"""

    return {"sessions": repo.list()}


def fetch_thread(repo: SessionRepository, key: str) -> JsonObject:
    """返回指定会话的完整线程快照。"""

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
    """同步提交一条用户消息并完整返回事件结果。

    中文说明：
    该函数保留给旧的 `POST /messages` 接口使用，会在一次 HTTP 请求内同步完成
    用户消息入库、工作流执行、事件落库和线程快照返回。新的 `/runs + SSE`
    也会复用同一套 `message_handler + emit` 协议，因此这里仍然是有效的兼容入口。
    """

    body = body or {}
    content = str(body.get("content") or "")
    if not content.strip() and not body.get("media"):
        raise ValueError("content is required")

    # 中文注释：先确认会话存在，避免将消息误写到不存在的会话里。
    repo.get(session_key)

    turn_id = str(body.get("turn_id") or uuid.uuid4().hex)
    media = list(body.get("media") or [])
    started_at = utc_now()
    events: list[JsonObject] = []
    assistant_buffer = AssistantMessageBuffer()
    resolved_handler = message_handler or _default_message_handler

    def emit(event: JsonObject) -> None:
        """收集同步运行中的事件，并顺手更新助手消息缓冲区。"""

        event_copy = copy.deepcopy(event)
        events.append(event_copy)
        assistant_buffer.apply(event_copy)

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

            resolved_handler(session_key, content, {"turn_id": turn_id, **body}, emit)

            # 中文注释：无论处理器是否主动收尾，都补齐 turn_end，避免前端状态悬空。
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
        # 中文注释：运行结束后必须清理运行状态，否则前端会一直认为会话仍在执行。
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
    events: list[JsonObject],
) -> None:
    """把运行期事件逐条写入事件存储。

    中文说明：
    这里仅负责 event 级别的持久化，不直接拼装 assistant message。
    这样同步接口和异步 SSE 运行都可以复用同样的聚合逻辑。
    """

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
    """默认消息处理器。

    中文说明：
    当真实论文工作流尚未注入时，该占位实现会生成一组最小事件，
    便于本地联调 `/messages` 与 `/runs` 两条调用链。
    """

    turn_id = frame.get("turn_id")
    emit(
        {
            "event": "reasoning_delta",
            "chat_id": chat_id,
            "session_key": chat_id,
            "content": "已收到主题，正在整理一个最小演示结果。",
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
