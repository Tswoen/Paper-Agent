from __future__ import annotations

import uuid
from typing import Any, Callable

from .sessions_api import SessionRepository, utc_now


JsonObject = dict[str, Any]
MessageHandler = Callable[[str, str, JsonObject], list[JsonObject]]


class HttpMessageGateway:
    """基于 FastAPI HTTP 请求的消息提交网关。

    它保留“用户消息 -> 后端事件列表 -> 前端聚合时间线”的协议形状，但传输层
    已经改为 POST /api/sessions/{key}/messages，不再需要 WebSocket 连接。
    """

    def __init__(
        self,
        sessions: SessionRepository,
        message_handler: MessageHandler | None = None,
    ):
        self.sessions = sessions
        self.message_handler = message_handler or self._default_message_handler

    def submit_message(self, session_key: str, body: JsonObject | None = None) -> JsonObject:
        """提交用户消息，返回本轮事件列表和最新线程快照。"""

        body = body or {}
        content = str(body.get("content") or "")
        if not content.strip() and not body.get("media"):
            raise ValueError("content is required")

        self.sessions.get(session_key)
        turn_id = str(body.get("turn_id") or uuid.uuid4().hex)
        media = list(body.get("media") or [])
        self.sessions.append_message(session_key, "user", content, media=media, turn_id=turn_id)

        started_at = utc_now()
        self.sessions.set_run_started_at(session_key, started_at)

        # 事件列表是 HTTP 响应体的一部分，前端无需维持长连接即可完成同样的 UI 聚合。
        events: list[JsonObject] = [
            {"event": "message", "chat_id": session_key, "role": "user", "content": content, "media": media, "turn_id": turn_id},
            {"event": "goal_status", "chat_id": session_key, "run_started_at": started_at, "turn_id": turn_id},
        ]
        events.extend(self.message_handler(session_key, content, {"turn_id": turn_id, **body}))
        if not any(event.get("event") == "turn_end" for event in events):
            events.append({"event": "turn_end", "chat_id": session_key, "turn_id": turn_id})

        self.sessions.set_run_started_at(session_key, None)
        return {
            "session_key": session_key,
            "turn_id": turn_id,
            "events": events,
            "thread": self.sessions.get(session_key).thread(),
        }

    @staticmethod
    def _default_message_handler(chat_id: str, content: str, frame: JsonObject) -> list[JsonObject]:
        """默认处理器用于演示协议闭环，真实 Agent 可注入自己的执行器。"""

        turn_id = frame.get("turn_id")
        return [
            {"event": "reasoning_delta", "chat_id": chat_id, "content": "收到问题，开始整理回答。", "turn_id": turn_id},
            {"event": "reasoning_end", "chat_id": chat_id, "turn_id": turn_id},
            {"event": "delta", "chat_id": chat_id, "content": f"已收到：{content}", "turn_id": turn_id},
            {"event": "stream_end", "chat_id": chat_id, "turn_id": turn_id},
        ]
