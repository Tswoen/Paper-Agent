from __future__ import annotations

import uuid
from typing import Any

from src.models.protocol import UIMessage


JsonObject = dict[str, Any]


class ChatStreamAggregator:
    """把后端事件聚合成前端可渲染的消息时间线。

    中文说明：
    后端返回的是事件流，前端真正想渲染的是稳定的消息列表。这个聚合器负责把
    delta、reasoning、tool、file_edit 等碎片事件整合成一条条 UIMessage。
    """

    def __init__(self):
        """初始化一个空的时间线聚合器。"""

        self.messages: list[UIMessage] = []
        self.is_streaming = False
        self.run_started_at: str | None = None
        self.goal_state: JsonObject | None = None
        self.stream_error: JsonObject | None = None
        self._active_assistant_id: str | None = None

    def add_optimistic_user_message(self, content: str, turn_id: str | None = None) -> UIMessage:
        """先把用户消息放入时间线，模拟前端乐观更新。"""

        message = UIMessage(id=uuid.uuid4().hex, role="user", content=content, turn_id=turn_id)
        self.messages.append(message)
        self.is_streaming = True
        return message

    def apply(self, event: JsonObject) -> None:
        """应用单个后端事件，增量更新当前时间线快照。"""

        event_name = event.get("event")
        if event_name == "delta":
            self._append_delta(str(event.get("content") or event.get("delta") or ""), event)
            return
        if event_name == "reasoning_delta":
            self._append_reasoning(str(event.get("content") or event.get("delta") or ""), event)
            return
        if event_name == "reasoning_end":
            assistant = self._active_assistant()
            if assistant:
                assistant.reasoning_streaming = False
            return
        if event_name == "message":
            self._append_message_event(event)
            return
        if event_name == "file_edit":
            self._append_file_edit(event)
            return
        if event_name == "goal_status":
            self.run_started_at = event.get("run_started_at")
            return
        if event_name == "goal_state":
            self.goal_state = dict(event.get("goal_state") or event)
            return
        if event_name == "stream_end":
            assistant = self._active_assistant()
            if assistant:
                assistant.is_streaming = False
            return
        if event_name == "turn_end":
            self.is_streaming = False
            assistant = self._active_assistant()
            if assistant:
                assistant.is_streaming = False
                assistant.reasoning_streaming = False
            self._active_assistant_id = None
            return
        if event_name == "error":
            self.stream_error = dict(event)
            self.is_streaming = False

    def snapshot(self) -> JsonObject:
        """返回当前前端时间线快照。"""

        return {
            "messages": [message.to_dict() for message in self.messages],
            "is_streaming": self.is_streaming,
            "run_started_at": self.run_started_at,
            "goal_state": self.goal_state,
            "stream_error": self.stream_error,
        }

    def _ensure_active_assistant(self, event: JsonObject | None = None) -> UIMessage:
        """确保当前轮次存在一条 assistant 消息用于承接增量内容。"""

        assistant = self._active_assistant()
        if assistant:
            return assistant
        message = UIMessage(
            id=uuid.uuid4().hex,
            role="assistant",
            is_streaming=True,
            turn_id=(event or {}).get("turn_id"),
            turn_phase=(event or {}).get("turn_phase"),
            turn_seq=(event or {}).get("turn_seq"),
        )
        self.messages.append(message)
        self._active_assistant_id = message.id
        self.is_streaming = True
        return message

    def _active_assistant(self) -> UIMessage | None:
        """查找当前正在聚合的 assistant 消息。"""

        if not self._active_assistant_id:
            return None
        return next((message for message in self.messages if message.id == self._active_assistant_id), None)

    def _append_delta(self, text: str, event: JsonObject) -> None:
        """把回答正文增量追加到当前 assistant 消息。"""

        # 中文注释：delta 只追加到当前 assistant 消息，避免每个片段都生成一条新消息。
        assistant = self._ensure_active_assistant(event)
        assistant.content += text
        assistant.is_streaming = True
        self.is_streaming = True

    def _append_reasoning(self, text: str, event: JsonObject) -> None:
        """把 reasoning 增量追加到当前 assistant 消息。"""

        assistant = self._ensure_active_assistant(event)
        assistant.reasoning += text
        assistant.reasoning_streaming = True
        assistant.is_streaming = True
        self.is_streaming = True

    def _append_message_event(self, event: JsonObject) -> None:
        """处理普通 message 事件，并按 role 写入时间线。"""

        kind = str(event.get("kind") or "message")
        if kind in {"tool_hint", "progress", "tool"}:
            assistant = self._ensure_active_assistant(event)
            assistant.tool_events.append(dict(event))
            return
        role = str(event.get("role") or "assistant")
        if role == "assistant":
            assistant = self._ensure_active_assistant(event)
            assistant.content = str(event.get("content") or assistant.content)
            assistant.media.extend(event.get("media") or [])
            return
        self.messages.append(
            UIMessage(
                id=str(event.get("id") or uuid.uuid4().hex),
                role="user" if role == "user" else "system",
                content=str(event.get("content") or ""),
                kind=kind,
                media=list(event.get("media") or []),
                turn_id=event.get("turn_id"),
            )
        )

    def _append_file_edit(self, event: JsonObject) -> None:
        """把文件编辑事件挂到当前 assistant 消息上。"""

        assistant = self._ensure_active_assistant(event)
        assistant.file_edits.append(dict(event))
