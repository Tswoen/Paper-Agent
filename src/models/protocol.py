from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


JsonObject = dict[str, Any]


@dataclass(slots=True)
class UIMessage:
    """前端时间线里的单条消息模型。

    中文说明：
    这个结构只保留前端真正会消费的字段，用于承载用户消息、助手消息、
    reasoning 增量、工具事件与文件改动等展示数据，避免让前端直接依赖
    后端原始事件结构。
    """

    id: str
    role: Literal["user", "assistant", "system"]
    content: str = ""
    kind: str = "message"
    is_streaming: bool = False
    reasoning: str = ""
    reasoning_streaming: bool = False
    tool_events: list[JsonObject] = field(default_factory=list)
    file_edits: list[JsonObject] = field(default_factory=list)
    media: list[JsonObject] = field(default_factory=list)
    turn_id: str | None = None
    turn_phase: str | None = None
    turn_seq: int | None = None

    def to_dict(self) -> JsonObject:
        """把消息对象转换成前端可直接消费的普通字典。"""

        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "kind": self.kind,
            "is_streaming": self.is_streaming,
            "reasoning": self.reasoning,
            "reasoning_streaming": self.reasoning_streaming,
            "tool_events": self.tool_events,
            "file_edits": self.file_edits,
            "media": self.media,
            "turn_id": self.turn_id,
            "turn_phase": self.turn_phase,
            "turn_seq": self.turn_seq,
        }
