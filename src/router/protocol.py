from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


JsonObject = dict[str, Any]


@dataclass(slots=True)
class UIMessage:
    """前端时间线里的单条消息模型。

    这个结构只保留前端渲染真正会用到的字段，负责承载用户消息、
    assistant 消息以及流式过程中附带的 reasoning、tool、media 等内容。
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
        """把消息对象转换成前端可直接消费的普通字典。

        这里保留显式转换方法，是为了让 `stream_aggregator.py` 在生成快照时
        不需要关心 dataclass 细节。
        """

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
