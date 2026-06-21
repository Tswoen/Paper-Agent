from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


JsonObject = dict[str, Any]
ConnectionState = Literal["idle", "connecting", "open", "reconnecting", "closed", "error"]


@dataclass(slots=True)
class ApiResponse:
    """框架无关的 HTTP 响应对象。"""

    status: int
    body: JsonObject
    headers: dict[str, str] = field(default_factory=lambda: {"content-type": "application/json"})


@dataclass(slots=True)
class BootstrapPayload:
    """前端启动所需的最小网关信息。"""

    expires_in: int
    api_base: str = ""
    runtime_surface: str = "paper_agent_workspace"
    runtime_capabilities: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        """转换为前端可直接消费的 JSON 字典。"""

        return {
            "expires_in": self.expires_in,
            "api_base": self.api_base,
            "runtime_surface": self.runtime_surface,
            "runtime_capabilities": self.runtime_capabilities,
        }


@dataclass(slots=True)
class UIMessage:
    """前端时间线消息模型。"""

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
        """转换为前端渲染层使用的普通 JSON 字典。"""

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
