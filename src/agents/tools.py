from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


JsonObject = dict[str, Any]


class ToolCallable(Protocol):
    def __call__(self, **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Agent 可调用工具的声明，后续可直接映射成 LLM function schema。"""

    name: str
    description: str
    parameters_schema: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class Tool:
    spec: ToolSpec
    handler: ToolCallable

    def call(self, **kwargs: Any) -> Any:
        return self.handler(**kwargs)


class ToolRegistry:
    """按名称管理工具；编排器会为每个 Agent 注入它声明过的工具子集。"""

    def __init__(self, tools: dict[str, Tool] | None = None):
        self._tools = dict(tools or {})

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def require(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ValueError(f"unknown tool: {name}") from exc

    def select(self, names: tuple[str, ...]) -> "ToolRegistry":
        # 只把当前 Agent 声明过的工具传进去，避免工具能力在步骤之间泄漏。
        return ToolRegistry({name: self.require(name) for name in names})

    def as_llm_tools(self) -> list[JsonObject]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.spec.name,
                    "description": tool.spec.description,
                    "parameters": tool.spec.parameters_schema,
                },
            }
            for tool in self._tools.values()
        ]


def not_implemented_tool(name: str, description: str, parameters_schema: JsonObject | None = None) -> Tool:
    """生成占位工具，先把架构接口立住，真实数据库接入后替换 handler 即可。"""

    def _handler(**_: Any) -> Any:
        raise NotImplementedError(f"tool {name} is not implemented")

    return Tool(ToolSpec(name, description, parameters_schema or {}), _handler)
