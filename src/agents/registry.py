from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from .base import AgentSpec, BaseAgent


@dataclass(slots=True)
class RegisteredAgent:
    spec: AgentSpec
    agent_cls: Type[BaseAgent]


@dataclass(slots=True)
class AgentRegistry:
    """管理系统内可用 Agent 类型，编排器只依赖这个注册表创建实例。"""

    _agents: dict[str, RegisteredAgent] = field(default_factory=dict)

    def register(self, agent_cls: Type[BaseAgent]) -> None:
        self._agents[agent_cls.spec.name] = RegisteredAgent(agent_cls.spec, agent_cls)

    def require(self, name: str) -> RegisteredAgent:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise ValueError(f"unknown agent: {name}") from exc

    def specs(self) -> list[AgentSpec]:
        return [item.spec for item in self._agents.values()]
