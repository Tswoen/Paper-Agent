from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.llm import ProviderSnapshot

from .contracts import AgentRole, JsonObject
from src.llm.base import normalize_token_usage
from .environment import AgentEnvironment
from .skills import SkillRegistry
from .tools import ToolRegistry


# 后端节点只允许从前端配置的三个固定模型档位中选择模型。
SUPPORTED_LLM_PROFILES = frozenset({"default_agent", "luna_agent", "solar_agent"})


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """声明 Agent 的静态能力与职责边界。

    当前保留 `input_keys` 只是为了表达这个 Agent 通常依赖哪些状态字段，
    它不再驱动一个独立的输入包装对象，而是用于阅读、调试和后续校验扩展。
    """

    name: str
    role: AgentRole
    description: str
    llm_profile: str = "default_agent"
    tools: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    input_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # 提前拦截拼写错误的 profile，避免运行到节点时才发现模型配置无法加载。
        if self.llm_profile not in SUPPORTED_LLM_PROFILES:
            allowed = ", ".join(sorted(SUPPORTED_LLM_PROFILES))
            raise ValueError(f"llm_profile must be one of: {allowed}")


@dataclass(slots=True)
class AgentContext:
    """Agent 运行时上下文。

    这里仅暴露当前 Agent 允许访问的能力，例如 LLM、工具、技能与环境信息，
    而具体业务输入统一从图状态 `State` 中读取。
    """

    spec: AgentSpec = None
    llm: ProviderSnapshot | None = None
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    skills: SkillRegistry = field(default_factory=SkillRegistry)
    environment: AgentEnvironment = field(default_factory=AgentEnvironment)
    trace: JsonObject = field(default_factory=dict)
    # 模型调用完成后由节点提供回调，把真实用量写入对应的运行卡片。
    usage_callback: Any | None = None


class BaseAgent(ABC):
    """所有论文流程 Agent 的统一基类。

    新设计下，Agent 不再依赖独立的 `AgentRunInput/Output` 契约，
    而是直接接收共享状态，并返回一个待合并的局部状态更新字典。
    """

    spec: AgentSpec

    def __init__(self, context: AgentContext):
        """初始化 Agent，并保存编排层注入的能力上下文。"""

        self.context = context
        self.spec = context.spec

    def run(self, state: JsonObject) -> JsonObject:
        """执行单步任务并返回局部状态更新。

        子类只读取自己声明所需的状态字段，不直接操作整个流程图的控制逻辑。
        真正的状态合并、异常处理和后续步骤推进由 graph node 负责。
        """

        self._validate_state(state)
        return self._run(state)

    def _validate_state(self, state: JsonObject) -> None:
        """校验当前状态是否包含 Agent 声明依赖的关键字段。"""

        missing = [key for key in self.spec.input_keys if key not in state]
        if missing:
            raise ValueError(f"{self.spec.name} missing required state keys: {missing}")

    def report_usage(self, response: object, callback: Any | None = None) -> None:
        """把模型返回的 usage 交给当前阶段，绝不根据文本长度估算 token。"""

        usage = normalize_token_usage(getattr(response, "usage", None))
        target = callback or self.context.usage_callback
        if callable(target):
            target(usage)

    @abstractmethod
    def _run(self, state: JsonObject) -> JsonObject:
        """由具体 Agent 实现自身领域逻辑，并返回局部状态更新。"""

        raise NotImplementedError
