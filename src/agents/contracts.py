from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.paper_retrieval.models import PaperDocument


JsonObject = dict[str, Any]
AgentRole = Literal[
    "search",
    "screen",
    "read",
    "analyze",
    "plan",
    "write",
    "critique",
]


@dataclass(slots=True)
class EvidenceItem:
    """表示从论文中抽取出的证据片段。

    当前搜索阶段暂时不会生产该对象，但后续阅读节点、分析节点和写作节点
    都可能依赖这种结构化证据，因此这里保留为共享领域模型。
    """

    paper_id: str
    claim: str
    quote: str | None = None
    section: str | None = None
    page: int | None = None
    confidence: float | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class ReviewRequest:
    """描述用户发起的一次综述任务请求。

    按当前场景，用户只需要给出主题与少量约束条件，例如时间范围、
    论文数量和数据源偏好。更细的研究问题可由后续规划 Agent 再生成，
    不再要求用户在入口阶段手工填写。
    """

    topic: str
    constraints: JsonObject = field(default_factory=dict)
    language: str = "zh"


# 中文注释：保留旧名称作为兼容别名，避免现有导入点在重构过程中立即失效。
ReviewTask = ReviewRequest


@dataclass(slots=True)
class ReviewArtifact:
    """兼容旧流水线的阶段产物对象。

    当前新的图执行主流程已经转为直接读写共享 `State`，
    不再把它作为节点之间的首选交换结构。这里保留它仅用于兼容旧接口
    或平滑迁移期间的外围调用，不建议新代码继续依赖。
    """

    step: str
    role: AgentRole
    summary: str
    papers: list[PaperDocument] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    content: str | None = None
    data: JsonObject = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentRunInput:
    """兼容旧 Agent 接口的输入包装。

    新的实现建议 Agent 直接从共享 `State` 读取所需字段。
    这里保留该对象是为了兼容可能尚未迁移的外部调用方。
    """

    state: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class AgentRunOutput:
    """兼容旧 Agent 接口的输出包装。

    新的实现更推荐 Agent 返回一段待合并到 `State` 的局部更新字典，
    因此这里的 `state_update` 也采用同样的表达方式。
    """

    state_update: JsonObject = field(default_factory=dict)
