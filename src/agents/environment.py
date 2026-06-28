from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


JsonObject = dict[str, Any]


@dataclass(slots=True)
class AgentEnvironment:
    """Agent 的运行边界。

    这里先只记录 workspace/cache/tmp 等路径，后续可以扩展成沙箱、网络策略、
    文件白名单、数据库连接池等更严格的执行环境。
    """

    workspace: Path = field(default_factory=lambda: Path.cwd())
    cache_dir: Path = field(default_factory=lambda: Path("D:/tmp/papers-agents-cache"))
    tmp_dir: Path = field(default_factory=lambda: Path("D:/tmp/papers-agents-runs"))
    variables: JsonObject = field(default_factory=dict)

    def child(self, name: str) -> "AgentEnvironment":
        # 每个 Agent 使用独立 tmp 子目录，避免中间文件互相污染。
        return AgentEnvironment(
            workspace=self.workspace,
            cache_dir=self.cache_dir,
            tmp_dir=self.tmp_dir / name,
            variables=dict(self.variables),
        )
