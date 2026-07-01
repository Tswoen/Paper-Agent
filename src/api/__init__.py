"""API 入口包。

中文说明：
这个包只暴露应用装配入口，避免上层代码继续直接依赖旧的 `src.router`
实现细节。后续如果还要扩展更多 HTTP 入口，也统一从这里向外导出。
"""

from .app import GatewayConfig, create_app

__all__ = ["GatewayConfig", "create_app"]
