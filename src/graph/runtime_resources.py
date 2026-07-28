from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx


if TYPE_CHECKING:
    from src.llm import ProviderSnapshot
    from src.repositories.chroma.read_vector_store import EmbeddingConnection


@dataclass(slots=True)
class RuntimeConcurrencyLimits:
    """保存单次 run 里统一使用的并发上限。"""

    # 中文注释：这一轮先把并发上限统一收口到这里，后面阅读节点、下载模块、
    # 模型调用、embedding 调用都从这里拿值，避免每个模块再各自写死一个数字。
    search_source_concurrency: int = 4
    paper_task_concurrency: int = 3
    download_concurrency: int = 3
    read_model_concurrency: int = 2
    embedding_concurrency: int = 1


@dataclass(slots=True)
class WorkflowRuntimeResources:
    """保存单次 run 共享的并发控制对象和公共资源。"""

    limits: RuntimeConcurrencyLimits = field(default_factory=RuntimeConcurrencyLimits)
    # 中文注释：搜索节点查询多个来源时，也统一走这里的信号量控制。
    # 这样后面如果想调搜索并发，只需要改一处，不用再去搜索模块里找写死的数字。
    search_source_semaphore: asyncio.Semaphore = field(init=False)
    # 中文注释：论文级并发这一轮先不真正启用，但先把信号量放到统一资源对象里，
    # 后面做“多篇论文并发处理”时可以直接复用，不需要再重新设计入口。
    paper_task_semaphore: asyncio.Semaphore = field(init=False)
    # 中文注释：下载并发这一轮会真正接入异步下载入口，避免下载模块继续自己偷偷限流。
    download_semaphore: asyncio.Semaphore = field(init=False)
    # 中文注释：阅读模型并发和 embedding 并发这轮先统一归口，暂时不往 provider
    # 内部和索引实现里强行下沉，避免一下改太多模块。
    read_model_semaphore: asyncio.Semaphore = field(init=False)
    embedding_semaphore: asyncio.Semaphore = field(init=False)
    # 中文注释：同一个 run 内共用一个 AsyncClient，后续多个异步下载可以复用连接，
    # 不需要每次下载都重新 new 一个 client。
    http_client: httpx.AsyncClient = field(init=False)
    # 中文注释：旧的同步图节点还没全部改完，短期需要用同一个 Runner
    # 跑异步下载主入口，避免每次下载都新建事件循环导致共享 client 难以复用。
    _sync_runner: asyncio.Runner | None = field(init=False, default=None)
    # 中文注释：embedding 相关对象放在这里缓存，避免阅读节点每处理一篇论文就重复
    # 解析一次模型配置、重复创建 provider 和连接对象。
    embedding_snapshot: ProviderSnapshot | None = None
    embedding_connection: EmbeddingConnection | None = None
    embedding_error: str | None = None

    def __post_init__(self) -> None:
        """根据统一配置生成本次 run 要复用的资源对象。"""

        self.search_source_semaphore = asyncio.Semaphore(max(1, int(self.limits.search_source_concurrency)))
        self.paper_task_semaphore = asyncio.Semaphore(max(1, int(self.limits.paper_task_concurrency)))
        self.download_semaphore = asyncio.Semaphore(max(1, int(self.limits.download_concurrency)))
        self.read_model_semaphore = asyncio.Semaphore(max(1, int(self.limits.read_model_concurrency)))
        self.embedding_semaphore = asyncio.Semaphore(max(1, int(self.limits.embedding_concurrency)))
        self.http_client = httpx.AsyncClient(
            follow_redirects=True,
            max_redirects=5,
        )

    def run_from_sync(self, awaitable: Awaitable[Any]) -> Any:
        """给还没改成 async 的同步节点临时运行异步资源调用。"""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # 中文注释：同步工作线程里没有正在运行的事件循环，可以安全复用同一个 Runner。
            # 下载模块会在这个 Runner 里使用共享 AsyncClient，后面关闭时也走同一个 Runner。
            if self._sync_runner is None:
                self._sync_runner = asyncio.Runner()
            return self._sync_runner.run(awaitable)
        # 中文注释：如果已经在 async 环境里，就不要再套一层同步桥。
        # 这种场景说明调用方可以直接 await 异步主入口。
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise RuntimeError("同步运行时资源桥不能在已有事件循环中调用，请直接 await 对应的异步方法")

    def close_from_sync(self) -> None:
        """在线程里的同步工作流结束时关闭当前 run 的异步资源。"""

        if self._sync_runner is None:
            # 中文注释：如果这次 run 没有通过同步桥使用过共享 AsyncClient，
            # 就临时创建一个 Runner 完成关闭动作，保证同步调用方不用关心事件循环细节。
            with asyncio.Runner() as runner:
                runner.run(self.aclose())
            return
        try:
            self._sync_runner.run(self.aclose())
        finally:
            self._sync_runner.close()
            self._sync_runner = None

    async def aclose(self) -> None:
        """关闭当前 run 里复用的异步客户端。"""

        # 中文注释：AsyncClient 持有网络连接池。run 结束后要主动关闭，避免连接一直挂着。
        await self.http_client.aclose()
