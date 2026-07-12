from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from ..models import PaperDocument, SearchRequest


class PaperSearchConnector(ABC):
    """论文检索 connector 抽象基类。

    每个外部来源都实现同一个 `search` 接口，
    编排层因此可以按统一协议调用，而不依赖具体站点的实现细节。
    """

    source_name: str

    @abstractmethod
    def search(self, request: SearchRequest) -> list[PaperDocument]:
        """执行单源检索并返回统一论文模型列表。"""

        raise NotImplementedError

    async def async_search(self, request: SearchRequest) -> list[PaperDocument]:
        """异步检索入口；未单独实现的来源先用线程包住同步方法兜底。"""

        # 这样新增 async 编排时不会逼所有 connector 一次性重写，后续可逐个替换为真正异步 HTTP。
        return await asyncio.to_thread(self.search, request)
