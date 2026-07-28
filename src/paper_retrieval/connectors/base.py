from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import httpx

from ..models import PaperDocument, SearchRequest


class PaperMetadataNormalizer(ABC):
    """把某个论文来源的原始记录转换成统一论文对象的接口。

    中文注释：每个网站返回的字段名都不一样，比如有的叫 DOI，有的叫 id，有的把期刊信息放在嵌套字段里。
    这个接口要求每个来源自己把这些差异整理好，检索服务和后面的节点只看统一后的 PaperDocument。
    """

    @abstractmethod
    def normalize_paper(self, raw: Any) -> PaperDocument | None:
        """把单条来源记录转换成统一论文对象，无法转换时返回 None。"""

        raise NotImplementedError


class PaperSearchConnector(PaperMetadataNormalizer, ABC):
    """论文检索 connector 抽象基类。

    每个外部来源都实现同一个 `search` 接口，
    编排层因此可以按统一协议调用，而不依赖具体站点的实现细节。
    """

    source_name: str

    def normalize_paper(self, raw: Any) -> PaperDocument | None:
        """默认只接受已经统一好的论文对象。

        中文注释：真实外部来源会覆盖这个方法；测试或内存来源如果本来就返回 PaperDocument，
        就不用额外写一遍空转换代码。
        """

        return raw if isinstance(raw, PaperDocument) else None

    @abstractmethod
    def search(self, request: SearchRequest) -> list[PaperDocument]:
        """执行单源检索并返回统一论文模型列表。"""

        raise NotImplementedError

    async def async_search(
        self,
        request: SearchRequest,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> list[PaperDocument]:
        """异步检索入口；未单独实现的来源先用线程包住同步方法兜底。"""

        # 中文注释：`client` 是给已经异步化的来源复用共享连接池用的。
        # 如果某个来源还没改造，这里就继续走线程兜底，不要求一次性全部重写完。
        return await asyncio.to_thread(self.search, request)
