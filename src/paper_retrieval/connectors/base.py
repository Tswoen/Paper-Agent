from __future__ import annotations

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
