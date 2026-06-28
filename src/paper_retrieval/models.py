from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JsonObject = dict[str, Any]


@dataclass(slots=True)
class PaperDocument:
    """统一的论文领域模型。

    这一层负责把不同数据源返回的字段差异压平，避免上游流程直接依赖某个站点的私有字段。
    """

    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    doi: str | None = None
    source: str | None = None
    metadata: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        """把领域对象转成普通字典，便于工具层、调试和接口输出直接消费。"""

        return {
            "id": self.id,
            "title": self.title,
            "authors": list(self.authors),
            "abstract": self.abstract,
            "year": self.year,
            "venue": self.venue,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "doi": self.doi,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class SearchRequest:
    """统一检索请求。

    这层同时兼容两类输入：
    1. `query`：上层直接传入的原始检索串，方便兼容旧接口；
    2. `topic` / `keywords`：由 SearchAgent 产出的结构化意图，便于 connector 自己拼查询串。

    这样就能把“模型生成什么”与“具体怎么检索”彻底隔离。
    """

    query: str = ""
    topic: str = ""
    keywords: list[str] = field(default_factory=list)
    source: str | None = None
    limit: int = 10
    year_from: int | None = None
    year_to: int | None = None
    excluded_terms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SearchResponse:
    """统一检索响应。

    除了论文列表以外，还保留来源统计和错误信息，便于上层做诊断和展示。
    """

    query: str
    sources_used: list[str] = field(default_factory=list)
    source_results: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    papers: list[PaperDocument] = field(default_factory=list)

    @property
    def total(self) -> int:
        """返回去重后的论文数量。"""

        return len(self.papers)

    def to_dict(self) -> JsonObject:
        """把检索响应转成普通字典，便于调试和接口透传。"""

        return {
            "query": self.query,
            "sources_used": list(self.sources_used),
            "source_results": dict(self.source_results),
            "errors": dict(self.errors),
            "papers": [paper.to_dict() for paper in self.papers],
            "total": self.total,
        }
