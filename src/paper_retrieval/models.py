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
    paperId: str | None = None
    publication_date: str = ""
    journal_conference: str = ""
    volume: str = ""
    issue: str = ""
    language: str = ""
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        """补齐统一字段的默认值。

        中文注释：旧代码通常只传 `id` 和 `venue`；新检索链路会使用 `paperId` 和
        `journal_conference`。这里把两个体系轻轻对齐，避免一次改动牵连太多下游代码。
        如果 connector 明确传入 `paperId=""`，表示这个来源没有可靠唯一编号，不会被这里覆盖。
        """

        if self.paperId is None:
            self.paperId = self.doi or self.id
        if not self.journal_conference and self.venue:
            self.journal_conference = self.venue

    def to_dict(self) -> JsonObject:
        """把领域对象转成普通字典，便于工具层、调试和接口输出直接消费。"""

        return {
            "id": self.id,
            "paperId": self.paperId or "",
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year or "",
            "abstract": self.abstract or "",
            "source": self.source or "",
            "url": self.url or "",
            "publication_date": self.publication_date,
            "journal/conference": self.journal_conference,
            "journal_conference": self.journal_conference,
            "volume": self.volume,
            "issue": self.issue,
            "language": self.language,
            "venue": self.venue,
            "pdf_url": self.pdf_url,
            "doi": self.doi,
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
    keyword_expression: str = ""
    source: str | None = None
    sources: list[str] = field(default_factory=list)
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
