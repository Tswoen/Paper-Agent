from __future__ import annotations

from datetime import datetime
from xml.etree import ElementTree as ET

import httpx

from ..models import PaperDocument, SearchRequest
from .base import PaperSearchConnector


class ArxivPaperConnector(PaperSearchConnector):
    """arXiv connector。

    这里负责把结构化意图拼成 arXiv Atom API 可接受的查询表达式，
    具体的检索语句组合规则不再暴露给上层 Agent。
    """

    source_name = "arxiv"
    _endpoint = "https://export.arxiv.org/api/query"
    _atom_ns = {"atom": "http://www.w3.org/2005/Atom"}

    def __init__(self, client: httpx.Client | None = None):
        """初始化 HTTP 客户端。"""

        self.headers = {
            "User-Agent": "papers-agents/0.1 paper-retrieval",
            "Accept": "application/atom+xml, application/xml;q=0.9, */*;q=0.8",
        }
        self.client = client or httpx.Client(
            timeout=20.0,
            headers=self.headers,
        )

    def search(self, request: SearchRequest) -> list[PaperDocument]:
        """执行 arXiv 检索，并在 connector 内完成查询拼装。"""

        query = self._build_query(request)
        response = self.client.get(
            self._endpoint,
            params={
                "search_query": query,
                "start": 0,
                "max_results": max(1, request.limit),
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        response.raise_for_status()
        return self._parse_response_text(response.text, request)

    async def async_search(
        self,
        request: SearchRequest,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> list[PaperDocument]:
        """异步执行 arXiv 检索，避免在异步编排里阻塞事件循环。"""

        query = self._build_query(request)
        resolved_client = client or httpx.AsyncClient(timeout=20.0)
        owns_client = client is None
        try:
            response = await resolved_client.get(
                self._endpoint,
                params={
                    "search_query": query,
                    "start": 0,
                    "max_results": max(1, request.limit),
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
                headers=self.headers,
                timeout=20.0,
            )
        finally:
            if owns_client:
                await resolved_client.aclose()
        response.raise_for_status()
        return self._parse_response_text(response.text, request)

    def _parse_response_text(self, text: str, request: SearchRequest) -> list[PaperDocument]:
        """把 arXiv XML 响应解析成论文列表，同步和异步入口共用。"""

        root = ET.fromstring(text)
        papers: list[PaperDocument] = []
        for entry in root.findall("atom:entry", self._atom_ns):
            paper = self.normalize_paper(entry)
            if paper is None:
                continue
            if not self._within_year_range(paper, request):
                continue
            if self._contains_excluded_terms(paper, request.excluded_terms):
                continue
            papers.append(paper)
        return papers[: request.limit]

    def _build_query(self, request: SearchRequest) -> str:
        """把 topic / keywords / query 组合成 arXiv 的查询串。"""

        query_text = self._choose_query_text(request)
        if not query_text:
            return "all:*"
        return f"all:{self._escape_query(query_text)}"

    def _choose_query_text(self, request: SearchRequest) -> str:
        """优先使用上层原始 query，没有时再用 topic 和 keywords 兜底。"""

        if request.keyword_expression.strip():
            return request.keyword_expression.strip()
        if request.query.strip():
            return request.query.strip()
        parts: list[str] = []
        if request.topic.strip():
            parts.append(request.topic.strip())
        if request.keywords:
            parts.extend(request.keywords[:5])
        return " ".join(parts).strip()

    def _escape_query(self, query: str) -> str:
        """对 arXiv 查询串做最小化清理，避免空白字符导致语义不稳定。"""

        return " ".join(query.split())

    def normalize_paper(self, raw: object) -> PaperDocument | None:
        """把单个 Atom entry 解析成统一论文对象。"""

        if not isinstance(raw, ET.Element):
            return None
        entry = raw
        title = self._text(entry, "atom:title")
        if not title:
            return None
        paper_id = self._text(entry, "atom:id").rsplit("/", 1)[-1]
        authors = [author_name.text.strip() for author_name in entry.findall("atom:author/atom:name", self._atom_ns) if author_name.text]
        summary = self._text(entry, "atom:summary")
        published_text = self._text(entry, "atom:published")
        published_year = self._parse_year(published_text)
        pdf_url = ""
        doi = ""
        for link in entry.findall("atom:link", self._atom_ns):
            href = (link.attrib.get("href") or "").strip()
            title_attr = (link.attrib.get("title") or "").strip().lower()
            link_type = (link.attrib.get("type") or "").strip().lower()
            if link_type == "application/pdf" and href:
                pdf_url = href
            if title_attr == "doi" and href:
                doi = href.rsplit("/", 1)[-1]
        unique_id = doi or paper_id
        return PaperDocument(
            id=unique_id or title,
            paperId=unique_id,
            title=title,
            authors=authors,
            abstract=summary,
            year=published_year,
            venue="arXiv",
            url=self._text(entry, "atom:id"),
            pdf_url=pdf_url or None,
            doi=doi or None,
            source=self.source_name,
            publication_date=published_text,
            journal_conference="arXiv",
            language="en",
            metadata={"published": published_text, "arxiv_id": paper_id},
        )

    def _text(self, entry: ET.Element, path: str) -> str:
        """安全读取 XML 文本节点。"""

        node = entry.find(path, self._atom_ns)
        return node.text.strip() if node is not None and node.text else ""

    def _parse_year(self, value: str) -> int | None:
        """从发布时间中提取年份。"""

        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).year
        except ValueError:
            return None

    def _within_year_range(self, paper: PaperDocument, request: SearchRequest) -> bool:
        """按年份范围过滤结果。"""

        if paper.year is None:
            return True
        if request.year_from is not None and paper.year < request.year_from:
            return False
        if request.year_to is not None and paper.year > request.year_to:
            return False
        return True

    def _contains_excluded_terms(self, paper: PaperDocument, excluded_terms: list[str]) -> bool:
        """对标题和摘要做排除词过滤。"""

        haystack = f"{paper.title} {paper.abstract or ''}".lower()
        return any(term.strip().lower() in haystack for term in excluded_terms if term.strip())
