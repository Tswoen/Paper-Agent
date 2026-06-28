from __future__ import annotations

import httpx

from ..models import PaperDocument, SearchRequest
from .base import PaperSearchConnector


class OpenAlexPaperConnector(PaperSearchConnector):
    """OpenAlex connector。

    这里负责把结构化输入拼成 OpenAlex 可接受的搜索参数，
    上层只需要关心 topic 和 keywords，而不需要知道具体参数格式。
    """

    source_name = "openalex"
    _endpoint = "https://api.openalex.org/works"

    def __init__(self, client: httpx.Client | None = None):
        """初始化 HTTP 客户端。"""

        self.client = client or httpx.Client(
            timeout=20.0,
            headers={
                "User-Agent": "papers-agents/0.1 paper-retrieval",
                "Accept": "application/json",
            },
        )

    def search(self, request: SearchRequest) -> list[PaperDocument]:
        """执行 OpenAlex 检索，并在 connector 内完成查询拼装。"""

        params: dict[str, str | int] = {
            "search": self._build_query(request),
            "per-page": max(1, request.limit),
        }
        filters: list[str] = []
        if request.year_from is not None:
            filters.append(f"from_publication_date:{request.year_from}-01-01")
        if request.year_to is not None:
            filters.append(f"to_publication_date:{request.year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        response = self.client.get(self._endpoint, params=params)
        response.raise_for_status()
        payload = response.json()
        papers: list[PaperDocument] = []
        for item in payload.get("results", []) or []:
            paper = self._parse_item(item)
            if paper is None:
                continue
            if self._contains_excluded_terms(paper, request.excluded_terms):
                continue
            papers.append(paper)
        return papers[: request.limit]

    def _build_query(self, request: SearchRequest) -> str:
        """把 topic / keywords 组合成 OpenAlex 搜索串。"""

        if request.query.strip():
            return request.query.strip()
        parts: list[str] = []
        if request.topic.strip():
            parts.append(request.topic.strip())
        if request.keywords:
            parts.extend(request.keywords[:5])
        return " ".join(parts).strip()

    def _parse_item(self, item: dict[str, object]) -> PaperDocument | None:
        """把单条 OpenAlex work 记录解析成统一论文对象。"""

        title = str(item.get("title") or "").strip()
        if not title:
            return None
        authorships = item.get("authorships") or []
        authors: list[str] = []
        if isinstance(authorships, list):
            for authorship in authorships:
                if not isinstance(authorship, dict):
                    continue
                author = authorship.get("author") or {}
                if isinstance(author, dict):
                    display_name = str(author.get("display_name") or "").strip()
                    if display_name:
                        authors.append(display_name)
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") if isinstance(primary_location, dict) else {}
        open_access = item.get("open_access") or {}
        pdf_url = ""
        if isinstance(open_access, dict):
            pdf_url = str(open_access.get("oa_url") or "").strip()
        venue = ""
        if isinstance(source, dict):
            venue = str(source.get("display_name") or "").strip()
        doi = str(item.get("doi") or "").strip()
        if doi.startswith("https://doi.org/"):
            doi = doi.removeprefix("https://doi.org/")
        return PaperDocument(
            id=str(item.get("id") or title),
            title=title,
            authors=authors,
            abstract=None,
            year=self._maybe_int(item.get("publication_year")),
            venue=venue or None,
            url=str(item.get("id") or "").strip() or None,
            pdf_url=pdf_url or None,
            doi=doi or None,
            source=self.source_name,
            metadata={
                "cited_by_count": item.get("cited_by_count"),
                "type": item.get("type"),
            },
        )

    def _maybe_int(self, value: object) -> int | None:
        """安全转换可选年份字段。"""

        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _contains_excluded_terms(self, paper: PaperDocument, excluded_terms: list[str]) -> bool:
        """对标题和摘要做排除词过滤。"""

        haystack = f"{paper.title} {paper.abstract or ''}".lower()
        return any(term.strip().lower() in haystack for term in excluded_terms if term.strip())
