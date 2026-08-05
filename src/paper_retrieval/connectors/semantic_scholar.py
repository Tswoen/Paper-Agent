from __future__ import annotations

import httpx

from ..models import PaperDocument, SearchRequest
from .base import PaperSearchConnector


class SemanticScholarPaperConnector(PaperSearchConnector):
    """Semantic Scholar connector。

    该 connector 负责把结构化检索意图拼接成 Graph API 查询参数，
    并保留来源内部支持的过滤逻辑。
    """

    source_name = "semantic_scholar"
    _endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"
    _fields = ",".join(
        [
            "title",
            "abstract",
            "year",
            "authors",
            "venue",
            "url",
            "externalIds",
            "openAccessPdf",
            "publicationDate",
            "journal",
            "publicationVenue",
        ]
    )

    def __init__(self, client: httpx.Client | None = None, api_key: str | None = None):
        """初始化 HTTP 客户端，并在配置了密钥时注入 API Key。"""

        headers = {
            "User-Agent": "papers-agents/0.1 paper-retrieval",
            "Accept": "application/json",
        }
        # 中文说明：密钥由 PaperSearchService 从 config/system.yaml 读出后传入。
        # 配置为 null 时不添加 x-api-key，请求会按 Semantic Scholar 的匿名规则执行。
        resolved_key = (api_key or "").strip()
        if resolved_key:
            headers["x-api-key"] = resolved_key
        self.headers = headers
        self.client = client or httpx.Client(timeout=20.0, headers=headers)

    def search(self, request: SearchRequest) -> list[PaperDocument]:
        """执行 Semantic Scholar 检索，并在 connector 内完成查询拼装。"""

        response = self.client.get(self._endpoint, params=self._params(request))
        response.raise_for_status()
        return self._parse_payload(response.json(), request)

    async def async_search(
        self,
        request: SearchRequest,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> list[PaperDocument]:
        """异步执行 Semantic Scholar 检索，避免在异步编排里阻塞事件循环。"""

        resolved_client = client or httpx.AsyncClient(timeout=20.0)
        owns_client = client is None
        try:
            response = await resolved_client.get(
                self._endpoint,
                params=self._params(request),
                headers=self.headers,
                timeout=20.0,
            )
        finally:
            if owns_client:
                await resolved_client.aclose()
        response.raise_for_status()
        return self._parse_payload(response.json(), request)

    def _params(self, request: SearchRequest) -> dict[str, str | int]:
        """构造 Semantic Scholar 请求参数，同步和异步入口共用。"""

        return {
            "query": self._build_query(request),
            "limit": max(1, request.limit),
            "fields": self._fields,
        }

    def _parse_payload(self, payload: dict[str, object], request: SearchRequest) -> list[PaperDocument]:
        """把 Semantic Scholar JSON 响应解析成论文列表，同步和异步入口共用。"""

        papers: list[PaperDocument] = []
        for item in payload.get("data", []) or []:
            paper = self.normalize_paper(item)
            if paper is None:
                continue
            if not self._within_year_range(paper, request):
                continue
            if self._contains_excluded_terms(paper, request.excluded_terms):
                continue
            papers.append(paper)
        return papers[: request.limit]

    def _build_query(self, request: SearchRequest) -> str:
        """把 topic / keywords 合成 Semantic Scholar 的 query。"""

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

    def normalize_paper(self, raw: object) -> PaperDocument | None:
        """把单条 Semantic Scholar 记录解析成统一论文对象。"""

        if not isinstance(raw, dict):
            return None
        item = raw
        title = str(item.get("title") or "").strip()
        if not title:
            return None
        authors: list[str] = []
        authors_raw = item.get("authors") or []
        if isinstance(authors_raw, list):
            for author in authors_raw:
                if not isinstance(author, dict):
                    continue
                name = str(author.get("name") or "").strip()
                if name:
                    authors.append(name)
        external_ids = item.get("externalIds") or {}
        doi = ""
        arxiv_id = ""
        if isinstance(external_ids, dict):
            doi = str(external_ids.get("DOI") or "").strip()
            arxiv_id = str(external_ids.get("ArXiv") or external_ids.get("Arxiv") or "").strip()
        open_access_pdf = item.get("openAccessPdf") or {}
        pdf_url = ""
        if isinstance(open_access_pdf, dict):
            pdf_url = str(open_access_pdf.get("url") or "").strip()
        semantic_id = str(item.get("paperId") or "").strip()
        paper_id = doi or arxiv_id or semantic_id
        venue = self._venue(item)
        journal = item.get("journal") or {}
        volume = ""
        issue = ""
        if isinstance(journal, dict):
            volume = str(journal.get("volume") or "").strip()
            issue = str(journal.get("issue") or "").strip()
        return PaperDocument(
            id=paper_id or title,
            paperId=paper_id,
            title=title,
            authors=authors,
            abstract=str(item.get("abstract") or "").strip() or None,
            year=self._maybe_int(item.get("year")),
            venue=venue or None,
            url=str(item.get("url") or "").strip() or None,
            pdf_url=pdf_url or None,
            doi=doi or None,
            source=self.source_name,
            publication_date=str(item.get("publicationDate") or "").strip(),
            journal_conference=venue,
            volume=volume,
            issue=issue,
            metadata={"semantic_scholar_id": semantic_id, "arxiv_id": arxiv_id},
        )

    def _venue(self, item: dict[str, object]) -> str:
        """从 Semantic Scholar 的多个可能字段里取期刊或会议名称。"""

        venue = str(item.get("venue") or "").strip()
        if venue:
            return venue
        publication_venue = item.get("publicationVenue") or {}
        if isinstance(publication_venue, dict):
            name = str(publication_venue.get("name") or "").strip()
            if name:
                return name
        journal = item.get("journal") or {}
        if isinstance(journal, dict):
            return str(journal.get("name") or "").strip()
        return ""

    def _maybe_int(self, value: object) -> int | None:
        """安全转换可选年份字段。"""

        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
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
