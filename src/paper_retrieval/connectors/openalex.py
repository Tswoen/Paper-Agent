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

    def __init__(self, client: httpx.Client | None = None, api_key: str | None = None):
        """初始化 HTTP 客户端，并保存可选的 OpenAlex API Key。"""

        self.headers = {
            "User-Agent": "papers-agents/0.1 paper-retrieval",
            "Accept": "application/json",
        }
        # 中文说明：OpenAlex 把密钥放在请求参数 api_key 中；密钥为空时不发送该参数，
        # 因此 config/system.yaml 保持 null 也能继续使用匿名检索。
        self.api_key = (api_key or "").strip()
        self.client = client or httpx.Client(
            timeout=20.0,
            headers=self.headers,
        )

    def search(self, request: SearchRequest) -> list[PaperDocument]:
        """执行 OpenAlex 检索，并在 connector 内完成查询拼装。"""

        response = self.client.get(self._endpoint, params=self._params(request))
        response.raise_for_status()
        return self._parse_payload(response.json(), request)

    async def async_search(
        self,
        request: SearchRequest,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> list[PaperDocument]:
        """异步执行 OpenAlex 检索，避免在异步编排里阻塞事件循环。"""

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
        """构造 OpenAlex 请求参数，同步和异步入口共用。"""

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
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _parse_payload(self, payload: dict[str, object], request: SearchRequest) -> list[PaperDocument]:
        """把 OpenAlex JSON 响应解析成论文列表，同步和异步入口共用。"""

        papers: list[PaperDocument] = []
        for item in payload.get("results", []) or []:
            paper = self.normalize_paper(item)
            if paper is None:
                continue
            if self._contains_excluded_terms(paper, request.excluded_terms):
                continue
            papers.append(paper)
        return papers[: request.limit]

    def _build_query(self, request: SearchRequest) -> str:
        """把 topic / keywords 组合成 OpenAlex 搜索串。"""

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
        """把单条 OpenAlex work 记录解析成统一论文对象。"""

        if not isinstance(raw, dict):
            return None
        item = raw
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
        openalex_id = str(item.get("id") or "").strip()
        paper_id = doi or openalex_id
        publication_date = str(item.get("publication_date") or "").strip()
        biblio = item.get("biblio") or {}
        volume = ""
        issue = ""
        if isinstance(biblio, dict):
            volume = str(biblio.get("volume") or "").strip()
            issue = str(biblio.get("issue") or "").strip()
        return PaperDocument(
            id=paper_id or title,
            paperId=paper_id,
            title=title,
            authors=authors,
            abstract=self._abstract_from_inverted_index(item.get("abstract_inverted_index")),
            year=self._maybe_int(item.get("publication_year")),
            venue=venue or None,
            url=openalex_id or None,
            pdf_url=pdf_url or None,
            doi=doi or None,
            source=self.source_name,
            publication_date=publication_date,
            journal_conference=venue,
            volume=volume,
            issue=issue,
            language=str(item.get("language") or "").strip(),
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

    def _abstract_from_inverted_index(self, value: object) -> str | None:
        """把 OpenAlex 的摘要词表还原成普通摘要文本。"""

        if not isinstance(value, dict):
            return None
        positions: dict[int, str] = {}
        for word, raw_indexes in value.items():
            if not isinstance(raw_indexes, list):
                continue
            for raw_index in raw_indexes:
                try:
                    positions[int(raw_index)] = str(word)
                except (TypeError, ValueError):
                    continue
        if not positions:
            return None
        return " ".join(positions[index] for index in sorted(positions))

    def _contains_excluded_terms(self, paper: PaperDocument, excluded_terms: list[str]) -> bool:
        """对标题和摘要做排除词过滤。"""

        haystack = f"{paper.title} {paper.abstract or ''}".lower()
        return any(term.strip().lower() in haystack for term in excluded_terms if term.strip())
