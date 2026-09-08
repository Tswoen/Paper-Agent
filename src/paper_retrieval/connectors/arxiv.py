from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from xml.etree import ElementTree as ET

import httpx

from ..models import PaperDocument, SearchRequest
from .base import PaperSearchConnector

# arXiv 查询里允许出现的字段前缀白名单（小写比较）。
# 不在白名单里的前缀说明查询写错了，直接判定为不合法查询。
_ALLOWED_FIELD_PREFIXES = {
    "all:",
    "ti:",
    "abs:",
    "au:",
    "submitteddate:",
    "submitted_date:",
}

# arXiv API 只认大写逻辑词，小写 and/or/not 会被当成普通词参与检索。
_BOOLEAN_OPERATOR_PATTERN = re.compile(r"\b(and|or|not)\b", re.IGNORECASE)


class ArxivPaperConnector(PaperSearchConnector):
    """arXiv connector。

    这里负责把结构化意图拼成 arXiv Atom API 可接受的查询表达式，
    具体的检索语句组合规则不再暴露给上层 Agent。
    """

    source_name = "arxiv"
    _endpoint = "https://export.arxiv.org/api/query"
    _atom_ns = {"atom": "http://www.w3.org/2005/Atom"}

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        query_retry_limit: int = 1,
    ):
        """初始化 HTTP 客户端。

        query_retry_limit：主查询返回 0 篇时，用安全查询再试几次的上限。
        这个数字只是给 arXiv 的兜底次数，不是限制最多处理多少篇论文。
        """

        self.query_retry_limit = max(0, int(query_retry_limit))
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

        query = self._resolve_query(request)
        papers = self._parse_response_text(self._fetch_sync(request, query), request)
        if not papers:
            # 主查询一篇都没找到，可能是布尔表达式写得过于严格，退回安全查询再试一次。
            fallback = self._safe_fallback_query(request)
            if fallback != query:
                papers = self._parse_response_text(
                    self._fetch_sync(request, fallback),
                    request,
                )
        return papers

    async def async_search(
        self,
        request: SearchRequest,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> list[PaperDocument]:
        """异步执行 arXiv 检索，避免在异步编排里阻塞事件循环。"""

        query = self._resolve_query(request)
        resolved_client = client or httpx.AsyncClient(timeout=20.0)
        owns_client = client is None
        try:
            text = await self._fetch_async(request, query, resolved_client)
            papers = self._parse_response_text(text, request)
            for _ in range(self.query_retry_limit):
                if papers:
                    break
                # 主查询没结果就退一步，用关键词拼的安全查询再试。
                fallback = self._safe_fallback_query(request)
                if fallback == query:
                    break
                query = fallback
                text = await self._fetch_async(request, query, resolved_client)
                papers = self._parse_response_text(text, request)
            return papers
        finally:
            if owns_client:
                await resolved_client.aclose()

    def _fetch_sync(self, request: SearchRequest, query: str) -> str:
        """同步发起一次 arXiv 请求，带简单的重试退避。"""

        last_error: Exception | None = None
        for _ in range(3):
            try:
                response = self.client.get(
                    self._endpoint,
                    params=self._search_params(request, query),
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"arXiv 服务暂时不可用（{response.status_code}）",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.text
            except httpx.TimeoutException as exc:
                last_error = exc
            except httpx.TransportError as exc:
                last_error = exc
            except httpx.HTTPStatusError as exc:
                last_error = exc
            time.sleep(1.0)
        raise last_error if last_error is not None else RuntimeError("arXiv 请求失败")

    async def _fetch_async(self, request: SearchRequest, query: str, client: httpx.AsyncClient) -> str:
        """异步发起一次 arXiv 请求，带简单的重试退避。"""

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = await client.get(
                    self._endpoint,
                    params=self._search_params(request, query),
                    headers=self.headers,
                    timeout=20.0,
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"arXiv 服务暂时不可用（{response.status_code}）",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.text
            except httpx.TimeoutException as exc:
                last_error = exc
            except httpx.TransportError as exc:
                last_error = exc
            except httpx.HTTPStatusError as exc:
                last_error = exc
            if attempt < 2:
                await asyncio.sleep(0.5 * (2**attempt))
        raise last_error if last_error is not None else RuntimeError("arXiv 请求失败")

    def _search_params(self, request: SearchRequest, query: str) -> dict[str, object]:
        """整理一次 arXiv 请求的公共查询参数。"""

        return {
            "search_query": query,
            "start": 0,
            "max_results": max(1, request.limit),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

    def _resolve_query(self, request: SearchRequest) -> str:
        """生成最终发给 arXiv 的查询串。

        先按原始意图拼出查询，再做引号消毒和逻辑词大写归一；
        如果结果仍不合规，就用关键词拼一个安全查询兜底。
        """

        query = self._build_query(request)
        sanitized = self._sanitize_query(query)
        if self._validate_query(sanitized):
            return sanitized
        return self._safe_fallback_query(request)

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """把查询里的引号和逻辑词整理成 arXiv 能正确解析的写法。

        arXiv 不支持双引号短语（带引号的查询永远返回 0 篇），所以把
        ``all:"a b c"`` 展开成 ``all:a AND all:b AND all:c``，并删掉残留引号；
        同时把小写 and/or/not 统一成大写（小写的它们会被 arXiv 当成普通词）。
        """

        if not query:
            return query

        def _expand_field_phrase(match: "re.Match[str]") -> str:
            field, phrase = match.group(1), match.group(2)
            return ArxivPaperConnector._phrase_to_terms(field, phrase)

        def _expand_bare_phrase(match: "re.Match[str]") -> str:
            # 裸引号短语没有字段前缀，展开后每个词默认在全部字段里搜索。
            phrase = match.group(1)
            return ArxivPaperConnector._phrase_to_terms("all", phrase)

        # 先展开 field:"phrase" 形式（field 如 all:/ti:/abs:/au:），再展开裸 "phrase"。
        sanitized = re.sub(r'([a-zA-Z]+):"([^"]*)"', _expand_field_phrase, query)
        sanitized = re.sub(r'"([^"]*)"', _expand_bare_phrase, sanitized)
        # 删除任何残留的双引号（未配对或多余的空引号），避免 arXiv 返回 0 篇。
        sanitized = sanitized.replace('"', "")
        # 把逻辑词统一成大写，只有大写 AND/OR/NOT 才会被 arXiv 当作操作符。
        sanitized = _BOOLEAN_OPERATOR_PATTERN.sub(lambda m: m.group(1).upper(), sanitized)
        return re.sub(r"\s+", " ", sanitized).strip()

    @staticmethod
    def _phrase_to_terms(field: str, phrase: str) -> str:
        """把一对引号里的短语拆成逐词 AND 的词项。

        引号短语在 arXiv 里永远搜不到结果，所以拆成 ``field:词1 AND field:词2``；
        单词上的小括号去掉，短语里的 and/or/not 是内容不是操作符，一并丢掉，
        避免后面统一大写逻辑词时把它们误当成真正的 AND/OR。
        """

        words = [w.strip("()[]{}") for w in re.split(r"[\s,;]+", phrase.strip()) if w.strip()]
        words = [w for w in words if w.lower() not in ("and", "or", "not")]
        if not words:
            return f"{field}:"
        return " AND ".join(f"{field}:{w}" for w in words)

    @staticmethod
    def _validate_query(query: str) -> bool:
        """检查查询表达式是否合规。

        检查项：括号是否配对、字段前缀是否在白名单内、有没有控制字符。
        """

        if not query or not query.strip():
            return False

        # 括号平衡检查。
        depth = 0
        for ch in query:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth < 0:
                return False
        if depth != 0:
            return False

        # 提取所有字段前缀并核对白名单。
        prefixes = re.findall(r"(?<![a-zA-Z])[a-zA-Z]+:", query)
        for prefix in prefixes:
            if prefix.lower() not in _ALLOWED_FIELD_PREFIXES:
                return False

        # 禁止控制字符（arXiv API 会拒绝这类请求）。
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", query):
            return False

        return True

    def _safe_fallback_query(self, request: SearchRequest) -> str:
        """从关键词里拼一个最简单的安全查询，用于主查询失败时兜底。

        只保留普通词，去掉括号、引号和字段前缀，避免再触发语法问题。
        """

        candidates = [term for term in request.keywords if term.strip()]
        if not candidates and request.topic.strip():
            candidates = [request.topic.strip()]
        safe_words: list[str] = []
        for term in candidates:
            cleaned = re.sub(r'[()"":]', " ", term)
            for word in cleaned.split():
                word = word.strip()
                if not word or word.lower() in ("and", "or", "not"):
                    continue
                if word.lower() not in (w.lower() for w in safe_words):
                    safe_words.append(word)
        if not safe_words:
            return "all:*"
        return " AND ".join(f"all:{word}" for word in safe_words[:8])

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
