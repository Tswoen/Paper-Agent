from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils import get_logger, logging_context

from .connectors import ArxivPaperConnector, OpenAlexPaperConnector, PaperSearchConnector, SemanticScholarPaperConnector
from .models import PaperDocument, SearchRequest, SearchResponse


logger = get_logger(__name__)


class PaperSearchService:
    """论文检索编排层。

    这一层只负责三件事：
    1. 管理 connector 注册表；
    2. 把结构化检索意图转换成各 connector 需要的请求；
    3. 聚合、去重和截断结果。
    具体查询语句如何拼接，由各个 connector 自己决定。
    """

    def __init__(self, connectors: dict[str, PaperSearchConnector] | None = None):
        """初始化服务，并注入默认 connector 集合。"""

        resolved = connectors or self._build_default_connectors()
        self._connectors = dict(resolved)
        logger.info("论文检索服务初始化完成", extra={"sources": sorted(self._connectors.keys())})

    def search(
        self,
        query: str = "",
        source: str | None = None,
        limit: int = 10,
        year_from: int | None = None,
        year_to: int | None = None,
        excluded_terms: list[str] | None = None,
        topic: str = "",
        keywords: list[str] | None = None,
        truncate: bool = True,
    ) -> SearchResponse:
        """执行统一检索入口。

        兼容两种调用方式：
        1. 直接传 `query`，用于旧接口或手工检索；
        2. 传 `topic` + `keywords`，由 connector 自己组装最终查询串。
        中文注释：`truncate=False` 主要给图节点使用，让上层先拿到所有来源的候选，
        再做统一评分排序和最终截断。
        """

        request = SearchRequest(
            query=query,
            topic=topic,
            keywords=list(keywords or []),
            source=source,
            limit=max(1, limit),
            year_from=year_from,
            year_to=year_to,
            excluded_terms=list(excluded_terms or []),
        )
        selected = self._select_connectors(request.source)
        response = SearchResponse(query=self._request_summary(request))
        with logging_context(
            search_query=response.query,
            search_source=request.source or "all",
            limit=request.limit,
            year_from=request.year_from,
            year_to=request.year_to,
        ):
            logger.info(
                "开始执行论文检索",
                extra={"keyword_count": len(request.keywords), "excluded_term_count": len(request.excluded_terms)},
            )
            if not selected:
                response.errors["sources"] = "No valid paper retrieval source selected."
                logger.warning("未找到可用的论文数据源")
                return response
            if len(selected) == 1:
                source_name, connector = next(iter(selected.items()))
                try:
                    papers = connector.search(request)
                except Exception:
                    logger.exception("单源论文检索失败", extra={"source": source_name})
                    raise
                response.sources_used = [source_name]
                response.source_results[source_name] = len(papers)
                response.papers = self._deduplicate_papers(papers, request.limit if truncate else None)
                logger.info(
                    "单源论文检索完成",
                    extra={"source": source_name, "raw_count": len(papers), "deduped_count": len(response.papers)},
                )
                return response
            gathered = self._search_many(selected, request)
            response.sources_used = list(selected.keys())
            merged: list[PaperDocument] = []
            for source_name, outcome in gathered.items():
                if isinstance(outcome, Exception):
                    response.errors[source_name] = str(outcome)
                    response.source_results[source_name] = 0
                    continue
                response.source_results[source_name] = len(outcome)
                merged.extend(outcome)
            response.papers = self._deduplicate_papers(merged, request.limit if truncate else None)
            logger.info(
                "多源论文检索完成",
                extra={
                    "source_count": len(selected),
                    "merged_count": len(merged),
                    "deduped_count": len(response.papers),
                    "error_sources": sorted(response.errors.keys()),
                },
            )
            return response

    async def async_search(
        self,
        query: str = "",
        source: str | None = None,
        limit: int = 10,
        year_from: int | None = None,
        year_to: int | None = None,
        excluded_terms: list[str] | None = None,
        topic: str = "",
        keywords: list[str] | None = None,
        truncate: bool = True,
    ) -> SearchResponse:
        """异步执行统一检索入口，并限制多来源并发数量。"""

        request = SearchRequest(
            query=query,
            topic=topic,
            keywords=list(keywords or []),
            source=source,
            limit=max(1, limit),
            year_from=year_from,
            year_to=year_to,
            excluded_terms=list(excluded_terms or []),
        )
        selected = self._select_connectors(request.source)
        response = SearchResponse(query=self._request_summary(request))
        if not selected:
            response.errors["sources"] = "No valid paper retrieval source selected."
            return response
        if len(selected) == 1:
            source_name, connector = next(iter(selected.items()))
            papers = await connector.async_search(request)
            response.sources_used = [source_name]
            response.source_results[source_name] = len(papers)
            response.papers = self._deduplicate_papers(papers, request.limit if truncate else None)
            return response
        gathered = await self._async_search_many(selected, request)
        response.sources_used = list(selected.keys())
        merged: list[PaperDocument] = []
        for source_name, outcome in gathered.items():
            if isinstance(outcome, Exception):
                response.errors[source_name] = str(outcome)
                response.source_results[source_name] = 0
                continue
            response.source_results[source_name] = len(outcome)
            merged.extend(outcome)
        response.papers = self._deduplicate_papers(merged, request.limit if truncate else None)
        return response

    def available_sources(self) -> list[str]:
        """返回当前可用来源名称。"""

        return sorted(self._connectors.keys())

    def _build_default_connectors(self) -> dict[str, PaperSearchConnector]:
        """构建默认 connector 注册表。"""

        openalex = OpenAlexPaperConnector()
        semantic = SemanticScholarPaperConnector()
        arxiv = ArxivPaperConnector()
        return {
            "openalex": openalex,
            "semantic_scholar": semantic,
            "semantic": semantic,
            "arxiv": arxiv,
        }

    def _select_connectors(self, source: str | None) -> dict[str, PaperSearchConnector]:
        """根据 source 选择 connector。

        传入空 source 时，表示启用所有唯一 connector 做多源检索。
        """

        if source:
            normalized = source.strip().lower()
            connector = self._connectors.get(normalized)
            return {normalized: connector} if connector is not None else {}
        unique: dict[int, tuple[str, PaperSearchConnector]] = {}
        for name, connector in self._connectors.items():
            unique[id(connector)] = (name, connector)
        return {name: connector for name, connector in unique.values()}

    def _search_many(
        self,
        connectors: dict[str, PaperSearchConnector],
        request: SearchRequest,
    ) -> dict[str, list[PaperDocument] | Exception]:
        """并发执行多源检索。

        中文注释：每个来源都拿完整 limit，后续再统一聚合、去重和评分排序，避免早期平均分配
        导致高质量来源被截断。
        """

        results: dict[str, list[PaperDocument] | Exception] = {}
        with ThreadPoolExecutor(max_workers=min(len(connectors), 4)) as executor:
            future_map = {
                executor.submit(
                    connector.search,
                    SearchRequest(
                        query=request.query,
                        topic=request.topic,
                        keywords=list(request.keywords),
                        source=source_name,
                        limit=request.limit,
                        year_from=request.year_from,
                        year_to=request.year_to,
                        excluded_terms=list(request.excluded_terms),
                    ),
                ): source_name
                for source_name, connector in connectors.items()
            }
            for future in as_completed(future_map):
                source_name = future_map[future]
                try:
                    results[source_name] = future.result()
                    logger.debug(
                        "单个来源检索完成",
                        extra={"source": source_name, "result_count": len(results[source_name])},
                    )
                except Exception as exc:
                    results[source_name] = exc
                    logger.exception("单个来源检索失败", extra={"source": source_name})
        return results

    async def _async_search_many(
        self,
        connectors: dict[str, PaperSearchConnector],
        request: SearchRequest,
    ) -> dict[str, list[PaperDocument] | Exception]:
        """异步并发执行多源检索，并用信号量限制同时访问的来源数量。"""

        semaphore = asyncio.Semaphore(min(len(connectors), 4))

        async def _run_one(source_name: str, connector: PaperSearchConnector) -> tuple[str, list[PaperDocument] | Exception]:
            async with semaphore:
                try:
                    papers = await connector.async_search(
                        SearchRequest(
                            query=request.query,
                            topic=request.topic,
                            keywords=list(request.keywords),
                            source=source_name,
                            limit=request.limit,
                            year_from=request.year_from,
                            year_to=request.year_to,
                            excluded_terms=list(request.excluded_terms),
                        )
                    )
                    return source_name, papers
                except Exception as exc:
                    logger.exception("异步单个来源检索失败", extra={"source": source_name})
                    return source_name, exc

        pairs = await asyncio.gather(*[_run_one(source_name, connector) for source_name, connector in connectors.items()])
        return dict(pairs)

    def _deduplicate_papers(self, papers: list[PaperDocument], limit: int | None) -> list[PaperDocument]:
        """按 DOI 和标题做轻量去重。

        中文注释：limit 为 None 时只去重不截断，供上层统一排序后再取最终结果。
        """

        seen: set[str] = set()
        unique: list[PaperDocument] = []
        for paper in papers:
            key = self._paper_key(paper)
            if key in seen:
                continue
            seen.add(key)
            unique.append(paper)
            if limit is not None and len(unique) >= limit:
                break
        return unique

    def _paper_key(self, paper: PaperDocument) -> str:
        """生成稳定的去重键。"""

        doi = (paper.doi or "").strip().lower()
        if doi:
            return f"doi:{doi}"
        return f"title:{paper.title.strip().lower()}"

    def _request_summary(self, request: SearchRequest) -> str:
        """把结构化请求压缩成调试用摘要。"""

        if request.query.strip():
            return request.query.strip()
        parts: list[str] = []
        if request.topic.strip():
            parts.append(request.topic.strip())
        if request.keywords:
            parts.append("keywords=" + " ".join(request.keywords[:5]))
        return " | ".join(parts)
