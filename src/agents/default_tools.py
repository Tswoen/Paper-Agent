from __future__ import annotations

from .tools import Tool, ToolRegistry, ToolSpec
from src.paper_retrieval import PaperSearchService


def build_default_tool_registry(service: PaperSearchService | None = None) -> ToolRegistry:
    """构建当前工程的默认工具注册表。

    这里先把论文检索能力接入进来，后续如果有更多 Agent 级工具，
    可以继续在这个工厂函数里集中扩展。
    """

    registry = ToolRegistry()
    registry.register(build_paper_search_tool(service))
    return registry


def build_paper_search_tool(service: PaperSearchService | None = None) -> Tool:
    """把论文检索编排层封装成现有 Agent 可调用的 `paper_search` 工具。"""

    resolved_service = service or PaperSearchService()

    def _handler(
        query: str,
        source: str | None = None,
        limit: int = 10,
        year_from: int | None = None,
        year_to: int | None = None,
        excluded_terms: list[str] | None = None,
    ) -> list[dict[str, object]]:
        """执行检索并只返回论文列表。

        SearchAgent 当前按单源逐条 query 调用工具，因此这里保持最薄的返回结构，
        只把统一论文字典列表透传回去。
        """

        response = resolved_service.search(
            query=query,
            source=source,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
            excluded_terms=excluded_terms,
        )
        return [paper.to_dict() for paper in response.papers]

    return Tool(
        spec=ToolSpec(
            name="paper_search",
            description="从 arxiv、OpenAlex、Semantic Scholar 等来源检索论文，并返回统一字段结构。",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词或检索短语"},
                    "source": {"type": ["string", "null"], "description": "可选来源名，如 arxiv/openalex/semantic_scholar"},
                    "limit": {"type": "integer", "minimum": 1, "default": 10},
                    "year_from": {"type": ["integer", "null"]},
                    "year_to": {"type": ["integer", "null"]},
                    "excluded_terms": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "标题或摘要中命中这些词时会被过滤掉",
                    },
                },
                "required": ["query"],
            },
        ),
        handler=_handler,
    )
