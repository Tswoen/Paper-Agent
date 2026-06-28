from __future__ import annotations

import re
from dataclasses import dataclass

from src.agents.base import AgentContext
from src.agents.searchAgent import SearchAgent, SearchIntent, load_search_agent_llm
from src.graph.state_models import State
from src.llm import ProviderSnapshot
from src.paper_retrieval import PaperSearchService
from src.paper_retrieval.models import PaperDocument


@dataclass(slots=True)
class ScoredPaper:
    """表示带有相关性评分明细的候选论文。"""

    paper: PaperDocument
    score: float
    title_hits: int
    abstract_hits: int
    topic_in_title: bool
    topic_in_abstract: bool
    matched_terms: list[str]


def run_search_agent_node(
    service: PaperSearchService | None = None,
    llm: ProviderSnapshot | None | str = "auto",
):
    """生成搜索图中的搜索节点。

    中文说明：
    1. 先调用 `SearchAgent` 生成结构化检索意图；
    2. 再调用检索服务执行查询；
    3. 对候选论文打分、排序并截断；
    4. 只把后续节点真正需要的业务字段写回共享 `State`。
    """

    resolved_service = service or PaperSearchService()
    resolved_llm = load_search_agent_llm() if llm == "auto" else llm

    def _node(state: State) -> State:
        """执行搜索节点主逻辑，并返回新的局部状态更新。"""

        agent = SearchAgent(AgentContext(spec=SearchAgent.spec, llm=resolved_llm))
        agent_update = agent.run(state)
        intent = agent_update["search_intent"]
        papers = [] if agent_update.get("search_halted") else _execute_search_intent(resolved_service, intent)
        scored_papers = _score_papers(state["request"].topic, papers)
        max_results = max(1, intent.max_results)
        search_results = [item.paper for item in scored_papers[:max_results]]
        return State(
            request=state["request"],
            search_results=search_results,
            current_step="search",
        )

    return _node


def _execute_search_intent(service: PaperSearchService, intent: SearchIntent) -> list[PaperDocument]:
    """按照检索意图调用检索服务，并汇总去重后的论文结果。"""

    collected: list[PaperDocument] = []
    seen: set[str] = set()
    sources = intent.sources or [None]
    for source in sources:
        response = service.search(
            query="",
            topic=intent.topic,
            keywords=intent.keywords,
            source=source,
            limit=max(1, intent.max_results),
            year_from=intent.year_from,
            year_to=intent.year_to,
            excluded_terms=intent.excluded_terms,
            truncate=False,
        )
        for paper in response.papers:
            dedupe_key = _paper_dedupe_key(paper)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            collected.append(paper)
    return collected


def _score_papers(topic: str, papers: list[PaperDocument]) -> list[ScoredPaper]:
    """根据主题对候选论文打分并排序。"""

    tokens = _extract_query_terms(topic)
    topic_text = _normalize_text(topic)
    threshold = _score_threshold(topic)
    scored_items: list[ScoredPaper] = []
    for paper in papers:
        title_text = _normalize_text(paper.title)
        abstract_text = _normalize_text(paper.abstract or "")
        title_hits = sum(1 for token in tokens if token in title_text)
        abstract_hits = sum(1 for token in tokens if token in abstract_text)
        topic_in_title = bool(topic_text) and topic_text in title_text
        topic_in_abstract = bool(topic_text) and topic_text in abstract_text
        score = float(title_hits * 2.0 + abstract_hits * 1.0)
        if topic_in_title:
            score += 3.0
        if topic_in_abstract:
            score += 1.5
        matched_terms = [token for token in tokens if token in title_text or token in abstract_text]
        scored_items.append(
            ScoredPaper(
                paper=paper,
                score=score,
                title_hits=title_hits,
                abstract_hits=abstract_hits,
                topic_in_title=topic_in_title,
                topic_in_abstract=topic_in_abstract,
                matched_terms=matched_terms,
            )
        )
    selected = [item for item in scored_items if item.score >= threshold]
    if not selected:
        # 中文注释：当主题过短导致阈值筛选后为空时，回退到全部候选，避免出现“明明搜到了却全被过滤”的情况。
        selected = list(scored_items)
    selected.sort(
        key=lambda item: (
            item.score,
            item.paper.year or 0,
            len(item.paper.authors),
        ),
        reverse=True,
    )
    return selected


def _paper_dedupe_key(paper: PaperDocument) -> str:
    """为论文生成稳定的去重键，优先使用 DOI，其次回退到标题。"""

    doi = (paper.doi or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    return f"title:{paper.title.strip().lower()}"


def _extract_query_terms(text: str) -> list[str]:
    """从主题文本中抽取关键词，供评分阶段计算命中数。"""

    seen: set[str] = set()
    results: list[str] = []
    for token in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*", text.lower()):
        normalized = token.strip()
        if len(normalized) <= 1 or normalized in seen:
            continue
        seen.add(normalized)
        results.append(normalized)
    return results


def _normalize_text(text: str) -> str:
    """统一文本大小写和空白，减少匹配噪声。"""

    return " ".join(text.lower().split())


def _score_threshold(topic: str) -> float:
    """根据主题长度给出一个温和的最低分阈值。"""

    token_count = len(_extract_query_terms(topic))
    if token_count >= 6:
        return 4.0
    if token_count >= 3:
        return 2.5
    return 1.5
