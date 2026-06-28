from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.agents.base import AgentContext
from src.agents.searchAgent import SearchAgent, SearchIntent, load_search_agent_llm
from src.graph.state_models import State
from src.llm import ProviderSnapshot
from src.paper_retrieval import PaperSearchService
from src.paper_retrieval.models import PaperDocument


JsonObject = dict[str, Any]


@dataclass(slots=True)
class ScoredPaper:
    """表示带评分明细的候选论文。

    search node 在完成检索后，会按标题和摘要命中情况为每篇论文打分，
    这个结构用于保存排序和诊断所需的评分细节。
    """

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

    当前节点承担完整的编排职责：
    1. 调用 SearchAgent 生成搜索意图；
    2. 调用 PaperSearchService 执行查询；
    3. 对召回论文进行相关度打分、筛选和排序；
    4. 把最终结果写回共享 State。
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
        search_scores = [_scored_paper_to_dict(item) for item in scored_papers]
        diagnostics = dict(state.get("diagnostics") or {})
        diagnostics.update(agent_update.get("diagnostics") or {})
        diagnostics["search_node"] = {
            "candidate_count": len(papers),
            "selected_count": len(search_results),
            "score_threshold": _score_threshold(state["request"].topic),
            "search_halted": bool(agent_update.get("search_halted")),
            "scores": search_scores,
        }
        return State(
            request=state["request"],
            search_intent=intent,
            search_results=search_results,
            search_scores=search_scores,
            current_step="search",
            diagnostics=diagnostics,
            raw_model_output=agent_update.get("raw_model_output", ""),
        )

    return _node


def _execute_search_intent(service: PaperSearchService, intent: SearchIntent) -> list[PaperDocument]:
    """按检索意图调用检索服务，并汇总去重后的论文结果。

    中文注释：指定多个来源时，每个来源都使用完整 max_results 检索，不再按来源均分数量。
    这里仅负责召回与去重，最终排序和截断交给评分阶段统一处理。
    """

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
    """根据主题对候选论文打分并排序。

    评分规则沿用当前文件原始需求说明：
    - 标题命中词数 * 2.0
    - 摘要命中词数 * 1.0
    - 主题整句出现在标题中 +3.0
    - 主题整句出现在摘要中 +1.5
    """

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
        # 中文注释：若阈值过滤后为空，则保留全部候选，避免当前主题过短时结果被全部筛掉。
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


def _scored_paper_to_dict(item: ScoredPaper) -> JsonObject:
    """把评分结果转换成普通字典，便于写入状态和调试输出。"""

    payload = item.paper.to_dict()
    payload.update(
        {
            "score": item.score,
            "title_hits": item.title_hits,
            "abstract_hits": item.abstract_hits,
            "topic_in_title": item.topic_in_title,
            "topic_in_abstract": item.topic_in_abstract,
            "matched_terms": list(item.matched_terms),
        }
    )
    return payload


def _paper_dedupe_key(paper: PaperDocument) -> str:
    """为论文生成稳定去重键，优先使用 DOI，其次回退到标题。"""

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
