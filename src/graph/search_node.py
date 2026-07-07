from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, cast

from src.agents.base import AgentContext
from src.agents.searchAgent import SearchAgent, SearchIntent, load_search_agent_llm
from src.graph.search_persistence import SearchPersistenceSink
from src.graph.state_models import JsonObject, State
from src.llm import ProviderSnapshot
from src.paper_retrieval import PaperSearchService
from src.paper_retrieval.models import PaperDocument
from src.repositories.sessions.base import SessionRepository


class SearchNodeSink(Protocol):
    """
    描述检索节点可选的持久化 sink 协议。
    Protocol：无需显式继承，只要类拥有协议里定义的全部方法 / 签名，就自动视为实现了该协议
    """
    def persist(
        self,
        *,
        topic: str,
        intent: SearchIntent,
        raw_papers: list[PaperDocument],
        scored_papers: list[JsonObject],
        selected_papers: list[PaperDocument],
        agent_diagnostics: JsonObject,
        search_halted: bool,
    ): ...


@dataclass(slots=True)
class ScoredPaper:
    """表示带有相关性评分明细的候选论文。"""

    paper: PaperDocument
    score: float
    title_hits: int
    abstract_hits: int
    keyword_phrase_in_title: bool
    keyword_phrase_in_abstract: bool
    matched_terms: list[str]

    def to_dict(self) -> JsonObject:
        """把评分结果转换为便于落盘和回传的普通字典。"""

        return {
            "paper": self.paper.to_dict(),
            "score": self.score,
            "title_hits": self.title_hits,
            "abstract_hits": self.abstract_hits,
            "keyword_phrase_in_title": self.keyword_phrase_in_title,
            "keyword_phrase_in_abstract": self.keyword_phrase_in_abstract,
            "matched_terms": list(self.matched_terms),
        }


def run_search_agent_node():
    """生成搜索图中的论文检索节点。

    中文说明：
    1. 检索服务、LLM 与持久化 sink 都在节点内部解析；
    2. 默认情况下节点自行创建依赖；
    3. 只有测试或特殊调用时，才通过图状态放入覆盖项。
    """

    def _node(state: State) -> State:
        """执行检索节点主逻辑，并返回新的局部状态更新。"""

        # 中文注释：节点优先从 state 里读取覆盖依赖，若没有则回退到内部默认装配。
        resolved_service = PaperSearchService()
        resolved_llm = load_search_agent_llm()
        resolved_sink = _resolve_search_sink(state)

        agent = SearchAgent(AgentContext(llm=resolved_llm))
        agent_update = agent.run(state)
        intent = agent_update["search_intent"]
        search_halted = bool(agent_update.get("search_halted"))
        agent_diagnostics = dict(agent_update.get("diagnostics") or {})
        # 开始检索相关论文
        raw_papers = [] if search_halted else _execute_search_intent(resolved_service, intent)
        # 中文注释：排序阶段只复用搜索阶段已经解析出的英文关键词，
        # 明确避免把用户原始 topic 直接带入评分，防止中文主题匹配英文论文时整批归零。
        scored_papers = _score_papers(intent, raw_papers)
        max_results = max(1, intent.max_results)
        search_results = [item.paper for item in scored_papers[:max_results]]
        search_scores = [item.to_dict() for item in scored_papers]
        search_summary = {
            "topic": state["request"].topic,
            "search_halted": search_halted,
            "raw_paper_count": len(raw_papers),
            "selected_paper_count": len(search_results),
            "max_results": max_results,
            "sources": list(intent.sources),
        }
        search_artifact_refs: list[JsonObject] = []
        if resolved_sink is not None:
            persistence_result = resolved_sink.persist(
                topic=state["request"].topic,
                intent=intent,
                raw_papers=raw_papers,
                scored_papers=search_scores,
                selected_papers=search_results,
                agent_diagnostics=agent_diagnostics,
                search_halted=search_halted,
            )
            search_artifact_refs = persistence_result.to_state_refs()
            search_summary["manifest"] = dict(persistence_result.manifest)
        return State(
            request=state["request"],
            search_results=search_results,
            search_scores=search_scores,
            search_summary=search_summary,
            search_artifact_refs=search_artifact_refs,
            diagnostics={"agent": agent_diagnostics},
            current_step="search",
        )

    return _node


def _resolve_search_sink(state: State) -> SearchNodeSink | None:
    """解析检索节点的持久化 sink。"""

    session_repo = cast(SessionRepository | None, state.get("session_repo"))
    session_key = _normalize_optional_str(state.get("session_key"))
    turn_id = _normalize_optional_str(state.get("turn_id"))
    if session_repo is None or session_key is None or turn_id is None:
        return None
    # 中文注释：只有具备完整会话上下文时，才内部创建搜索结果持久化 sink。
    return SearchPersistenceSink(session_repo, session_key=session_key, turn_id=turn_id)


def _normalize_optional_str(value: Any) -> str | None:
    """把任意可选值规整为非空字符串。"""

    text = str(value).strip() if value is not None else ""
    return text or None


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


def _score_papers(intent: SearchIntent, papers: list[PaperDocument]) -> list[ScoredPaper]:
    """
    根据检索意图对候选论文打分并排序。

    中文说明：
    1. 评分阶段完全抛弃用户原始 topic，只使用 SearchAgent 生成的 keywords；
    2. 这样可以保证“检索使用什么语义，排序就使用什么语义”，避免输入语言不一致带来的噪声；
    3. 若后续需要排查排序问题，直接查看 keywords 与 matched_terms 即可，链路更单纯。
    """

    tokens = _build_scoring_tokens(intent.keywords)
    phrase_terms = _build_scoring_phrases(intent.keywords)
    threshold = _score_threshold(tokens)
    scored_items: list[ScoredPaper] = []
    for paper in papers:
        title_text = _normalize_text(paper.title)
        abstract_text = _normalize_text(paper.abstract or "")
        title_hits = sum(1 for token in tokens if token in title_text)
        abstract_hits = sum(1 for token in tokens if token in abstract_text)
        # 中文注释：这里直接使用新字段名，明确表达“关键词短语是否在标题、摘要中命中”。
        keyword_phrase_in_title = any(term in title_text for term in phrase_terms)
        keyword_phrase_in_abstract = any(term in abstract_text for term in phrase_terms)
        score = float(title_hits * 2.0 + abstract_hits * 1.0)
        if keyword_phrase_in_title:
            score += 3.0
        if keyword_phrase_in_abstract:
            score += 1.5
        matched_terms = _collect_matched_terms(tokens, phrase_terms, title_text, abstract_text)
        scored_items.append(
            ScoredPaper(
                paper=paper,
                score=score,
                title_hits=title_hits,
                abstract_hits=abstract_hits,
                keyword_phrase_in_title=keyword_phrase_in_title,
                keyword_phrase_in_abstract=keyword_phrase_in_abstract,
                matched_terms=matched_terms,
            )
        )
    selected = [item for item in scored_items if item.score >= threshold]
    if not selected:
        # 中文注释：当主题过短导致阈值筛选后为空时，回退到全部候选，避免“明明搜到了却全部被过滤”。
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
    """从主题文本中提取关键词，供评分阶段计算命中数。"""

    seen: set[str] = set()
    results: list[str] = []
    for token in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*", text.lower()):
        normalized = token.strip()
        if len(normalized) <= 1 or normalized in seen:
            continue
        seen.add(normalized)
        results.append(normalized)
    return results


def _build_scoring_tokens(keywords: list[str]) -> list[str]:
    """
    构建用于 token 命中的评分词列表。

    中文说明：
    1. 每个 keyword 会继续拆成 token，便于统计 title_hits / abstract_hits；
    2. 该函数返回去重后的稳定顺序列表，便于后续调试 matched_terms。
    """

    seen: set[str] = set()
    scoring_tokens: list[str] = []
    for candidate in keywords:
        for token in _extract_query_terms(candidate):
            if token in seen:
                continue
            seen.add(token)
            scoring_tokens.append(token)
    return scoring_tokens


def _build_scoring_phrases(keywords: list[str]) -> list[str]:
    """
    构建用于短语级命中的评分短语列表。

    中文说明：
    1. 这里保留完整 keyword 短语，主要服务于 keyword_phrase_in_title / keyword_phrase_in_abstract 这两个字段；
    2. 因为评分阶段已明确抛弃 topic，所以这里只看 keywords；
    3. 只保留长度大于 1 的非空短语，避免噪声匹配。
    """

    seen: set[str] = set()
    phrase_terms: list[str] = []
    for candidate in keywords:
        normalized = _normalize_text(candidate)
        if len(normalized) <= 1 or normalized in seen:
            continue
        seen.add(normalized)
        phrase_terms.append(normalized)
    return phrase_terms


def _collect_matched_terms(
    tokens: list[str],
    phrase_terms: list[str],
    title_text: str,
    abstract_text: str,
) -> list[str]:
    """
    汇总当前论文命中的关键词与短语。

    中文说明：
    1. 优先返回完整短语，方便观察“autonomous driving”这类核心主题是否直接命中；
    2. 再补充 token 级命中，便于解释 score 的来源；
    3. 保持返回顺序稳定，便于前端和 artifact 做差异对比。
    """

    matched_terms: list[str] = []
    seen: set[str] = set()
    for candidate in [*phrase_terms, *tokens]:
        if candidate in seen:
            continue
        if candidate in title_text or candidate in abstract_text:
            seen.add(candidate)
            matched_terms.append(candidate)
    return matched_terms


def _normalize_text(text: str) -> str:
    """统一文本大小写和空白，减少匹配噪声。"""

    return " ".join(text.lower().split())


def _score_threshold(tokens: list[str]) -> float:
    """
    根据评分 token 数量给出一个温和的最低分阈值。

    中文说明：
    这里不再只看原始 topic 的长度，而是看最终真正参与评分的 token 数量，
    这样阈值才能和新的“topic + keywords 联合打分”策略保持一致。
    """

    token_count = len(tokens)
    if token_count >= 6:
        return 4.0
    if token_count >= 3:
        return 2.5
    return 1.5
