from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, cast

from src.agents.base import AgentContext
from src.agents.searchAgent import SearchAgent, SearchIntent, load_search_agent_llm
from src.graph.runtime import WorkflowRuntimeContext
from src.graph.search_persistence import SearchPersistenceSink
from src.graph.state_models import JsonObject, State
from src.llm import ProviderSnapshot
from src.paper_retrieval import PaperSearchService
from src.paper_retrieval.models import PaperDocument
from src.repositories.sessions.base import SessionRepository


class SearchNodeSink(Protocol):
    """描述检索节点可选的写盘能力。"""

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
    """表示带评分细节的候选论文。"""

    paper: PaperDocument
    score: float
    title_hits: int
    abstract_hits: int
    keyword_phrase_in_title: bool
    keyword_phrase_in_abstract: bool
    matched_terms: list[str]

    def to_dict(self) -> JsonObject:
        """把评分结果转成普通字典，方便写盘和调试。"""

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
    """生成执行图里的检索节点。"""

    def _node(state: State) -> State:
        """执行检索节点，并在执行过程中直接同步中间结果。"""

        resolved_service = cast(PaperSearchService, state.get("search_node_service") or PaperSearchService())
        resolved_llm = cast(ProviderSnapshot | None | str, state.get("search_node_llm") or load_search_agent_llm())
        resolved_sink = _resolve_search_sink(state)
        reporter = _resolve_reporter(state)

        if reporter is not None:
            reporter.started("正在生成检索条件", stage="plan_search")
            reporter.reasoning_delta("正在根据主题生成检索关键词、来源范围和筛选条件。", stage="plan_search")

        agent = SearchAgent(AgentContext(llm=resolved_llm))
        agent_update = agent.run(state)
        intent = agent_update["search_intent"]
        search_halted = bool(agent_update.get("search_halted"))
        agent_diagnostics = dict(agent_update.get("diagnostics") or {})

        if reporter is not None:
            reporter.progress(
                "检索条件已准备完成",
                stage="intent_ready",
                search_halted=search_halted,
                keywords=list(intent.keywords),
                sources=list(intent.sources),
                max_results=intent.max_results,
            )
            if intent.keywords:
                reporter.reasoning_delta(
                    f"本次检索将重点使用这些关键词：{'、'.join(intent.keywords[:6])}",
                    stage="intent_ready",
                )

        if search_halted:
            raw_papers: list[PaperDocument] = []
        else:
            if reporter is not None:
                reporter.progress("正在从论文数据源拉取候选结果", stage="fetch_results")
            raw_papers = _execute_search_intent(resolved_service, intent)

        if reporter is not None:
            reporter.progress(
                f"已拿到 {len(raw_papers)} 篇原始候选论文",
                stage="raw_results_ready",
                raw_paper_count=len(raw_papers),
            )

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

        if reporter is not None:
            reporter.progress(
                f"排序和筛选已完成，保留 {len(search_results)} 篇论文",
                stage="rank_completed",
                selected_paper_count=len(search_results),
            )

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
            if reporter is not None:
                for artifact in search_artifact_refs:
                    reporter.artifact(artifact, stage="artifact_ready")

        if reporter is not None:
            reporter.reasoning_delta(
                f"检索阶段已完成：原始候选 {len(raw_papers)} 篇，最终保留 {len(search_results)} 篇。",
                stage="search_done",
            )
            reporter.reasoning_end(stage="search_done")
            reporter.completed(
                "论文检索节点已完成",
                stage="search_done",
                raw_paper_count=len(raw_papers),
                selected_paper_count=len(search_results),
                artifact_count=len(search_artifact_refs),
            )

        return State(
            request=state["request"],
            search_results=search_results,
            search_scores=search_scores,
            search_summary=search_summary,
            search_artifact_refs=search_artifact_refs,
            diagnostics={"agent": agent_diagnostics},
            current_step="search",
            session_repo=state.get("session_repo"),
            session_key=state.get("session_key"),
            turn_id=state.get("turn_id"),
            search_node_service=state.get("search_node_service"),
            search_node_llm=state.get("search_node_llm"),
            search_node_sink=state.get("search_node_sink"),
            runtime_context=state.get("runtime_context"),
            assistant_message=state.get("assistant_message", ""),
            assistant_message_metadata=dict(state.get("assistant_message_metadata") or {}),
        )

    return _node


def _resolve_reporter(state: State):
    """从共享状态里取出检索节点专用的上报器。"""

    runtime = cast(WorkflowRuntimeContext | None, state.get("runtime_context"))
    if runtime is None or runtime.sync_port is None:
        return None
    return runtime.sync_port.for_node("search", "论文检索")


def _resolve_search_sink(state: State) -> SearchNodeSink | None:
    """根据会话上下文决定是否启用检索产物写盘。"""

    session_repo = cast(SessionRepository | None, state.get("session_repo"))
    session_key = _normalize_optional_str(state.get("session_key"))
    turn_id = _normalize_optional_str(state.get("turn_id"))
    if session_repo is None or session_key is None or turn_id is None:
        return None
    return SearchPersistenceSink(session_repo, session_key=session_key, turn_id=turn_id)


def _normalize_optional_str(value: Any) -> str | None:
    """把任意可选值整理成非空字符串，没有值时返回 None。"""

    text = str(value).strip() if value is not None else ""
    return text or None


def _execute_search_intent(service: PaperSearchService, intent: SearchIntent) -> list[PaperDocument]:
    """按检索意图调用检索服务，并合并去重后的论文列表。"""

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
    """根据检索意图对候选论文打分并排序。"""

    tokens = _build_scoring_tokens(intent.keywords)
    phrase_terms = _build_scoring_phrases(intent.keywords)
    threshold = _score_threshold(tokens)
    scored_items: list[ScoredPaper] = []
    for paper in papers:
        title_text = _normalize_text(paper.title)
        abstract_text = _normalize_text(paper.abstract or "")
        title_hits = sum(1 for token in tokens if token in title_text)
        abstract_hits = sum(1 for token in tokens if token in abstract_text)
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
        # 中文注释：当阈值过高导致全部被筛掉时，这里退回所有候选，避免“搜到了却一条都没有”。
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
    """为论文生成稳定的去重键，优先使用 DOI。"""

    doi = (paper.doi or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    return f"title:{paper.title.strip().lower()}"


def _extract_query_terms(text: str) -> list[str]:
    """从文本里提取适合做打分的关键词。"""

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
    """把关键词拆成用于命中统计的 token 列表。"""

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
    """保留完整关键词短语，方便做短语级命中。"""

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
    """汇总当前论文命中的关键词和短语。"""

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
    """统一大小写和空白，减少命中判断时的噪声。"""

    return " ".join(text.lower().split())


def _score_threshold(tokens: list[str]) -> float:
    """根据关键词数量给出一个比较温和的最低分阈值。"""

    token_count = len(tokens)
    if token_count >= 6:
        return 4.0
    if token_count >= 3:
        return 2.5
    return 1.5

