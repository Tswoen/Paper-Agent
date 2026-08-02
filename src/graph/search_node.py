from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Protocol, cast

from src.agents.base import AgentContext
from src.agents.searchAgent import SearchAgent, SearchIntent, SearchSubtopic, load_search_agent_llm
from src.graph.runtime import WorkflowRuntimeContext
from src.graph.runtime_resources import WorkflowRuntimeResources
from src.repositories.node_persistence.search_persistence import SearchPersistenceSink
from src.graph.state_models import JsonObject, State
from src.llm import ProviderSnapshot
from src.paper_retrieval import PaperSearchService
from src.paper_retrieval.models import PaperDocument, SearchResponse
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
        search_summary: JsonObject,
        search_output: JsonObject,
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


@dataclass(slots=True)
class SearchExecutionResult:
    """保存一次按子主题检索后的合并结果。"""

    papers: list[PaperDocument]
    raw_candidate_count: int
    sources_used: list[str]
    source_results: dict[str, int]
    source_errors: dict[str, str]


def run_search_agent_node():
    """生成执行图里的检索节点。"""

    async def _node(state: State) -> State:
        """异步执行检索节点，让图层可以自然接入 ainvoke。"""

        resolved_service = cast(PaperSearchService, state.get("search_node_service") or PaperSearchService())
        resolved_llm = cast(ProviderSnapshot | None | str, state.get("search_node_llm") or load_search_agent_llm())
        resolved_sink = _resolve_search_sink(state)
        reporter = _resolve_reporter(state)
        runtime_resources = _resolve_runtime_resources(state)

        if reporter is not None:
            reporter.started("正在生成检索条件", stage="plan_search")
            reporter.reasoning_delta("正在根据主题生成检索关键词、来源范围和筛选条件。", stage="plan_search")

        def report_search_usage(usage: JsonObject) -> None:
            """把检索条件模型的真实 token 用量更新到检索子卡片。"""

            if reporter is not None:
                reporter.progress("检索条件模型调用完成", stage="plan_search", **usage)

        agent = SearchAgent(AgentContext(llm=resolved_llm, usage_callback=report_search_usage))
        agent_update = await agent.async_run(state)
        intent = agent_update["search_intent"]
        search_halted = bool(agent_update.get("search_halted"))
        agent_diagnostics = dict(agent_update.get("diagnostics") or {})

        search_execution = SearchExecutionResult(
            papers=[],
            raw_candidate_count=0,
            sources_used=[],
            source_results={},
            source_errors={},
        )
        if reporter is not None:
            reporter.progress(
                "检索条件已准备完成",
                stage="intent_ready",
                search_halted=search_halted,
                keywords=list(intent.keywords),
                # <question> 如果intent默认应该是全源检索，但intent这里却没有值
                检索来源=list(intent.sources),
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
            search_execution = await _execute_search_intent(
                resolved_service,
                intent,
                runtime_resources=runtime_resources,
            )
            raw_papers = list(search_execution.papers)

        if reporter is not None:
            reporter.progress(
                f"已拿到 {len(raw_papers)} 篇原始候选论文",
                stage="raw_results_ready",
                raw_paper_count=len(raw_papers),
            )

        # 中文注释：检索节点先做“粗筛”，没有唯一编号或没有摘要的论文直接排除。
        # 这样阅读节点拿到的论文都更稳定，也能减少后面下载和阅读时的无效工作。
        searchable_papers = _filter_searchable_papers(raw_papers)
        scored_papers = _score_papers(intent, searchable_papers)
        max_results = max(1, intent.max_results)
        search_results = [item.paper for item in scored_papers[:max_results]]
        search_scores = [item.to_dict() for item in scored_papers]
        drop_stats = _build_drop_stats(raw_papers, searchable_papers)
        search_output = _build_search_output(state["request"].topic, state["request"].constraints, intent, search_results)
        search_summary = {
            "topic": state["request"].topic,
            "search_halted": search_halted,
            "raw_candidate_count": search_execution.raw_candidate_count,
            "raw_paper_count": len(raw_papers),
            "deduplicated_paper_count": len(raw_papers),
            "abstract_paper_count": len(searchable_papers),
            "searchable_paper_count": len(searchable_papers),
            "removed_candidate_count": drop_stats["removed_candidate_count"],
            "dropped_no_paper_id_count": drop_stats["dropped_no_paper_id_count"],
            "dropped_no_abstract_count": drop_stats["dropped_no_abstract_count"],
            "selected_paper_count": len(search_results),
            "max_results": max_results,
            "sources": list(intent.sources),
            "sources_used": list(search_execution.sources_used),
            "source_results": dict(search_execution.source_results),
            "source_errors": dict(search_execution.source_errors),
            "subtopics": [
                {
                    "subtopic": subtopic.subtopic,
                    "keyword": subtopic.keyword,
                }
                for subtopic in intent.subtopics
            ],
        }

        if reporter is not None:
            reporter.progress(
                f"排序和筛选已完成，保留 {len(search_results)} 篇论文",
                stage="rank_completed",
                selected_paper_count=len(search_results),
            )

        search_artifact_refs: list[JsonObject] = []
        if resolved_sink is not None:
            persistence_result = await asyncio.to_thread(
                resolved_sink.persist,
                topic=state["request"].topic,
                intent=intent,
                raw_papers=raw_papers,
                scored_papers=search_scores,
                selected_papers=search_results,
                search_summary=search_summary,
                search_output=search_output,
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
            search_output=search_output,
            search_artifact_refs=search_artifact_refs,
            read_resume_checkpoint=state.get("read_resume_checkpoint", {}),
            diagnostics={"agent": agent_diagnostics},
            current_step="search",
            session_repo=state.get("session_repo"),
            session_key=state.get("session_key"),
            turn_id=state.get("turn_id"),
            search_node_service=state.get("search_node_service"),
            search_node_llm=state.get("search_node_llm"),
            read_node_llm=state.get("read_node_llm"),
            analysis_node_llm=state.get("analysis_node_llm"),
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


def _resolve_runtime_resources(state: State) -> WorkflowRuntimeResources | None:
    """从运行时上下文中取出搜索节点可复用的并发资源。"""

    runtime = cast(WorkflowRuntimeContext | None, state.get("runtime_context"))
    if runtime is None:
        return None
    return runtime.resources if isinstance(runtime.resources, WorkflowRuntimeResources) else None


def _normalize_optional_str(value: Any) -> str | None:
    """把任意可选值整理成非空字符串，没有值时返回 None。"""

    text = str(value).strip() if value is not None else ""
    return text or None


async def _execute_search_intent(
    service: PaperSearchService,
    intent: SearchIntent,
    *,
    runtime_resources: WorkflowRuntimeResources | None,
) -> SearchExecutionResult:
    """按每个子主题调用检索服务，并把所有候选论文合并到一起。"""

    subtopics = intent.subtopics or [SearchSubtopic(subtopic=intent.topic or "综合检索", keyword=" ".join(intent.keywords))]
    tasks = [
        _search_one_subtopic(
            service,
            intent,
            subtopic,
            runtime_resources=runtime_resources,
        )
        for subtopic in subtopics
    ]
    responses = await asyncio.gather(*tasks)
    return _merge_subtopic_search_responses(responses)


async def _search_one_subtopic(
    service: PaperSearchService,
    intent: SearchIntent,
    subtopic: SearchSubtopic,
    *,
    runtime_resources: WorkflowRuntimeResources | None,
) -> tuple[SearchSubtopic, SearchResponse]:
    """执行单个子主题检索，让数据源自己处理检索式细节。"""

    source = intent.sources[0] if len(intent.sources) == 1 else None
    sources = list(intent.sources) if len(intent.sources) > 1 else None
    response = await service.async_search(
        query="",
        topic=subtopic.subtopic,
        keywords=[subtopic.keyword],
        keyword_expression=subtopic.keyword,
        source=source,
        sources=sources,
        limit=max(1, intent.max_results),
        year_from=intent.year_from,
        year_to=intent.year_to,
        excluded_terms=intent.excluded_terms,
        truncate=False,
        runtime_resources=runtime_resources,
    )
    return subtopic, response


def _merge_subtopic_search_responses(responses: list[tuple[SearchSubtopic, SearchResponse]]) -> SearchExecutionResult:
    """合并多个子主题的检索响应，同时记录来源统计和错误信息。"""

    # papers_by_key 用于全局去重
    papers_by_key: dict[str, PaperDocument] = {}
    raw_candidate_count = 0
    sources_used: list[str] = []
    source_results: dict[str, int] = {}
    source_errors: dict[str, str] = {}
    for subtopic, response in responses:
        source_count_sum = sum(response.source_results.values())
        raw_candidate_count += source_count_sum if source_count_sum > 0 else len(response.papers)
        for source in response.sources_used:
            if source not in sources_used:
                sources_used.append(source)
        for source, count in response.source_results.items():
            source_results[source] = source_results.get(source, 0) + int(count)
        for source, message in response.errors.items():
            prefix = f"{subtopic.subtopic}: {message}"
            source_errors[source] = f"{source_errors[source]}; {prefix}" if source in source_errors else prefix
        for paper in response.papers:
            _attach_search_origin(paper, subtopic)
            key = _paper_dedupe_key(paper)
            # 同一子主题下的论文进行去重
            if key in papers_by_key:
                _merge_duplicate_paper(papers_by_key[key], paper)
                continue
            papers_by_key[key] = paper
    return SearchExecutionResult(
        papers=list(papers_by_key.values()),
        raw_candidate_count=raw_candidate_count,
        sources_used=sources_used,
        source_results=source_results,
        source_errors=source_errors,
    )


def _filter_papers_with_abstract(papers: list[PaperDocument]) -> list[PaperDocument]:
    """只保留带摘要的论文，让后面的打分能同时看标题和摘要。"""

    filtered: list[PaperDocument] = []
    for paper in papers:
        # 中文注释：粗筛阶段不再把“没有摘要但可能有全文链接”的论文交给阅读节点，
        # 因为阅读节点的精筛依赖标题和摘要一起判断相关性。
        if not (paper.abstract or "").strip():
            continue
        filtered.append(paper)
    return filtered


def _filter_searchable_papers(papers: list[PaperDocument]) -> list[PaperDocument]:
    """只保留有唯一编号、也有摘要的论文。"""

    filtered: list[PaperDocument] = []
    for paper in papers:
        # 中文注释：没有 paperId 的论文后面很难稳定去重和复用；没有摘要的论文也无法在检索节点做贴题粗筛。
        if not (paper.paperId or "").strip():
            continue
        if not (paper.abstract or "").strip():
            continue
        filtered.append(paper)
    return filtered


def _build_drop_stats(raw_papers: list[PaperDocument], kept_papers: list[PaperDocument]) -> JsonObject:
    """统计粗筛阶段删掉了多少论文，以及主要删除原因。"""

    kept_keys = {_paper_dedupe_key(paper) for paper in kept_papers}
    dropped_no_paper_id_count = 0
    dropped_no_abstract_count = 0
    for paper in raw_papers:
        if _paper_dedupe_key(paper) in kept_keys:
            continue
        if not (paper.paperId or "").strip():
            dropped_no_paper_id_count += 1
            continue
        if not (paper.abstract or "").strip():
            dropped_no_abstract_count += 1
    return {
        "removed_candidate_count": len(raw_papers) - len(kept_papers),
        "dropped_no_paper_id_count": dropped_no_paper_id_count,
        "dropped_no_abstract_count": dropped_no_abstract_count,
    }


def _build_search_output(
    topic: str,
    constraints: JsonObject,
    intent: SearchIntent,
    selected_papers: list[PaperDocument],
) -> JsonObject:
    """整理检索节点对后续流程和前端展示都友好的输出结构。"""

    subtopics: list[JsonObject] = []
    for subtopic in intent.subtopics:
        matched_papers = [
            paper.to_dict()
            for paper in selected_papers
            if _paper_has_subtopic_origin(paper, subtopic)
        ]
        subtopics.append(
            {
                "subtopic": subtopic.subtopic,
                "keyword": subtopic.keyword,
                "papers": matched_papers,
            }
        )
    return {
        "topic": topic,
        "constraint": dict(constraints or {}),
        "subtopics": subtopics,
    }


def _paper_has_subtopic_origin(paper: PaperDocument, subtopic: SearchSubtopic) -> bool:
    """判断论文是否来自指定子主题。"""

    origins = paper.metadata.get("search_subtopics") if isinstance(paper.metadata, dict) else []
    if not isinstance(origins, list):
        return False
    for origin in origins:
        if not isinstance(origin, dict):
            continue
        if origin.get("subtopic") == subtopic.subtopic and origin.get("keyword") == subtopic.keyword:
            return True
    return False


def _attach_search_origin(paper: PaperDocument, subtopic: SearchSubtopic) -> None:
    """给论文记录补上它来自哪个子主题，方便前端按方向展示。"""

    metadata = dict(paper.metadata or {})
    origins = metadata.get("search_subtopics")
    if not isinstance(origins, list):
        origins = []
    origin = {"subtopic": subtopic.subtopic, "keyword": subtopic.keyword}
    if origin not in origins:
        origins.append(origin)
    metadata["search_subtopics"] = origins
    found_sources = metadata.get("found_sources")
    if not isinstance(found_sources, list):
        found_sources = []
    if paper.source and paper.source not in found_sources:
        found_sources.append(paper.source)
    metadata["found_sources"] = found_sources
    paper.metadata = metadata


def _merge_duplicate_paper(target: PaperDocument, duplicate: PaperDocument) -> None:
    """把重复论文的来源信息补到已保留的论文上。"""

    target_metadata = dict(target.metadata or {})
    duplicate_metadata = dict(duplicate.metadata or {})
    target_origins = target_metadata.get("search_subtopics")
    duplicate_origins = duplicate_metadata.get("search_subtopics")
    if not isinstance(target_origins, list):
        target_origins = []
    if isinstance(duplicate_origins, list):
        for origin in duplicate_origins:
            if origin not in target_origins:
                target_origins.append(origin)
    target_metadata["search_subtopics"] = target_origins
    found_sources = target_metadata.get("found_sources")
    if not isinstance(found_sources, list):
        found_sources = [target.source] if target.source else []
    if duplicate.source and duplicate.source not in found_sources:
        found_sources.append(duplicate.source)
    target_metadata["found_sources"] = found_sources
    target.metadata = target_metadata
    if not (target.abstract or "").strip() and (duplicate.abstract or "").strip():
        target.abstract = duplicate.abstract
    if not target.url and duplicate.url:
        target.url = duplicate.url
    if not target.pdf_url and duplicate.pdf_url:
        target.pdf_url = duplicate.pdf_url


def _score_papers(intent: SearchIntent, papers: list[PaperDocument]) -> list[ScoredPaper]:
    """根据检索意图对候选论文打分并排序。"""

    # 中文注释：粗筛打分只看模型生成的关键词和检索式，标题命中权重大一些，摘要命中权重小一些。
    # 标题通常更直接说明论文是否贴题，所以这里给标题命中 2 倍分。
    scoring_sources = [*intent.keywords, *[subtopic.keyword for subtopic in intent.subtopics]]
    tokens = _build_scoring_tokens(scoring_sources)
    phrase_terms = _build_scoring_phrases(scoring_sources)
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
    """为论文生成稳定的去重键，优先使用统一后的 paperId。"""

    paper_id = (paper.paperId or "").strip().lower()
    if paper_id:
        return f"paper_id:{paper_id}"
    return f"title:{paper.title.strip().lower()}"


def _extract_query_terms(text: str) -> list[str]:
    """从文本里提取适合做打分的关键词。"""

    seen: set[str] = set()
    results: list[str] = []
    for token in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*", text.lower()):
        normalized = token.strip()
        if len(normalized) <= 1 or normalized in {"and", "or", "not"} or normalized in seen:
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
