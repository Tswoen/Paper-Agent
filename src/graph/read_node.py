from __future__ import annotations

import asyncio
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from src.agents.Prompts import READ_DEEP_READ_SELECTION_PROMPT
from src.agents.readAgent import ReadAgentModelUnavailableError, build_read_agent, load_read_agent_llm
from src.utils.read_utils.read_fulltext import async_convert_fulltext_to_markdown
# DEPRECATED: 2026-09-07
# 原因：三维等级 + 评分表退役，相关性判断改为模型直接输出二元 {relevant, reason}。
# 替代方案：改用 READ_RELEVANCE_STATUSES 判断精读名额状态。
# 计划移除：阅读阶段持续过滤机制稳定运行一段时间后清理
# from src.models.read_models import FullTextStatus, PaperReadResult, ReadNote, ReadRelevance, calculate_relevance_score, normalize_match_levels
from src.models.read_models import FullTextStatus, PaperReadResult, ReadNote, ReadRelevance, READ_RELEVANCE_STATUSES
from src.repositories.node_persistence.read_persistence import ReadPersistenceSink, read_request_fingerprint
from src.repositories.chroma.read_vector_store import EmbeddingConnection, async_index_chunk_file
from src.trace import trace_tool
from src.graph.runtime import WorkflowRuntimeContext
from src.graph.runtime_resources import WorkflowRuntimeResources
from src.graph.state_models import JsonObject, State
from src.llm import ModelConfig, ProviderSnapshot, SystemConfig, make_provider
from src.paper_retrieval.download import async_download_paper_fulltext
from src.paper_retrieval.models import PaperDocument
from src.repositories.sessions.base import SessionRepository
from src.utils.read_utils.chunkers import async_build_chunks_file
from src.utils.read_utils.extraction import async_extract_paper_from_chunks, empty_extraction, extraction_payload
from src.utils.read_utils.verify_extraction import async_verify_extraction


if TYPE_CHECKING:
    from langgraph.runtime import Runtime


# 中文说明：全文切分后默认只保留 chunk.json，不调用向量嵌入服务。
# 以后需要恢复向量检索时，只需把这里改成 True；无需改动全文下载、转换和切分流程。
ENABLE_FULL_TEXT_EMBEDDING = False

# 中文说明：这两种状态都表示全文处理已经结束。indexed 表示已写入向量库，
# chunks_saved 表示只保存了切分后的 chunk.json。
_FULL_TEXT_COMPLETED_STATUSES = frozenset({"indexed", "chunks_saved"})


class ReadResourceUnavailableError(RuntimeError):
    """表示阅读节点依赖的外部资源不可用，按普通失败中断当前节点。

    中文注释：只用于错误分类（区分"资源不可用"与"单篇处理异常"）。恢复由官方
    Checkpointer 负责：用户重试时重跑本节点，note 跳过已完成论文。
    """


class ReadModelUnavailableError(ReadResourceUnavailableError):
    """表示阅读模型当前不可用，按普通失败中断当前节点。

    这个异常只用于“用户修好模型后可以继续”的场景，例如：
    1. 没有配置阅读模型；
    2. 模型接口返回 HTTP/鉴权/限流/服务端错误；
    3. 调用模型时发生网络、超时、鉴权等异常。

    它和普通论文处理异常刻意区分：普通异常只跳过当前论文；模型返回了内容但
    内容质量不好时，也不算模型不可用，阅读节点会改用保守笔记继续处理。
    """


@dataclass(slots=True)
class PaperTaskInput:
    """保存一篇论文并发处理时需要的固定输入。"""

    paper: PaperDocument
    position: int
    restored_result: PaperReadResult | None = None


class CompletedPaperCounter:
    """记录已经完成的论文数量，方便并发任务实时上报进度。"""

    def __init__(self, initial_count: int):
        """用已经恢复的完成数量初始化计数器。"""

        self._count = max(0, int(initial_count))

    def current(self) -> int:
        """返回当前已经完成的论文数量。"""

        return self._count

    def increment(self) -> int:
        """在一篇论文真正完成并落盘后加一，并返回新值。"""

        self._count += 1
        return self._count


def run_read_node():
    """生成执行图中的阅读节点，直接返回异步节点实现。"""

    async def _node(state: State, runtime: Runtime[WorkflowRuntimeContext] | None) -> State:
        """阅读节点已经具备 async 主流程，图层改成 ainvoke 后这里直接 await 即可。"""

        return await _run_read_node_async(state, runtime)

    return _node


async def _run_read_node_async(state: State, runtime: Runtime[WorkflowRuntimeContext] | None) -> State:
    """并发处理多篇论文；论文级幂等由 note 负责，节点级恢复由官方 Checkpointer 负责。"""

    request = state.get("request")
    if request is None:
        raise ValueError("阅读节点缺少用户请求，无法判断论文主题")
    # 中文注释：State 里的 request 是纯 dict；领域对象只在节点内部使用。
    request_payload = request if isinstance(request, dict) else {}
    request_topic = str(request_payload.get("topic") or "")
    request_constraints = dict(request_payload.get("constraints") or {})
    request_fingerprint = read_request_fingerprint(request_payload)
    system_config = SystemConfig.load()
    config = system_config.read
    # 中文注释：State 里存的是 dict，阅读链路内部继续使用 PaperDocument，这里统一还原一次。
    # 论文列表只有当前检索结果这一个来源；续跑时由官方 Checkpointer 恢复 search_results。
    papers = _deduplicate_papers(_papers_from_payload(list(state.get("search_results") or [])))

    reporter = _resolve_reporter(runtime)
    sink = _resolve_sink(runtime, request_fingerprint=request_fingerprint)
    runtime_resources = _resolve_runtime_resources(runtime)
    llm = load_read_agent_llm(config.agent_name)
    if ENABLE_FULL_TEXT_EMBEDDING:
        embedding_connection, embedding_error = _resolve_embedding_connection(
            system_config,
            config.download_timeout_seconds,
            runtime_resources=runtime_resources,
        )
    else:
        # 中文说明：关闭向量嵌入时不读取 embedding 配置，避免配置缺失影响全文分块保存。
        embedding_connection, embedding_error = None, None

    # 中文注释：论文级恢复只有当前格式 note 这一个来源。state.read_results 只在
    # 节点完成时写回供下游消费，不作为恢复来源；全文中间产物按 paper_cache 文件复用。
    results_by_paper_id: dict[str, PaperReadResult] = {}
    artifact_refs: list[JsonObject] = list(state.get("read_artifact_refs") or [])
    note_stages, note_relevance = await _restore_paper_notes(
        sink,
        papers,
        results_by_paper_id=results_by_paper_id,
        artifact_refs=artifact_refs,
    )
    paper_runtime_statuses = _initial_paper_runtime_statuses(papers, results_by_paper_id)
    deep_read_limit = _deep_read_limit(request_constraints, len(papers))
    completed_counter = CompletedPaperCounter(len(results_by_paper_id))

    if reporter is not None:
        reporter.started(
            f"准备阅读 {len(papers)} 篇论文",
            stage="read_start",
            total=len(papers),
            completed=len(results_by_paper_id),
            resumed_paper_count=len(results_by_paper_id),
        )

    # 中文注释：资源不可用按普通失败处理——异常直接冒泡让 run 失败，已完成论文
    # 的 note 早已逐篇落盘；用户重试时由官方 Checkpointer 重跑本节点，note 跳过
    # 已完成论文。不再保存手写恢复现场。
    deep_read_count = await _process_papers_concurrently(
        papers=papers,
        topic=request_topic,
        constraints=request_constraints,
        config=config,
        llm=llm,
        embedding_connection=embedding_connection,
        embedding_error=embedding_error,
        reporter=reporter,
        sink=sink,
        runtime_resources=runtime_resources,
        results_by_paper_id=results_by_paper_id,
        artifact_refs=artifact_refs,
        note_stages=note_stages,
        note_relevance=note_relevance,
        paper_runtime_statuses=paper_runtime_statuses,
        completed_counter=completed_counter,
        deep_read_limit=deep_read_limit,
    )

    results = _ordered_results(papers, results_by_paper_id)
    summary = _build_summary(results, deep_read_count)
    if sink is not None:
        try:
            persisted = await asyncio.to_thread(sink.persist_summary, summary, results)
            _append_artifact_refs(artifact_refs, persisted.artifacts)
            summary["manifest"] = persisted.artifacts[0] if persisted.artifacts else {}
            if reporter is not None:
                for artifact in persisted.artifacts:
                    reporter.artifact(artifact, stage="read_artifact_ready")
        except Exception as exc:
            summary["persistence_error"] = str(exc)

    if papers and not any(result.failure is None for result in results):
        raise RuntimeError("论文阅读阶段失败：所有论文均未产出可用笔记")

    if reporter is not None:
        reporter.completed(
            "论文阅读节点已完成",
            stage="read_done",
            total=len(papers),
            completed=len(results),
            deep_read_paper_count=summary["deep_read_paper_count"],
            deep_read_papers=summary["deep_read_papers"],
            discarded_paper_count=summary["discarded_paper_count"],
            indexed_paper_count=summary["indexed_paper_count"],
        )
    # 中文注释：软丢弃的论文不进下游——汇总和产物里保留完整记录（含丢弃理由），
    # 但写回 state 的 read_results 只包含未被丢弃的论文，分析/写作节点自然收窄。
    kept_results = [result for result in results if result.failure is None and not result.relevance.discarded]
    # 中文注释：只返回本节点真正改动的键。
    updated: dict[str, Any] = {
        "read_results": [result.to_dict() for result in kept_results],
        "read_summary": summary,
        "read_artifact_refs": artifact_refs,
        "diagnostics": dict(state.get("diagnostics") or {}),
        "current_step": "read",
    }
    return cast(State, updated)


async def _restore_paper_notes(
    sink: ReadPersistenceSink | None,
    papers: list[PaperDocument],
    *,
    results_by_paper_id: dict[str, PaperReadResult],
    artifact_refs: list[JsonObject],
) -> tuple[dict[str, str], dict[str, JsonObject]]:
    """逐篇加载当前格式 note——论文级恢复的唯一来源。

    有效 note 直接进结果池；返回各篇的 stage 与 relevance，供全局选择后判断
    是否需要把新的选择结果写回磁盘。
    """

    restored_stages: dict[str, str] = {}
    restored_relevance: dict[str, JsonObject] = {}
    if sink is None:
        return restored_stages, restored_relevance
    for paper in papers:
        loaded = await asyncio.to_thread(sink.load_paper_note, paper.id)
        if loaded is None or loaded.stage not in {"abstract_complete", "fulltext_complete", "discarded"}:
            continue
        result = _read_result_from_payload(loaded.result, papers)
        if result is None or result.failure is not None:
            continue
        results_by_paper_id[paper.id] = result
        restored_stages[paper.id] = loaded.stage
        restored_relevance[paper.id] = result.relevance.to_dict()
        _append_artifact_refs(artifact_refs, [loaded.artifact])
    return restored_stages, restored_relevance


def _read_result_from_payload(value: Any, papers: list[PaperDocument]) -> PaperReadResult | None:
    """把 note 中单篇论文阅读结果恢复为 PaperReadResult。"""

    if not isinstance(value, dict):
        return None
    paper_payload = value.get("paper")
    paper = _paper_from_payload(paper_payload) if isinstance(paper_payload, dict) else None
    if paper is None:
        return None
    # 中文注释：优先使用当前论文列表中的对象，避免 note 里的旧字段覆盖新检索结果。
    papers_by_id = {item.id: item for item in papers}
    paper = papers_by_id.get(paper.id, paper)
    return PaperReadResult(
        paper=paper,
        note=_read_note_from_payload(value.get("note")),
        relevance=_read_relevance_from_payload(value.get("relevance")),
        full_text=_full_text_from_payload(value.get("full_text")),
        extraction=_json_object(value.get("extraction")),
        verification=_json_object(value.get("verification")),
        warnings=_string_list(value.get("warnings")),
        failure=_failure_from_payload(value.get("failure")),
    )


def _append_artifact_refs(artifact_refs: list[JsonObject], new_refs: list[JsonObject]) -> None:
    """按稳定 artifact_id 合并引用，摘要和全文阶段重复保存也不会累积重复项。"""

    positions = {
        str(item.get("artifact_id") or ""): index
        for index, item in enumerate(artifact_refs)
        if item.get("artifact_id")
    }
    for ref in new_refs:
        artifact_id = str(ref.get("artifact_id") or "")
        if artifact_id and artifact_id in positions:
            artifact_refs[positions[artifact_id]] = ref
        else:
            if artifact_id:
                positions[artifact_id] = len(artifact_refs)
            artifact_refs.append(ref)


def _papers_from_payload(value: Any) -> list[PaperDocument]:
    """把论文 JSON 列表恢复成 PaperDocument 列表。"""

    if not isinstance(value, list):
        return []
    papers: list[PaperDocument] = []
    for item in value:
        paper = _paper_from_payload(item)
        if paper is not None:
            papers.append(paper)
    return papers


def _paper_from_payload(value: Any) -> PaperDocument | None:
    """把论文 JSON 安全恢复为领域对象。"""

    if not isinstance(value, dict):
        return None
    paper_id = str(value.get("id") or "").strip()
    title = str(value.get("title") or "").strip()
    if not paper_id or not title:
        return None
    return PaperDocument(
        id=paper_id,
        title=title,
        authors=_string_list(value.get("authors")),
        abstract=str(value.get("abstract")) if value.get("abstract") is not None else None,
        year=_optional_int(value.get("year")),
        venue=str(value.get("venue")) if value.get("venue") is not None else None,
        url=str(value.get("url")) if value.get("url") is not None else None,
        pdf_url=str(value.get("pdf_url")) if value.get("pdf_url") is not None else None,
        doi=str(value.get("doi")) if value.get("doi") is not None else None,
        source=str(value.get("source")) if value.get("source") is not None else None,
        paperId=str(value.get("paperId")) if value.get("paperId") is not None else None,
        publication_date=str(value.get("publication_date") or ""),
        journal_conference=str(value.get("journal_conference") or value.get("journal/conference") or ""),
        volume=str(value.get("volume") or ""),
        issue=str(value.get("issue") or ""),
        language=str(value.get("language") or ""),
        metadata=dict(value.get("metadata") or {}) if isinstance(value.get("metadata"), dict) else {},
    )


def _read_note_from_payload(value: Any) -> ReadNote:
    """把 note 中的笔记 JSON 恢复为 ReadNote。"""

    payload = value if isinstance(value, dict) else {}
    return ReadNote(
        main_question=_text_value(payload.get("main_question")),
        methods=_string_list(payload.get("methods")),
        datasets=_string_list(payload.get("datasets")),
        contributions=_string_list(payload.get("contributions")),
        limitations=_string_list(payload.get("limitations")),
        main_results=_string_list(payload.get("main_results")),
        short_summary=_text_value(payload.get("short_summary")),
        evidence_level=_text_value(payload.get("evidence_level")) or "metadata",
    )


def _read_relevance_from_payload(value: Any) -> ReadRelevance:
    """把 note 中的相关性 JSON 恢复为 ReadRelevance。

    中文注释：结构是 {relevant, reason, discarded, uncertain, status}。
    relevant 缺失时按"拿不准判有关"保底；status 非法时回退 pending。
    """

    payload = value if isinstance(value, dict) else {}
    relevant = payload.get("relevant")
    if not isinstance(relevant, bool):
        relevant = True
    status = _text_value(payload.get("status")) or "pending"
    if status not in READ_RELEVANCE_STATUSES:
        status = "pending"
    return ReadRelevance(
        relevant=relevant,
        reason=_text_value(payload.get("reason")),
        discarded=bool(payload.get("discarded")),
        uncertain=bool(payload.get("uncertain")),
        status=status,
    )


def _full_text_from_payload(value: Any) -> FullTextStatus:
    """把 note 中的全文状态 JSON 恢复为 FullTextStatus。"""

    payload = value if isinstance(value, dict) else {}
    return FullTextStatus(
        status=_text_value(payload.get("status")) or "not_requested",
        reason=_text_value(payload.get("reason")),
        source_url=_optional_text(payload.get("source_url")),
        source_path=_optional_text(payload.get("source_path")),
        markdown_path=_optional_text(payload.get("markdown_path")),
        page_count=_optional_int(payload.get("page_count")),
        chunk_count=_score_value(payload.get("chunk_count")),
    )


def _optional_text(value: Any) -> str | None:
    """把可选字段恢复为非空字符串或 None。"""

    text = str(value).strip() if isinstance(value, str) else ""
    return text or None


def _optional_int(value: Any) -> int | None:
    """把可选数字字段恢复为整数或 None。"""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_object(value: Any) -> JsonObject:
    """把可能缺失的字段整理成普通字典。

    中文注释：note 内容损坏时不能假设字段一定规整。如果 extraction 不是字典，
    就返回空字典，避免一篇坏记录影响整批阅读恢复。
    """

    return dict(value) if isinstance(value, dict) else {}


def _failure_from_payload(value: Any) -> JsonObject | None:
    """failure 缺失或不是字典时保持 None，避免把缺失字段误判成失败。"""

    return dict(value) if isinstance(value, dict) else None


def _read_agent_from_llm(llm: ProviderSnapshot | None, usage_callback: Any | None = None):
    """根据节点已经解析出的模型快照创建阅读 Agent。"""

    return build_read_agent(llm, usage_callback=usage_callback)


# 中文说明：全文精读预算的默认值。相关论文不超过这个数时全部精读（零新增调用）；
# 超过时触发一次批量比较选择。用户可通过 constraints 的 deep_read_limit 覆盖。
DEFAULT_DEEP_READ_LIMIT = 10


def _deep_read_limit(constraints: JsonObject, total: int) -> int:
    """读取用户可选的精读数量限制，未提供时默认最多精读 10 篇。"""

    raw = constraints.get("deep_read_limit", constraints.get("max_deep_read", DEFAULT_DEEP_READ_LIMIT))
    try:
        return max(0, min(int(raw), total))
    except (TypeError, ValueError):
        return min(DEFAULT_DEEP_READ_LIMIT, total)


def _deduplicate_papers(papers: list[PaperDocument]) -> list[PaperDocument]:
    """按 DOI 优先、标题其次去除重复论文，保持检索节点原有排序。"""

    seen: set[str] = set()
    unique: list[PaperDocument] = []
    for paper in papers:
        key = f"doi:{paper.doi.strip().lower()}" if (paper.doi or "").strip() else f"title:{paper.title.strip().lower()}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(paper)
    return unique

def _resolve_runtime_resources(runtime: Runtime[WorkflowRuntimeContext] | None) -> WorkflowRuntimeResources | None:
    """从 graph context 里取出单次 run 共享的资源对象。"""

    context = runtime.context if runtime is not None else None
    if context is None:
        return None
    return context.resources if isinstance(context.resources, WorkflowRuntimeResources) else None


def _resolve_embedding_connection(
    system_config: SystemConfig,
    timeout_seconds: int,
    *,
    runtime_resources: WorkflowRuntimeResources | None = None,
) -> tuple[EmbeddingConnection | None, str | None]:
    """从运行时资源或本地模型配置中读取 embedding 服务信息。"""

    if runtime_resources is not None:
        if runtime_resources.embedding_connection is not None:
            return runtime_resources.embedding_connection, runtime_resources.embedding_error
        if runtime_resources.embedding_error:
            return None, runtime_resources.embedding_error
    model_path = Path("config/model.json")
    if not model_path.exists():
        error = "未找到模型配置，全文尚未写入向量库"
        if runtime_resources is not None:
            runtime_resources.embedding_error = error
        return None, error
    try:
        model_config = ModelConfig.from_dict(json.loads(model_path.read_text(encoding="utf-8")), system_config)
        profile = model_config.resolve_embedding_profile()
        snapshot = make_provider(model_config, embedding_profile_name=model_config.default_embedding_profile, timeout_s=float(max(1, timeout_seconds)))
        connection = EmbeddingConnection(
            provider=snapshot.provider,
            model_name=profile.model_name,
            dimensions=profile.dimensions,
            batch_size=int(profile.batch_size or 32),
        )
        if runtime_resources is not None:
            runtime_resources.embedding_snapshot = snapshot
            runtime_resources.embedding_connection = connection
            runtime_resources.embedding_error = None
        return connection, None
    except Exception as exc:
        error = f"embedding 服务配置不可用：{exc}"
        if runtime_resources is not None:
            runtime_resources.embedding_error = error
        return None, error


def _resolve_sink(
    runtime: Runtime[WorkflowRuntimeContext] | None,
    *,
    request_fingerprint: str,
) -> ReadPersistenceSink | None:
    """仅在运行时上下文完整时启用阅读产物写盘，普通脚本调用仍可直接运行。"""

    context = runtime.context if runtime is not None else None
    if context is None:
        return None
    repo = context.session_repo
    session_key = str(context.session_key or "").strip()
    turn_id = str(context.turn_id or "").strip()
    if not isinstance(repo, SessionRepository) or not session_key or not turn_id:
        return None
    thread_id = str(context.thread_id or "").strip() or f"{session_key}:{turn_id}"
    return ReadPersistenceSink(
        repo,
        session_key=session_key,
        turn_id=turn_id,
        thread_id=thread_id,
        request_fingerprint=request_fingerprint,
    )


def _resolve_reporter(runtime: Runtime[WorkflowRuntimeContext] | None):
    """从 graph context 里取出阅读节点上报器，没有前端同步接口时返回空值。"""

    context = runtime.context if runtime is not None else None
    if context is None or context.sync_port is None:
        return None
    return context.sync_port.for_node("read", "论文阅读")


def _build_summary(results: list[PaperReadResult], deep_read_count: int) -> JsonObject:
    """汇总每篇论文的处理结果；彻底失败与可用但部分失败分开统计。"""

    failed_results = [item for item in results if item.failure is not None]
    usable_results = [item for item in results if item.failure is None]
    global_statistics = {
        "total_papers_received": len(results),
        "passed_abstract_filter": sum(not item.relevance.discarded for item in usable_results),
        "fulltext_downloaded": sum(bool(item.full_text.source_path) for item in results),
        "extraction_succeeded": sum(bool(item.extraction and any(str(value).strip() for value in item.extraction.values())) for item in results),
        "vectorized": sum(item.full_text.status == "indexed" for item in results),
        # 中文说明：这里的统计只反映“是否跑过验证”，与验证结果好坏分开看。
        "verified_paper_count": sum(bool(item.verification and item.verification.get("status") == "completed") for item in results),
        "verification_failed_paper_count": sum(bool(item.verification and not item.verification.get("passed")) for item in results),
        "errors": _read_errors(results),
    }
    return {
        "total_paper_count": len(results),
        "deep_read_attempt_count": deep_read_count,
        "deep_read_candidate_count": sum(not item.relevance.discarded for item in usable_results),
        # 中文说明：只要被统一排序选中，就保留在精读清单中；下载、解析或建立
        # 向量索引失败不会改变它已经获得全文名额这一事实。
        "deep_read_paper_count": sum(item.relevance.status == "selected_for_deep_read" for item in results),
        "deep_read_papers": [
            _deep_read_paper_item(item)
            for item in results
            if item.relevance.status == "selected_for_deep_read"
        ],
        # 中文说明：软丢弃清单单独汇总，方便审计"哪些论文被 AI 判无关、为什么"。
        "discarded_paper_count": sum(item.relevance.discarded for item in results),
        "discarded_papers": [
            {
                "paperId": item.paper.paperId or item.paper.id,
                "title": item.paper.title,
                "reason": item.relevance.reason,
            }
            for item in results
            if item.relevance.discarded
        ],
        "failed_paper_count": len(failed_results),
        "failed_papers": [
            {
                "paperId": item.paper.paperId or item.paper.id,
                "title": item.paper.title,
                **dict(item.failure or {}),
            }
            for item in failed_results
        ],
        "uncertain_paper_count": sum(item.relevance.uncertain for item in usable_results),
        "indexed_paper_count": sum(item.full_text.status == "indexed" for item in results),
        "failed_fulltext_count": sum(item.full_text.status in {"download_failed", "parse_failed"} for item in results),
        "subtopics": _read_subtopics(usable_results),
        "global_statistics": global_statistics,
    }



def _text_value(value: Any) -> str:
    """把可能为空的模型字段安全整理为字符串。"""

    return str(value).strip() if isinstance(value, str) else ""


def _read_subtopics(results: list[PaperReadResult]) -> list[JsonObject]:
    """按检索子主题整理阅读结果。

    中文注释：检索节点会把论文来自哪个子主题放进 metadata.search_subtopics。
    如果没有这个信息，就放到“综合阅读”里，保证输出结构稳定。
    """

    grouped: dict[str, list[JsonObject]] = {}
    for result in results:
        names = _paper_subtopic_names(result.paper) or ["综合阅读"]
        for name in names:
            grouped.setdefault(name, []).append(_summary_paper_item(result))
    return [{"subtopic": name, "papers": papers} for name, papers in grouped.items()]


def _paper_subtopic_names(paper: PaperDocument) -> list[str]:
    """从论文 metadata 中取出子主题名称。"""

    origins = paper.metadata.get("search_subtopics") if isinstance(paper.metadata, dict) else []
    if not isinstance(origins, list):
        return []
    names: list[str] = []
    for origin in origins:
        if not isinstance(origin, dict):
            continue
        name = str(origin.get("subtopic") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _summary_paper_item(result: PaperReadResult) -> JsonObject:
    """整理 read_summary.subtopics[].papers[] 中的单篇论文。"""

    return {
        "paperId": result.paper.paperId or result.paper.id,
        "title": result.paper.title,
        "year": result.paper.year,
        "extraction": dict(result.extraction or empty_extraction()),
        "processing_status": "completed" if result.full_text.status in _FULL_TEXT_COMPLETED_STATUSES else result.full_text.status,
    }


def _deep_read_paper_item(result: PaperReadResult) -> JsonObject:
    """整理已经进入全文精读流程的论文，供汇总信息和前端详情展示。"""

    # 中文说明：这里同时保留全文状态，方便用户看出“已精读但未成功建立索引”的情况。
    return {
        "paperId": result.paper.paperId or result.paper.id,
        "title": result.paper.title,
        "year": result.paper.year,
        "relevance_reason": result.relevance.reason,
        "uncertain": result.relevance.uncertain,
        "selection_status": result.relevance.status,
        "full_text_status": result.full_text.status,
    }


def _paper_completion_runtime_status(result: PaperReadResult) -> str:
    """根据单篇论文结果判断卡片最终显示完成、丢弃还是失败。"""

    # 中文注释：软丢弃优先判断——被 AI 判无关的论文卡片显示"已丢弃"，
    # 不算失败（判断本身是正常业务结果），也不算完成（没有进入下游）。
    if result.relevance.discarded:
        return "discarded"
    # 中文注释：下载、转换、索引失败不再让整批阅读中断，
    # 但对这篇论文来说确实有阶段失败，所以卡片用 failed 更醒目。
    if result.full_text.status in {"download_failed", "no_url", "parse_failed"}:
        return "failed"
    if result.full_text.reason and result.full_text.status not in _FULL_TEXT_COMPLETED_STATUSES | {"not_requested"}:
        return "failed"
    if any("失败" in warning or "无法" in warning for warning in result.warnings):
        return "failed"
    return "completed"


def _paper_completion_message(result: PaperReadResult) -> str:
    """把单篇论文最终结果整理成用户能看懂的一句话。"""

    # 中文注释：丢弃卡片直接展示 AI 给出的判断理由，方便用户核对是否误杀。
    if result.relevance.discarded:
        return f"已丢弃：{result.relevance.reason}" if result.relevance.reason else "已丢弃：判定与主题无关"
    status = result.full_text.status
    if status in {"download_failed", "no_url"}:
        return "全文下载失败，已保留摘要阅读结果"
    if status == "parse_failed":
        return "Markdown 转换失败，已保留摘要阅读结果"
    if result.full_text.reason and status not in _FULL_TEXT_COMPLETED_STATUSES | {"not_requested"}:
        return "全文索引失败，已保留阅读结果"
    if any("全文结构化提取失败" in warning for warning in result.warnings):
        return "全文结构化信息提取失败，已保存基础阅读结果"
    if status == "not_requested":
        return result.full_text.reason or "摘要阅读完成，当前论文不需要全文精读"
    return "论文阅读完成"


def _paper_completion_error_message(result: PaperReadResult) -> str:
    """提取单篇论文最终失败原因，没有失败时返回空字符串。"""

    # 中文注释：软丢弃不是错误，不产生 error 信息。
    if result.relevance.discarded:
        return ""
    if result.full_text.status in {"download_failed", "no_url", "parse_failed"}:
        return result.full_text.reason or result.full_text.status
    if result.full_text.reason and result.full_text.status not in _FULL_TEXT_COMPLETED_STATUSES | {"not_requested"}:
        return result.full_text.reason
    for warning in result.warnings:
        if "失败" in warning or "无法" in warning:
            return warning
    return ""


def _read_errors(results: list[PaperReadResult]) -> list[JsonObject]:
    """把下载、解析、提取和向量化失败整理成统一错误列表。"""

    errors: list[JsonObject] = []
    for result in results:
        # 中文注释：被软丢弃的论文不算错误，跳过统计。
        if result.relevance.discarded:
            continue
        paper_id = result.paper.paperId or result.paper.id
        status = result.full_text.status
        if status in {"download_failed", "no_url"}:
            errors.append({"paperId": paper_id, "stage": "download", "reason": result.full_text.reason})
        elif status == "parse_failed":
            errors.append({"paperId": paper_id, "stage": "parse", "reason": result.full_text.reason})
        elif result.full_text.reason and status not in _FULL_TEXT_COMPLETED_STATUSES:
            errors.append({"paperId": paper_id, "stage": "vectorize", "reason": result.full_text.reason})
        for warning in result.warnings:
            if "全文结构化提取失败" in warning:
                errors.append({"paperId": paper_id, "stage": "extraction", "reason": warning})
    return errors


def _string_list(value: Any) -> list[str]:
    """只保留模型返回列表中的非空字符串，避免错误类型进入结构化笔记。"""

    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _score_value(value: Any) -> int:
    """把模型相关性分数限制到 0 到 100，异常值统一当作零分。"""

    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


_FALLBACK_PAPER_TASK_SEMAPHORE = asyncio.Semaphore(3)
_FALLBACK_READ_MODEL_SEMAPHORE = asyncio.Semaphore(2)


async def _process_papers_concurrently(
    *,
    papers: list[PaperDocument],
    topic: str,
    constraints: JsonObject,
    config: Any,
    llm: ProviderSnapshot | None,
    embedding_connection: EmbeddingConnection | None,
    embedding_error: str | None,
    reporter: Any,
    sink: ReadPersistenceSink | None,
    runtime_resources: WorkflowRuntimeResources | None,
    results_by_paper_id: dict[str, PaperReadResult],
    artifact_refs: list[JsonObject],
    note_stages: dict[str, str],
    note_relevance: dict[str, JsonObject],
    paper_runtime_statuses: dict[str, JsonObject],
    completed_counter: CompletedPaperCounter,
    deep_read_limit: int,
) -> int:
    """先并发阅读全部摘要，再统一选择全文精读论文。"""

    # 中文说明：第一阶段只调用一次摘要模型，不会下载任何全文。这样模型较早返回的论文
    # 也不能提前占用全文名额，所有论文都拿到固定分数后才会一起排序。
    abstract_items = [
        PaperTaskInput(paper=paper, position=position)
        for position, paper in enumerate(papers, start=1)
        if paper.id not in results_by_paper_id or results_by_paper_id[paper.id].failure is not None
    ]
    await _run_paper_task_batch(
        abstract_items,
        topic=topic,
        constraints=constraints,
        config=config,
        llm=llm,
        embedding_connection=embedding_connection,
        embedding_error=embedding_error,
        reporter=reporter,
        sink=sink,
        runtime_resources=runtime_resources,
        results_by_paper_id=results_by_paper_id,
        artifact_refs=artifact_refs,
        paper_runtime_statuses=paper_runtime_statuses,
        completed_counter=completed_counter,
        total_paper_count=len(papers),
        process_full_text=False,
    )

    needs_selection = bool(abstract_items) or any(
        result.failure is None and result.relevance.status == "pending"
        for result in results_by_paper_id.values()
    )
    if needs_selection:
        selected_paper_ids = await _assign_deep_read_selection(
            papers,
            results_by_paper_id,
            paper_runtime_statuses,
            deep_read_limit,
            topic=topic,
            llm=llm,
        )
    else:
        # 所有笔记都已带着选择结果恢复时，不再额外调用一次全局比较模型。
        selected_paper_ids = {
            paper_id
            for paper_id, result in results_by_paper_id.items()
            if result.failure is None and result.relevance.status == "selected_for_deep_read"
        }

    # 全局选择可能改变恢复笔记的 relevance；只有 stage 和选择内容都没变才跳过写盘。
    for position, paper in enumerate(papers, start=1):
        result = results_by_paper_id[paper.id]
        note_is_current = (
            paper.id in note_stages
            and note_relevance.get(paper.id) == result.relevance.to_dict()
        )
        if paper.id not in selected_paper_ids:
            if note_is_current:
                continue
            await _finalize_paper_result(
                result,
                paper_position=position,
                results_by_paper_id=results_by_paper_id,
                paper_runtime_statuses=paper_runtime_statuses,
                completed_counter=completed_counter,
                total_paper_count=len(papers),
                reporter=reporter,
                sink=sink,
                artifact_refs=artifact_refs,
            )
        elif not note_is_current:
            # 新摘要或全局选择改变了 relevance 时，把最新稳定点写回 note。
            await _persist_note_async(
                result,
                stage="abstract_complete" if _deep_read_needs_processing(result) else None,
                sink=sink,
                artifact_refs=artifact_refs,
            )

    # 中文说明：第二阶段只处理统一排序选中的论文。恢复运行时，已经下载或转换过的
    # 论文会从保存的文件继续，而不是再次调用摘要模型。
    deep_read_items = [
        PaperTaskInput(
            paper=paper,
            position=position,
            restored_result=results_by_paper_id[paper.id],
        )
        for position, paper in enumerate(papers, start=1)
        if paper.id in selected_paper_ids and _deep_read_needs_processing(results_by_paper_id[paper.id])
    ]
    await _run_paper_task_batch(
        deep_read_items,
        topic=topic,
        constraints=constraints,
        config=config,
        llm=llm,
        embedding_connection=embedding_connection,
        embedding_error=embedding_error,
        reporter=reporter,
        sink=sink,
        runtime_resources=runtime_resources,
        results_by_paper_id=results_by_paper_id,
        artifact_refs=artifact_refs,
        paper_runtime_statuses=paper_runtime_statuses,
        completed_counter=completed_counter,
        total_paper_count=len(papers),
        process_full_text=True,
    )
    return len(selected_paper_ids)


async def _run_paper_task_batch(
    work_items: list[PaperTaskInput],
    *,
    topic: str,
    constraints: JsonObject,
    config: Any,
    llm: ProviderSnapshot | None,
    embedding_connection: EmbeddingConnection | None,
    embedding_error: str | None,
    reporter: Any,
    sink: ReadPersistenceSink | None,
    runtime_resources: WorkflowRuntimeResources | None,
    results_by_paper_id: dict[str, PaperReadResult],
    artifact_refs: list[JsonObject],
    paper_runtime_statuses: dict[str, JsonObject],
    completed_counter: CompletedPaperCounter,
    total_paper_count: int,
    process_full_text: bool,
) -> None:
    """执行一个阶段的论文任务；资源不可用时取消同批任务并冒泡原始异常。"""

    if not work_items:
        return
    semaphore = runtime_resources.paper_task_semaphore if runtime_resources is not None else _FALLBACK_PAPER_TASK_SEMAPHORE
    tasks = {
        asyncio.create_task(
            _run_one_paper_task(
                item,
                topic=topic,
                constraints=constraints,
                config=config,
                llm=llm,
                embedding_connection=embedding_connection,
                embedding_error=embedding_error,
                reporter=reporter,
                runtime_resources=runtime_resources,
                paper_runtime_statuses=paper_runtime_statuses,
                completed_counter=completed_counter,
                total_paper_count=total_paper_count,
                semaphore=semaphore,
                process_full_text=process_full_text,
            )
        ): item
        for item in work_items
    }
    pending = set(tasks)
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        resource_error: ReadResourceUnavailableError | None = None
        for task in done:
            item = tasks[task]
            try:
                result = task.result()
            except ReadResourceUnavailableError as exc:
                resource_error = resource_error or exc
                continue
            results_by_paper_id[item.paper.id] = result
            if not process_full_text:
                _set_paper_runtime_status(
                    paper_runtime_statuses,
                    item.paper,
                    status="failed" if result.failure is not None else "running",
                    current_stage="abstract_failed" if result.failure is not None else "abstract_ready",
                    result=result,
                    error_message=str((result.failure or {}).get("message") or ""),
                )
                # 每篇摘要一完成就落盘，不能等全部摘要和全局选择结束。
                await _persist_note_async(
                    result,
                    stage=None if result.failure is not None else "abstract_complete",
                    sink=sink,
                    artifact_refs=artifact_refs,
                )
                continue
            await _finalize_paper_result(
                result,
                paper_position=item.position,
                results_by_paper_id=results_by_paper_id,
                paper_runtime_statuses=paper_runtime_statuses,
                completed_counter=completed_counter,
                total_paper_count=total_paper_count,
                reporter=reporter,
                sink=sink,
                artifact_refs=artifact_refs,
            )
        if resource_error is not None:
            # 中文注释：资源不可用时取消同批其余任务，避免继续浪费模型调用，
            # 然后把原始异常冒泡成普通节点失败（fail-fast 并发控制）。
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise resource_error


async def _finalize_paper_result(
    result: PaperReadResult,
    *,
    paper_position: int,
    results_by_paper_id: dict[str, PaperReadResult],
    paper_runtime_statuses: dict[str, JsonObject],
    completed_counter: CompletedPaperCounter,
    total_paper_count: int,
    reporter: Any,
    sink: ReadPersistenceSink | None,
    artifact_refs: list[JsonObject],
) -> None:
    """保存一篇最终结果，并更新前端看到的完成进度。"""

    results_by_paper_id[result.paper.id] = result
    completion_status = _paper_completion_runtime_status(result)
    completion_message = _paper_completion_message(result)
    completion_error = _paper_completion_error_message(result)
    _set_paper_runtime_status(
        paper_runtime_statuses,
        result.paper,
        status=completion_status,
        current_stage="paper_completed",
        result=result,
        error_message=completion_error,
    )
    completed = completed_counter.increment()
    _report_progress(
        reporter,
        result.paper,
        "paper_completed",
        completed,
        total_paper_count,
        runtime_status=completion_status,
        message=completion_message,
        current_status=result.full_text.status,
        paper_position=paper_position,
        error_message=completion_error,
    )
    _, save_error = await _persist_completed_paper_async(
        result,
        sink=sink,
        artifact_refs=artifact_refs,
        reporter=reporter,
        completed=completed,
        total=total_paper_count,
        paper_position=paper_position,
    )
    _set_paper_runtime_status(
        paper_runtime_statuses,
        result.paper,
        # 中文注释：软丢弃的论文卡片状态保持 discarded，不伪装成 completed；
        # 保存出错或阶段失败才是 failed，其余为 completed。
        status=(
            "failed"
            if save_error or completion_status == "failed"
            else "discarded"
            if completion_status == "discarded"
            else "completed"
        ),
        current_stage="paper_artifact_ready",
        result=result,
        error_message=save_error or completion_error,
    )


# 中文说明：保底回捞阈值——判相关的论文少于这个数时，从软丢弃池按顺序回捞，
# 避免窄主题下因误判导致最终报告空壳。
MIN_USABLE_PAPERS = 3


async def _assign_deep_read_selection(
    papers: list[PaperDocument],
    results_by_paper_id: dict[str, PaperReadResult],
    paper_runtime_statuses: dict[str, JsonObject],
    deep_read_limit: int,
    *,
    topic: str,
    llm: ProviderSnapshot | None,
) -> set[str]:
    """摘要阅读后统一分配去留与全文精读名额。

    中文注释：新版不再按分数排序，而是三步：
    1. 模型判无关（relevant=False）→ 软丢弃（标记 discarded，不进下游，笔记照常落盘）；
    2. 判相关的论文少于保底数量 → 从丢弃池按候选池顺序回捞，标注"相关性存疑"；
    3. 相关论文超过精读预算 → 触发一次批量比较，由模型在同一上下文里挑出精读对象；
       不超过预算则全部精读，零新增调用。
    """

    candidates: list[tuple[int, PaperReadResult]] = []
    discarded_items: list[tuple[int, PaperReadResult]] = []
    for position, paper in enumerate(papers, start=1):
        result = results_by_paper_id[paper.id]
        if result.failure is not None:
            # 没有可用笔记的论文不参与候选、回捞或精读名额分配。
            continue
        if not result.relevance.relevant:
            # 软丢弃：只标记，不删除对象；note.json 会带 discarded 字段照常落盘。
            result.relevance.discarded = True
            result.relevance.status = "discarded"
            result.full_text = FullTextStatus(status="not_requested", reason="判定与主题无关，已丢弃")
            discarded_items.append((position, result))
            continue
        candidates.append((position, result))

    # 保底回捞：可用论文太少时按候选池顺序从丢弃池捞回，标注存疑。
    if len(candidates) < MIN_USABLE_PAPERS and discarded_items:
        for position, result in list(discarded_items):
            if len(candidates) >= MIN_USABLE_PAPERS:
                break
            discarded_items.remove((position, result))
            result.relevance.discarded = False
            result.relevance.uncertain = True
            result.relevance.status = "pending"
            result.relevance.reason = f"{result.relevance.reason}；回捞保留，相关性存疑"
            result.full_text = FullTextStatus(status="not_requested", reason="")
            candidates.append((position, result))

    # 精读名额分配：预算内全取；超预算时一次批量比较选择。
    candidate_ids = [result.paper.id for _, result in candidates]
    if len(candidates) <= deep_read_limit:
        selected_paper_ids = set(candidate_ids)
    else:
        selected_paper_ids = await _select_deep_read_by_llm(
            topic=topic,
            candidates=[result for _, result in candidates],
            limit=deep_read_limit,
            llm=llm,
        )
        # 模型比较失败时 _select_deep_read_by_llm 已回退为池内顺序取前 N。
    for _, result in candidates:
        if result.paper.id not in selected_paper_ids:
            result.relevance.status = "not_selected_due_to_limit"
            result.full_text = FullTextStatus(status="not_requested", reason="全文精读名额已分配给更贴近主题的论文")
            continue
        result.relevance.status = "selected_for_deep_read"
        current = dict(paper_runtime_statuses.get(result.paper.id) or {})
        if _deep_read_needs_processing(result) and not str(current.get("status") or "").startswith("waiting_"):
            _set_paper_runtime_status(
                paper_runtime_statuses,
                result.paper,
                status="pending",
                current_stage="selected_for_deep_read",
                result=result,
                error_message="",
                deep_read_reserved=True,
            )
    return selected_paper_ids


async def _select_deep_read_by_llm(
    *,
    topic: str,
    candidates: list[PaperReadResult],
    limit: int,
    llm: ProviderSnapshot | None,
) -> set[str]:
    """相关论文超出精读预算时，让模型在同一上下文里比较并挑出精读对象。

    中文注释：这是"一次批量比较"而不是"逐篇打分"——所有候选放进同一个请求，
    模型可以直接对比它们的相对价值。任何环节失败都回退为按候选池顺序取前 N 篇，
    保证精读流程不会因为比较调用出错而中断。
    """

    fallback_ids = {result.paper.id for result in candidates[:limit]}
    if llm is None:
        return fallback_ids
    lines = []
    for index, result in enumerate(candidates):
        summary = (result.note.short_summary or "").strip() or (result.paper.abstract or "").strip()[:150]
        lines.append(f"[{index}] {result.paper.title}\n    {summary}")
    user_prompt = (
        f"用户研究主题：{topic}\n\n"
        f"精读名额：{limit}\n\n以下是全部相关论文的编号、标题和摘要要点：\n\n" + "\n".join(lines)
    )
    messages: list[JsonObject] = [
        {"role": "system", "content": READ_DEEP_READ_SELECTION_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        content = await _call_read_llm(llm.provider, messages)
    except Exception:
        # 比较调用失败时按池内顺序回退，宁可按原排序精读，也不中断流程。
        return fallback_ids
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return fallback_ids
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return fallback_ids
    selected_indices = payload.get("selected") if isinstance(payload, dict) else None
    if not isinstance(selected_indices, list):
        return fallback_ids
    selected_ids = {
        candidates[index].paper.id
        for index in selected_indices
        if isinstance(index, int) and 0 <= index < len(candidates)
    }
    if not selected_ids:
        return fallback_ids
    return set(list(selected_ids)[:limit])


async def _call_read_llm(provider: Any, messages: list[JsonObject]) -> str:
    """调用阅读侧模型并取回文本内容，优先走异步 chat 接口。"""

    chat = getattr(provider, "chat", None)
    if callable(chat):
        result = chat(messages, temperature=0)
        if inspect.isawaitable(result):
            result = await result
        return getattr(result, "content", "") or ""
    result = await asyncio.to_thread(provider.chat_with_retry, messages, temperature=0)
    return getattr(result, "content", "") or ""


def _deep_read_needs_processing(result: PaperReadResult) -> bool:
    """判断已选中论文是否还需要继续全文流程。"""

    # 中文说明：下载、转换失败已经是这一轮的最终结果；Markdown 已生成或正在等待
    # embedding 时则返回真，让恢复流程从现有文件继续。
    if result.full_text.status in _FULL_TEXT_COMPLETED_STATUSES:
        # 中文注释：只有切块文件、还没完成全文结构化摘要时不算全部完成；
        # 只有笔记明确标记为 full_text 后，才把这篇精读论文视为全部完成。
        return result.note.evidence_level != "full_text"
    return result.full_text.status not in {"download_failed", "parse_failed", "no_url"}


async def _run_one_paper_task(
    item: PaperTaskInput,
    *,
    topic: str,
    constraints: JsonObject,
    config: Any,
    llm: ProviderSnapshot | None,
    embedding_connection: EmbeddingConnection | None,
    embedding_error: str | None,
    reporter: Any,
    runtime_resources: WorkflowRuntimeResources | None,
    paper_runtime_statuses: dict[str, JsonObject],
    completed_counter: CompletedPaperCounter,
    total_paper_count: int,
    semaphore: asyncio.Semaphore,
    process_full_text: bool,
) -> PaperReadResult:
    """在线程池外层控制论文级并发，并把单篇异常隔离在当前论文内部。"""

    async with semaphore:
        try:
            return await _read_one_paper(
                item,
                topic=topic,
                constraints=constraints,
                llm=llm,
                config=config,
                embedding_connection=embedding_connection,
                embedding_error=embedding_error,
                reporter=reporter,
                runtime_resources=runtime_resources,
                paper_runtime_statuses=paper_runtime_statuses,
                completed_counter=completed_counter,
                total_paper_count=total_paper_count,
                process_full_text=process_full_text,
            )
        except ReadResourceUnavailableError as exc:
            _mark_paper_waiting_resource(paper_runtime_statuses, item.paper, exc)
            # 中文注释：资源不可用会让整个阅读节点按普通失败中断。这里先更新当前
            # 论文卡片，让用户一眼看到具体是哪篇论文卡在模型或向量服务上。
            _report_progress(
                reporter,
                item.paper,
                _paper_stage_from_status(paper_runtime_statuses.get(item.paper.id)),
                completed_counter.current(),
                total_paper_count,
                runtime_status="failed",
                message=str(exc),
                paper_position=item.position,
                error_message=str(exc),
            )
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = f"当前论文处理失败：{exc}"
            if process_full_text and item.restored_result is not None:
                # 全文异常但摘要笔记仍可用，属于部分失败；保留摘要并允许进入下游。
                result = item.restored_result
                result.full_text.status = "parse_failed"
                result.full_text.reason = message
                result.warnings.append(message)
                result.failure = None
            else:
                result = PaperReadResult(
                    paper=item.paper,
                    full_text=FullTextStatus(status="not_requested", reason="当前论文处理发生异常"),
                    warnings=[message],
                    failure={
                        "stage": "fulltext" if process_full_text else "abstract",
                        "error_kind": type(exc).__name__,
                        "message": str(exc),
                        "attempts": 1,
                    },
                )
            _set_paper_runtime_status(
                paper_runtime_statuses,
                item.paper,
                status="failed",
                current_stage=_paper_stage_from_status(paper_runtime_statuses.get(item.paper.id)),
                result=result,
                error_message=message,
            )
            _report_progress(
                reporter,
                item.paper,
                _paper_stage_from_status(paper_runtime_statuses.get(item.paper.id)),
                completed_counter.current(),
                total_paper_count,
                runtime_status="failed",
                message="当前论文处理失败",
                current_status=result.full_text.status,
                paper_position=item.position,
                error_message=message,
            )
            return result


async def _read_one_paper(
    item: PaperTaskInput,
    *,
    topic: str,
    constraints: JsonObject,
    llm: ProviderSnapshot | None,
    config: Any,
    embedding_connection: EmbeddingConnection | None,
    embedding_error: str | None,
    reporter: Any,
    runtime_resources: WorkflowRuntimeResources | None,
    paper_runtime_statuses: dict[str, JsonObject],
    completed_counter: CompletedPaperCounter,
    total_paper_count: int,
    process_full_text: bool,
) -> PaperReadResult:
    """执行摘要阶段或已选中论文的全文阶段。"""

    paper = item.paper
    title = paper.title.strip()
    paper_usage = {"input_tokens": 0, "output_tokens": 0}
    if not title:
        result = PaperReadResult(
            paper=paper,
            full_text=FullTextStatus(status="not_requested", reason="论文没有标题，无法可靠阅读"),
            warnings=["论文缺少标题，未交给模型判断"],
            failure={
                "stage": "validation",
                "error_kind": "ValueError",
                "message": "论文缺少标题，未交给模型判断",
                "attempts": 1,
            },
        )
        _report_progress(
            reporter,
            paper,
            "paper_completed",
            completed_counter.current(),
            total_paper_count,
            runtime_status="failed",
            message="论文缺少标题，无法阅读",
            current_status=result.full_text.status,
            paper_position=item.position,
            error_message="论文缺少标题，未交给模型判断",
        )
        return result

    result = item.restored_result
    if not process_full_text:
        _set_paper_runtime_status(
            paper_runtime_statuses,
            paper,
            status="running",
            current_stage="reading_abstract",
            result=result,
            error_message="",
        )
        _report_progress(
            reporter,
            paper,
            "reading_abstract",
            completed_counter.current(),
            total_paper_count,
            paper_position=item.position,
        )
        def report_abstract_usage(usage: JsonObject) -> None:
            """把单篇论文摘要模型的真实 token 用量写入该论文卡片。"""

            paper_usage["input_tokens"] += int(usage.get("input_tokens") or 0)
            paper_usage["output_tokens"] += int(usage.get("output_tokens") or 0)
            _report_progress(
                reporter,
                paper,
                "reading_abstract",
                completed_counter.current(),
                total_paper_count,
                **paper_usage,
            )

        note, relevance, warnings = await _build_abstract_note(
            paper,
            topic=topic,
            constraints=constraints,
            llm=llm,
            runtime_resources=runtime_resources,
            usage_callback=report_abstract_usage,
        )
        result = PaperReadResult(paper=paper, note=note, relevance=relevance, warnings=warnings)
        return result

    if result is None:
        raise ValueError("全文阶段缺少摘要阅读结果")
    if result.relevance.status != "selected_for_deep_read":
        return result

    _set_paper_runtime_status(
        paper_runtime_statuses,
        paper,
        status="running",
        current_stage="selected_for_deep_read",
        result=result,
        error_message="",
        deep_read_reserved=True,
    )

    full_text = result.full_text
    source_path = Path(full_text.source_path) if full_text.source_path else None
    if source_path is None or not source_path.is_file():
        _set_paper_runtime_status(
            paper_runtime_statuses,
            paper,
            status="running",
            current_stage="downloading_full_text",
            result=result,
            error_message="",
        )
        _report_progress(
            reporter,
            paper,
            "downloading_full_text",
            completed_counter.current(),
            total_paper_count,
            paper_position=item.position,
        )
        # 中文注释：trace 只包外围——下载调用与后续失败分支语义一字不动。
        with trace_tool(
            "download_fulltext",
            args={"paper_id": paper.paperId},
            stage="downloading_full_text",
        ) as call:
            downloaded = await async_download_paper_fulltext(
                paper,
                cache_dir=config.paper_cache_dir,
                connect_timeout_seconds=config.connect_timeout_seconds,
                download_timeout_seconds=config.download_timeout_seconds,
                max_file_size_mb=config.max_file_size_mb,
                runtime_resources=runtime_resources,
            )
            call.set_result(
                {
                    "status": downloaded.status,
                    "reason": downloaded.reason,
                    "source_url": downloaded.source_url,
                    "file_path": str(downloaded.file_path) if downloaded.file_path else None,
                },
                status="completed" if downloaded.file_path is not None else "failed",
                error=None if downloaded.file_path is not None else {"message": downloaded.reason or "全文下载失败"},
            )
        full_text = FullTextStatus(status=downloaded.status, reason=downloaded.reason, source_url=downloaded.source_url)
        if downloaded.file_path is None:
            result.full_text = full_text
            _report_progress(
                reporter,
                paper,
                "paper_completed",
                completed_counter.current(),
                total_paper_count,
                runtime_status="failed",
                message="全文下载失败，已保留摘要阅读结果",
                current_status=result.full_text.status,
                paper_position=item.position,
                error_message=full_text.reason or "全文下载失败",
            )
            return result
        source_path = downloaded.file_path
        full_text.source_path = str(source_path)
    else:
        full_text.source_path = str(source_path)

    markdown_path = Path(full_text.markdown_path) if full_text.markdown_path else None
    if markdown_path is None or not markdown_path.is_file():
        _set_paper_runtime_status(
            paper_runtime_statuses,
            paper,
            status="running",
            current_stage="converting_markdown",
            result=result,
            error_message="",
        )
        _report_progress(
            reporter,
            paper,
            "converting_markdown",
            completed_counter.current(),
            total_paper_count,
            paper_position=item.position,
        )
        with trace_tool(
            "convert_markdown",
            args={"paper_id": paper.paperId},
            stage="converting_markdown",
        ) as call:
            converted = await async_convert_fulltext_to_markdown(
                paper,
                source_path=source_path,
                source_url=full_text.source_url,
            )
            call.set_result(
                {
                    "markdown_path": str(converted.markdown_path) if converted.markdown_path else None,
                    "page_count": converted.page_count,
                    "warnings": list(converted.warnings),
                },
                status="completed" if converted.markdown_path is not None else "failed",
                error=None
                if converted.markdown_path is not None
                else {"message": "；".join(converted.warnings) or "无法将全文转换为 Markdown"},
            )
        if converted.markdown_path is None:
            result.full_text = FullTextStatus(
                status="parse_failed",
                reason="；".join(converted.warnings) or "无法将全文转换为 Markdown",
                source_url=full_text.source_url,
                source_path=str(source_path),
            )
            _report_progress(
                reporter,
                paper,
                "paper_completed",
                completed_counter.current(),
                total_paper_count,
                runtime_status="failed",
                message="Markdown 转换失败，已保留摘要阅读结果",
                current_status=result.full_text.status,
                paper_position=item.position,
                error_message=result.full_text.reason or "无法将全文转换为 Markdown",
            )
            return result
        full_text.status = "markdown_ready"
        full_text.markdown_path = str(converted.markdown_path)
        full_text.page_count = converted.page_count
        if converted.warnings:
            result.warnings.extend(converted.warnings)
        markdown_path = converted.markdown_path
    else:
        full_text.status = "markdown_ready"
        full_text.markdown_path = str(markdown_path)

    result.full_text = full_text
    _set_paper_runtime_status(
        paper_runtime_statuses,
        paper,
        status="running",
        current_stage="chunking_full_text",
        result=result,
        error_message="",
    )
    _report_progress(
        reporter,
        paper,
        "chunking_full_text",
        completed_counter.current(),
        total_paper_count,
        paper_position=item.position,
    )
    with trace_tool(
        "build_chunks",
        args={"paper_id": paper.paperId},
        stage="chunking_full_text",
    ) as call:
        chunk_build = await async_build_chunks_file(paper, markdown_path=markdown_path)
        call.set_result(
            {
                "chunk_count": len(chunk_build.chunks),
                "chunks_path": str(chunk_build.chunks_path),
            }
        )
    full_text.chunk_count = len(chunk_build.chunks)
    result.full_text = full_text

    _set_paper_runtime_status(
        paper_runtime_statuses,
        paper,
        status="running",
        current_stage="extracting_full_text",
        result=result,
        error_message="",
    )
    _report_progress(
        reporter,
        paper,
        "extracting_full_text",
        completed_counter.current(),
        total_paper_count,
        paper_position=item.position,
    )
    if llm is None:
        raise ReadModelUnavailableError("未配置可用的阅读模型，无法生成全文结构化摘要")
    try:
        # 中文注释：先提取全文结构化信息，再回读原文做证据验证；验证不通过的
        # 字段会反馈给模型重提，最多按配置重试几次。模型不可用仍会抛错，按普通失败中断节点。
        # 提取提示词里同时要求模型判断全文是否真的与主题相关，明显无关时返回
        # irrelevant 标记，这里就地止损，跳过后续提取核查（只中断本篇）。
        # 中文注释：trace 壳套在现有 try 内部——异常先留 failed 牌，再落进
        # 下方原有 except 分支，转换/降级语义一字不动。
        with trace_tool(
            "extract_fulltext",
            args={"paper_id": paper.paperId},
            stage="extracting_full_text",
        ) as call:
            extraction_record, verification = await _extract_and_verify(
                paper,
                topic=topic,
                chunks_path=chunk_build.chunks_path,
                markdown_path=markdown_path,
                llm=llm,
                runtime_resources=runtime_resources,
                config=config,
            )
            call.set_result(
                {
                    "status": extraction_record.get("status") if isinstance(extraction_record, dict) else None,
                    "reason": extraction_record.get("reason") if isinstance(extraction_record, dict) else None,
                    "verification": {
                        "status": (verification or {}).get("status"),
                        "passed": (verification or {}).get("passed"),
                        "checked_fields": len((verification or {}).get("items") or []),
                    },
                }
            )
    except RuntimeError as exc:
        # 中文注释：这里的运行错误来自全文摘要模型调用。暂停后保留已下载、转换和
        # 切分的文件，用户修好模型配置后可直接从 chunk.json 继续处理。
        raise ReadModelUnavailableError(f"全文结构化摘要模型不可用：{exc}") from exc
    except ValueError as exc:
        # 中文注释：模型返回的 JSON 或引用不合格时，正文块仍可用于后续人工核对和
        # 写作，因此保留摘要阶段结果并记录提醒，而不是丢弃整篇论文。
        result.extraction = empty_extraction()
        result.warnings.append(f"全文结构化摘要生成失败：{exc}")
    else:
        if isinstance(extraction_record, dict) and extraction_record.get("status") == "irrelevant":
            # 中文注释：⑤ 全文止损——提取模型通读全文后判定与主题无关，
            # 本篇直接软丢弃，跳过核查；其他论文的精读照常进行，不做级联停批。
            reason = str(extraction_record.get("reason") or "全文判定与主题无关")
            result.relevance.relevant = False
            result.relevance.discarded = True
            result.relevance.reason = f"全文阶段：{reason}"
            result.relevance.status = "discarded"
            result.full_text.status = "chunks_saved"
            result.full_text.reason = ""
            return result
        if isinstance(extraction_record, dict) and extraction_record.get("status") == "extraction_failed":
            # 修复：提取格式违规已在 _extract_and_verify 内重试仍未通过。这里保留摘要阶段
            # 结果、置空全文提取，并把 verification 落成 skipped_extraction_failed（而非空 {}），
            # 让"核查因提取失败被跳过"可观测。论文不丢弃，继续走分块入库。
            reason = str(extraction_record.get("reason") or "全文提取未通过格式校验")
            result.extraction = empty_extraction()
            result.verification = verification
            result.warnings.append(f"全文结构化摘要生成失败（已重试）：{reason}")
        else:
            result.extraction = extraction_payload(extraction_record)
            result.note = _full_text_note_from_extraction(result.extraction)
            result.verification = verification
        if verification:
            _report_progress(
                reporter,
                paper,
                "verifying_full_text",
                completed_counter.current(),
                total_paper_count,
                paper_position=item.position,
                verify_status=str(verification.get("status") or ""),
                verify_passed=bool(verification.get("passed")),
            )
            if verification.get("status") == "failed_after_retries" and not verification.get("passed"):
                failed_fields = [
                    str(item.get("field") or "")
                    for item in verification.get("items", [])
                    if not item.get("verified")
                ]
                result.warnings.append(f"全文结构化信息经核查未通过：{'、'.join(failed_fields)}")

    if not ENABLE_FULL_TEXT_EMBEDDING:
        # 中文说明：async_build_chunks_file 已经把分块内容写到论文缓存目录的 chunk.json。
        # 关闭开关后到这里就结束，既不要求 embedding 配置，也不发起任何向量服务请求。
        full_text.status = "chunks_saved"
        full_text.reason = ""
        return result

    _set_paper_runtime_status(
        paper_runtime_statuses,
        paper,
        status="running",
        current_stage="saving_chunks",
        result=result,
        error_message="",
    )
    _report_progress(
        reporter,
        paper,
        "saving_chunks",
        completed_counter.current(),
        total_paper_count,
        paper_position=item.position,
    )
    # 中文注释：embedding 不可用或索引失败按单篇部分失败处理——摘要与切分成果
    # 照常保留，不再保存"等待 embedding 恢复"的中间现场。
    if embedding_connection is None:
        full_text.reason = embedding_error or "未配置可用的 embedding 服务，全文尚未写入向量库"
        result.full_text = full_text
        _report_progress(
            reporter,
            paper,
            "paper_completed",
            completed_counter.current(),
            total_paper_count,
            runtime_status="failed",
            message="全文索引失败，已保留阅读结果",
            current_status=result.full_text.status,
            paper_position=item.position,
            error_message=full_text.reason,
        )
        return result
    try:
        index_result = await async_index_chunk_file(
            paper,
            chunks_path=chunk_build.chunks_path,
            source_url=full_text.source_url,
            vector_store_path=config.vector_store_path,
            collection_name=config.vector_store_collection,
            embedding_connection=embedding_connection,
            runtime_resources=runtime_resources,
        )
    except (RuntimeError, OSError, ValueError) as exc:
        full_text.reason = f"全文已转成 Markdown，但建立索引失败：{exc}"
        result.full_text = full_text
        _report_progress(
            reporter,
            paper,
            "paper_completed",
            completed_counter.current(),
            total_paper_count,
            runtime_status="failed",
            message="全文索引失败，已保留阅读结果",
            current_status=result.full_text.status,
            paper_position=item.position,
            error_message=full_text.reason,
        )
        return result
    full_text.status = "indexed"
    full_text.reason = ""
    full_text.chunk_count = index_result.chunk_count
    result.full_text = full_text
    return result


def _full_text_note_from_extraction(extraction: JsonObject) -> ReadNote:
    """把精读后的结构化摘要同步到阅读笔记，供界面直接展示。"""

    # 中文注释：粗读论文继续使用摘要模型生成的笔记；只有真正完成全文精读的论文
    # 才会走到这里。因此界面中的精读摘要会自然带有 [chunkId]，粗读摘要不会混入它。
    research_topic = str(extraction.get("research_topic") or "").strip()
    research_object = str(extraction.get("research_object") or "").strip()
    methods = str(extraction.get("methods") or "").strip()
    conclusions = str(extraction.get("conclusions") or "").strip()
    contributions = str(extraction.get("contributions") or "").strip()
    limitations = str(extraction.get("limitations") or "").strip()
    return ReadNote(
        main_question=research_topic,
        methods=[methods] if methods else [],
        datasets=[research_object] if research_object else [],
        contributions=[contributions] if contributions else [],
        limitations=[limitations] if limitations else [],
        main_results=[conclusions] if conclusions else [],
        short_summary=conclusions or research_topic,
        evidence_level="full_text",
    )


async def _extract_and_verify(
    paper: PaperDocument,
    *,
    topic: str,
    chunks_path: Path,
    markdown_path: Path,
    llm: ProviderSnapshot,
    runtime_resources: WorkflowRuntimeResources | None,
    config: Any,
) -> tuple[JsonObject, JsonObject]:
    """提取全文结构化信息，再回读原文做证据验证，验证不通过会按配置重提。

    返回 (extraction_record, verification) 两个结果：
    - extraction_record：和原来一样是 extraction.json 的记录；模型通读全文后
      判定与主题无关时，返回带 status="irrelevant" 的记录，调用方据此止损；
    - verification：证据验证结论，启用验证时才非空。

    中文注释：第一次提取用缓存，验证不通过后的重提才强制重新提取，并把
    上次没通过核查的字段反馈给模型，让它针对性地修正。
    """

    verify_enabled = bool(getattr(config, "verify_enabled", True))
    max_retries = int(getattr(config, "verify_max_retries", 1))
    optional_fields = tuple(getattr(config, "verify_optional_fields", ("limitations",)))
    max_chars = int(getattr(config, "verify_max_chars", 120000))

    feedback: str | None = None
    extraction_record: JsonObject | None = None
    verification: JsonObject = {}
    for attempt in range(max_retries + 1):
        try:
            extraction_record = await async_extract_paper_from_chunks(
                paper,
                chunks_path=chunks_path,
                llm=llm,
                topic=topic,
                runtime_resources=runtime_resources,
                force=attempt > 0,
                feedback=feedback,
            )
        except ValueError as exc:
            # 修复：提取阶段的格式违规（缺引用、字段不合规等）以前会直接冒泡到调用方，
            # 被吞成空 extraction + 一条笼统 warning，重试机制一次都跑不到（见 EVAL_NOTES.md 第六节）。
            # 现在把它纳入重试：把具体错误通过 feedback 回灌给模型，下一轮针对性修正。
            # 注意：只捕获 ValueError（格式问题）；RuntimeError（模型不可用）不在此捕获，
            #       继续冒泡到调用方转 ReadModelUnavailableError，按普通失败中断节点。
            if attempt >= max_retries:
                # 重试耗尽仍不合格：返回结构化失败，让调用方能区分"提取失败"与"核查跳过"，
                # 不再静默留空 verification。
                return (
                    {"status": "extraction_failed", "reason": str(exc)},
                    {"status": "skipped_extraction_failed", "passed": True, "items": [], "reason": str(exc)},
                )
            feedback = (
                f"上一次提取未通过格式校验：{exc}。"
                "请为每个非空字段在句末标注输入中真实存在的 [chunkId]，"
                "并且只输出 research_topic、research_object、methods、conclusions、contributions、limitations 六个字段的 JSON。"
            )
            continue
        # 中文注释：全文阶段止损判定优先于核查——无关论文不再浪费核查调用。
        if isinstance(extraction_record, dict) and extraction_record.get("status") == "irrelevant":
            return extraction_record, {}
        if not verify_enabled:
            return extraction_record, {}
        extraction = extraction_payload(extraction_record)
        verification = await async_verify_extraction(
            paper,
            extraction=extraction,
            markdown_path=markdown_path,
            llm=llm,
            runtime_resources=runtime_resources,
            optional_fields=optional_fields,
            max_chars=max_chars,
        )
        if verification.get("status") != "completed":
            # 验证不可用（例如模型调用失败），本轮先不重试，保留一次提取结果。
            break
        failed_fields = [
            str(item.get("field") or "")
            for item in verification.get("items", [])
            if not item.get("verified") and item.get("field") not in optional_fields
        ]
        if not failed_fields:
            break
        if attempt >= max_retries:
            # 重试次数用完了仍没通过，把状态标出来，让阅读节点能提示用户。
            verification = dict(verification)
            verification["status"] = "failed_after_retries"
            break
        # 把没通过核查的字段反馈给模型，下一轮提取时针对性地修正。
        feedback = "以下字段未通过核查：" + "、".join(failed_fields)
    return extraction_record, verification


async def _build_abstract_note(
    paper: PaperDocument,
    *,
    topic: str,
    constraints: JsonObject,
    llm: ProviderSnapshot | None,
    runtime_resources: WorkflowRuntimeResources | None,
    usage_callback: Any | None = None,
) -> tuple[ReadNote, ReadRelevance, list[str]]:
    """通过 ReadAgent 完成异步摘要阅读，节点只保留流程控制和暂停恢复逻辑。"""

    semaphore = runtime_resources.read_model_semaphore if runtime_resources is not None else _FALLBACK_READ_MODEL_SEMAPHORE
    try:
        result = await _read_agent_from_llm(llm, usage_callback=usage_callback).async_read_abstract(
            paper,
            topic=topic,
            constraints=constraints,
            semaphore=semaphore,
            usage_callback=usage_callback,
        )
        return result.as_tuple()
    except ReadAgentModelUnavailableError as exc:
        raise ReadModelUnavailableError(str(exc)) from exc


async def _persist_note_async(
    result: PaperReadResult,
    *,
    stage: str | None,
    sink: ReadPersistenceSink | None,
    artifact_refs: list[JsonObject],
) -> None:
    """保存一个阅读稳定点，不额外发送“论文已完成”的误导事件。"""

    if sink is None:
        return
    try:
        persisted = await asyncio.to_thread(sink.persist_paper, result, stage=stage)
        _append_artifact_refs(artifact_refs, persisted.artifacts)
    except Exception as exc:
        result.warnings.append(f"阅读稳定点无法保存到会话目录：{exc}")


async def _persist_completed_paper_async(
    result: PaperReadResult,
    *,
    sink: ReadPersistenceSink | None,
    artifact_refs: list[JsonObject],
    reporter: Any,
    completed: int,
    total: int,
    paper_position: int,
) -> tuple[bool, str]:
    """在 async 阅读流程里把单篇结果落盘，并更新这篇论文自己的卡片。"""

    if sink is None:
        return True, ""
    try:
        persisted = await asyncio.to_thread(sink.persist_paper, result)
        _append_artifact_refs(artifact_refs, persisted.artifacts)
        if reporter is not None:
            # 中文注释：产物列表仍然需要 artifact 事件，但这里关闭自动生成保存步骤卡片。
            # 保存成功这件事由下面的 _report_progress 更新当前论文卡片，避免前端多出一张卡片。
            for artifact in persisted.artifacts:
                reporter.artifact(artifact, stage="paper_artifact_ready", emit_runtime_event=False)
        completion_status = _paper_completion_runtime_status(result)
        _report_progress(
            reporter,
            result.paper,
            "paper_artifact_ready",
            completed,
            total,
            runtime_status=completion_status,
            message="单篇阅读结果已保存" if completion_status == "completed" else "单篇阅读结果已保存，但阅读过程中有阶段失败",
            current_status=result.full_text.status,
            paper_position=paper_position,
            artifact_count=len(persisted.artifacts),
            error_message=_paper_completion_error_message(result),
        )
        return True, ""
    except Exception as exc:
        message = f"阅读结果无法保存到会话目录：{exc}"
        result.warnings.append(message)
        _report_progress(
            reporter,
            result.paper,
            "paper_artifact_ready",
            completed,
            total,
            runtime_status="failed",
            message="单篇阅读结果保存失败",
            current_status=result.full_text.status,
            paper_position=paper_position,
            error_message=message,
        )
        return False, message


def _initial_paper_runtime_statuses(
    papers: list[PaperDocument],
    results_by_paper_id: dict[str, PaperReadResult],
) -> dict[str, JsonObject]:
    """初始化每篇论文的实时进度卡片（纯内存，不序列化、不从 checkpoint 恢复）。

    已从 note 恢复的论文按实际完成状态回填，被软丢弃的保持 discarded，
    不能统一标成 completed，否则前端卡片和后续判断都会出错。
    """

    statuses = {paper.id: _default_paper_runtime_status(paper) for paper in papers}
    for paper_id, result in results_by_paper_id.items():
        statuses[paper_id] = {
            **statuses.get(paper_id, _default_paper_runtime_status(result.paper)),
            "paper_id": result.paper.id,
            "paper_title": result.paper.title,
            "status": _paper_completion_runtime_status(result),
            "current_stage": "paper_completed",
            "source_path": result.full_text.source_path,
            "markdown_path": result.full_text.markdown_path,
            "error_message": "",
            "deep_read_reserved": _infer_deep_read_reserved(result),
            "result": result.to_dict(),
        }
    return statuses


def _ordered_results(papers: list[PaperDocument], results_by_paper_id: dict[str, PaperReadResult]) -> list[PaperReadResult]:
    """把并发完成的论文结果重新按原论文顺序排好，方便后续节点稳定消费。"""

    return [results_by_paper_id[paper.id] for paper in papers if paper.id in results_by_paper_id]


def _default_paper_runtime_status(paper: PaperDocument) -> JsonObject:
    """给每篇论文准备一个最基础的运行状态记录。"""

    return {
        "paper_id": paper.id,
        "paper_title": paper.title,
        "status": "pending",
        "current_stage": "pending",
        "source_path": None,
        "markdown_path": None,
        "error_message": "",
        "deep_read_reserved": False,
        "result": None,
    }


def _set_paper_runtime_status(
    paper_runtime_statuses: dict[str, JsonObject],
    paper: PaperDocument,
    *,
    status: str,
    current_stage: str,
    result: PaperReadResult | None,
    error_message: str,
    deep_read_reserved: bool | None = None,
) -> None:
    """更新某篇论文的实时进度卡片（纯内存，供前端展示与选择逻辑使用）。"""

    current = dict(paper_runtime_statuses.get(paper.id) or _default_paper_runtime_status(paper))
    current.update(
        paper_id=paper.id,
        paper_title=paper.title,
        status=status,
        current_stage=current_stage,
        error_message=error_message,
    )
    if result is not None:
        current["result"] = result.to_dict()
        current["source_path"] = result.full_text.source_path
        current["markdown_path"] = result.full_text.markdown_path
    if deep_read_reserved is not None:
        current["deep_read_reserved"] = bool(deep_read_reserved)
    paper_runtime_statuses[paper.id] = current


def _paper_stage_from_status(status: JsonObject | None) -> str:
    """从论文状态里取当前阶段，缺失时给前端一个稳定的默认阶段。"""

    # 中文注释：异常处理可能发生在任意位置。这里不猜测复杂流程，
    # 只读取已经记录过的 current_stage；如果还没记录，就默认算作摘要阅读阶段。
    stage = str((status or {}).get("current_stage") or "").strip()
    return stage or "reading_abstract"


def _mark_paper_waiting_resource(
    paper_runtime_statuses: dict[str, JsonObject],
    paper: PaperDocument,
    error: ReadResourceUnavailableError,
) -> None:
    """在模型或 embedding 不可用时，把当前论文的中间结果和失败原因记入实时卡片。"""

    payload = paper_runtime_statuses.get(paper.id) or {}
    result_payload = payload.get("result")
    result = _read_result_from_payload(result_payload, [paper]) if isinstance(result_payload, dict) else None
    current_stage = str(payload.get("current_stage") or "reading_abstract")
    _set_paper_runtime_status(
        paper_runtime_statuses,
        paper,
        status="failed",
        current_stage=current_stage,
        result=result,
        error_message=str(error),
        deep_read_reserved=_infer_deep_read_reserved(result) if result is not None else bool(payload.get("deep_read_reserved")),
    )


def _infer_deep_read_reserved(result: PaperReadResult | None) -> bool:
    """只要论文已经进入全文链路，就视为这个名额已经被占用。"""

    if result is None:
        return False
    full_text = result.full_text
    return bool(full_text.source_path or full_text.markdown_path or full_text.status in _FULL_TEXT_COMPLETED_STATUSES | {"downloaded", "markdown_ready", "download_failed", "parse_failed"})


def _report_progress(reporter: Any, paper: PaperDocument, stage: str, completed: int, total: int, **extra: Any) -> None:
    """按论文维度上报实时进度，让同一篇论文始终更新同一张卡片。"""

    if reporter is None:
        return
    # 中文注释：这里的 key 一定要稳定。同一篇论文的摘要、下载、转换、保存等阶段
    # 都使用同一个 event_key，前端收到后就会更新旧卡片，而不是新增一堆零散卡片。
    messages = {
        "reading_abstract": "正在阅读论文摘要",
        "downloading_full_text": "正在下载论文全文",
        "converting_markdown": "正在转换 Markdown",
        "chunking_full_text": "正在切分全文内容",
        "extracting_full_text": "正在提取论文结构化信息",
        "verifying_full_text": "正在核实论文提取结果",
        "saving_chunks": "正在写入全文索引",
        "paper_completed": "论文阅读完成",
        "paper_artifact_ready": "单篇阅读结果已保存",
    }
    runtime_status = str(extra.pop("runtime_status", "running") or "running")
    message = str(extra.pop("message", messages.get(stage, "正在处理论文")))
    reporter.progress(
        message,
        stage=stage,
        event_key=f"paper:{paper.id}",
        stage_title=paper.title,
        runtime_status=runtime_status,
        show_content=message,
        paper_id=paper.id,
        paper_title=paper.title,
        current_title=paper.title,
        completed=completed,
        total=total,
        **extra,
    )
