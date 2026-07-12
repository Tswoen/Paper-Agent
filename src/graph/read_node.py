from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, NoReturn, cast

from src.utils.read_utils.read_fulltext import convert_fulltext_to_markdown
from src.models.read_models import FullTextStatus, PaperReadResult, ReadNote, ReadRelevance
from src.repositories.node_persistence.read_persistence import ReadPersistenceSink
from src.repositories.chroma.read_vector_store import EmbeddingConnection, index_markdown_chunks
from src.graph.checkpoint_halt import halt_with_checkpoint
from src.graph.runtime import WorkflowRuntimeContext
from src.graph.state_models import JsonObject, State
from src.llm import ModelConfig, ProviderSnapshot, SystemConfig, make_provider
from src.llm.base import LLMResponse
from src.paper_retrieval.download import download_paper_fulltext
from src.paper_retrieval.models import PaperDocument
from src.repositories.sessions.base import SessionRepository


class ReadResourceUnavailableError(RuntimeError):
    """表示阅读节点依赖的外部资源不可用，需要保存现场后等待用户处理。"""

    recovery_status = "waiting_resource"
    current_step = "read_waiting_resource"
    failure_stage = "read_resource_unavailable"
    diagnostic_key = "read_resource_unavailable"

    def __init__(self, message: str, *, details: JsonObject | None = None):
        """保存可展示错误信息和用于恢复现场的结构化细节。"""

        super().__init__(message)
        # 中文注释：details 只保存能写进 JSON 的普通数据，不保存 provider、repo
        # 这类运行时对象，避免 checkpoint 写盘或前端传递时无法序列化。
        self.details = dict(details or {})


class ReadModelUnavailableError(ReadResourceUnavailableError):
    """表示阅读模型当前不可用，需要保存现场后等待用户处理。

    这个异常只用于“用户修好模型后可以继续”的场景，例如：
    1. 没有配置阅读模型；
    2. 模型接口返回 HTTP/鉴权/限流/服务端错误；
    3. 调用模型时发生网络、超时、鉴权等异常。

    它和普通论文处理异常刻意区分：普通异常只跳过当前论文；模型返回了内容但
    内容质量不好时，也不算模型不可用，阅读节点会改用保守笔记继续处理。
    """

    recovery_status = "waiting_model"
    current_step = "read_waiting_model"
    failure_stage = "read_model_unavailable"
    diagnostic_key = "read_model_unavailable"


class ReadEmbeddingUnavailableError(ReadResourceUnavailableError):
    """表示 embedding 服务不可用，需要保存 Markdown 现场后等待用户处理。"""

    recovery_status = "waiting_embedding"
    current_step = "read_waiting_embedding"
    failure_stage = "read_embedding_unavailable"
    diagnostic_key = "read_embedding_unavailable"

    def __init__(self, message: str, *, pending_result: PaperReadResult, details: JsonObject | None = None):
        """保存已经转换到 Markdown、但还没有写入向量库的论文结果。"""

        super().__init__(message, details=details)
        # 中文注释：pending_result 是当前论文的中间成果，里面包含 Markdown 路径。
        # 恢复时优先从这个 Markdown 继续建索引，避免重新下载和转换全文。
        self.pending_result = pending_result

def run_read_node():
    """生成执行图中的阅读节点，按论文顺序完成摘要、全文和索引处理。"""

    def _node(state: State) -> State:
        """读取检索结果并保留全部已有状态，模型不可用时保存现场并中断。"""

        request = state.get("request")
        if request is None:
            raise ValueError("阅读节点缺少用户请求，无法判断论文主题")
        system_config = SystemConfig.load()
        config = system_config.read

        # 中文注释：恢复执行时，调用方会把上次中断保存的 read_resume_checkpoint
        # 放回 state。此时 search_results 可能为空，所以论文列表优先读当前 state，
        # 读不到再从 checkpoint 还原，确保可以跳过检索节点直接回到阅读节点。
        checkpoint = _checkpoint_from_state(state)
        papers = _deduplicate_papers(list(state.get("search_results") or _papers_from_payload(checkpoint.get("search_results"))))

        reporter = _resolve_reporter(state)
        sink = _resolve_sink(state)
        llm = _resolve_llm(state, config.agent_name)
        embedding_connection, embedding_error = _resolve_embedding_connection(system_config, config.download_timeout_seconds)

        # 中文注释：恢复执行的核心是不重复处理已经成功完成的论文。
        # checkpoint/read_results 中的条目会被恢复成 PaperReadResult，循环从
        # start_position 继续；这样用户验证模型可用后，不会重新下载、转换或重写
        # 已经完成的前几篇论文。
        recovered_results = _restore_read_results(state, papers, checkpoint)
        results: list[PaperReadResult] = list(recovered_results)
        artifact_refs: list[JsonObject] = list(state.get("read_artifact_refs") or [])
        deep_read_count = _restore_deep_read_count(results, checkpoint)
        deep_read_limit = _deep_read_limit(request.constraints, len(papers))

        if reporter is not None:
            reporter.started(
                f"准备阅读 {len(papers)} 篇论文",
                stage="read_start",
                total=len(papers),
                completed=len(results),
                resumed_paper_count=len(results),
            )

        # 中文注释：如果上次是在“全文已转成 Markdown，但 embedding 不可用”时中断，
        # checkpoint 里会保存 pending_read_result。这里优先复用这份 Markdown 继续建索引，
        # 避免重新下载和转换同一篇论文。
        deep_read_count = _resume_pending_index(
            state,
            papers=papers,
            results=results,
            artifact_refs=artifact_refs,
            checkpoint=checkpoint,
            config=config,
            embedding_connection=embedding_connection,
            embedding_error=embedding_error,
            sink=sink,
            reporter=reporter,
            deep_read_count=deep_read_count,
            deep_read_limit=deep_read_limit,
        )
        start_position = len(results) + 1

        for position, paper in enumerate(papers[start_position - 1 :], start=start_position):
            _report_progress(reporter, paper, "reading_abstract", position - 1, len(papers))
            try:
                result, used_deep_read = _read_one_paper(
                    paper,
                    topic=request.topic,
                    constraints=request.constraints,
                    llm=llm,
                    config=config,
                    embedding_connection=embedding_connection,
                    embedding_error=embedding_error,
                    allow_deep_read=deep_read_count < deep_read_limit,
                    reporter=reporter,
                )
                if used_deep_read:
                    deep_read_count += 1
            except ReadResourceUnavailableError as exc:
                # 中文注释：模型或 embedding 这类外部资源不可用时，不是“当前论文失败”，
                # 而是需要用户修好配置后继续。这里保存已完成结果和恢复位置后中断。
                pending_deep_read_count = deep_read_count + (1 if isinstance(exc, ReadEmbeddingUnavailableError) else 0)
                _halt_for_resource_unavailable(
                    state,
                    papers=papers,
                    results=results,
                    artifact_refs=artifact_refs,
                    position=position,
                    error=exc,
                    sink=sink,
                    reporter=reporter,
                    deep_read_count=pending_deep_read_count,
                    deep_read_limit=deep_read_limit,
                )
            except Exception as exc:
                # 中文注释：未知异常也只影响当前论文，不能让一次批量阅读全部中断。
                result = PaperReadResult(
                    paper=paper,
                    full_text=FullTextStatus(status="not_requested", reason="当前论文处理发生异常"),
                    warnings=[f"当前论文处理失败：{exc}"],
                )
            results.append(result)
            _persist_completed_paper(result, sink=sink, artifact_refs=artifact_refs, reporter=reporter)
            _report_progress(
                reporter,
                paper,
                "paper_completed",
                position,
                len(papers),
                current_status=result.full_text.status,
            )

        summary = _build_summary(results, deep_read_count)
        if sink is not None:
            try:
                persisted = sink.persist_summary(summary, results)
                artifact_refs.extend(persisted.artifacts)
                summary["manifest"] = persisted.artifacts[0] if persisted.artifacts else {}
                if reporter is not None:
                    for artifact in persisted.artifacts:
                        reporter.artifact(artifact, stage="read_artifact_ready")
            except Exception as exc:
                summary["persistence_error"] = str(exc)

        if reporter is not None:
            reporter.completed(
                "论文阅读节点已完成",
                stage="read_done",
                total=len(papers),
                completed=len(papers),
                indexed_paper_count=summary["indexed_paper_count"],
            )
        updated = dict(state)
        updated.update(
            read_results=[result.to_dict() for result in results],
            read_summary=summary,
            read_artifact_refs=artifact_refs,
            current_step="read",
        )
        if checkpoint:
            # 中文注释：恢复运行可能没有经过检索节点，因此需要把 checkpoint
            # 还原出来的论文列表写回最终 state，供回复节点和调用方继续展示。
            updated["search_results"] = papers
        # 中文注释：阅读节点成功跑完后，旧 checkpoint 已经失效，避免后续节点
        # 或下一次调用误以为仍处于“等待模型恢复”的状态。
        updated.pop("read_resume_checkpoint", None)
        return cast(State, updated)

    return _node


def _halt_for_resource_unavailable(
    state: State,
    *,
    papers: list[PaperDocument],
    results: list[PaperReadResult],
    artifact_refs: list[JsonObject],
    position: int,
    error: ReadResourceUnavailableError,
    sink: ReadPersistenceSink | None,
    reporter: Any,
    deep_read_count: int,
    deep_read_limit: int,
) -> NoReturn:
    """保存阅读节点现场并抛出明确错误，等待用户修好外部资源后恢复。"""

    # 中文注释：checkpoint 必须包含“继续执行所需的最小信息”：原始请求、全部论文、
    # 已完成结果、产物引用、当前位置和精读计数。具体字段由阅读节点负责构造，
    # 通用模块只负责写盘、上报、写回 state 和抛错。
    checkpoint = _build_resume_checkpoint(
        state,
        papers=papers,
        results=results,
        artifact_refs=artifact_refs,
        position=position,
        error=error,
        deep_read_count=deep_read_count,
        deep_read_limit=deep_read_limit,
    )
    halt_with_checkpoint(
        state,
        checkpoint=checkpoint,
        error=error,
        persist_checkpoint=sink.persist_checkpoint if sink is not None else None,
        reporter=reporter,
        results_payload=[result.to_dict() for result in results],
        artifact_refs=artifact_refs,
        checkpoint_key="read_resume_checkpoint",
        results_key="read_results",
        artifact_refs_key="read_artifact_refs",
        diagnostics_key=error.diagnostic_key,
        current_step=error.current_step,
        failure_stage=error.failure_stage,
        recovery_status=error.recovery_status,
        total=len(papers),
        completed=len(results),
        next_position=position,
    )


def _build_resume_checkpoint(
    state: State,
    *,
    papers: list[PaperDocument],
    results: list[PaperReadResult],
    artifact_refs: list[JsonObject],
    position: int,
    error: ReadResourceUnavailableError,
    deep_read_count: int,
    deep_read_limit: int,
) -> JsonObject:
    """构造可以通过 state_overrides 注入并继续执行的阅读现场。"""

    request = state.get("request")
    # 中文注释：这里保存的是纯 JSON，不保存 provider、repo、reporter 等运行时对象。
    # 运行时对象在恢复请求中重新注入；checkpoint 只负责描述“业务进度”和
    # “还原输入”，因此可以安全写入文件、事件 metadata 或前端状态。
    checkpoint = {
        "recovery_status": error.recovery_status,
        "current_step": error.current_step,
        "message": str(error),
        "error_details": dict(error.details),
        "next_position": position,
        "completed_count": len(results),
        "total_count": len(papers),
        "deep_read_count": deep_read_count,
        "deep_read_limit": deep_read_limit,
        "request": {
            "topic": getattr(request, "topic", ""),
            "constraints": dict(getattr(request, "constraints", {}) or {}),
            "language": getattr(request, "language", "zh"),
        },
        "search_results": [paper.to_dict() for paper in papers],
        "read_results": [result.to_dict() for result in results],
        "read_artifact_refs": list(artifact_refs),
        "resume_hint": "外部资源验证可用后，把此 checkpoint 作为 read_resume_checkpoint 注入即可从 next_position 继续阅读。",
    }
    if isinstance(error, ReadModelUnavailableError):
        checkpoint["resume_hint"] = "模型验证可用后，把此 checkpoint 作为 read_resume_checkpoint 注入即可从 next_position 继续阅读。"
    if isinstance(error, ReadEmbeddingUnavailableError):
        # 中文注释：embedding 失败发生在当前论文中间，所以当前论文不能放进
        # read_results，否则恢复时会被当成“已完成”而跳过。这里单独保存 pending。
        checkpoint.update(
            pending_read_result=error.pending_result.to_dict(),
            pending_position=position,
            pending_resume_phase="index_markdown",
            resume_hint="embedding 服务验证可用后，把此 checkpoint 作为 read_resume_checkpoint 注入即可从已保存的 Markdown 继续建立索引。",
        )
    return checkpoint


def _checkpoint_from_state(state: State) -> JsonObject:
    """读取调用方注入的阅读恢复现场。"""

    checkpoint = state.get("read_resume_checkpoint")
    return dict(checkpoint) if isinstance(checkpoint, dict) else {}


def _restore_read_results(state: State, papers: list[PaperDocument], checkpoint: JsonObject | None = None) -> list[PaperReadResult]:
    """从状态或 checkpoint 恢复已完成论文，避免重跑已保存的阅读结果。"""

    checkpoint = checkpoint or {}
    # 中文注释：优先使用 state 中较新的 read_results；如果恢复运行直接从
    # checkpoint 进入阅读节点，state 可能还没被 graph 合并，则退回 checkpoint。
    raw_results = list(state.get("read_results") or checkpoint.get("read_results") or [])
    papers_by_id = {paper.id: paper for paper in papers}
    restored: list[PaperReadResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        paper_payload = item.get("paper")
        paper = _paper_from_payload(paper_payload) if isinstance(paper_payload, dict) else None
        if paper is None:
            continue
        # 中文注释：如果当前检索结果里已经有同 ID 论文，使用当前对象，避免
        # checkpoint 中的旧 metadata 覆盖新版检索节点补充的字段。
        paper = papers_by_id.get(paper.id, paper)
        restored.append(
            PaperReadResult(
                paper=paper,
                note=_read_note_from_payload(item.get("note")),
                relevance=_read_relevance_from_payload(item.get("relevance")),
                full_text=_full_text_from_payload(item.get("full_text")),
                warnings=_string_list(item.get("warnings")),
            )
        )
    return restored[: len(papers)]


def _resume_pending_index(
    state: State,
    *,
    papers: list[PaperDocument],
    results: list[PaperReadResult],
    artifact_refs: list[JsonObject],
    checkpoint: JsonObject,
    config: Any,
    embedding_connection: EmbeddingConnection | None,
    embedding_error: str | None,
    sink: ReadPersistenceSink | None,
    reporter: Any,
    deep_read_count: int,
    deep_read_limit: int,
) -> int:
    """如果 checkpoint 里有等待入库的 Markdown，就优先从这里继续建立索引。"""

    if checkpoint.get("pending_resume_phase") != "index_markdown":
        return deep_read_count
    pending_position = _optional_int(checkpoint.get("pending_position")) or 0
    expected_position = len(results) + 1
    pending_result = _read_result_from_payload(checkpoint.get("pending_read_result"), papers)
    if pending_result is None or pending_position != expected_position or not pending_result.full_text.markdown_path:
        # 中文注释：pending 现场不完整时不能硬继续，否则可能把错误论文写入向量库。
        # 这里只记录诊断信息，然后退回普通流程，从当前论文重新阅读和转换。
        diagnostics = dict(state.get("diagnostics") or {})
        diagnostics["read_pending_resume_skipped"] = {
            "reason": "checkpoint 中等待入库的论文现场不完整，已退回普通阅读流程",
            "pending_position": pending_position,
            "expected_position": expected_position,
        }
        state["diagnostics"] = diagnostics
        return deep_read_count

    _report_progress(reporter, pending_result.paper, "saving_chunks", pending_position - 1, len(papers))
    if embedding_connection is None:
        reason = embedding_error or "未配置可用的 embedding 服务，全文尚未写入向量库"
        pending_result.full_text.reason = reason
        _halt_for_resource_unavailable(
            state,
            papers=papers,
            results=results,
            artifact_refs=artifact_refs,
            position=pending_position,
            error=ReadEmbeddingUnavailableError(
                reason,
                pending_result=pending_result,
                details=_embedding_error_details(pending_result, stage="embedding_config", message=reason),
            ),
            sink=sink,
            reporter=reporter,
            deep_read_count=deep_read_count,
            deep_read_limit=deep_read_limit,
        )
    try:
        index_result = index_markdown_chunks(
            pending_result.paper,
            markdown_path=Path(pending_result.full_text.markdown_path),
            source_url=pending_result.full_text.source_url,
            collection_path=Path(config.vector_store_path) / config.vector_store_collection,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            embedding_connection=embedding_connection,
        )
    except RuntimeError as exc:
        # 中文注释：恢复时如果 embedding 服务仍然调用失败，就再次保存同一份 Markdown
        # 现场，让用户修好服务后还能继续从这里恢复。
        pending_result.full_text.reason = f"全文已转成 Markdown，但 embedding 服务不可用：{exc}"
        _halt_for_resource_unavailable(
            state,
            papers=papers,
            results=results,
            artifact_refs=artifact_refs,
            position=pending_position,
            error=ReadEmbeddingUnavailableError(
                pending_result.full_text.reason,
                pending_result=pending_result,
                details=_embedding_error_details(pending_result, stage="embedding_call", message=str(exc)),
            ),
            sink=sink,
            reporter=reporter,
            deep_read_count=deep_read_count,
            deep_read_limit=deep_read_limit,
        )
    except (OSError, ValueError) as exc:
        pending_result.full_text.reason = f"全文已转成 Markdown，但建立索引失败：{exc}"
    else:
        pending_result.full_text.status = "indexed"
        pending_result.full_text.reason = ""
        pending_result.full_text.chunk_count = index_result.chunk_count
    results.append(pending_result)
    _persist_completed_paper(pending_result, sink=sink, artifact_refs=artifact_refs, reporter=reporter)
    _report_progress(
        reporter,
        pending_result.paper,
        "paper_completed",
        pending_position,
        len(papers),
        current_status=pending_result.full_text.status,
    )
    return deep_read_count


def _read_result_from_payload(value: Any, papers: list[PaperDocument]) -> PaperReadResult | None:
    """把 checkpoint 中单篇论文阅读结果恢复为 PaperReadResult。"""

    if not isinstance(value, dict):
        return None
    paper_payload = value.get("paper")
    paper = _paper_from_payload(paper_payload) if isinstance(paper_payload, dict) else None
    if paper is None:
        return None
    # 中文注释：优先使用当前论文列表中的对象，避免 checkpoint 里的旧字段覆盖新检索结果。
    papers_by_id = {item.id: item for item in papers}
    paper = papers_by_id.get(paper.id, paper)
    return PaperReadResult(
        paper=paper,
        note=_read_note_from_payload(value.get("note")),
        relevance=_read_relevance_from_payload(value.get("relevance")),
        full_text=_full_text_from_payload(value.get("full_text")),
        warnings=_string_list(value.get("warnings")),
    )


def _persist_completed_paper(
    result: PaperReadResult,
    *,
    sink: ReadPersistenceSink | None,
    artifact_refs: list[JsonObject],
    reporter: Any,
) -> None:
    """单篇论文完成后立刻写入产物目录，避免后续论文失败导致结果丢失。"""

    if sink is None:
        return
    try:
        persisted = sink.persist_paper(result)
        artifact_refs.extend(persisted.artifacts)
        if reporter is not None:
            for artifact in persisted.artifacts:
                reporter.artifact(artifact, stage="paper_artifact_ready")
    except Exception as exc:
        # 中文注释：写单篇结果失败不应该让整次阅读中断；把原因放进 warnings，
        # 后续汇总或回复时仍能看到这篇论文处理过但保存失败。
        result.warnings.append(f"阅读结果无法保存到会话目录：{exc}")


def _papers_from_payload(value: Any) -> list[PaperDocument]:
    """把 checkpoint 里的论文 JSON 恢复成 PaperDocument 列表。"""

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
        metadata=dict(value.get("metadata") or {}) if isinstance(value.get("metadata"), dict) else {},
    )


def _read_note_from_payload(value: Any) -> ReadNote:
    """把 checkpoint 中的笔记 JSON 恢复为 ReadNote。"""

    payload = value if isinstance(value, dict) else {}
    return ReadNote(
        main_question=_text_value(payload.get("main_question")),
        methods=_string_list(payload.get("methods")),
        datasets=_string_list(payload.get("datasets")),
        contributions=_string_list(payload.get("contributions")),
        limitations=_string_list(payload.get("limitations")),
        main_results=_string_list(payload.get("main_results")),
        short_summary=_text_value(payload.get("short_summary")),
        missing_information=_string_list(payload.get("missing_information")),
        evidence_level=_text_value(payload.get("evidence_level")) or "metadata",
    )


def _read_relevance_from_payload(value: Any) -> ReadRelevance:
    """把 checkpoint 中的相关性 JSON 恢复为 ReadRelevance。"""

    payload = value if isinstance(value, dict) else {}
    decision = _text_value(payload.get("decision")) or "insufficient"
    if decision not in {"deep_read", "abstract_only", "insufficient"}:
        decision = "insufficient"
    return ReadRelevance(
        score=_score_value(payload.get("score")),
        decision=decision,
        reason=_text_value(payload.get("reason")) or "资料不足，无法可靠判断",
    )


def _full_text_from_payload(value: Any) -> FullTextStatus:
    """把 checkpoint 中的全文状态 JSON 恢复为 FullTextStatus。"""

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


def _restore_deep_read_count(results: list[PaperReadResult], checkpoint: JsonObject) -> int:
    """恢复已经消耗的全文精读次数，兼容 embedding 中断时的 pending 论文。"""

    restored_count = _restored_deep_read_count(results)
    checkpoint_count = _optional_int(checkpoint.get("deep_read_count"))
    if checkpoint_count is None:
        return restored_count
    # 中文注释：embedding 失败时，当前论文已经下载并转成 Markdown，但不会放进
    # read_results。checkpoint 里的 deep_read_count 会把这篇 pending 论文也算进去，
    # 这里取较大值，避免恢复后超过本次全文精读数量上限。
    return max(restored_count, checkpoint_count)


def _restored_deep_read_count(results: list[PaperReadResult]) -> int:
    """根据已恢复结果估算已经占用的全文精读数量。"""

    # 中文注释：恢复时不能简单用 relevance.decision 统计精读次数，因为有些论文
    # 判断为 deep_read 但因上限没有真正下载；只统计已经进入全文链路的状态，
    # 保证恢复后 deep_read_limit 仍然按原规则生效。
    counted_statuses = {"downloaded", "markdown_ready", "indexed", "download_failed", "parse_failed"}
    return sum(result.full_text.status in counted_statuses for result in results)


def _embedding_error_details(result: PaperReadResult, *, stage: str, message: str) -> JsonObject:
    """整理 embedding 不可用时写入 checkpoint 的诊断信息。"""

    # 中文注释：这里只放论文编号、标题、Markdown 路径和错误阶段等普通字段，
    # 不保存 embedding_connection 这类可能包含密钥或无法序列化的运行时对象。
    return {
        "resource": "embedding",
        "stage": stage,
        "message": message,
        "paper_id": result.paper.id,
        "paper_title": result.paper.title,
        "source_path": result.full_text.source_path,
        "markdown_path": result.full_text.markdown_path,
        "source_url": result.full_text.source_url,
    }


def _model_unavailable_message(response: LLMResponse) -> str:
    """把模型错误响应整理为用户可理解的不可用提示。"""

    detail = response.content or response.error_code or response.error_type or response.error_kind or "未知错误"
    status = f"HTTP {response.error_status_code}" if response.error_status_code else response.error_kind or "模型调用失败"
    return f"阅读模型当前不可用（{status}）：{detail}。请在模型设置中验证可用后继续执行。"


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


def _read_one_paper(
    paper: PaperDocument,
    *,
    topic: str,
    constraints: JsonObject,
    llm: ProviderSnapshot | None,
    config: Any,
    embedding_connection: EmbeddingConnection | None,
    embedding_error: str | None,
    allow_deep_read: bool,
    reporter: Any,
) -> tuple[PaperReadResult, bool]:
    """处理一篇论文：先做摘要笔记，再按判断尝试下载、转换和建立全文索引。"""

    title = paper.title.strip()
    if not title:
        return (
            PaperReadResult(
                paper=paper,
                full_text=FullTextStatus(status="not_requested", reason="论文没有标题，无法可靠阅读"),
                warnings=["论文缺少标题，未交给模型判断"],
            ),
            False,
        )
    note, relevance, warnings = _build_abstract_note(paper, topic=topic, constraints=constraints, llm=llm)
    result = PaperReadResult(paper=paper, note=note, relevance=relevance, warnings=warnings)
    # 中文注释：检索节点已经保证进入这里的论文都有摘要。
    # 阅读节点这里只根据模型给出的相关性判断是否需要全文精读，不再处理“没有摘要”的情况。
    should_deep_read = relevance.decision == "deep_read" and relevance.score >= config.deep_score_threshold
    if not should_deep_read:
        result.full_text = FullTextStatus(status="not_requested", reason="当前论文只保留摘要笔记")
        return result, False
    if not allow_deep_read:
        result.full_text = FullTextStatus(status="not_requested", reason="已达到本次全文精读数量上限")
        return result, False

    _report_progress(reporter, paper, "downloading_full_text", 0, 0)
    downloaded = download_paper_fulltext(
        paper,
        cache_dir=config.paper_cache_dir,
        connect_timeout_seconds=config.connect_timeout_seconds,
        download_timeout_seconds=config.download_timeout_seconds,
        max_file_size_mb=config.max_file_size_mb,
    )
    full_text = FullTextStatus(status=downloaded.status, reason=downloaded.reason, source_url=downloaded.source_url)
    # 只有下载成功的file_path才不为None,且file_path=data/cache/{sha256编码}/{source.pdf/source.html}
    if downloaded.file_path is None:
        result.full_text = full_text
        return result, True
    full_text.source_path = str(downloaded.file_path)

    _report_progress(reporter, paper, "converting_markdown", 0, 0)
    converted = convert_fulltext_to_markdown(
        paper,
        source_path=downloaded.file_path,
        source_url=downloaded.source_url,
    )
    if converted.markdown_path is None:
        result.full_text = FullTextStatus(
            status="parse_failed",
            reason="；".join(converted.warnings) or "无法将全文转换为 Markdown",
            source_url=downloaded.source_url,
            source_path=str(downloaded.file_path),
        )
        return result, True
    full_text.status = "markdown_ready"
    full_text.markdown_path = str(converted.markdown_path)
    full_text.page_count = converted.page_count
    if converted.warnings:
        result.warnings.extend(converted.warnings)

    _report_progress(reporter, paper, "saving_chunks", 0, 0)
    result.full_text = full_text
    if embedding_connection is None:
        # 中文注释：能走到这里说明全文已经下载并转成 Markdown，只差写入向量库。
        # embedding 不可用时不能把这篇论文当成完成，否则用户修好配置后无法补建索引。
        full_text.reason = embedding_error or "未配置可用的 embedding 服务，全文尚未写入向量库"
        raise ReadEmbeddingUnavailableError(
            full_text.reason,
            pending_result=result,
            details=_embedding_error_details(result, stage="embedding_config", message=full_text.reason),
        )
    try:
        index_result = index_markdown_chunks(
            paper,
            markdown_path=converted.markdown_path,
            source_url=downloaded.source_url,
            collection_path=Path(config.vector_store_path) / config.vector_store_collection,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            embedding_connection=embedding_connection,
        )
    except RuntimeError as exc:
        # 中文注释：read_vector_store.py 用 RuntimeError 表示 embedding 接口调用失败、
        # 返回格式不对或向量数量不一致。这类问题通常需要用户修好 embedding 服务后恢复。
        full_text.reason = f"全文已转成 Markdown，但 embedding 服务不可用：{exc}"
        raise ReadEmbeddingUnavailableError(
            full_text.reason,
            pending_result=result,
            details=_embedding_error_details(result, stage="embedding_call", message=str(exc)),
        ) from exc
    except (OSError, ValueError) as exc:
        full_text.reason = f"全文已转成 Markdown，但建立索引失败：{exc}"
        result.full_text = full_text
        return result, True
    full_text.status = "indexed"
    full_text.chunk_count = index_result.chunk_count
    result.full_text = full_text
    return result, True


def _build_abstract_note(
    paper: PaperDocument,
    *,
    topic: str,
    constraints: JsonObject,
    llm: ProviderSnapshot | None,
) -> tuple[ReadNote, ReadRelevance, list[str]]:
    """只使用标题、摘要和基本信息整理笔记，不根据常识补写论文未说明的内容。"""

    if not (paper.abstract or "").strip():
        # 中文注释：没有摘要时不需要调用模型，直接记录“资料不足”。这不是
        # 模型不可用，因此不会触发 checkpoint；后续如果有全文 URL，仍可按
        # missing_abstract 规则尝试下载全文。
        note = ReadNote(
            short_summary=f"《{paper.title}》只有标题和基本信息，尚无摘要可供整理。",
            missing_information=["研究问题", "研究方法", "数据集", "实验结果", "论文不足"],
            evidence_level="metadata",
        )
        return note, ReadRelevance(reason="论文没有摘要，现有资料不足"), []
    if llm is None:
        # 中文注释：没有配置模型，或者配置文件无法组装出模型，才算“模型不可用”。
        # 这种情况用户需要先去设置页处理，所以这里保存现场并中断整次阅读。
        raise ReadModelUnavailableError("未配置可用的阅读模型，请在模型设置中验证可用后继续执行。")
    try:
        response = llm.provider.chat_with_retry(_abstract_messages(paper, topic, constraints), temperature=0)
    except Exception as exc:
        # 中文注释：能走到这里，说明模型调用本身没有成功完成，例如网络中断、
        # 请求超时、SDK 报错等。这属于“配置了模型但无法成功调用”，需要中断。
        raise ReadModelUnavailableError(f"阅读模型调用失败，请验证当前模型可用后继续执行：{exc}") from exc
    if not response.ok:
        # 中文注释：provider 明确告诉我们 HTTP、鉴权、限流或服务端失败，说明
        # 当前模型无法正常调用。这是模型不可用，需要交给 checkpoint 恢复流程处理。
        raise ReadModelUnavailableError(_model_unavailable_message(response))
    # 中文注释：只要模型成功返回内容，就说明模型“可用”。如果内容不是合格 JSON，
    # 那只是这篇论文的模型效果不好，不能中断整次阅读；改用保守笔记继续处理。
    parsed = _parse_model_result(response)
    if parsed is None:
        return _fallback_abstract_note(paper, warning="阅读模型返回内容格式不符合要求，已改用保守摘要笔记")
    try:
        # 中文注释：字段缺失或类型不理想时，_note_from_model 会尽量补空值；如果
        # 仍出现意外错误，也只影响当前论文，不能把它当成模型不可用。
        return _note_from_model(parsed)
    except Exception as exc:
        return _fallback_abstract_note(paper, warning=f"阅读模型返回内容无法整理，已改用保守摘要笔记：{exc}")


def _fallback_abstract_note(paper: PaperDocument, *, warning: str) -> tuple[ReadNote, ReadRelevance, list[str]]:
    """模型已成功返回但内容不可用时，生成一份保守笔记继续处理当前论文。"""

    # 中文注释：走到这个函数时，模型接口已经调用成功，所以不能把它当成
    # “模型不可用”去中断整次任务。这里不猜论文没写的内容，只把标题和摘要
    # 原文整理成最保守的笔记，让后面的论文还能继续处理。
    abstract = (paper.abstract or "").strip()
    title = paper.title.strip() or "未命名论文"
    summary_source = abstract or title
    # 中文注释：摘要可能很长，保守笔记只取前面一小段，避免把完整摘要原封不动
    # 塞进结果里，也避免后续回复节点展示时过长。
    short_summary = summary_source[:500]
    if len(summary_source) > 500:
        short_summary += "……"
    note = ReadNote(
        short_summary=f"《{title}》的模型阅读结果不可用，以下仅保留论文原始摘要片段：{short_summary}",
        missing_information=["研究问题", "研究方法", "数据集", "实验结果", "论文不足"],
        evidence_level="abstract" if abstract else "metadata",
    )
    # 中文注释：因为没有拿到可信的模型判断，这里把相关性设为 insufficient。
    # 这样系统会跳过这篇论文的模型精读判断，不会因为兜底内容误判为高相关。
    relevance = ReadRelevance(score=0, decision="insufficient", reason="模型已返回内容，但内容格式或字段无法可靠使用")
    return note, relevance, [warning]


def _abstract_messages(paper: PaperDocument, topic: str, constraints: JsonObject) -> list[JsonObject]:
    """构造摘要阅读提示，明确要求模型只使用提供内容并返回固定 JSON。"""

    payload = {
        "用户主题": topic,
        "用户要求": constraints,
        "论文": {"标题": paper.title, "摘要": paper.abstract, "作者": paper.authors, "年份": paper.year, "发表位置": paper.venue},
    }
    instruction = """你是论文摘要阅读助手。只能依据给出的 JSON 内容，不能猜测论文没有写明的信息。
请只返回一个 JSON 对象，字段必须包含：main_question、methods、datasets、contributions、limitations、main_results、short_summary、missing_information、evidence_level、score、decision、reason。
methods、datasets、contributions、limitations、main_results、missing_information 必须是字符串数组；没有证据就用空数组。
evidence_level 只能是 metadata 或 abstract。score 是 0 到 100 的整数。decision 只能是 deep_read、abstract_only、insufficient。"""
    return [{"role": "system", "content": instruction}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _parse_model_result(response: LLMResponse) -> JsonObject | None:
    """从模型文本中取出 JSON 对象，格式不合格时返回空值让调用方触发恢复中断。"""

    if not response.ok or not response.content.strip():
        return None
    text = response.content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _note_from_model(payload: JsonObject) -> tuple[ReadNote, ReadRelevance, list[str]]:
    """检查模型 JSON 的字段类型并补齐缺失项，避免不规范响应污染后续状态。"""

    note = ReadNote(
        main_question=_text_value(payload.get("main_question")),
        methods=_string_list(payload.get("methods")),
        datasets=_string_list(payload.get("datasets")),
        contributions=_string_list(payload.get("contributions")),
        limitations=_string_list(payload.get("limitations")),
        main_results=_string_list(payload.get("main_results")),
        short_summary=_text_value(payload.get("short_summary"))[:800],
        missing_information=_string_list(payload.get("missing_information")),
        evidence_level="abstract" if payload.get("evidence_level") == "abstract" else "metadata",
    )
    score = _score_value(payload.get("score"))
    decision = str(payload.get("decision") or "insufficient")
    if decision not in {"deep_read", "abstract_only", "insufficient"}:
        decision = "insufficient"
    relevance = ReadRelevance(score=score, decision=decision, reason=_text_value(payload.get("reason")) or "模型未提供判断原因")
    return note, relevance, []


def _deep_read_limit(constraints: JsonObject, total: int) -> int:
    """读取用户可选的精读数量限制，未提供时允许处理全部高相关论文。"""

    raw = constraints.get("deep_read_limit", constraints.get("max_deep_read", total))
    try:
        return max(0, min(int(raw), total))
    except (TypeError, ValueError):
        return total


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

def _resolve_llm(state: State, agent_name: str) -> ProviderSnapshot | None:
    """优先使用状态注入的阅读模型，正式运行时再从本地配置加载默认模型。"""

    injected = state.get("read_node_llm")
    if isinstance(injected, ProviderSnapshot):
        return injected
    if injected is None:
        return load_read_node_llm(agent_name)
    return None


def load_read_node_llm(agent_name: str | None = None) -> ProviderSnapshot | None:
    """读取模型配置并装配阅读摘要所需的模型，配置错误时返回空值继续本地处理。"""

    model_path = Path("config/model.json")
    if not model_path.exists():
        return None
    try:
        config = ModelConfig.from_dict(json.loads(model_path.read_text(encoding="utf-8")), SystemConfig.load())
        return make_provider(config, agent_name)
    except Exception:
        return None


def _resolve_embedding_connection(system_config: SystemConfig, timeout_seconds: int) -> tuple[EmbeddingConnection | None, str | None]:
    """从本地模型配置读取 embedding 服务信息，缺失或错误时返回可展示的原因。"""

    model_path = Path("config/model.json")
    if not model_path.exists():
        return None, "未找到模型配置，全文尚未写入向量库"
    try:
        model_config = ModelConfig.from_dict(json.loads(model_path.read_text(encoding="utf-8")), system_config)
        profile, provider = model_config.resolve_embedding_provider_config()
        return (
            EmbeddingConnection(
                api_base=str(provider.api_base),
                api_key=provider.api_key,
                model_name=profile.model_name,
                extra_headers=dict(provider.extra_headers),
                dimensions=profile.dimensions,
                batch_size=int(profile.batch_size or 32),
                timeout_seconds=timeout_seconds,
            ),
            None,
        )
    except Exception as exc:
        return None, f"embedding 服务配置不可用：{exc}"


def _resolve_sink(state: State) -> ReadPersistenceSink | None:
    """仅在会话信息完整时启用阅读产物写盘，普通脚本调用仍可直接运行。"""

    repo = state.get("session_repo")
    session_key = str(state.get("session_key") or "").strip()
    turn_id = str(state.get("turn_id") or "").strip()
    if not isinstance(repo, SessionRepository) or not session_key or not turn_id:
        return None
    return ReadPersistenceSink(repo, session_key=session_key, turn_id=turn_id)


def _resolve_reporter(state: State):
    """从运行上下文中取出阅读节点上报器，没有前端同步接口时返回空值。"""

    runtime = state.get("runtime_context")
    if not isinstance(runtime, WorkflowRuntimeContext) or runtime.sync_port is None:
        return None
    return runtime.sync_port.for_node("read", "论文阅读")


def _report_progress(reporter: Any, paper: PaperDocument, stage: str, completed: int, total: int, **extra: Any) -> None:
    """统一上报当前论文和处理数量，避免不同阶段缺少前端需要的关键信息。"""

    if reporter is None:
        return
    messages = {
        "reading_abstract": "正在阅读论文摘要",
        "downloading_full_text": "正在下载论文全文",
        "converting_markdown": "正在将论文转换为 Markdown",
        "saving_chunks": "正在建立全文检索索引",
        "paper_completed": "论文阅读完成",
    }
    reporter.progress(
        messages.get(stage, "正在处理论文"),
        stage=stage,
        current_title=paper.title,
        completed=completed,
        total=total,
        **extra,
    )


def _build_summary(results: list[PaperReadResult], deep_read_count: int) -> JsonObject:
    """汇总每篇论文的处理结果，让后续节点不必重新遍历全部阅读详情。"""

    return {
        "total_paper_count": len(results),
        "deep_read_attempt_count": deep_read_count,
        "deep_read_candidate_count": sum(item.relevance.decision == "deep_read" for item in results),
        "indexed_paper_count": sum(item.full_text.status == "indexed" for item in results),
        "failed_fulltext_count": sum(item.full_text.status in {"download_failed", "parse_failed"} for item in results),
        "insufficient_paper_count": sum(item.relevance.decision == "insufficient" for item in results),
    }



def _text_value(value: Any) -> str:
    """把可能为空的模型字段安全整理为字符串。"""

    return str(value).strip() if isinstance(value, str) else ""


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
