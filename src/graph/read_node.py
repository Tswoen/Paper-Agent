from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from src.utils.read_utils.read_fulltext import async_convert_fulltext_to_markdown
from src.models.read_models import FullTextStatus, PaperReadResult, ReadNote, ReadRelevance
from src.repositories.node_persistence.read_persistence import ReadPersistenceSink
from src.repositories.chroma.read_vector_store import EmbeddingConnection, async_index_chunk_file, async_index_markdown_chunks
from src.graph.checkpoint_halt import halt_with_checkpoint
from src.graph.runtime import WorkflowRuntimeContext
from src.graph.runtime_resources import WorkflowRuntimeResources
from src.graph.state_models import JsonObject, State
from src.llm import ModelConfig, ProviderSnapshot, SystemConfig, make_provider
from src.llm.base import LLMResponse
from src.paper_retrieval.download import async_download_paper_fulltext
from src.paper_retrieval.models import PaperDocument
from src.repositories.sessions.base import SessionRepository
from src.utils.read_utils.chunkers import async_build_chunks_file
from src.utils.read_utils.extraction import async_extract_paper_from_chunks, async_write_failed_extraction, empty_extraction, extraction_payload


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


@dataclass(slots=True)
class PaperTaskInput:
    """保存一篇论文并发处理时需要的固定输入和恢复现场。"""

    paper: PaperDocument
    position: int
    restored_status: JsonObject
    restored_result: PaperReadResult | None = None


class DeepReadQuota:
    """统一控制本次阅读批次还剩多少全文精读名额。"""

    def __init__(self, used_count: int, limit: int):
        """记录已经占用的名额和总上限，供并发论文任务共享。"""

        self._used_count = max(0, int(used_count))
        self._limit = max(0, int(limit))
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        """尝试占用一个全文精读名额，成功时返回真。"""

        async with self._lock:
            if self._used_count >= self._limit:
                return False
            self._used_count += 1
            return True

    @property
    def used_count(self) -> int:
        """返回当前已经占用的全文精读名额。"""

        return self._used_count


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


def _legacy_run_read_node_serial():
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
        runtime_resources = _resolve_runtime_resources(state)
        llm = _resolve_llm(state, config.agent_name)
        embedding_connection, embedding_error = _resolve_embedding_connection(
            system_config,
            config.download_timeout_seconds,
            runtime_resources=runtime_resources,
        )

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
            runtime_resources=runtime_resources,
        )
        start_position = len(results) + 1

        for position, paper in enumerate(papers[start_position - 1 :], start=start_position):
            _report_progress(reporter, paper, "reading_abstract", position - 1, len(papers))
            try:
                # 阅读每一篇论文，返回论文阅读的结果，以及是否采用了全文精读
                result, used_deep_read = _legacy_read_one_paper(
                    paper,
                    topic=request.topic,
                    constraints=request.constraints,
                    llm=llm,
                    config=config,
                    embedding_connection=embedding_connection,
                    embedding_error=embedding_error,
                    allow_deep_read=deep_read_count < deep_read_limit,
                    reporter=reporter,
                    runtime_resources=runtime_resources,
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


def run_read_node():
    """生成执行图中的阅读节点，直接返回异步节点实现。"""

    async def _node(state: State) -> State:
        """阅读节点已经具备 async 主流程，图层改成 ainvoke 后这里直接 await 即可。"""

        return await _run_read_node_async(state)

    return _node


async def _run_read_node_async(state: State) -> State:
    """并发处理多篇论文，并在资源不可用时按论文状态保存恢复现场。"""

    request = state.get("request")
    if request is None:
        raise ValueError("阅读节点缺少用户请求，无法判断论文主题")
    system_config = SystemConfig.load()
    config = system_config.read
    checkpoint = _checkpoint_from_state(state)
    papers = _deduplicate_papers(list(state.get("search_results") or _papers_from_payload(checkpoint.get("search_results"))))

    reporter = _resolve_reporter(state)
    sink = _resolve_sink(state)
    runtime_resources = _resolve_runtime_resources(state)
    llm = _resolve_llm(state, config.agent_name)
    embedding_connection, embedding_error = _resolve_embedding_connection(
        system_config,
        config.download_timeout_seconds,
        runtime_resources=runtime_resources,
    )

    recovered_results = _restore_read_results(state, papers, checkpoint)
    results_by_paper_id = {result.paper.id: result for result in recovered_results}
    artifact_refs: list[JsonObject] = list(state.get("read_artifact_refs") or checkpoint.get("read_artifact_refs") or [])
    paper_runtime_statuses = _restore_paper_runtime_statuses(papers, checkpoint, recovered_results)
    deep_read_count = _restore_deep_read_count(recovered_results, checkpoint)
    deep_read_limit = _deep_read_limit(request.constraints, len(papers))
    completed_counter = CompletedPaperCounter(len(recovered_results))

    if reporter is not None:
        reporter.started(
            f"准备阅读 {len(papers)} 篇论文",
            stage="read_start",
            total=len(papers),
            completed=len(recovered_results),
            resumed_paper_count=len(recovered_results),
        )

    try:
        deep_read_count = await _process_papers_concurrently(
            papers=papers,
            topic=request.topic,
            constraints=request.constraints,
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
            deep_read_count=deep_read_count,
            deep_read_limit=deep_read_limit,
        )
    except ReadResourceUnavailableError as exc:
        deep_read_count = int(getattr(exc, "deep_read_count", deep_read_count))
        _halt_for_resource_unavailable(
            state,
            papers=papers,
            results=_ordered_results(papers, results_by_paper_id),
            artifact_refs=artifact_refs,
            position=_next_pending_position(papers, paper_runtime_statuses),
            error=exc,
            sink=sink,
            reporter=reporter,
            deep_read_count=deep_read_count,
            deep_read_limit=deep_read_limit,
            paper_runtime_statuses=paper_runtime_statuses,
        )

    results = _ordered_results(papers, results_by_paper_id)
    summary = _build_summary(results, deep_read_count)
    if sink is not None:
        try:
            persisted = await asyncio.to_thread(sink.persist_summary, summary, results)
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
            completed=len(results),
            indexed_paper_count=summary["indexed_paper_count"],
        )
    updated = dict(state)
    updated.update(
        read_results=[result.to_dict() for result in results],
        read_summary=summary,
        read_artifact_refs=artifact_refs,
        read_paper_statuses=_ordered_paper_runtime_statuses(papers, paper_runtime_statuses),
        current_step="read",
    )
    if checkpoint:
        updated["search_results"] = papers
    updated.pop("read_resume_checkpoint", None)
    return cast(State, updated)


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
    paper_runtime_statuses: dict[str, JsonObject],
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
        paper_runtime_statuses=paper_runtime_statuses,
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


def _legacy_build_resume_checkpoint(
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
                extraction=_json_object(item.get("extraction")),
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
    runtime_resources: WorkflowRuntimeResources | None = None,
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
            vector_store_path=config.vector_store_path,
            collection_name=config.vector_store_collection,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            embedding_connection=embedding_connection,
            runtime_resources=runtime_resources,
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
        extraction=_json_object(value.get("extraction")),
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
        paperId=str(value.get("paperId")) if value.get("paperId") is not None else None,
        publication_date=str(value.get("publication_date") or ""),
        journal_conference=str(value.get("journal_conference") or value.get("journal/conference") or ""),
        volume=str(value.get("volume") or ""),
        issue=str(value.get("issue") or ""),
        language=str(value.get("language") or ""),
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


def _json_object(value: Any) -> JsonObject:
    """把可能来自旧状态的字段整理成普通字典。

    中文注释：恢复 checkpoint 时不能假设旧数据一定规整。如果 extraction 不是字典，
    就返回空字典，避免一篇旧记录影响整批阅读恢复。
    """

    return dict(value) if isinstance(value, dict) else {}


def _legacy_read_one_paper(
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
    runtime_resources: WorkflowRuntimeResources | None = None,
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
    note, relevance, warnings = _legacy_build_abstract_note(paper, topic=topic, constraints=constraints, llm=llm)
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
        runtime_resources=runtime_resources,
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
            vector_store_path=config.vector_store_path,
            collection_name=config.vector_store_collection,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            embedding_connection=embedding_connection,
            runtime_resources=runtime_resources,
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


def _legacy_build_abstract_note(
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

def _resolve_runtime_resources(state: State) -> WorkflowRuntimeResources | None:
    """从运行时上下文里取出单次 run 共享的资源对象。"""

    runtime = state.get("runtime_context")
    if not isinstance(runtime, WorkflowRuntimeContext):
        return None
    return runtime.resources if isinstance(runtime.resources, WorkflowRuntimeResources) else None


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

    global_statistics = {
        "total_papers_received": len(results),
        "passed_abstract_filter": sum(item.relevance.decision == "deep_read" for item in results),
        "fulltext_downloaded": sum(bool(item.full_text.source_path) for item in results),
        "extraction_succeeded": sum(bool(item.extraction and any(str(value).strip() for value in item.extraction.values())) for item in results),
        "vectorized": sum(item.full_text.status == "indexed" for item in results),
        "errors": _read_errors(results),
    }
    return {
        "total_paper_count": len(results),
        "deep_read_attempt_count": deep_read_count,
        "deep_read_candidate_count": sum(item.relevance.decision == "deep_read" for item in results),
        "indexed_paper_count": sum(item.full_text.status == "indexed" for item in results),
        "failed_fulltext_count": sum(item.full_text.status in {"download_failed", "parse_failed"} for item in results),
        "insufficient_paper_count": sum(item.relevance.decision == "insufficient" for item in results),
        "subtopics": _read_subtopics(results),
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
        "processing_status": "completed" if result.full_text.status == "indexed" else result.full_text.status,
    }


def _read_errors(results: list[PaperReadResult]) -> list[JsonObject]:
    """把下载、解析、提取和向量化失败整理成统一错误列表。"""

    errors: list[JsonObject] = []
    for result in results:
        paper_id = result.paper.paperId or result.paper.id
        status = result.full_text.status
        if status in {"download_failed", "no_url"}:
            errors.append({"paperId": paper_id, "stage": "download", "reason": result.full_text.reason})
        elif status == "parse_failed":
            errors.append({"paperId": paper_id, "stage": "parse", "reason": result.full_text.reason})
        elif result.full_text.reason and status != "indexed":
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
    paper_runtime_statuses: dict[str, JsonObject],
    completed_counter: CompletedPaperCounter,
    deep_read_count: int,
    deep_read_limit: int,
) -> int:
    """按论文建立任务池，哪篇先完成就先落盘、先更新进度。"""

    quota = DeepReadQuota(deep_read_count, deep_read_limit)
    work_items = _build_pending_paper_tasks(papers, paper_runtime_statuses, results_by_paper_id)
    if not work_items:
        return quota.used_count

    semaphore = runtime_resources.paper_task_semaphore if runtime_resources is not None else _FALLBACK_PAPER_TASK_SEMAPHORE
    tasks: dict[asyncio.Task[PaperReadResult], PaperTaskInput] = {}
    for item in work_items:
        task = asyncio.create_task(
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
                deep_read_quota=quota,
                total_paper_count=len(papers),
                semaphore=semaphore,
            )
        )
        tasks[task] = item

    pending = set(tasks)
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        resource_error: ReadResourceUnavailableError | None = None
        for task in done:
            item = tasks[task]
            try:
                result = task.result()
            except ReadResourceUnavailableError as exc:
                if resource_error is None:
                    resource_error = exc
            else:
                results_by_paper_id[item.paper.id] = result
                await _persist_completed_paper_async(result, sink=sink, artifact_refs=artifact_refs, reporter=reporter)
                _set_paper_runtime_status(
                    paper_runtime_statuses,
                    item.paper,
                    status="completed",
                    current_stage="paper_completed",
                    result=result,
                    error_message="",
                )
                completed = completed_counter.increment()
                _report_progress(
                    reporter,
                    item.paper,
                    "paper_completed",
                    completed,
                    len(papers),
                    current_status=result.full_text.status,
                    paper_position=item.position,
                )
        if resource_error is not None:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            setattr(resource_error, "deep_read_count", quota.used_count)
            raise resource_error
    return quota.used_count


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
    deep_read_quota: DeepReadQuota,
    total_paper_count: int,
    semaphore: asyncio.Semaphore,
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
                deep_read_quota=deep_read_quota,
                total_paper_count=total_paper_count,
            )
        except ReadResourceUnavailableError as exc:
            _mark_paper_waiting_resource(paper_runtime_statuses, item.paper, exc)
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return PaperReadResult(
                paper=item.paper,
                full_text=FullTextStatus(status="not_requested", reason="当前论文处理发生异常"),
                warnings=[f"当前论文处理失败：{exc}"],
            )


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
    deep_read_quota: DeepReadQuota,
    total_paper_count: int,
) -> PaperReadResult:
    """处理一篇论文，单篇内部仍保持顺序执行，只是批次层改成并发。"""

    paper = item.paper
    title = paper.title.strip()
    if not title:
        return PaperReadResult(
            paper=paper,
            full_text=FullTextStatus(status="not_requested", reason="论文没有标题，无法可靠阅读"),
            warnings=["论文缺少标题，未交给模型判断"],
        )

    restored_stage = str(item.restored_status.get("current_stage") or "").strip()
    result = item.restored_result
    if result is None or restored_stage not in {"downloading_full_text", "converting_markdown", "chunking_full_text", "extracting_full_text", "saving_chunks"}:
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
        note, relevance, warnings = await _build_abstract_note(
            paper,
            topic=topic,
            constraints=constraints,
            llm=llm,
            runtime_resources=runtime_resources,
        )
        result = PaperReadResult(paper=paper, note=note, relevance=relevance, warnings=warnings)
    else:
        _set_paper_runtime_status(
            paper_runtime_statuses,
            paper,
            status="running",
            current_stage=restored_stage,
            result=result,
            error_message="",
        )

    should_deep_read = result.relevance.decision == "deep_read" and result.relevance.score >= config.deep_score_threshold
    if not should_deep_read:
        result.full_text = FullTextStatus(status="not_requested", reason="当前论文只保留摘要笔记")
        return result

    if not _paper_has_reserved_deep_read(paper_runtime_statuses, paper):
        if not await deep_read_quota.try_acquire():
            result.full_text = FullTextStatus(status="not_requested", reason="已达到本次全文精读数量上限")
            return result
        _set_paper_runtime_status(
            paper_runtime_statuses,
            paper,
            status="running",
            current_stage=restored_stage or "reading_abstract",
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
        downloaded = await async_download_paper_fulltext(
            paper,
            cache_dir=config.paper_cache_dir,
            connect_timeout_seconds=config.connect_timeout_seconds,
            download_timeout_seconds=config.download_timeout_seconds,
            max_file_size_mb=config.max_file_size_mb,
            runtime_resources=runtime_resources,
        )
        full_text = FullTextStatus(status=downloaded.status, reason=downloaded.reason, source_url=downloaded.source_url)
        if downloaded.file_path is None:
            result.full_text = full_text
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
        converted = await async_convert_fulltext_to_markdown(
            paper,
            source_path=source_path,
            source_url=full_text.source_url,
        )
        if converted.markdown_path is None:
            result.full_text = FullTextStatus(
                status="parse_failed",
                reason="；".join(converted.warnings) or "无法将全文转换为 Markdown",
                source_url=full_text.source_url,
                source_path=str(source_path),
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
    chunk_build = await async_build_chunks_file(paper, markdown_path=markdown_path)
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
    if llm is None:
        raise ReadModelUnavailableError("未配置可用的阅读模型，无法从全文 chunks 提取论文结构化信息")
    try:
        extraction_record = await async_extract_paper_from_chunks(
            paper,
            chunks_path=chunk_build.chunks_path,
            llm=llm,
            runtime_resources=runtime_resources,
        )
        result.extraction = extraction_payload(extraction_record)
    except RuntimeError as exc:
        raise ReadModelUnavailableError(f"全文提取模型调用失败，请验证阅读模型可用后继续执行：{exc}") from exc
    except (OSError, ValueError) as exc:
        result.extraction = empty_extraction()
        result.warnings.append(f"全文结构化提取失败：{exc}")
        await async_write_failed_extraction(paper, chunks_path=chunk_build.chunks_path, reason=str(exc))

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
    if embedding_connection is None:
        full_text.reason = embedding_error or "未配置可用的 embedding 服务，全文尚未写入向量库"
        result.full_text = full_text
        raise ReadEmbeddingUnavailableError(
            full_text.reason,
            pending_result=result,
            details=_embedding_error_details(result, stage="embedding_config", message=full_text.reason),
        )
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
    except RuntimeError as exc:
        full_text.reason = f"全文已转成 Markdown，但 embedding 服务不可用：{exc}"
        result.full_text = full_text
        raise ReadEmbeddingUnavailableError(
            full_text.reason,
            pending_result=result,
            details=_embedding_error_details(result, stage="embedding_call", message=str(exc)),
        ) from exc
    except (OSError, ValueError) as exc:
        full_text.reason = f"全文已转成 Markdown，但建立索引失败：{exc}"
        result.full_text = full_text
        return result
    full_text.status = "indexed"
    full_text.reason = ""
    full_text.chunk_count = index_result.chunk_count
    result.full_text = full_text
    return result


async def _build_abstract_note(
    paper: PaperDocument,
    *,
    topic: str,
    constraints: JsonObject,
    llm: ProviderSnapshot | None,
    runtime_resources: WorkflowRuntimeResources | None,
) -> tuple[ReadNote, ReadRelevance, list[str]]:
    """异步读取摘要并整理笔记，模型失败时保留原有的保守降级逻辑。"""

    if not (paper.abstract or "").strip():
        note = ReadNote(
            short_summary=f"《{paper.title}》只有标题和基本信息，暂无摘要可供整理。",
            missing_information=["研究问题", "研究方法", "数据集", "实验结果", "论文不足"],
            evidence_level="metadata",
        )
        return note, ReadRelevance(reason="论文没有摘要，现有资料不足"), []
    if llm is None:
        raise ReadModelUnavailableError("未配置可用的阅读模型，请在模型设置中验证可用后继续执行。")
    try:
        response = await _call_read_model_async(
            llm,
            _abstract_messages(paper, topic, constraints),
            runtime_resources=runtime_resources,
        )
    except Exception as exc:
        raise ReadModelUnavailableError(f"阅读模型调用失败，请验证当前模型可用后继续执行：{exc}") from exc
    if not response.ok:
        raise ReadModelUnavailableError(_model_unavailable_message(response))
    parsed = _parse_model_result(response)
    if parsed is None:
        return _fallback_abstract_note(paper, warning="阅读模型返回内容格式不符合要求，已改用保守摘要笔记")
    try:
        return _note_from_model(parsed)
    except Exception as exc:
        return _fallback_abstract_note(paper, warning=f"阅读模型返回内容无法整理，已改用保守摘要笔记：{exc}")


async def _call_read_model_async(
    llm: ProviderSnapshot,
    messages: list[JsonObject],
    *,
    runtime_resources: WorkflowRuntimeResources | None,
) -> LLMResponse:
    """统一控制摘要模型并发，避免阅读批次并发后把同一个模型打满。"""

    semaphore = runtime_resources.read_model_semaphore if runtime_resources is not None else _FALLBACK_READ_MODEL_SEMAPHORE
    async with semaphore:
        return await llm.provider.chat(messages, temperature=0)


async def _persist_completed_paper_async(
    result: PaperReadResult,
    *,
    sink: ReadPersistenceSink | None,
    artifact_refs: list[JsonObject],
    reporter: Any,
) -> None:
    """在 async 阅读流程里把单篇结果落盘，但文件写入本身仍放到线程里。"""

    if sink is None:
        return
    try:
        persisted = await asyncio.to_thread(sink.persist_paper, result)
        artifact_refs.extend(persisted.artifacts)
        if reporter is not None:
            for artifact in persisted.artifacts:
                reporter.artifact(artifact, stage="paper_artifact_ready")
    except Exception as exc:
        result.warnings.append(f"阅读结果无法保存到会话目录：{exc}")


def _restore_paper_runtime_statuses(
    papers: list[PaperDocument],
    checkpoint: JsonObject,
    recovered_results: list[PaperReadResult],
) -> dict[str, JsonObject]:
    """从 checkpoint 中恢复每篇论文自己的处理状态，供并发任务继续运行。"""

    statuses = {paper.id: _default_paper_runtime_status(paper) for paper in papers}
    for item in list(checkpoint.get("paper_runtime_statuses") or []):
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id") or "").strip()
        paper = next((candidate for candidate in papers if candidate.id == paper_id), None)
        if paper is None:
            continue
        statuses[paper_id] = _merge_runtime_status(statuses[paper_id], item, paper)

    # 中文注释：这里兼容旧版本 checkpoint 里“pending_read_result + pending_resume_phase”的写法。
    pending_result = _read_result_from_payload(checkpoint.get("pending_read_result"), papers)
    if pending_result is not None:
        statuses[pending_result.paper.id] = {
            **statuses.get(pending_result.paper.id, _default_paper_runtime_status(pending_result.paper)),
            "paper_id": pending_result.paper.id,
            "paper_title": pending_result.paper.title,
            "status": str(checkpoint.get("recovery_status") or "waiting_embedding"),
            "current_stage": str(checkpoint.get("pending_resume_phase") or "saving_chunks"),
            "source_path": pending_result.full_text.source_path,
            "markdown_path": pending_result.full_text.markdown_path,
            "error_message": str(checkpoint.get("message") or ""),
            "deep_read_reserved": True,
            "result": pending_result.to_dict(),
        }

    for result in recovered_results:
        statuses[result.paper.id] = {
            **statuses.get(result.paper.id, _default_paper_runtime_status(result.paper)),
            "paper_id": result.paper.id,
            "paper_title": result.paper.title,
            "status": "completed",
            "current_stage": "paper_completed",
            "source_path": result.full_text.source_path,
            "markdown_path": result.full_text.markdown_path,
            "error_message": "",
            "deep_read_reserved": _infer_deep_read_reserved(result),
            "result": result.to_dict(),
        }
    return statuses


def _build_pending_paper_tasks(
    papers: list[PaperDocument],
    paper_runtime_statuses: dict[str, JsonObject],
    results_by_paper_id: dict[str, PaperReadResult],
) -> list[PaperTaskInput]:
    """找出还没完成的论文，整理成待投递的并发任务输入。"""

    tasks: list[PaperTaskInput] = []
    for position, paper in enumerate(papers, start=1):
        if paper.id in results_by_paper_id:
            continue
        status = dict(paper_runtime_statuses.get(paper.id) or _default_paper_runtime_status(paper))
        if str(status.get("status") or "") == "completed":
            continue
        tasks.append(
            PaperTaskInput(
                paper=paper,
                position=position,
                restored_status=status,
                restored_result=_paper_result_from_runtime_status(status, paper),
            )
        )
    return tasks


def _ordered_results(papers: list[PaperDocument], results_by_paper_id: dict[str, PaperReadResult]) -> list[PaperReadResult]:
    """把并发完成的论文结果重新按原论文顺序排好，方便后续节点稳定消费。"""

    return [results_by_paper_id[paper.id] for paper in papers if paper.id in results_by_paper_id]


def _ordered_paper_runtime_statuses(
    papers: list[PaperDocument],
    paper_runtime_statuses: dict[str, JsonObject],
) -> list[JsonObject]:
    """把每篇论文的运行状态按原顺序输出，便于恢复和前端展示。"""

    return [dict(paper_runtime_statuses.get(paper.id) or _default_paper_runtime_status(paper)) for paper in papers]


def _next_pending_position(papers: list[PaperDocument], paper_runtime_statuses: dict[str, JsonObject]) -> int:
    """找出当前批次里第一篇未完成论文的位置，兼容旧前端依赖的 next_position 字段。"""

    for position, paper in enumerate(papers, start=1):
        status = str((paper_runtime_statuses.get(paper.id) or {}).get("status") or "pending")
        if status != "completed":
            return position
    return len(papers) + 1


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


def _merge_runtime_status(base: JsonObject, payload: JsonObject, paper: PaperDocument) -> JsonObject:
    """把 checkpoint 里恢复出来的论文状态整理成统一结构。"""

    result_payload = payload.get("result")
    result = _paper_result_from_runtime_status({"result": result_payload}, paper)
    return {
        **base,
        "paper_id": paper.id,
        "paper_title": paper.title,
        "status": str(payload.get("status") or base.get("status") or "pending"),
        "current_stage": str(payload.get("current_stage") or base.get("current_stage") or "pending"),
        "source_path": _optional_text(payload.get("source_path")) or (result.full_text.source_path if result is not None else base.get("source_path")),
        "markdown_path": _optional_text(payload.get("markdown_path")) or (result.full_text.markdown_path if result is not None else base.get("markdown_path")),
        "error_message": str(payload.get("error_message") or base.get("error_message") or ""),
        "deep_read_reserved": bool(payload.get("deep_read_reserved") if payload.get("deep_read_reserved") is not None else (result is not None and _infer_deep_read_reserved(result))),
        "result": result.to_dict() if result is not None else base.get("result"),
    }


def _paper_result_from_runtime_status(payload: JsonObject, paper: PaperDocument) -> PaperReadResult | None:
    """从单篇论文状态里恢复出中间结果，便于从下载后或 Markdown 后继续执行。"""

    result_payload = payload.get("result")
    if not isinstance(result_payload, dict):
        return None
    merged = dict(result_payload)
    merged["paper"] = paper.to_dict()
    return _read_result_from_payload(merged, [paper])


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
    """更新某篇论文的当前状态，保证 checkpoint 里随时都有可恢复的信息。"""

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


def _mark_paper_waiting_resource(
    paper_runtime_statuses: dict[str, JsonObject],
    paper: PaperDocument,
    error: ReadResourceUnavailableError,
) -> None:
    """在模型或 embedding 不可用时，把当前论文的中间结果和等待原因记下来。"""

    result = error.pending_result if isinstance(error, ReadEmbeddingUnavailableError) else _paper_result_from_runtime_status(
        paper_runtime_statuses.get(paper.id) or {},
        paper,
    )
    current_stage = str((paper_runtime_statuses.get(paper.id) or {}).get("current_stage") or "reading_abstract")
    _set_paper_runtime_status(
        paper_runtime_statuses,
        paper,
        status=error.recovery_status,
        current_stage=current_stage,
        result=result,
        error_message=str(error),
        deep_read_reserved=_infer_deep_read_reserved(result) if result is not None else bool((paper_runtime_statuses.get(paper.id) or {}).get("deep_read_reserved")),
    )


def _paper_has_reserved_deep_read(paper_runtime_statuses: dict[str, JsonObject], paper: PaperDocument) -> bool:
    """判断这篇论文是否已经占用了全文精读名额，避免恢复时重复扣减。"""

    payload = paper_runtime_statuses.get(paper.id) or {}
    if bool(payload.get("deep_read_reserved")):
        return True
    result = _paper_result_from_runtime_status(payload, paper)
    return _infer_deep_read_reserved(result)


def _infer_deep_read_reserved(result: PaperReadResult | None) -> bool:
    """只要论文已经进入全文链路，就视为这个名额已经被占用。"""

    if result is None:
        return False
    full_text = result.full_text
    return bool(full_text.source_path or full_text.markdown_path or full_text.status in {"downloaded", "markdown_ready", "indexed", "download_failed", "parse_failed"})


def _legacy_report_progress(reporter: Any, paper: PaperDocument, stage: str, completed: int, total: int, **extra: Any) -> None:
    """按论文维度上报实时进度，避免并发后多个论文互相覆盖同一个阶段事件。"""

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
        event_key=f"paper:{paper.id}",
        stage_title=paper.title,
        paper_id=paper.id,
        paper_title=paper.title,
        current_title=paper.title,
        completed=completed,
        total=total,
        **extra,
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
    paper_runtime_statuses: dict[str, JsonObject],
) -> JsonObject:
    """新的恢复现场按论文保存状态，不再主要依赖串行位置。"""

    request = state.get("request")
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
        "paper_runtime_statuses": _ordered_paper_runtime_statuses(papers, paper_runtime_statuses),
        "resume_hint": "外部资源验证可用后，把此 checkpoint 作为 read_resume_checkpoint 注入即可继续未完成的论文。",
    }
    if isinstance(error, ReadModelUnavailableError):
        checkpoint["resume_hint"] = "阅读模型验证可用后，把此 checkpoint 作为 read_resume_checkpoint 注入即可继续未完成的论文。"
    if isinstance(error, ReadEmbeddingUnavailableError):
        checkpoint["resume_hint"] = "embedding 服务验证可用后，把此 checkpoint 作为 read_resume_checkpoint 注入即可继续未完成的论文。"
    return checkpoint


def _report_progress(reporter: Any, paper: PaperDocument, stage: str, completed: int, total: int, **extra: Any) -> None:
    """按论文维度上报实时进度，避免并发后多个论文互相覆盖同一个阶段事件。"""

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
        event_key=f"paper:{paper.id}",
        stage_title=paper.title,
        paper_id=paper.id,
        paper_title=paper.title,
        current_title=paper.title,
        completed=completed,
        total=total,
        **extra,
    )
