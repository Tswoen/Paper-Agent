from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from src.graph.read_fulltext import convert_fulltext_to_markdown
from src.graph.read_models import FullTextStatus, PaperReadResult, ReadNote, ReadRelevance
from src.graph.read_persistence import ReadPersistenceSink
from src.graph.read_vector_store import EmbeddingConnection, index_markdown_chunks
from src.graph.runtime import WorkflowRuntimeContext
from src.graph.state_models import JsonObject, State
from src.llm import ModelConfig, ProviderSnapshot, SystemConfig, make_provider
from src.llm.base import LLMResponse
from src.paper_retrieval.download import download_paper_fulltext
from src.paper_retrieval.models import PaperDocument
from src.repositories.sessions.base import SessionRepository

def run_read_node():
    """生成执行图中的阅读节点，按论文顺序完成摘要、全文和索引处理。"""

    def _node(state: State) -> State:
        """读取检索结果并保留全部已有状态，单篇论文出错时继续处理下一篇。"""

        request = state.get("request")
        if request is None:
            raise ValueError("阅读节点缺少用户请求，无法判断论文主题")
        system_config = SystemConfig.load()
        config = system_config.read
        papers = _deduplicate_papers(list(state.get("search_results") or []))
        reporter = _resolve_reporter(state)
        sink = _resolve_sink(state)
        llm = _resolve_llm(state, config.agent_name)
        embedding_connection, embedding_error = _resolve_embedding_connection(system_config, config.download_timeout_seconds)
        results: list[PaperReadResult] = []
        artifact_refs: list[JsonObject] = []
        deep_read_count = 0
        deep_read_limit = _deep_read_limit(request.constraints, len(papers))

        if reporter is not None:
            reporter.started(
                f"准备阅读 {len(papers)} 篇论文",
                stage="read_start",
                total=len(papers),
                completed=0,
            )

        for position, paper in enumerate(papers, start=1):
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
            except Exception as exc:
                # 中文注释：未知异常也只影响当前论文，不能让一次批量阅读全部中断。
                result = PaperReadResult(
                    paper=paper,
                    full_text=FullTextStatus(status="not_requested", reason="当前论文处理发生异常"),
                    warnings=[f"当前论文处理失败：{exc}"],
                )
            results.append(result)
            if sink is not None:
                try:
                    persisted = sink.persist_paper(result)
                    artifact_refs.extend(persisted.artifacts)
                    if reporter is not None:
                        for artifact in persisted.artifacts:
                            reporter.artifact(artifact, stage="paper_artifact_ready")
                except Exception as exc:
                    result.warnings.append(f"阅读结果无法保存到会话目录：{exc}")
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
        return cast(State, updated)

    return _node


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
    missing_abstract = not (paper.abstract or "").strip()
    if missing_abstract and not _has_fulltext_candidate(paper):
        result.full_text = FullTextStatus(status="no_url", reason="论文没有摘要，也没有可尝试的全文地址")
        return result, False
    should_deep_read = missing_abstract or (
        relevance.decision == "deep_read" and relevance.score >= config.deep_score_threshold
    )
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
    if embedding_connection is None:
        full_text.reason = embedding_error or "未配置可用的 embedding 服务，全文尚未写入向量库"
        result.full_text = full_text
        return result, True
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
        note = ReadNote(
            short_summary=f"《{paper.title}》只有标题和基本信息，尚无摘要可供整理。",
            missing_information=["研究问题", "研究方法", "数据集", "实验结果", "论文不足"],
            evidence_level="metadata",
        )
        return note, ReadRelevance(reason="论文没有摘要，现有资料不足"), []
    if llm is not None:
        try:
            for _ in range(3):
                # 中文注释：模型格式偶尔不稳定时最多重新请求两次，避免一篇论文无限阻塞整个任务。
                response = llm.provider.chat_with_retry(_abstract_messages(paper, topic, constraints), temperature=0)
                parsed = _parse_model_result(response)
                if parsed is not None:
                    return _note_from_model(parsed)
            note, relevance = _fallback_abstract_note(paper, topic)
            return note, relevance, ["摘要模型返回格式不符合要求，已使用本地保守规则"]
        except Exception as exc:
            warning = f"摘要模型整理失败，已改用本地保守规则：{exc}"
            note, relevance = _fallback_abstract_note(paper, topic)
            return note, relevance, [warning]
    note, relevance = _fallback_abstract_note(paper, topic)
    return note, relevance, ["未配置可用的阅读模型，已使用本地保守规则"]


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
    """从模型文本中取出 JSON 对象，格式不合格时返回空值让调用方使用保守结果。"""

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


def _fallback_abstract_note(paper: PaperDocument, topic: str) -> tuple[ReadNote, ReadRelevance]:
    """模型不可用时生成不臆测字段的基础笔记，并用关键词给出保守相关性判断。"""

    abstract = " ".join((paper.abstract or "").split())
    title_and_abstract = f"{paper.title} {abstract}".lower()
    topic_terms = _terms(topic)
    matched = [term for term in topic_terms if term in title_and_abstract]
    score = min(100, round(100 * len(matched) / max(1, len(topic_terms))))
    decision = "deep_read" if score >= 75 else "abstract_only" if score >= 25 else "insufficient"
    note = ReadNote(
        short_summary=abstract[:800],
        missing_information=["数据集", "论文不足"],
        evidence_level="abstract",
    )
    reason = "摘要与主题存在关键词重合" if matched else "摘要中未找到与主题直接重合的关键词"
    return note, ReadRelevance(score=score, decision=decision, reason=reason)


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


def _has_fulltext_candidate(paper: PaperDocument) -> bool:
    """判断论文是否至少提供一个可以尝试下载的全文地址，不在这里假设地址一定可用。"""

    metadata = paper.metadata or {}
    candidates = [paper.pdf_url, paper.url, metadata.get("open_access_pdf"), metadata.get("openAccessPdf")]
    for candidate in candidates:
        if isinstance(candidate, dict):
            candidate = candidate.get("url")
        if str(candidate or "").strip():
            return True
    return False


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


def _terms(value: str) -> list[str]:
    """从主题中提取简单关键词，供模型不可用时做保守的相关性判断。"""

    return [item for item in re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9][a-zA-Z0-9_-]+", value.lower()) if len(item) > 1]


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
