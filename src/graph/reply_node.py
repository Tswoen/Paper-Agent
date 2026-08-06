from __future__ import annotations

import asyncio
from typing import Any

from src.graph.runtime import WorkflowRuntimeContext
from src.graph.state_models import JsonObject, State
from src.repositories.sessions.base import SessionRepository


FINAL_ARTIFACT_VERSION = "1.0"


def run_compose_reply_node():
    """生成工作流里最后一条助手回复，并在节点内直接发给前端。"""

    async def _node(state: State) -> State:
        """根据检索结果拼装最终回复，同时把结果写成实时事件。"""

        runtime = state.get("runtime_context")
        reporter = _resolve_reporter(runtime)
        papers = list(state.get("search_results") or [])
        read_results = list(state.get("read_results") or [])
        summary = dict(state.get("search_summary") or {})
        read_summary = dict(state.get("read_summary") or {})
        analysis_report = dict(state.get("analysis_report") or {})
        writing_outline = dict(state.get("writing_outline") or {})
        writing_outline_report = dict(state.get("writing_outline_report") or {})
        writing_sections = list(state.get("writing_sections") or [])
        writing_report = dict(state.get("writing_report") or {})
        artifact_refs = list(state.get("search_artifact_refs") or [])
        read_artifact_refs = list(state.get("read_artifact_refs") or [])
        analysis_artifact_refs = list(state.get("analysis_artifact_refs") or [])
        writing_outline_artifact_refs = list(state.get("writing_outline_artifact_refs") or [])
        writing_artifact_refs = list(state.get("writing_artifact_refs") or [])
        final_artifact_refs = list(state.get("final_artifact_refs") or [])
        diagnostics = dict(state.get("diagnostics") or {})

        # 中文说明：最终文件只从写作节点已经完成的内容中拼接，不再调用模型，
        # 这样文件内容与用户看到的每个小节、摘要和参考文献保持完全一致。
        final_markdown = _build_final_markdown(
            topic=str(getattr(state.get("request"), "topic", "") or ""),
            writing_report=writing_report,
            writing_sections=writing_sections,
        )

        if reporter is not None:
            reporter.started("正在整理最终回复", stage="compose_start")
            reporter.progress("正在生成最终的 Markdown 论文文件", stage="compose_reply")

        if final_markdown:
            persisted = await _persist_final_markdown_if_possible(state, final_markdown, writing_report)
            if persisted is not None:
                final_artifact_refs.append(persisted)
                if reporter is not None:
                    reporter.artifact(persisted, stage="final_artifact_ready")

        if final_markdown:
            # 中文说明：最终回复直接展示完整 Markdown，前端可以预览，文件产物则用于下载和长期保存。
            assistant_text = final_markdown
        elif not papers:
            assistant_text = "未检索到符合条件的论文结果。"
        else:
            lines = ["已完成论文检索与阅读，结果如下："]
            if analysis_report:
                metadata = dict(analysis_report.get("execution_metadata") or {})
                lines[0] = (
                    "已完成论文检索、阅读与分析，结果如下："
                    f"\n分析覆盖 {metadata.get('total_papers_analyzed', 0)} 篇论文、"
                    f"{metadata.get('subtopic_count', 0)} 个子主题。"
                )
            if writing_outline:
                lines.append(f"写作大纲已生成，共 {len(writing_outline)} 章，可在 writing_outline 字段中查看结构化对象。")
            if writing_report:
                lines.append(
                    f"正文写作已完成，共 {len(writing_sections)} 个小节，"
                    f"引用 {len(writing_report.get('cited_paper_ids') or [])} 篇论文。"
                )
                if writing_report.get("abstract"):
                    lines.append(f"摘要已生成，参考文献已整理 {len(writing_report.get('references') or [])} 条。")
            results_by_paper_id = {str(item.get("paper", {}).get("id") or ""): item for item in read_results}
            for index, paper in enumerate(papers[:5], start=1):
                result = results_by_paper_id.get(paper.id, {})
                relevance = dict(result.get("relevance") or {})
                note = dict(result.get("note") or {})
                full_text = dict(result.get("full_text") or {})
                selection_status = relevance.get("status") or "not_eligible"
                score = relevance.get("score") if relevance.get("score") is not None else "-"
                short_summary = str(note.get("short_summary") or "暂无可用摘要笔记")
                lines.append(
                    f"{index}. {paper.title} | 匹配分数 {score} | {selection_status} | 全文状态："
                    f"{full_text.get('status') or 'not_requested'}\n   {short_summary}"
                )
            assistant_text = "\n".join(lines)

        diagnostics["compose_reply"] = {
            "status": "ok",
            "final_markdown": bool(final_markdown),
            "final_artifact_count": len(final_artifact_refs),
        }

        assistant_metadata: JsonObject = {
            "diagnostics": diagnostics,
            "search_summary": summary,
            "search_artifact_refs": artifact_refs,
            "read_summary": read_summary,
            "read_artifact_refs": read_artifact_refs,
            "analysis_report": analysis_report,
            "analysis_artifact_refs": analysis_artifact_refs,
            "writing_outline": writing_outline,
            "writing_outline_report": writing_outline_report,
            "writing_outline_artifact_refs": writing_outline_artifact_refs,
            "writing_sections": writing_sections,
            "writing_report": writing_report,
            "writing_artifact_refs": writing_artifact_refs,
            "final_artifact_refs": final_artifact_refs,
            "final_markdown": final_markdown,
        }

        if reporter is not None:
            reporter.message(
                role="assistant",
                content=assistant_text,
                metadata=assistant_metadata,
                stage="compose_reply",
            )
            reporter.completed(
                "最终回复整理完成",
                stage="compose_done",
                selected_paper_count=summary.get("selected_paper_count", 0),
                final_artifact_count=len(final_artifact_refs),
            )

        return State(
            request=state["request"],
            search_results=papers,
            search_scores=list(state.get("search_scores") or []),
            search_summary=summary,
            search_artifact_refs=artifact_refs,
            read_results=read_results,
            read_summary=read_summary,
            read_artifact_refs=read_artifact_refs,
            analysis_report=analysis_report,
            analysis_artifact_refs=analysis_artifact_refs,
            writing_outline=writing_outline,
            writing_outline_report=writing_outline_report,
            writing_outline_artifact_refs=writing_outline_artifact_refs,
            writing_sections=writing_sections,
            writing_report=writing_report,
            writing_artifact_refs=writing_artifact_refs,
            final_artifact_refs=final_artifact_refs,
            read_resume_checkpoint=state.get("read_resume_checkpoint", {}),
            diagnostics=diagnostics,
            current_step="reply",
            session_repo=state.get("session_repo"),
            session_key=state.get("session_key"),
            turn_id=state.get("turn_id"),
            search_node_service=state.get("search_node_service"),
            search_node_llm=state.get("search_node_llm"),
            read_node_llm=state.get("read_node_llm"),
            analysis_node_llm=state.get("analysis_node_llm"),
            writing_outline_node_llm=state.get("writing_outline_node_llm"),
            writing_node_llm=state.get("writing_node_llm"),
            search_node_sink=state.get("search_node_sink"),
            runtime_context=runtime,
            assistant_message=assistant_text,
            assistant_message_metadata=assistant_metadata,
        )

    return _node


def _resolve_reporter(runtime: Any):
    """从运行上下文里安全取出回复节点的上报器。"""

    if not isinstance(runtime, WorkflowRuntimeContext):
        return None
    if runtime.sync_port is None:
        return None
    return runtime.sync_port.for_node("compose_reply", "回复整理")


def _build_final_markdown(*, topic: str, writing_report: JsonObject, writing_sections: list[JsonObject]) -> str:
    """把写作节点的全部已完成内容拼成一份可以直接保存的 Markdown 论文。"""

    sections = list(writing_report.get("sections") or writing_sections)
    abstract = str(writing_report.get("abstract") or "").strip()
    references = list(writing_report.get("references") or [])
    blocks: list[str] = []
    title = topic.strip() or str(writing_report.get("topic") or "文献综述").strip()
    if title:
        blocks.append(f"# {title}")
    if abstract:
        blocks.append(f"## 摘要\n\n{abstract}")

    current_chapter = ""
    for section in sections:
        if not isinstance(section, dict):
            continue
        chapter_key = str(section.get("chapter_key") or "").strip()
        chapter_title = str(section.get("chapter_title") or chapter_key).strip()
        if chapter_key and chapter_key != current_chapter:
            blocks.append(f"## {chapter_title or chapter_key}")
            current_chapter = chapter_key
        section_title = str(section.get("section_title") or section.get("section_id") or "小节").strip()
        content = str(section.get("content") or "").strip()
        if content:
            blocks.append(f"### {section_title}\n\n{content}")

    reference_lines = [
        f"[{item.get('index')}] {item.get('citation')}"
        for item in references
        if isinstance(item, dict) and str(item.get("citation") or "").strip()
    ]
    if reference_lines:
        blocks.append("## 参考文献\n\n" + "\n".join(reference_lines))
    return "\n\n".join(blocks).strip() + ("\n" if blocks else "")


async def _persist_final_markdown_if_possible(
    state: State,
    content: str,
    writing_report: JsonObject,
) -> JsonObject | None:
    """把最终 Markdown 写入当前会话，并返回前端可识别的产物引用。"""

    repository = state.get("session_repo")
    session_key = str(state.get("session_key") or "").strip()
    turn_id = str(state.get("turn_id") or "").strip()
    # 中文说明：这里使用仓储协议实际需要的方法判断，便于脚本和测试传入精简实现。
    if repository is None or (
        not isinstance(repository, SessionRepository) and not hasattr(repository, "write_artifact")
    ):
        return None
    try:
        record = await asyncio.to_thread(
            repository.write_artifact,
            session_key,
            "final_review",
            "literature_review.md",
            content,
            relative_path=f"artifacts/final_review/{turn_id}/literature_review.md",
            metadata={
                "turn_id": turn_id,
                "format": "markdown",
                "artifact_version": FINAL_ARTIFACT_VERSION,
                "section_count": len(writing_report.get("sections") or []),
                "reference_count": len(writing_report.get("references") or []),
            },
        )
    except Exception:
        return None
    return {
        "artifact_id": str(record["id"]),
        "artifact_type": str(record["artifact_type"]),
        "name": str(record["name"]),
        "path": str(record["path"]),
        "size": int(record["size"]),
        "created_at": str(record["created_at"]),
        "metadata": dict(record.get("metadata") or {}),
    }

