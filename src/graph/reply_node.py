from __future__ import annotations

from typing import Any

from src.graph.runtime import WorkflowRuntimeContext
from src.graph.state_models import JsonObject, State


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
        artifact_refs = list(state.get("search_artifact_refs") or [])
        read_artifact_refs = list(state.get("read_artifact_refs") or [])
        analysis_artifact_refs = list(state.get("analysis_artifact_refs") or [])
        writing_outline_artifact_refs = list(state.get("writing_outline_artifact_refs") or [])
        diagnostics = dict(state.get("diagnostics") or {})

        if reporter is not None:
            reporter.started("正在整理最终回复", stage="compose_start")
            reporter.progress("正在把检索结果整理成会话可展示的摘要", stage="compose_reply")

        if not papers:
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
            results_by_paper_id = {str(item.get("paper", {}).get("id") or ""): item for item in read_results}
            for index, paper in enumerate(papers[:5], start=1):
                result = results_by_paper_id.get(paper.id, {})
                relevance = dict(result.get("relevance") or {})
                note = dict(result.get("note") or {})
                full_text = dict(result.get("full_text") or {})
                decision = relevance.get("decision") or "insufficient"
                score = relevance.get("score") if relevance.get("score") is not None else "-"
                short_summary = str(note.get("short_summary") or "暂无可用摘要笔记")
                lines.append(
                    f"{index}. {paper.title} | 相关度 {score} | {decision} | 全文状态："
                    f"{full_text.get('status') or 'not_requested'}\n   {short_summary}"
                )
            assistant_text = "\n".join(lines)

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

