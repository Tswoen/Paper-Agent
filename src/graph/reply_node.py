from __future__ import annotations

from typing import Any

from src.graph.runtime import WorkflowRuntimeContext
from src.graph.state_models import JsonObject, State


def run_compose_reply_node():
    """生成工作流里最后一条助手回复，并在节点内直接发给前端。"""

    def _node(state: State) -> State:
        """根据检索结果拼装最终回复，同时把结果写成实时事件。"""

        runtime = state.get("runtime_context")
        reporter = _resolve_reporter(runtime)
        papers = list(state.get("search_results") or [])
        summary = dict(state.get("search_summary") or {})
        artifact_refs = list(state.get("search_artifact_refs") or [])
        diagnostics = dict(state.get("diagnostics") or {})

        if reporter is not None:
            reporter.started("正在整理最终回复", stage="compose_start")
            reporter.progress("正在把检索结果整理成会话可展示的摘要", stage="compose_reply")

        if not papers:
            assistant_text = "未检索到符合条件的论文结果。"
        else:
            lines = ["已完成论文检索，候选结果如下："]
            for index, paper in enumerate(papers[:5], start=1):
                author_text = "、".join(paper.authors[:3]) if paper.authors else "作者未知"
                year_text = str(paper.year) if paper.year is not None else "年份未知"
                source_text = paper.source or "unknown"
                lines.append(f"{index}. {paper.title} | {author_text} | {year_text} | {source_text}")
            assistant_text = "\n".join(lines)

        assistant_metadata: JsonObject = {
            "diagnostics": diagnostics,
            "search_summary": summary,
            "search_artifact_refs": artifact_refs,
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
            diagnostics=diagnostics,
            current_step="reply",
            session_repo=state.get("session_repo"),
            session_key=state.get("session_key"),
            turn_id=state.get("turn_id"),
            search_node_service=state.get("search_node_service"),
            search_node_llm=state.get("search_node_llm"),
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

