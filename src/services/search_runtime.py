from __future__ import annotations

from typing import Any

from src.agents import ReviewRequest
from src.graph import run_graph
from src.models.sessions import utc_now
from src.repositories.sessions.base import SessionRepository
from src.services.sessions import RuntimeEventEmitter


JsonObject = dict[str, Any]


def build_search_message_handler(repo: SessionRepository):
    """构建基于论文处理图的流式消息处理器。

    中文说明：
    该处理器把用户输入的论文主题交给既有图执行层处理，并在关键阶段持续
    `emit(...)` 事件，供同步接口和新的 `/runs + SSE` 统一复用。
    """

    def _handler(chat_id: str, content: str, frame: JsonObject, emit: RuntimeEventEmitter) -> None:
        """执行论文检索工作流，并把过程事件增量发给上层。"""

        turn_id = str(frame.get("turn_id") or "")
        emit(
            {
                "event": "message",
                "kind": "progress",
                "role": "system",
                "content": "正在解析主题并规划检索策略",
                "step": "plan",
                "turn_id": turn_id,
                "timestamp": utc_now(),
            }
        )
        emit(
            {
                "event": "reasoning_delta",
                "content": "正在根据主题生成检索关键词、来源约束与筛选范围。",
                "turn_id": turn_id,
                "timestamp": utc_now(),
            }
        )

        result = run_graph(
            ReviewRequest(topic=content),
            session_repo=repo,
            session_key=chat_id,
            turn_id=turn_id,
        )
        papers = result.papers
        summary = result.state.get("search_summary") or {}
        artifact_refs = list(result.state.get("search_artifact_refs") or [])

        emit(
            {
                "event": "reasoning_delta",
                "content": (
                    f"检索已完成：原始候选 {summary.get('raw_paper_count', 0)} 篇，"
                    f"最终保留 {summary.get('selected_paper_count', 0)} 篇。"
                ),
                "turn_id": turn_id,
                "timestamp": utc_now(),
            }
        )
        emit(
            {
                "event": "reasoning_end",
                "turn_id": turn_id,
                "timestamp": utc_now(),
            }
        )
        emit(
            {
                "event": "message",
                "kind": "progress",
                "role": "system",
                "content": "正在整理候选论文摘要与产物清单",
                "step": "summarize",
                "turn_id": turn_id,
                "timestamp": utc_now(),
            }
        )

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

        emit(
            {
                "event": "message",
                "role": "assistant",
                "content": assistant_text,
                "turn_id": turn_id,
                "kind": "message",
                "metadata": {
                    "diagnostics": result.diagnostics,
                    "search_summary": summary,
                    "search_artifact_refs": artifact_refs,
                },
                "timestamp": utc_now(),
            }
        )

        # 中文注释：artifact 事件不负责重新写文件，只是把已落库的产物信息实时推给前端。
        for artifact in artifact_refs:
            emit(
                {
                    "event": "artifact",
                    "artifact": artifact,
                    "turn_id": turn_id,
                    "timestamp": utc_now(),
                }
            )

        emit(
            {
                "event": "turn_end",
                "status": "completed",
                "turn_id": turn_id,
                "timestamp": utc_now(),
            }
        )

    return _handler
