from __future__ import annotations

from typing import Any

from src.agents import ReviewRequest
from src.graph import run_graph
from src.repositories.sessions.base import SessionRepository


JsonObject = dict[str, Any]


def build_search_message_handler(repo: SessionRepository):
    """构建基于论文处理图的消息处理器。

    该处理器目前仍然服务于“搜索”场景，但底层执行入口已经统一切换为
    通用图接口，方便后续把筛选、总结和综述生成步骤平滑纳入同一流程。
    """

    def _handler(chat_id: str, content: str, frame: JsonObject) -> list[JsonObject]:
        """将一次用户消息转换为图执行结果并产出前端事件流。"""

        turn_id = str(frame.get("turn_id") or "")
        result = run_graph(
            ReviewRequest(topic=content),
            session_repo=repo,
            session_key=chat_id,
            turn_id=turn_id,
        )
        papers = result.papers
        summary = result.state.get("search_summary") or {}
        artifact_refs = result.state.get("search_artifact_refs") or []

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

        reasoning_text = (
            f"已完成论文检索：原始候选 {summary.get('raw_paper_count', 0)} 篇，"
            f"最终保留 {summary.get('selected_paper_count', 0)} 篇。"
        )
        return [
            {
                "event": "reasoning_delta",
                "chat_id": chat_id,
                "content": reasoning_text,
                "turn_id": turn_id,
            },
            {"event": "reasoning_end", "chat_id": chat_id, "turn_id": turn_id},
            {
                "event": "message",
                "chat_id": chat_id,
                "role": "assistant",
                "content": assistant_text,
                "turn_id": turn_id,
                "kind": "message",
                "metadata": {
                    "diagnostics": result.diagnostics,
                    "search_summary": summary,
                    "search_artifact_refs": artifact_refs,
                },
            },
            {
                "event": "stream_end",
                "chat_id": chat_id,
                "turn_id": turn_id,
            },
        ]

    return _handler
