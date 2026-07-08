from __future__ import annotations

from typing import Any

from src.agents import ReviewRequest
from src.graph import run_graph
from src.graph.runtime import InlineWorkflowSyncPort, WorkflowRuntimeContext
from src.repositories.sessions.base import SessionRepository
from src.services.sessions import RuntimeEventEmitter


JsonObject = dict[str, Any]


def build_search_message_handler(repo: SessionRepository):
    """构建基于论文工作流的消息处理器。"""

    def _handler(chat_id: str, content: str, frame: JsonObject, emit: RuntimeEventEmitter) -> None:
        """把会话输入交给工作流执行，并把同步能力注入图状态。"""

        turn_id = str(frame.get("turn_id") or "")
        run_id = str(frame.get("run_id") or "") or None
        runtime = frame.get("runtime_context")
        if not isinstance(runtime, WorkflowRuntimeContext):
            # 中文注释：同步接口没有独立的 run service，所以这里兜底创建一个本地运行上下文。
            runtime = WorkflowRuntimeContext(
                session_key=chat_id,
                run_id=run_id,
                turn_id=turn_id,
                workflow_name="paper_graph",
                sync_port=InlineWorkflowSyncPort(
                    emit,
                    session_key=chat_id,
                    run_id=run_id,
                    turn_id=turn_id,
                    workflow_name="paper_graph",
                ),
            )

        run_graph(
            ReviewRequest(topic=content),
            runtime=runtime,
            session_repo=repo,
            session_key=chat_id,
            turn_id=turn_id,
        )

    return _handler
