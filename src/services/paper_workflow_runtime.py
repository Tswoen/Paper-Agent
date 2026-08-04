from __future__ import annotations

from typing import Any

from src.agents import ReviewRequest
from src.graph import run_graph
from src.graph.runtime import InlineWorkflowSyncPort, WorkflowCancellation, WorkflowRuntimeContext
from src.repositories.sessions.base import SessionRepository
from src.services.sessions import RuntimeEventEmitter


JsonObject = dict[str, Any]


def build_paper_workflow_message_handler(repo: SessionRepository):
    """构建把会话消息交给论文工作流执行的处理器。"""

    async def _handler(chat_id: str, content: str, frame: JsonObject, emit: RuntimeEventEmitter) -> None:
        """把会话输入交给异步工作流执行，并把运行时能力注入图状态。"""

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
                cancellation=frame.get("cancellation") if isinstance(frame.get("cancellation"), WorkflowCancellation) else None,
            )

        checkpoint = frame.get("read_resume_checkpoint")
        # 中文注释：新请求的约束由前端放在请求体的 constraints 字段中，
        # 这里把它装进 ReviewRequest，后续图状态就会通过 state["request"].constraints 使用。
        request = _request_from_checkpoint(checkpoint) or ReviewRequest(
            topic=content,
            constraints=_constraints_from_frame(frame),
        )
        state_overrides = {"read_resume_checkpoint": checkpoint} if isinstance(checkpoint, dict) else None
        await run_graph(
            request,
            runtime=runtime,
            session_repo=repo,
            session_key=chat_id,
            turn_id=turn_id,
            state_overrides=state_overrides,
        )

    return _handler


def _constraints_from_frame(frame: JsonObject) -> JsonObject:
    """从一次会话请求中取出约束；没有传约束时返回空字典。"""

    value = frame.get("constraints")
    return dict(value) if isinstance(value, dict) else {}


def _request_from_checkpoint(checkpoint: Any) -> ReviewRequest | None:
    """恢复执行时优先沿用 checkpoint 里的原始请求。"""

    if not isinstance(checkpoint, dict):
        return None
    payload = checkpoint.get("request")
    if not isinstance(payload, dict):
        return None
    topic = str(payload.get("topic") or "").strip()
    if not topic:
        return None
    return ReviewRequest(
        topic=topic,
        constraints=dict(payload.get("constraints") or {}),
        language=str(payload.get("language") or "zh"),
    )
