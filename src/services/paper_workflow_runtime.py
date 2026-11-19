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
                session_repo=repo,
            )

        # 中文注释：存档能力来自运行时上下文；直接调用 handler（没有 run service）
        # 时 bundle 为空，图就退回“不落盘”的行为。
        checkpointer = runtime.checkpoint_bundle.saver if runtime.checkpoint_bundle is not None else None
        thread_id = runtime.thread_id
        # 中文注释：写入时机跟随 bundle 的配置；没有 bundle 时传 None，表示不配置持久化。
        durability = (
            runtime.checkpoint_bundle.settings.durability
            if runtime.checkpoint_bundle is not None
            else None
        )
        # 中文注释：只有同时具备 saver 和线程时才认续跑标记；旧入口即使收到
        # 客户端塞进来的 resume 标记，也退回普通运行，避免在没有存档时误触发。
        resume = bool(frame.get("resume_from_checkpoint")) and checkpointer is not None and thread_id is not None
        # 中文注释：续跑时输入必须来自官方 checkpoint，不能再塞新请求；
        # request=None 由 run_graph 的续跑分支接受。
        request = (
            None
            if resume
            else ReviewRequest(
                topic=content,
                constraints=_constraints_from_frame(frame),
            )
        )
        await run_graph(
            request,
            runtime=runtime,
            session_repo=repo,
            session_key=chat_id,
            turn_id=turn_id,
            checkpointer=checkpointer,
            thread_id=thread_id,
            resume=resume,
            durability=durability,
        )

    return _handler


def _constraints_from_frame(frame: JsonObject) -> JsonObject:
    """从一次会话请求中取出约束；没有传约束时返回空字典。"""

    value = frame.get("constraints")
    return dict(value) if isinstance(value, dict) else {}
