from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.repositories.sessions.base import SessionRepository
from src.services.session_runs import SessionRunService, encode_sse
from src.services.sessions import (
    MessageHandler,
    create_session,
    delete_session,
    fetch_thread,
    list_sessions,
    submit_message,
)


JsonObject = dict[str, Any]


def create_sessions_router(
    repo: SessionRepository,
    message_handler: MessageHandler | None = None,
    run_service: SessionRunService | None = None,
) -> APIRouter:
    """创建会话相关的 FastAPI 路由。

    中文说明：
    该模块只负责请求解析与响应适配，不直接承载会话落库、run 编排和
    SSE 事件管理逻辑，核心业务统一委托给 service 层处理。
    """

    router = APIRouter(prefix="/api/sessions", tags=["sessions"])
    resolved_run_service = run_service or SessionRunService(repo=repo, message_handler=message_handler)

    @router.get("")
    async def get_sessions() -> JsonObject:
        """返回会话列表。"""

        return list_sessions(repo)

    @router.post("", status_code=201)
    async def post_session(request: Request) -> JsonObject:
        """创建一个新会话。"""

        return create_session(repo, await _json_body(request))

    @router.get("/{session_key}/webui-thread")
    async def get_thread(session_key: str) -> JsonObject:
        """读取指定会话的完整线程快照。"""

        return fetch_thread(repo, session_key)

    @router.delete("/{session_key}")
    async def remove_session(session_key: str) -> JsonObject:
        """删除指定会话。"""

        return delete_session(repo, session_key)

    @router.post("/{session_key}/messages")
    async def post_message(session_key: str, request: Request) -> JsonObject:
        """兼容旧版同步消息接口，提交后等待整次运行完成。"""

        body = await _json_body(request)
        # 旧接口会等待完整工作流结束，放到线程里执行，避免卡住其他异步请求。
        return await asyncio.to_thread(submit_message, repo, session_key, body, message_handler=message_handler)

    @router.post("/{session_key}/runs", status_code=202)
    async def post_run(session_key: str, request: Request) -> JsonObject:
        """创建一次新的后台运行，并返回对应的流地址。"""

        return await resolved_run_service.start_run(session_key, await _json_body(request))

    @router.post("/{session_key}/runs/{run_id}/cancel", status_code=202)
    async def cancel_run(session_key: str, run_id: str) -> JsonObject:
        """接收用户主动停止请求，后台任务会在当前等待点尽快结束。"""

        return await resolved_run_service.cancel_run(session_key, run_id)

    @router.get("/{session_key}/runs/{run_id}/stream")
    async def stream_run(session_key: str, run_id: str) -> StreamingResponse:
        """以 SSE 形式持续返回指定 run 的实时事件。"""

        async def _event_generator():
            """持续输出 SSE 事件，同时定期发送心跳避免连接被中间层回收。"""

            event_iterator = resolved_run_service.stream_events(session_key, run_id)
            while True:
                try:
                    event = await anext(event_iterator)
                except StopAsyncIteration:
                    break
                yield encode_sse(event)
            yield ": stream closed\n\n"

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router


async def _json_body(request: Request) -> JsonObject:
    """读取 JSON body；空 body 按空对象处理。"""

    try:
        payload = await request.json()
    except Exception:
        return {}
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload
