from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from .sessions_api import (
    MessageHandler,
    SessionRepository,
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
) -> APIRouter:
    """创建会话相关的 FastAPI 路由。"""

    router = APIRouter(prefix="/api/sessions", tags=["sessions"])

    @router.get("")
    async def get_sessions() -> JsonObject:
        return list_sessions(repo)

    @router.post("", status_code=201)
    async def post_session(request: Request) -> JsonObject:
        return create_session(repo, await _json_body(request))

    @router.get("/{session_key}/webui-thread")
    async def get_thread(session_key: str) -> JsonObject:
        return fetch_thread(repo, session_key)

    @router.delete("/{session_key}")
    async def remove_session(session_key: str) -> JsonObject:
        return delete_session(repo, session_key)

    @router.post("/{session_key}/messages")
    async def post_message(session_key: str, request: Request) -> JsonObject:
        # 中文注释：消息提交和其他 session 接口一样，直接下沉到 sessions_api 应用服务层处理。
        return submit_message(repo, session_key, await _json_body(request), message_handler=message_handler)

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
