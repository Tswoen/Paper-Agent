from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from src.repositories.sessions.base import SessionRepository
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
) -> APIRouter:
    """创建会话相关的 FastAPI 路由。

    中文说明：
    这里是纯路由适配层，只负责：
    1. 解析 HTTP 请求。
    2. 调用会话服务函数。
    3. 返回标准 JSON 结构。
    不在这里直接写持久化与业务编排逻辑。
    """

    router = APIRouter(prefix="/api/sessions", tags=["sessions"])

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
        """向指定会话提交一条用户消息。"""

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
