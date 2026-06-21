from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from .settings_api import (
    SettingsError,
    SettingsRepository,
    create_or_update_agent,
    provider_models_payload,
    settings_payload,
    update_agent_settings,
    update_embedding_profile,
    update_provider_settings,
)


JsonObject = dict[str, Any]


def create_settings_router(repo: SettingsRepository) -> APIRouter:
    """创建模型设置相关的 FastAPI 路由。"""

    router = APIRouter(prefix="/api/settings", tags=["settings"])

    @router.get("")
    async def get_settings() -> JsonObject:
        return settings_payload(repo)

    @router.get("/provider-models", response_model=None)
    async def get_provider_models(provider: str = Query(..., min_length=1)):
        try:
            return provider_models_payload(repo, provider)
        except SettingsError as exc:
            return _settings_error_response(exc)

    @router.put("/agent")
    @router.post("/agent")
    async def save_agent_settings(request: Request):
        try:
            return update_agent_settings(repo, await _json_body(request))
        except SettingsError as exc:
            return _settings_error_response(exc)

    @router.put("/agents/{name}")
    @router.post("/agents/{name}")
    async def save_named_agent(name: str, request: Request):
        try:
            return create_or_update_agent(repo, name, await _json_body(request))
        except SettingsError as exc:
            return _settings_error_response(exc)

    @router.put("/embedding-profiles/{name}")
    @router.post("/embedding-profiles/{name}")
    async def save_embedding_profile(name: str, request: Request):
        try:
            return update_embedding_profile(repo, name, await _json_body(request))
        except SettingsError as exc:
            return _settings_error_response(exc)

    @router.put("/providers/{name}")
    async def save_provider_settings(name: str, request: Request):
        try:
            return update_provider_settings(repo, name, await _json_body(request))
        except SettingsError as exc:
            return _settings_error_response(exc)

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
        raise SettingsError("request body must be a JSON object")
    return payload


def _settings_error_response(exc: SettingsError) -> JSONResponse:
    """把业务错误转换成统一的前端错误结构。"""

    return JSONResponse(status_code=exc.status, content={"error": {"message": str(exc), "status": exc.status}})
