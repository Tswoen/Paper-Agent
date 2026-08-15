from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from src.repositories.settings.json import SettingsRepository
from src.services.settings import (
    SettingsError,
    async_model_connectivity_payload,
    async_provider_models_payload,
    create_or_update_agent,
    delete_provider_settings,
    settings_payload,
    update_agent_settings,
    update_embedding_profile,
    update_provider_settings,
)


JsonObject = dict[str, Any]


def create_settings_router(repo: SettingsRepository) -> APIRouter:
    """创建模型设置相关的 FastAPI 路由。

    中文说明：
    这里保持“薄路由”原则，只做请求参数接收与错误响应适配，
    真正的配置读取、更新与校验全部下沉到 `src.services.settings`。
    """

    router = APIRouter(prefix="/api/settings", tags=["settings"])

    @router.get("")
    async def get_settings() -> JsonObject:
        """返回完整设置快照。"""

        return settings_payload(repo)

    @router.get("/provider-models", response_model=None)
    async def get_provider_models(provider: str = Query(..., min_length=1)):
        """读取某个 provider 的模型目录。"""

        try:
            # 模型目录会访问远端服务，现在 provider 已经是真 async，直接 await 即可，不再整段丢进线程。
            return await async_provider_models_payload(repo, provider)
        except SettingsError as exc:
            return _settings_error_response(exc)

    @router.post("/model-connectivity")
    async def post_model_connectivity(request: Request):
        """按当前保存的模型配置做一次真实连通性测试。"""

        try:
            body = await _json_body(request)
            # 连通性测试会真实调用模型；provider 层已经改成 async，所以这里直接 await。
            return await async_model_connectivity_payload(
                repo,
                str(body.get("target_type") or body.get("targetType") or ""),
                str(body.get("name") or ""),
            )
        except SettingsError as exc:
            return _settings_error_response(exc)

    @router.put("/agent")
    @router.post("/agent")
    async def save_agent_settings(request: Request):
        """保存默认 agent 的设置。"""

        try:
            return update_agent_settings(repo, await _json_body(request))
        except SettingsError as exc:
            return _settings_error_response(exc)

    @router.put("/agents/{name}")
    @router.post("/agents/{name}")
    async def save_named_agent(name: str, request: Request):
        """按名称创建或更新一个 agent 配置。"""

        try:
            return create_or_update_agent(repo, name, await _json_body(request))
        except SettingsError as exc:
            return _settings_error_response(exc)

    @router.put("/embedding-profiles/{name}")
    @router.post("/embedding-profiles/{name}")
    async def save_embedding_profile(name: str, request: Request):
        """保存嵌入模型配置。"""

        try:
            return update_embedding_profile(repo, name, await _json_body(request))
        except SettingsError as exc:
            return _settings_error_response(exc)

    @router.put("/providers/{name}")
    async def save_provider_settings(name: str, request: Request):
        """保存 provider 配置。"""

        try:
            return update_provider_settings(repo, name, await _json_body(request))
        except SettingsError as exc:
            return _settings_error_response(exc)

    @router.delete("/providers/{name}")
    async def remove_provider(name: str):
        """删除一个 provider 配置（引用它的智能体和嵌入模型也会一起删掉）。"""

        try:
            return delete_provider_settings(repo, name)
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
