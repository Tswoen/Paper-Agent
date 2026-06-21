from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .gateway import GatewayConfig, bootstrap_payload
from .realtime import HttpMessageGateway
from .sessions_api import SessionError, SessionRepository
from .sessions_router import create_sessions_router
from .settings_api import SettingsRepository
from .settings_router import create_settings_router


JsonObject = dict[str, Any]


def create_app(
    settings_repo: SettingsRepository | None = None,
    sessions_repo: SessionRepository | None = None,
    config: GatewayConfig | None = None,
    message_handler: Callable[[str, str, JsonObject], list[JsonObject]] | None = None,
) -> FastAPI:
    """创建面向前端的 FastAPI 应用。"""

    settings_repo = settings_repo or SettingsRepository(_default_settings_path())
    sessions_repo = sessions_repo or SessionRepository()
    config = config or GatewayConfig()
    message_gateway = HttpMessageGateway(sessions_repo, message_handler=message_handler)

    app = FastAPI(title="Papers Agents API")
    app.include_router(create_settings_router(settings_repo))
    app.include_router(create_sessions_router(sessions_repo, message_gateway))

    @app.exception_handler(SessionError)
    async def session_error_handler(_: Request, exc: SessionError) -> JSONResponse:
        """把会话业务错误转换成前端统一错误结构。"""

        return JSONResponse(status_code=exc.status, content={"error": {"message": str(exc), "status": exc.status}})

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        """把请求体格式错误转换成 400。"""

        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "status": 400}})

    @app.get("/webui/bootstrap")
    async def bootstrap() -> JsonObject:
        """前端启动入口：返回本地运行时能力声明。"""

        return bootstrap_payload(config)

    front_dir = Path("front")
    if front_dir.exists():
        # 前端与 API 同源挂载，浏览器可以直接请求 /api 与 /webui/bootstrap。
        app.mount("/", StaticFiles(directory=str(front_dir), html=True), name="front")

    return app


def _default_settings_path() -> Path | None:
    """优先使用真实配置文件；缺失时让仓库以空内存配置启动。"""

    path = Path("config/model.json")
    return path if path.exists() else None
