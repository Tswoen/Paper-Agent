from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
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
    """创建面向前端工作台的 FastAPI 应用。"""

    settings_repo = settings_repo or SettingsRepository(_default_settings_path())
    sessions_repo = sessions_repo or SessionRepository()
    config = config or GatewayConfig()
    message_gateway = HttpMessageGateway(sessions_repo, message_handler=message_handler)

    app = FastAPI(title="Papers Agents API")
    app.include_router(create_settings_router(settings_repo))
    app.include_router(create_sessions_router(sessions_repo, message_gateway))

    @app.exception_handler(SessionError)
    async def session_error_handler(_: Request, exc: SessionError) -> JSONResponse:
        """把会话业务错误转换成统一的前端错误结构。"""

        return JSONResponse(status_code=exc.status, content={"error": {"message": str(exc), "status": exc.status}})

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        """把请求体格式错误转换成 400。"""

        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "status": 400}})

    @app.get("/webui/bootstrap")
    async def bootstrap() -> JsonObject:
        """前端启动入口：返回本地运行时能力声明。"""

        return bootstrap_payload(config)

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """挂载前端构建产物，保留 SPA 路由回退能力。"""

    dist_dir = Path("front/dist")
    index_file = dist_dir / "index.html"
    assets_dir = dist_dir / "assets"

    if assets_dir.exists():
        # 中文注释：构建产物里的 assets 使用单独挂载，浏览器可以直接请求指纹文件。
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="front-assets")

    if not index_file.exists():
        return

    @app.get("/", include_in_schema=False)
    async def front_index() -> FileResponse:
        return FileResponse(index_file)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def front_routes(full_path: str) -> FileResponse:
        candidate = dist_dir / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        # 中文注释：前端路由交给 React 应用处理，未知路径统一回退到 index.html。
        return FileResponse(index_file)


def _default_settings_path() -> Path | None:
    """优先使用真实配置文件；缺失时让仓库以内存配置启动。"""

    path = Path("config/model.json")
    return path if path.exists() else None
