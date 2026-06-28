from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.utils import get_logger, logging_context, setup_logging

from .gateway import GatewayConfig, bootstrap_payload
from .realtime import HttpMessageGateway
from .sessions_api import SessionError, SessionRepository
from .sessions_router import create_sessions_router
from .settings_api import SettingsRepository
from .settings_router import create_settings_router


JsonObject = dict[str, Any]
logger = get_logger(__name__)


def create_app(
    settings_repo: SettingsRepository | None = None,
    sessions_repo: SessionRepository | None = None,
    config: GatewayConfig | None = None,
    message_handler: Callable[[str, str, JsonObject], list[JsonObject]] | None = None,
) -> FastAPI:
    """创建面向前端工作台的 FastAPI 应用。

    该入口除了组装路由外，还负责在应用层接入统一日志系统和请求访问日志，这样无论
    是业务模块、错误处理还是静态资源访问，都能落到同一套日志观测链路里。
    """

    setup_logging()
    settings_repo = settings_repo or SettingsRepository(_default_settings_path())
    sessions_repo = sessions_repo or SessionRepository()
    config = config or GatewayConfig()
    message_gateway = HttpMessageGateway(sessions_repo, message_handler=message_handler)

    app = FastAPI(title="Papers Agents API")
    app.include_router(create_settings_router(settings_repo))
    app.include_router(create_sessions_router(sessions_repo, message_gateway))

    @app.middleware("http")
    async def access_log_middleware(request: Request, call_next):
        """记录每一次 HTTP 请求的访问日志。

        中文注释：这里把 request_id、方法、路径、状态码和耗时统一记录下来，后续排查
        “某次请求为什么慢、为什么报错、落到了哪个会话”时会顺手很多。
        """

        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        started_at = time.perf_counter()
        client_host = request.client.host if request.client else None
        with logging_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_host=client_host,
        ):
            try:
                response = await call_next(request)
            except Exception:
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                logger.exception("HTTP 请求处理失败", extra={"duration_ms": duration_ms, "status_code": 500})
                raise
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            response.headers["X-Request-ID"] = request_id
            log_level = "warning" if response.status_code >= 400 else "info"
            getattr(logger, log_level)(
                "HTTP 请求完成",
                extra={"duration_ms": duration_ms, "status_code": response.status_code},
            )
            return response

    @app.exception_handler(SessionError)
    async def session_error_handler(_: Request, exc: SessionError) -> JSONResponse:
        """把会话业务错误转换成统一的前端错误结构。"""

        return JSONResponse(status_code=exc.status, content={"error": {"message": str(exc), "status": exc.status}})

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        """把请求体格式错误转换成 400 响应。"""

        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "status": 400}})

    @app.get("/webui/bootstrap")
    async def bootstrap() -> JsonObject:
        """前端启动入口：返回本地运行时能力声明。"""

        logger.debug("返回前端 bootstrap 配置")
        return bootstrap_payload(config)

    _mount_frontend(app)
    logger.info(
        "FastAPI 应用创建完成",
        extra={
            "title": app.title,
            "has_frontend_dist": (Path("front/dist") / "index.html").exists(),
            "settings_file": str(_default_settings_path()) if _default_settings_path() else None,
        },
    )
    return app


def _mount_frontend(app: FastAPI) -> None:
    """挂载前端构建产物，并保留 SPA 路由回退能力。"""

    dist_dir = Path("front/dist")
    index_file = dist_dir / "index.html"
    assets_dir = dist_dir / "assets"

    if assets_dir.exists():
        # 中文注释：构建产物里的静态资源单独挂载，浏览器可以直接按指纹路径命中资源文件。
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="front-assets")

    if not index_file.exists():
        logger.warning("未发现前端构建产物，SPA 静态页面不会被挂载", extra={"dist_dir": str(dist_dir)})
        return

    @app.get("/", include_in_schema=False)
    async def front_index() -> FileResponse:
        """返回前端首页。"""

        return FileResponse(index_file)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def front_routes(full_path: str) -> FileResponse:
        """为前端静态文件与 SPA 路由提供统一出口。"""

        candidate = dist_dir / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        # 中文注释：未知前端路由统一回退到 index.html，由前端路由系统接管页面切换。
        return FileResponse(index_file)


def _default_settings_path() -> Path | None:
    """优先使用真实配置文件；缺失时允许仓库以内存配置启动。"""

    path = Path("config/model.json")
    return path if path.exists() else None
