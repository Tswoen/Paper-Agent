from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.repositories.sessions.base import SessionRepository
from src.repositories.sessions.sqlite import SQLiteSessionRepository
from src.repositories.settings.json import SettingsRepository
from src.services.session_runs import SessionRunService
from src.services.paper_workflow_runtime import build_paper_workflow_message_handler
from src.services.sessions import MessageHandler, SessionError
from src.utils import get_logger, logging_context, setup_logging

from .routers.sessions import create_sessions_router
from .routers.settings import create_settings_router


JsonObject = dict[str, Any]
logger = get_logger(__name__)


@dataclass(slots=True)
class GatewayConfig:
    """前端网关配置对象。

    中文说明：
    这个数据类只负责承载前端在启动阶段需要读取的最小运行时配置，
    目前主要是 `api_base`。之所以单独保留一层对象，是为了后续补充
    bootstrap 字段时仍然维持统一的应用装配入口。
    """

    api_base: str = ""


def create_app(
    settings_repo: SettingsRepository | None = None,
    sessions_repo: SessionRepository | None = None,
    config: GatewayConfig | None = None,
    message_handler: MessageHandler | None = None,
) -> FastAPI:
    """创建面向前端工作台的 FastAPI 应用。

    中文说明：
    该函数是后端 HTTP 层唯一的总装配入口，负责：
    1. 初始化统一日志系统。
    2. 组装设置仓储与会话仓储。
    3. 注册 settings/sessions 路由。
    4. 统一配置异常处理与静态前端挂载。
    """

    setup_logging()
    settings_repo = settings_repo or SettingsRepository(_default_settings_path())
    sessions_repo = sessions_repo or SQLiteSessionRepository()
    config = config or GatewayConfig()
    message_handler = message_handler or build_paper_workflow_message_handler(sessions_repo)
    run_service = SessionRunService(repo=sessions_repo, message_handler=message_handler)

    app = FastAPI(title="Papers Agents API")
    app.include_router(create_settings_router(settings_repo))
    app.include_router(
        create_sessions_router(
            sessions_repo,
            message_handler=message_handler,
            run_service=run_service,
        )
    )

    @app.middleware("http")
    async def access_log_middleware(request: Request, call_next):
        """记录每一次 HTTP 请求的访问日志。

        中文说明：
        这里统一补充 request_id、方法、路径、客户端地址、状态码与耗时，
        方便后续排查前后端联调问题以及某次请求落到了哪个会话。
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
        """把会话业务异常转换成统一的前端错误结构。"""

        return JSONResponse(status_code=exc.status, content={"error": {"message": str(exc), "status": exc.status}})

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        """把请求体格式错误转换成 400 响应。"""

        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "status": 400}})

    @app.get("/webui/bootstrap")
    async def bootstrap() -> JsonObject:
        """返回前端启动所需的最小运行时能力声明。

        中文说明：
        这里不做鉴权协商，只告诉前端当前后端支持哪些调用方式，
        让单机版界面可以在启动时一次性拿到能力快照。
        """

        logger.debug("返回前端 bootstrap 配置")
        return {
            "expires_in": 0,
            "api_base": config.api_base,
            "runtime_surface": "paper_agent_workspace",
            "runtime_capabilities": {
                "fastapi_rest": True,
                "rest_management": True,
                "http_message_submit": True,
                "session_runs": True,
                "sse_streaming": True,
                "multi_chat_socket": False,
                "settings_snapshot": True,
                "auth_required": False,
            },
        }

    _mount_frontend(app)
    logger.info(
        "FastAPI 应用创建完成",
        extra={
            "title": app.title,
            "has_frontend_dist": (Path("front/dist") / "index.html").exists(),
            "settings_file": str(_default_settings_path()),
        },
    )
    return app


def _mount_frontend(app: FastAPI) -> None:
    """挂载前端构建产物，并保留 SPA 路由回退能力。"""

    dist_dir = Path("front/dist")
    index_file = dist_dir / "index.html"
    assets_dir = dist_dir / "assets"

    if assets_dir.exists():
        # 中文注释：构建产物中的静态资源单独挂载，浏览器可以直接命中资源文件。
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
        # 中文注释：未知前端路径统一回退到 index.html，由前端路由系统接管。
        return FileResponse(index_file)


def _default_settings_path() -> Path:
    """返回本地模型配置文件路径。

    中文说明：
    即使文件还不存在也返回固定路径，这样设置页保存时会自动创建
    config/model.json 并落盘；如果这里返回 None，保存只会写进内存，
    重启后配置就丢了。
    """

    return Path("config/model.json")
