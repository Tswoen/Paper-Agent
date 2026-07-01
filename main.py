from src.utils import get_logger, setup_logging


setup_logging()

from src.api import create_app


app = create_app()
logger = get_logger(__name__)


def main() -> None:
    """本地启动 FastAPI 服务。

    这里显式复用统一日志系统，确保直接运行脚本和被 uvicorn 导入时都能拿到一致的
    日志行为。
    """

    import uvicorn

    logger.info("准备启动本地 FastAPI 服务", extra={"host": "127.0.0.1", "port": 8000, "reload": True})
    # 中文注释：统一使用 FastAPI 暴露 API 和前端构建产物，减少本地联调时的入口分裂。
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
