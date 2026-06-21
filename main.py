from src.router import create_app


app = create_app()


def main() -> None:
    """本地启动 FastAPI 服务。"""

    import uvicorn

    # 中文注释：统一由 FastAPI 暴露 API 和前端构建产物，减少本地联调时的入口分裂。
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
