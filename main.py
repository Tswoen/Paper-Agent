from src.router import create_app


app = create_app()


def main() -> None:
    """本地启动 FastAPI 服务。"""

    import uvicorn

    # 统一从 FastAPI app 暴露前端接口，避免再启用手写 HTTP 或 WebSocket 服务。
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
