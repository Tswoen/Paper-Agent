import asyncio
import json
import os
import socket
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from src.agents.userproxy_agent import userProxyAgent
from src.core.config_router import configuration
from src.core.state_models import BackToFrontData
from src.knowledge.knowledge_router import knowledge
from src.utils.log_utils import setup_logger

logger = setup_logger(name='main', log_file='project.log')

DEFAULT_BACKEND_PORT = 8000
FALLBACK_PORT_END = 8099
BACKEND_BIND_HOST = "0.0.0.0"
BACKEND_RUNTIME_HOST = "127.0.0.1"
RUNTIME_CONFIG_PATH = Path(__file__).resolve().parent / ".runtime" / "backend.json"
PORT_CHECK_TARGETS = (
    ("127.0.0.1", socket.AF_INET),
    ("::1", socket.AF_INET6),
)


def _get_preferred_backend_port() -> int:
    raw_port = os.getenv("BACKEND_PORT")
    if not raw_port:
        return DEFAULT_BACKEND_PORT
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise RuntimeError(f"BACKEND_PORT 必须是整数，当前值为: {raw_port}") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError(f"BACKEND_PORT 必须在 1-65535 之间，当前值为: {port}")
    return port


def _is_port_available(host: str, port: int, family: socket.AddressFamily) -> bool:
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _is_backend_port_available(port: int) -> bool:
    return all(
        _is_port_available(host, port, family)
        for host, family in PORT_CHECK_TARGETS
    )


def _select_backend_port(preferred_port: int) -> int:
    if _is_backend_port_available(preferred_port):
        return preferred_port

    scan_start = (
        preferred_port + 1
        if DEFAULT_BACKEND_PORT <= preferred_port < FALLBACK_PORT_END
        else DEFAULT_BACKEND_PORT
    )
    for port in range(scan_start, FALLBACK_PORT_END + 1):
        if port == preferred_port:
            continue
        if _is_backend_port_available(port):
            logger.warning("首选后端端口 %s 被占用，自动切换到端口 %s", preferred_port, port)
            return port

    raise RuntimeError(
        f"端口 {preferred_port} 被占用，且 {scan_start}-{FALLBACK_PORT_END} 范围内没有可用端口"
    )


def _write_backend_runtime_config(port: int) -> None:
    runtime_config = {
        "host": BACKEND_RUNTIME_HOST,
        "port": port,
        "url": f"http://{BACKEND_RUNTIME_HOST}:{port}",
    }
    RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_CONFIG_PATH.write_text(
        json.dumps(runtime_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("后端运行时端口配置已写入: %s", RUNTIME_CONFIG_PATH)


app = FastAPI()
app.include_router(knowledge)
app.include_router(configuration)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state_queue = asyncio.Queue()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "paper-agent"}


@app.post("/send_input")
async def send_input(data: dict):
    user_input = data.get("input")
    userProxyAgent.set_user_input(user_input)
    return JSONResponse({"status": 200, "msg": "已收到人工输入"})


@app.get('/api/research')
async def research_stream(query: str):
    from src.agents.orchestrator import PaperAgentOrchestrator

    async def event_generator():
        while True:
            state = await state_queue.get()
            yield {"data": f"{state.model_dump_json()}"}

    event_source = EventSourceResponse(event_generator(), media_type="text/event-stream")

    orchestrator = PaperAgentOrchestrator(state_queue=state_queue)

    async def run_orchestrator():
        try:
            await orchestrator.run(user_request=query)
        except Exception as exc:
            logger.exception("Research workflow failed")
            await state_queue.put(
                BackToFrontData(
                    step="failed",
                    state="failed",
                    data=str(exc),
                )
            )

    asyncio.create_task(run_orchestrator())

    return event_source


if __name__ == "__main__":
    import uvicorn

    preferred_port = _get_preferred_backend_port()
    port = _select_backend_port(preferred_port)
    _write_backend_runtime_config(port)
    logger.info("Paper-Agent 后端启动: preferred_port=%s, actual_port=%s", preferred_port, port)
    uvicorn.run(app, host=BACKEND_BIND_HOST, port=port)
