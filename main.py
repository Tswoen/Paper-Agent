from time import sleep
from src.utils.log_utils import setup_logger
from src.utils.tool_utils import handlerChunk
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from src.agents.userproxy_agent import WebUserProxyAgent, userProxyAgent
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.core.config_router import configuration
from src.knowledge.knowledge_router import knowledge
from fastapi import APIRouter

import asyncio
from src.core.state_models import BackToFrontData
# 设置日志
logger = setup_logger(name='main', log_file='project.log')

app = FastAPI()
app.include_router(knowledge)
app.include_router(configuration)
# === CORS 配置（开发时可用 "*"，生产请限定具体域名） ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state_queue = asyncio.Queue()

# agent = WebUserProxyAgent("user_proxy")

@app.post("/send_input")
async def send_input(data: dict):
    user_input = data.get("input")
    userProxyAgent.set_user_input(user_input)
    return JSONResponse({"status": 200, "msg": "已收到人工输入"})

@app.get('/api/research')
async def research_stream(query: str):
    from src.agents.orchestrator import PaperAgentOrchestrator
    from src.core.state_models import State,ExecutionState
    async def event_generator():
        while True:
            state = await state_queue.get()
            yield {"data": f"{state.model_dump_json()}"}
    
    # 启动事件生成器（此时已开始监听队列）
    event_source = EventSourceResponse(event_generator(), media_type="text/event-stream")

    # 初始化业务流程控制器
    orchestrator = PaperAgentOrchestrator(state_queue = state_queue)
    
    # 启动异步任务
    asyncio.create_task(orchestrator.run(user_request=query))

    return event_source

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
