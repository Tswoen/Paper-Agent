from time import sleep
from src.utils.log_utils import setup_logger
from src.utils.tool_utils import handlerChunk
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from src.agents.userproxy_agent import WebUserProxyAgent, userProxyAgent
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import asyncio
from src.core.state_models import BackToFrontData
# 设置日志
logger = setup_logger(name='main', log_file='project.log')


app = FastAPI()
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
            # yield {"data": f"{state.model_dump_json()}\n\n"}
            yield {"data": f"{state.model_dump_json()}"}
            # state_queue.task_done()  # 标记任务完成
    
    # 启动事件生成器（此时已开始监听队列）
    event_source = EventSourceResponse(event_generator(), media_type="text/event-stream")

    # 初始化业务流程控制器
    orchestrator = PaperAgentOrchestrator(state_queue = state_queue)
    
    # 启动异步任务
    asyncio.create_task(orchestrator.run(user_request=query))

    return event_source


async def main():
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_core.models import ModelInfo, UserMessage
    from autogen_agentchat.base import TaskResult

    model_client = OpenAIChatCompletionClient(
        model="Qwen/Qwen3-32B",
        api_key="sk-mvjhwoypajnggqoasejoqumfaabvifdrvztgvmxdpdyukggy", # Optional if you have an OPENAI_API_KEY environment variable set.
        base_url="https://api.siliconflow.cn/v1",
        model_info=ModelInfo(
            vision=True,
            function_calling=True,
            json_output=True,
            family="Qwen",
            structured_output=True
        )
    )
    agent = AssistantAgent(
        name="search_agent",
        model_client=model_client,
        system_message="你是一个智能助手，可以回答用户的问题。",
        model_client_stream=True
    )

    user_message = "请你给我写一篇50字的关于春天的文章"

    # 让代理处理用户消息
    response = agent.run_stream(task=user_message)
    async for chunk in response: 
        if not isinstance(chunk, TaskResult):
            if chunk.type == "TextMessage" or chunk.type == "ThoughtEvent":
                continue
            yield chunk.content


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    