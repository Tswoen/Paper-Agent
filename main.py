from time import sleep
from src.utils.log_utils import setup_logger
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
# from src.agents.orchestrator import test

import asyncio
from src.core.state_models import BackToFrontData
# 设置日志
logger = setup_logger(name='main', log_file='project.log')


app = FastAPI()
state_queue = asyncio.Queue()


@app.get('/api/research')
async def research_stream(query: str):
    from src.agents.orchestrator import PaperAgentOrchestrator
    
    # 初始化业务流程控制器
    orchestrator = PaperAgentOrchestrator(state_queue = state_queue)
    
    # 启动异步任务
    asyncio.create_task(orchestrator.run(user_request=query))
    
    async def event_generator():
        while True:
            state = await state_queue.get()
            yield {"data": state.model_dump_json()}

    return EventSourceResponse(event_generator(), media_type="text/event-stream")

# async def update_state(data: BackToFrontData):
#     await state_queue.put(data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    