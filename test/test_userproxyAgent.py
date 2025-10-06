import asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from src.agents.userproxy_agent import WebUserProxyAgent
from autogen_core import CancellationToken
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# === CORS 配置（开发时可用 "*"，生产请限定具体域名） ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = WebUserProxyAgent("user_proxy")

@app.post("/send_input")
async def send_input(data: dict):
    user_input = data.get("input")
    agent.set_user_input(user_input)
    return JSONResponse({"status": "ok", "msg": "已收到人工输入"})

@app.get("/run")
async def run_task():
    from autogen_agentchat.messages import TextMessage
    print("开始任务...")
    result = await agent.on_messages(
        [TextMessage(content="请人工审核：是否批准该任务？", source="AI")],
        cancellation_token=CancellationToken()
    )
    return JSONResponse({"result": result.content})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)