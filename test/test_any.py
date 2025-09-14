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

async def main():
    # 创建一个用户消息
    user_message = "请你给我写一篇50字的关于春天的文章"

    # 让代理处理用户消息
    response = agent.run_stream(task=user_message)
    async for chunk in response:
        if not isinstance(chunk, TaskResult):
            print(chunk.content,end="")
        else:
            print(chunk)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())