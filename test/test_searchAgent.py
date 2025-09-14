from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo, UserMessage
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console

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

def add(x:int,y:int):
    """Add two numbers.
        参数描述：
        x: 第一个加数
        y: 第二个加数
    """
    return x+y

from src.tasks.paper_search import PaperSearcher
paper_searcher = PaperSearcher()
search_papers = paper_searcher.search_papers

search_paper_tool = FunctionTool(search_papers, description="查询论文")
add_tool = FunctionTool(add, description="专门用于计算加法的计算器")


agent = AssistantAgent(
    name="search_agent",
    model_client=model_client,
    tools=[search_paper_tool],
    system_message="你是一个论文查询助手，请根据用户的需求，进行语义分析，提取关键字作为查询条件，注意查询条件必须是英文，然后调用工具进行论文查询。",
    model_client_stream=True
)
# agent = AssistantAgent(
#     name="search_agent",
#     model_client=model_client,
#     tools=[add_tool],
#     system_message="你是一个计算助手，请根据用户的需求，计算数值，你必须使用工具来计算。",
# )


async def main():
    # 创建一个用户消息
    user_message = "请帮我查询关于大语言模型的最新论文"
    # user_message = "请帮我计算100+1的值"

    
    # 让代理处理用户消息
    await Console(
        agent.run_stream(task=user_message)
    )
    # response = await agent.run(task=user_message)
    
    # 打印响应
    # print("代理响应:")
    # print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())