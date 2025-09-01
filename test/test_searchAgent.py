from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo, UserMessage

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

from src.tasks.paper_search import PaperSearcher
paper_searcher = PaperSearcher()

agent = AssistantAgent(
    name="search-agent",
    model_client=model_client,
    tools=[web_search],
    system_message="Use tools to solve tasks.",
)



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())