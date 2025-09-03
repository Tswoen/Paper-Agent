import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from src.agents.reading_agent import read_agent
from src.agents.search_agent import search_agent
from autogen_agentchat.messages import StructuredMessage
from src.agents.reading_agent import ExtractedPapersDataList

# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("APPROVE")

# Create a team with the primary and critic agents.

team = RoundRobinGroupChat(
    [search_agent,read_agent], 
    termination_condition=text_termination,
     custom_message_types=[StructuredMessage[ExtractedPapersDataList]]
)

async def main():

    # 让代理处理用户消息
    try:
        await Console(team.run_stream(task="请帮我查询关于大语言模型的最新论文"))  # Stream the messages to the console.
    except Exception as e:
        return
    


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
