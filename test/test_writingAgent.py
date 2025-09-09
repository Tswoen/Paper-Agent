from typing import List, Sequence

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from src.agents.sub_writing_agent.writing_agent import writing_agent
from src.agents.sub_writing_agent.writing_director_agent import writing_director_agent
from src.agents.sub_writing_agent.retrieval_agent import retrieval_agent
from src.core.model_client import create_default_client

termination = TextMentionTermination("TERMINATE")

selector_prompt = """选择一个代理来执行任务。

{roles}

当前对话上下文：
{history}

阅读上述对话，然后从{participants}中选择一个代理来执行下一个任务。

大致的选择思路是这样的：
让writing_director_agent确定写作的章节，然后选择writing_agent让其写作，如果writing_agent需要外部资料资料，则选择retrieval_agent。再回到writing_agent继续完成某章节的写作。
如此循环往复，直到所有章节撰写完毕。
"""

model_client = create_default_client()

def candidate_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> List[str]:
    # keep planning_agent first one to plan out the tasks
    if messages[-1].source == retrieval_agent.name:
        return [writing_agent.name]
    return [writing_director_agent.name, retrieval_agent.name, writing_agent.name]

team = SelectorGroupChat(
    [writing_director_agent,writing_agent, retrieval_agent],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    candidate_func=candidate_func
    #allow_repeated_speaker=True,  # Allow an agent to speak multiple turns in a row.
)


async def main():

    task = """
    下面是提供的写作大纲：
        1.摘要
        2.引言
        3.方案
        4.实验
        5.结论
    下面是提供的写作任务：
        写一篇关于RAG的技术报告
"""

    await Console(team.run_stream(task=task))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())