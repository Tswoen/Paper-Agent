from typing import List, Sequence

from autogen_agentchat.agents import AssistantAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from src.agents.sub_writing_agent.writing_agent import writing_agent
from src.agents.sub_writing_agent.writing_director_agent import writing_director_agent
from src.agents.sub_writing_agent.retrieval_agent import retrieval_agent
from src.core.model_client import create_default_client

from src.utils.log_utils import setup_logger


logger = setup_logger(__name__)

filtered_writing_director_agent = MessageFilterAgent(
    name="writing_director_agent",
    wrapped_agent=writing_director_agent,
    filter=MessageFilterConfig(
        per_source=[
            PerSourceFilter(source="user", position="last", count=1),
            PerSourceFilter(source="writing_agent", position="last", count=1)
        ])
)

filtered_writing_agent = MessageFilterAgent(
    name="writing_agent",
    wrapped_agent=writing_agent,
    filter=MessageFilterConfig(
        per_source=[
            PerSourceFilter(source="retrieval_agent", position="last", count=1),
            PerSourceFilter(source="writing_director_agent", position="last", count=1),
            PerSourceFilter(source="writing_agent")
        ]),
)

filtered_retrieval_agent = MessageFilterAgent(
    name="retrieval_agent",
    wrapped_agent=retrieval_agent,
    filter=MessageFilterConfig(per_source=[PerSourceFilter(source="writing_agent", position="last", count=1)]),
)

end_agent = AssistantAgent(
    name="end_agent",
    system_message="你是一个终结代理，轮到你发言时，不管什么情况，你只需输出TERMINATE",
    model_client=create_default_client(),
)

# Build graph with conditional loop
builder = DiGraphBuilder()

# builder.add_node(filtered_writing_director_agent).add_node(filtered_writing_agent).add_node(filtered_retrieval_agent)
# builder.add_edge(filtered_writing_director_agent, filtered_writing_agent)
# builder.add_edge(filtered_writing_agent, filtered_retrieval_agent)



builder.add_node(filtered_writing_director_agent).add_node(filtered_writing_agent).add_node(filtered_retrieval_agent).add_node(end_agent)
# builder.add_edge(filtered_writing_director_agent, filtered_writing_agent,condition=lambda msg: "TERMINATE" not in msg.to_model_text())
builder.add_edge(filtered_writing_director_agent, filtered_writing_agent,condition=lambda msg: "TERMINATE" in "TERMINATE")
builder.add_edge(filtered_writing_director_agent, end_agent,condition=lambda msg: "TERMINATE" in msg.to_model_text())
builder.add_edge(filtered_writing_agent, filtered_writing_director_agent, condition=lambda msg: "APPROVE" in msg.to_model_text())
builder.add_edge(filtered_writing_agent, filtered_retrieval_agent, condition=lambda msg: "APPROVE" not in msg.to_model_text())
builder.add_edge(filtered_retrieval_agent, filtered_writing_agent)
builder.set_entry_point(filtered_writing_director_agent)  

graph = builder.build()

termination_condition = TextMentionTermination("TERMINATE")
# termination_condition = MaxMessageTermination(5)

# Create the flow
flow = GraphFlow(
    participants=builder.get_participants(),
    graph=graph,
    termination_condition=termination_condition
)


async def main():

    task = """
    下面是提供的写作大纲：
        1.介绍
        2.方法
        3.实验
        4.结论
    下面是提供的写作任务：
        写一篇关于AutoGen的技术报告
"""

    # Run the flow and pretty print the output in the console
    async for item in flow.run_stream(task=task):
        # 处理每一步返回的item（例如打印、存储等）
        logger.info(f"收到结果:\n {item}")
    # await flow.run_stream(task=task)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())