from typing import Dict, List, Optional, Union, AsyncGenerator, Sequence
from autogen_agentchat.agents import BaseChatAgent
import autogen
import asyncio
from src.utils.log_utils import setup_logger
from src.agents.reading_agent import ExtractedPapersData
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_agentchat.base import Response
from autogen_core import CancellationToken
from src.agents.sub_analyse_agent.cluster_agent import PaperClusterAgent
from src.agents.sub_analyse_agent.deep_analyse_agent import DeepAnalyseAgent
from src.agents.sub_analyse_agent.global_analyse_agent import GlobalanalyseAgent
from src.core.model_client import create_default_client
import json

logger = setup_logger(__name__)

class AnalyseAgent(BaseChatAgent):
    """基于AutoGen框架的论文分析智能体"""
    
    def __init__(self, name: str = "analyse_agent"):
        super().__init__(name, "A simple agent that counts down.")
        """初始化论文分析系列智能体"""
        # 创建聚类智能体
        self.cluster_agent = PaperClusterAgent()
        # 创建深度分析智能体
        self.deep_analyse_agent = DeepAnalyseAgent()
        # 创建全局分析智能体
        self.global_analyse_agent = GlobalanalyseAgent()
    
        self.model_client = create_default_client()
    
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        # Calls the on_messages_stream.
        response: Response | None = None
        
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                response = message
        assert response is not None
        return response

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        inner_messages: List[BaseAgentEvent | BaseChatMessage] = []
        # 1. 调用聚类智能体进行论文聚类
        cluster_results = await self.cluster_agent.run(messages[-1].content)
        
        # 2. 调用深度分析智能体分析每个聚类的论文
        deep_analysis_results = []
        analyse_tasks = [self.deep_analyse_agent.run(cluster) for cluster in cluster_results]
        deep_analysis_results = await asyncio.gather(*analyse_tasks)
        
        # 3. 调用全局分析智能体生成整体分析报告
        global_analysis = await self.global_analyse_agent.run(deep_analysis_results)
        
        inner_messages.append(global_analysis)
        # 4. 生成最终分析报告草稿
        # report_draft = {
        #     'summary': global_analysis.get('summary', ''),
        #     'key_findings': global_analysis.get('key_findings', []),
        #     'recommendations': global_analysis.get('recommendations', []),
        #     'clusters': cluster_results
        # }
        

        # 返回分析结果
        yield Response(
            chat_message=TextMessage(
                content=json.dumps(global_analysis, ensure_ascii=False, indent=2),
                source=self.name
            ),
            inner_messages=inner_messages
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass
   