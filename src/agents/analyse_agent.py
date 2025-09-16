from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Sequence,get_type_hints,TypeAlias
from autogen_agentchat.agents import BaseChatAgent
import asyncio

from starlette.routing import Route
from src.utils.log_utils import setup_logger
from src.agents.reading_agent import ExtractedPapersData
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_agentchat.base import Response
from autogen_core import CancellationToken, RoutedAgent
from src.agents.sub_analyse_agent.cluster_agent import PaperClusterAgent
from src.agents.sub_analyse_agent.deep_analyse_agent import DeepAnalyseAgent
from src.agents.sub_analyse_agent.global_analyse_agent import GlobalanalyseAgent
from src.core.model_client import create_default_client
from src.core.state_models import BackToFrontData
from main import update_state
import json

from src.core.state_models import State,ExecutionState
from autogen_core import message_handler

logger = setup_logger(__name__)
# BaseChatAgent
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

    @message_handler
    async def on_messages(self, message: ExtractedPapersData, cancellation_token: CancellationToken) -> Response:
        """处理分析消息并返回响应
        
        Args:
            message: 提取的论文数据
            cancellation_token: 取消令牌
            
        Returns:
            Response: 包含分析结果的响应对象
        """
        # Calls the on_messages_stream.
        response: Response | None = None
        
        async for msg in self.on_messages_stream(message, cancellation_token):
            if isinstance(msg, Response):
                response = msg
        assert response is not None
        return response

    @message_handler
    async def on_messages_stream(self, message: ExtractedPapersData, cancellation_token: CancellationToken) -> Any:
        """流式处理分析消息
        
        Args:
            message: 提取的论文数据
            cancellation_token: 取消令牌
            
        Yields:
            生成分析过程中的事件或消息
            AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]
        """
        inner_messages: List[BaseAgentEvent | BaseChatMessage] = []
        # 1. 调用聚类智能体进行论文聚类
        cluster_results = await self.cluster_agent.run(message[-1].content)
        
        # 2. 调用深度分析智能体分析每个聚类的论文
        deep_analysis_results = []
        analyse_tasks = [self.deep_analyse_agent.run(cluster) for cluster in cluster_results]
        deep_analysis_results = await asyncio.gather(*analyse_tasks)
        
        # 3. 调用全局分析智能体生成整体分析报告
        global_analysis = await self.global_analyse_agent.run(deep_analysis_results)
        
        inner_messages.append(global_analysis)
    
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
   
async def analyse_node(state: State) -> State:
    """搜索论文节点"""
    state_queue = None
    try:
        state_queue = state["state_queue"]
        current_state = state["value"]
        current_state.current_step = ExecutionState.ANALYZING
        await state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="processing",data=None))
        extracted_apers = current_state.extracted_data
        analyse_agent = AnalyseAgent()
        response = await analyse_agent.run(task=extracted_apers)

        analyse_results = response.messages[-1].content
        
        current_state.analyse_results = analyse_results
        await state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="completed",data=analyse_results))

        return {"value": current_state}
            
    except Exception as e:
        err_msg = f"Analyse failed: {str(e)}"
        state["value"].error.analyse_node_error = err_msg
        await state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="error",data=err_msg))
        return state

# if __name__ == "__main__":
#     analyse_agent = AnalyseAgent()
#     print(get_type_hints(analyse_agent.on_messages_stream))
