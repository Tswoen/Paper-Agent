import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录（agents目录）
src_parent_dir = os.path.dirname(os.path.dirname(current_dir))  # 向上两级找到 Paper-Agents 目录

# 将路径添加到 Python 搜索路径
sys.path.append(src_parent_dir)


from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Sequence,get_type_hints,TypeAlias
from autogen_agentchat.agents import BaseChatAgent
import asyncio

from starlette.routing import Route
from src.utils.log_utils import setup_logger
from src.utils.tool_utils import handlerChunk
from src.agents.reading_agent import ExtractedPapersData,KeyMethodology,ExtractedPaperData
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage,StructuredMessage
from autogen_agentchat.base import Response
from autogen_core import CancellationToken, RoutedAgent
from src.agents.sub_analyse_agent.cluster_agent import PaperClusterAgent
from src.agents.sub_analyse_agent.deep_analyse_agent import DeepAnalyseAgent
from src.agents.sub_analyse_agent.global_analyse_agent import GlobalanalyseAgent
from src.core.model_client import create_default_client
from src.core.state_models import BackToFrontData
import json

from src.core.state_models import State,ExecutionState
from autogen_core import message_handler

logger = setup_logger(__name__)
# BaseChatAgent
class AnalyseAgent(BaseChatAgent):
    """基于AutoGen框架的论文分析智能体"""
    
    def __init__(self, name: str = "analyse_agent", state_queue: asyncio.Queue = None):
        super().__init__(name, "A simple agent that counts down.")
        """初始化论文分析系列智能体"""
        # 创建聚类智能体
        self.cluster_agent = PaperClusterAgent()
        # 创建深度分析智能体
        self.deep_analyse_agent = DeepAnalyseAgent()
        # 创建全局分析智能体
        self.global_analyse_agent = GlobalanalyseAgent()
    
        self.model_client = create_default_client()
        self.state_queue = state_queue
    
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    # @message_handler
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """处理分析消息并返回响应
        
        Args:
            message: 提取的论文数据
            cancellation_token: 取消令牌
            
        Returns:
            Response: 包含分析结果的响应对象
        """
        # Calls the on_messages_stream.
        response: Response | None = None
        stream_message = messages[-1].content
        # async for msg in self.on_messages_stream(stream_message, cancellation_token):
        #     if isinstance(msg, Response):
        #         response = msg
        response = await self.on_messages_stream(stream_message, cancellation_token)
        assert response is not None
        return response

    # @message_handler
    async def on_messages_stream(self, message: ExtractedPapersData, cancellation_token: CancellationToken) -> Any:
        """流式处理分析消息
        
        Args:
            message: 提取的论文数据
            cancellation_token: 取消令牌
            
        Yields:
            生成分析过程中的事件或消息
            AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]
        """
        # 1. 调用聚类智能体进行论文聚类
        await self.state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="thinking",data="正在进行论文聚类分析\n"))
        cluster_results = await self.cluster_agent.run(message)
        await self.state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="thinking",data=f"论文聚类分析完成，共形成 {len(cluster_results)} 个聚类\n"))

        # 2. 调用深度分析智能体分析每个聚类的论文
        deep_analysis_results = []
        await self.state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="thinking",data="正在进行论文深度分析\n"))
        deep_analysis_results = await asyncio.gather(*[self.deep_analyse_agent.run(cluster) for cluster in cluster_results])
        await self.state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="thinking",data="论文深度分析完成\n"))
        
        # 3. 调用全局分析智能体生成整体分析报告
        await self.state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="thinking",data="等待全局分析\n"))
        is_thinking = None
        async for chunk in self.global_analyse_agent.run(deep_analysis_results):
            if isinstance(chunk, Dict):
                if not chunk.get("isSuccess", False):
                    await self.state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="error",data=chunk.get("global_analyse", "Unknown error")))
                    break
                global_analysis = chunk
                break
            state,is_thinking = handlerChunk(is_thinking,chunk)
            if state is None:
                continue
            await self.state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state=state,data=chunk))
            
        return Response(
            chat_message=TextMessage(
                content=json.dumps(global_analysis, ensure_ascii=False, indent=2),
                 source=self.name
            )
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass
   
async def analyse_node(state: State) -> State:
    """搜索论文节点"""
    try:
        state_queue = state["state_queue"]
        current_state = state["value"]
        current_state.current_step = ExecutionState.ANALYZING
        await state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="initializing",data=None))
        extracted_papers = current_state.extracted_data

        analyse_agent = AnalyseAgent(state_queue=state_queue)
        task = StructuredMessage(content=extracted_papers, source="User")
        # task = TextMessage(content=json.dumps(extracted_papers.model_dump(),ensure_ascii=False), source="User")
        response = await analyse_agent.run(task=task)

        analyse_results = response.messages[-1].content
        
        current_state.analyse_results = analyse_results
        await state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="completed",data=analyse_results))

        return {"value": current_state}
            
    except Exception as e:
        err_msg = f"Analyse failed: {str(e)}"
        state["value"].error.analyse_node_error = err_msg
        await state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="error",data=err_msg))
        return state

def main():
    """主函数"""
    asyncio.run(analyse_node(state))

if __name__ == "__main__":
    pass
    # from src.core.state_models import PaperAgentState,NodeError
    # state_queue = asyncio.Queue()
    # initial_state = PaperAgentState(
    #         user_request="帮我写一篇关于人工智能的调研报告",
    #         max_papers=2,
    #         error=NodeError(),
    #         config={}  # 可以传入各种配置
    #     )
    # state = {"state_queue": state_queue, "value": initial_state}
    # analyse_agent = AnalyseAgent()
    # main()
