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
        cluster_results = await self.cluster_agent.run(message)
        
        # 2. 调用深度分析智能体分析每个聚类的论文
        deep_analysis_results = []
        deep_analysis_results = await asyncio.gather(*[self.deep_analyse_agent.run(cluster) for cluster in cluster_results])
        
        # 3. 调用全局分析智能体生成整体分析报告
        global_analysis = await self.global_analyse_agent.run(deep_analysis_results)
            
        # 返回分析结果
        # yield Response(
        #     chat_message=TextMessage(
        #         content=json.dumps(global_analysis, ensure_ascii=False, indent=2),
        #         source=self.name
        #     ),
        #     inner_messages=inner_messages
        # )
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
    state_queue = None
    try:
        state_queue = state["state_queue"]
        current_state = state["value"]
        current_state.current_step = ExecutionState.ANALYZING
        await state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="processing",data=None))
        extracted_papers = current_state.extracted_data
        extracted_papers = ExtractedPapersData(
        papers=[
        ExtractedPaperData(core_problem='尽管RAI、伦理AI和AI中的伦理问题的研究旨在提高AI的可信度，但此类工作仍可能带来意外的负面影响。', key_methodology=KeyMethodology(name='Impact statements', principle='主张采用类似医学和社会科学领域的影响力声明（impact statements）来识别、阐述和缓解AI研究对社会的潜在影响。', novelty='未明确声明'), datasets_used=['未提及具体数据集'], evaluation_metrics=['未提及具体评估指标'], main_results='主要提出AI研究同样需要刊登影响声明，以避免和评估潜在的不良后果，并通过跨领域比较强调其必要性与合理性。', limitations='未明确具体内容，但可能包含对现有影响声明应用模式的局限性分析', contributions=['We argue that researchers in RAI, ethical AI, and AI ethics can also cause unintended adverse consequences.', 'Highlighting the need for mechanisms to assess and mitigate the impact of RAI-related work.', 'Drawing parallels between impact statements in medical and social sciences and their potential use in AI research.']) ,
        ExtractedPaperData(core_problem='Despite advancements in AI/ML, the field lacks a unified approach to ensuring confidence in model predictions and results.', key_methodology=KeyMethodology(name='Confident AI', principle='Confident AI is structured around four tenets: Repeatability, Believability, Sufficiency, and Adaptability, each addressing systemic issues in AI/ML.', novelty="We propose 'Confident AI'"), datasets_used=['Not explicitly mentioned'], evaluation_metrics=['None specified'], main_results='Not numerically stated as the paper focuses on framework definition rather than specific experiments or results.', limitations='Not explicitly mentioned', contributions=["Proposed 'Confident AI' as a framework for designing AI and ML systems.", 'Defined four basic tenets: Repeatability, Believability, Sufficiency, and Adaptability.', 'Utilized the tenets to explore fundamental issues in current AI/ML systems.', 'Provided an overall approach to building AI systems with algorithm and user confidence.'])
           ]
        )

        analyse_agent = AnalyseAgent()
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
    from src.core.state_models import PaperAgentState,NodeError
    state_queue = asyncio.Queue()
    initial_state = PaperAgentState(
            user_request="帮我写一篇关于人工智能的调研报告",
            max_papers=2,
            error=NodeError(),
            config={}  # 可以传入各种配置
        )
    state = {"state_queue": state_queue, "value": initial_state}
    analyse_agent = AnalyseAgent()
    main()
