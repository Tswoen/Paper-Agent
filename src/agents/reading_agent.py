from autogen_agentchat.agents import AssistantAgent
from pydantic import BaseModel
from typing import List, Optional
from src.utils.log_utils import setup_logger
from src.core.prompts import reading_agent_prompt
from src.core.model_client import create_default_client
from src.core.state_models import BackToFrontData
from main import update_state

from src.core.state_models import State,ExecutionState
from src.services.chroma_client import ChromaClient
import asyncio

logger = setup_logger(__name__)

class KeyMethodology(BaseModel):
    name: Optional[str] = Field(default=None, description="方法名称（如“Transformer-based Sentiment Classifier”）")
    principle: Optional[str] = Field(default=None, description="核心原理")
    novelty: Optional[str] = Field(default=None, description="创新点（如“首次引入领域自适应预训练”）")


class ExtractedPaperData(BaseModel):
    # paper_id: str = Field(default=None, description="论文ID")
    core_problem: str = Field(default=None, description="核心问题")
    key_methodology: KeyMethodology = Field(default=None, description="关键方法")
    datasets_used: List[str] = Field(default=[], description="使用的数据集")
    evaluation_metrics: List[str] = Field(default=[], description="评估指标")
    main_results: str = Field(default="", description="主要结果")
    limitations: str = Field(default="", description="局限性")
    contributions: List[str] = Field(default=[], description="贡献")
    # author_institutions: Optional[str]  # 如“Stanford University, Department of CS”

# 创建一个新的Pydantic模型来包装列表
class ExtractedPapersData(BaseModel):
    papers: List[ExtractedPaperData]

model_client = create_default_client()

read_agent = AssistantAgent(
    name="read_agent",
    model_client=model_client,
    system_message=reading_agent_prompt,
    output_content_type=ExtractedPapersData,
)

async def reading_node(state: State) -> State:
    """搜索论文节点"""
    try:
        current_state = state["value"]
        current_state.current_step = ExecutionState.READING
        await update_state(BackToFrontData(step=ExecutionState.READING,state="processing",data=None))

        papers = current_state.search_results

        # 将papers合理分割成多个任务，交给多个read_agent并行执行，最后合并结果
        # 并行执行任务，使用asyncio.gather
        results = await asyncio.gather(*[read_agent.run(task=str(paper)) for paper in papers])

        # 合并结果
        extracted_papers = ExtractedPapersData(papers=[])
        for result in results:
            for parsed_paper in result.messages[-1].content.papers:
                extracted_papers.papers.append(parsed_paper)     

        # 还得存入向量数据库中
        chroma_client = ChromaClient()
        chroma_client.add_documents(
            documents=[paper.model_dump() for paper in extracted_papers.papers],
            metadatas=[paper for paper in papers],
        )   
        
        current_state.extracted_data = extracted_papers
        await update_state(BackToFrontData(step=ExecutionState.READING,state="completed",data=extracted_papers))

        return {"value": current_state}
            
    except Exception as e:
        err_msg = f"Reading failed: {str(e)}"
        state["value"].error.reading_node_error = err_msg
        await update_state(BackToFrontData(step=ExecutionState.READING,state="error",data=err_msg))
        return state