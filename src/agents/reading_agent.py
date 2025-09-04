from autogen_agentchat.agents import AssistantAgent
from pydantic import BaseModel
from typing import List, Optional
from src.utils.log_utils import setup_logger
from src.core.prompts import reading_agent_prompt
from src.core.model_client import create_default_client

logger = setup_logger(__name__)

class KeyMethodology(BaseModel):
    name: str  # 方法名称（如“Transformer-based Sentiment Classifier”）
    principle: str  # 核心原理
    novelty: str  # 创新点（如“首次引入领域自适应预训练”）


class ExtractedPaperData(BaseModel):
    paper_id: str  # 关联搜索结果的paper_id
    core_problem: str
    key_methodology: KeyMethodology
    datasets_used: List[str]  # 如["IMDB Dataset (50k reviews)", "SST-2"]
    evaluation_metrics: List[str]
    main_results: str  # 含关键数值，如“在IMDB上Accuracy达92.5%，优于BERT的89.3%”
    limitations: str
    contributions: List[str]
    author_institutions: Optional[str]  # 如“Stanford University, Department of CS”
    # extract_source: dict  # 溯源：记录每个维度的提取章节，如{"core_problem": "Abstract, Introduction"}

# 创建一个新的Pydantic模型来包装列表
class ExtractedPapersDataList(BaseModel):
    papers: List[ExtractedPaperData]

model_client = create_default_client()

read_agent = AssistantAgent(
    name="read_agent",
    model_client=model_client,
    system_message=reading_agent_prompt,
    output_content_type=ExtractedPapersDataList,
)
