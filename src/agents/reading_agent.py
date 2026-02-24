from autogen_agentchat.agents import AssistantAgent
# from pydantic import BaseModel, Field
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional,Dict,Any
from src.utils.log_utils import setup_logger
from src.core.prompts import reading_agent_prompt
from src.core.model_client import create_default_client, create_reading_model_client
from src.core.state_models import BackToFrontData
from src.core.state_models import State,ExecutionState
from src.services.chroma_client import ChromaClient
from src.knowledge.knowledge import knowledge_base
from src.core.config import config
import re, json, ast
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
    
    # 清理空字符串和列表
    @field_validator("datasets_used", "evaluation_metrics", "contributions", mode="before")
    @classmethod
    def _validate_list_fields(cls, v):
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("core_problem", "main_results", "limitations", mode="before")
    @classmethod
    def _validate_str_fields(cls, v):
        if v is None:
            return ""
        return str(v)

# 创建一个新的Pydantic模型来包装列表
class ExtractedPapersData(BaseModel):
    papers: List[ExtractedPaperData] = Field(default=[], description="提取的论文数据列表")

model_client = create_reading_model_client()

read_agent = AssistantAgent(
    name="read_agent",
    model_client=model_client,
    system_message=reading_agent_prompt,
    output_content_type=ExtractedPaperData,
    model_client_stream=True
)

def sanitize_metadata(paper: Dict[str, Any]) -> Dict[str, Any]:
    new_meta = {}
    for k, v in paper.items():
        if v is None:
            continue
        if isinstance(v, list):
            new_meta[k] = ", ".join(str(x) for x in v)
        elif isinstance(v, dict):
            new_meta[k] = json.dumps(v, ensure_ascii=False)
        else:
            new_meta[k] = v
    return new_meta


async def add_papers_to_kb(papers:Optional[List[Dict[str, Any]]], extracted_papers: ExtractedPapersData):
    """将提取的论文数据添加到知识库"""
    embedding_dic = config.get("embedding-model")
    embedding_provider = embedding_dic.get("model-provider")
    provider_dic = config.get(embedding_provider)
    
    embed_info = {
        "name": embedding_dic.get("model"),
        "dimension": embedding_dic.get("dimension"),
        "base_url": provider_dic.get("base_url"),
        "api_key": provider_dic.get("api_key"),
    }
    kb_type = config.get("KB_TYPE")
    database_info = await knowledge_base.create_database(
        "临时知识库", "用于存储临时提取的论文数据，仅用于本次报告的生成，用完即删", kb_type=kb_type, embed_info=embed_info, llm_info=None,
    )
    db_id = database_info["db_id"]
    config.set("tmp_db_id", db_id) # 记录临时知识库的db_id，后面retrieval_agent中使用
    
    # 注释掉原本的代码，因为papers中包含了一些None值，导致报错
    # documents = [json.dumps(paper.model_dump(), ensure_ascii=False) for paper in extracted_papers.papers],
    # metadatas = [paper for paper in papers],
    # ids = [str(i) for i in range(len(papers))]
    
    documents=[json.dumps(paper.model_dump(),ensure_ascii=False) for paper in extracted_papers.papers]
    sanitized_metadatas = []
    if papers:
        for paper in papers:
           # new_meta = {}
           # for k, v in paper.items():
            #     if isinstance(v, list):
            #         new_meta[k] = ", ".join(str(x) for x in v)
            #     else:
            #         new_meta[k] = v
            # sanitized_metadatas.append(new_meta)
            sanitized_metadatas.append(sanitize_metadata(paper))          
    metadatas = sanitized_metadatas
    
    # # 确保 ids, metadatas 和 documents 长度一致
    # # 注意：这里假设 extracted_papers.papers 和 papers 是一一对应的
    # min_len = min(len(documents), len(metadatas))
    # documents = documents[:min_len]
    # metadatas = metadatas[:min_len]
    # ids = [str(i) for i in range(min_len)]
    ids = [str(i) for i in range(len(documents))] 
    
    data = {
        "documents": documents,
        "metadatas": metadatas,
        "ids": ids,
    }

    await knowledge_base.add_processed_content(db_id, data)


async def reading_node(state: State) -> State:
    """搜索论文节点"""
    state_queue = state["state_queue"]
    current_state = state["value"]
    current_state.current_step = ExecutionState.READING
    await state_queue.put(BackToFrontData(step=ExecutionState.READING,state="initializing",data=None))

    papers = current_state.search_results

    # 将papers合理分割成多个任务，交给多个read_agent并行执行，最后合并结果
    # 并行执行任务，使用asyncio.gather
    results = await asyncio.gather(*[read_agent.run(task=str(paper)) for paper in papers])

    # 合并结果
    extracted_papers = ExtractedPapersData()
    # 注释掉原本的代码，防止数据格式导致报错
    # for result in results:
    #     if result.messages[-1].content:
    #         parsed_paper = result.messages[-1].content
    #         extracted_papers.papers.append(parsed_paper)   
    
    # 清洗和预处理获取的数据    
    successful_papers = []
    for i, result in enumerate(results):
        raw_content = result.messages[-1].content
        # logger.info(f"Reading Agent Raw Output: {raw_content}") # 打印原始输出
        
        if isinstance(raw_content, ExtractedPaperData):
            extracted_papers.papers.append(raw_content)
            successful_papers.append(papers[i])
            continue
        if isinstance(raw_content, dict):
            data = raw_content
        elif isinstance(raw_content, str):
            clean_content = raw_content.strip()
            if clean_content.startswith("```"):
                clean_content = re.sub(r"^```(?:json)?\s*", "", clean_content)
                clean_content = re.sub(r"\s*```$", "", clean_content)
            try:
                data = json.loads(clean_content)
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(clean_content)
                except Exception:
                    logger.error(f"Failed to parse content as JSON or Python dict: {clean_content}")
                    continue
        else:
            logger.error(f"Unsupported content type: {type(raw_content)}")
            continue

        # 清理 Markdown 代码块
        # 3. 数据结构修正（处理列表包裹或 {"papers": ...} 包裹）
        if isinstance(data, list):
            if len(data) > 0:
                data = data[0] # 取第一个
            else:
                logger.warning("Parsed content is an empty list.")
                continue
        
        if isinstance(data, dict):
            # 如果被包裹在 "papers" 键中
            if "papers" in data and isinstance(data["papers"], list):
                if len(data["papers"]) > 0:
                    data = data["papers"][0]
            # 如果被包裹在 "paper" 键中
            elif "paper" in data and isinstance(data["paper"], dict):
                data = data["paper"]
        
        try:
            # 4. 验证并转换
            parsed_paper = ExtractedPaperData.model_validate(data)
            extracted_papers.papers.append(parsed_paper)
            successful_papers.append(papers[i])
        except Exception as e:
            logger.error(f"Validation failed for data: {data}. Error: {e}")
            # extracted_papers.papers.append(ExtractedPaperData()) 


     # 还得存入向量数据库中
    # await add_papers_to_kb(papers,extracted_papers)
    await add_papers_to_kb(successful_papers,extracted_papers)
        
    current_state.extracted_data = extracted_papers
    await state_queue.put(BackToFrontData(step=ExecutionState.READING,state="completed",data=f"论文阅读完成，共阅读 {len(extracted_papers.papers)} 篇论文"))
    return {"value": current_state}


if __name__ == "__main__":
    paper = {
        'core_problem': 'Despite the rapid introduction of autonomous vehicles, public misunderstanding and mistrust are prominent issues hindering their acceptance.'
    }
    chroma_client = ChromaClient()
    chroma_client.add_documents(
        documents=[paper],
        metadatas=[paper],
    )   
