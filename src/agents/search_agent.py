from autogen_agentchat.agents import AssistantAgent
from pydantic import BaseModel, Field
from typing import Optional

from src.utils.log_utils import setup_logger
from src.tasks.paper_search import PaperSearcher
from src.core.state_models import State,ExecutionState
from src.core.prompts import search_agent_prompt
from main import update_state
from src.core.state_models import BackToFrontData

from src.core.model_client import create_default_client

logger = setup_logger(__name__)


model_client = create_default_client()

# 创建一个查询条件类，包括查询内容、主题、时间范围等信息，用于存储用户的查询需求
class SearchQuery(BaseModel):
    """查询条件类，存储用户查询需求"""
    query: str = Field(default=None, description="查询关键字")
    start_date: Optional[str] = Field(default=None, description="开始时间, 格式: YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="结束时间, 格式: YYYY-MM-DD")

search_agent = AssistantAgent(
    name="search_agent",
    model_client=model_client,
    system_message=search_agent_prompt,
    output_content_type=SearchQuery
)

async def search_node(state: State) -> State:
    
    """搜索论文节点"""
    try:
        current_state = state["value"]
        current_state.current_step = ExecutionState.SEARCHING
        await update_state(BackToFrontData(step=ExecutionState.SEARCHING,state="processing",data=None))

        prompt = f"""
        请根据用户查询需求，生成检索查询条件。
        用户查询需求：{current_state.user_request}
        """
        response = await search_agent.run(task = prompt)
        search_query = response.messages[-1].content
        # 调用检索服务
        paper_searcher = PaperSearcher()
        results = await paper_searcher.search_papers(
            query = search_query.query,
            start_date = search_query.start_date,
            end_date = search_query.end_date,
        )
        
        current_state.search_results = results
        await update_state(BackToFrontData(step=ExecutionState.SEARCHING,state="completed",data=results))
            
        return {"value": current_state}
            
    except Exception as e:
        err_msg = f"Search failed: {str(e)}"
        state["value"].error.search_node_error = err_msg
        await update_state(BackToFrontData(step=ExecutionState.SEARCHING,state="error",data=err_msg))
        return state