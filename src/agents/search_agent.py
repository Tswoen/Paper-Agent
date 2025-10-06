from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from src.agents.userproxy_agent import WebUserProxyAgent,userProxyAgent
from pydantic import BaseModel, Field
from typing import Optional,List
import re
import ast

from src.utils.log_utils import setup_logger
from src.tasks.paper_search import PaperSearcher
from src.core.state_models import State,ExecutionState
from src.core.prompts import search_agent_prompt
from src.core.state_models import BackToFrontData

from src.core.model_client import create_search_model_client

logger = setup_logger(__name__)


model_client = create_search_model_client()

# 创建一个查询条件类，包括查询内容、主题、时间范围等信息，用于存储用户的查询需求
class SearchQuery(BaseModel):
    """查询条件类，存储用户查询需求"""
    querys: List[str] = Field(default=None, description="查询条件列表")
    start_date: Optional[str] = Field(default=None, description="开始时间, 格式: YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="结束时间, 格式: YYYY-MM-DD")

search_agent = AssistantAgent(
    name="search_agent",
    model_client=model_client,
    system_message=search_agent_prompt,
    output_content_type=SearchQuery
)

def parse_search_query(s: str) -> SearchQuery:
    """将前端传回的字符串转为 SearchQuery 对象"""
    # 提取 querys（使用 ast.literal_eval 保证安全）
    querys_match = re.search(r"querys\s*=\s*(\[[^\]]*\])", s)
    start_match = re.search(r"start_date\s*=\s*'([^']*)'", s)
    end_match = re.search(r"end_date\s*=\s*'([^']*)'", s)

    querys = []
    if querys_match:
        try:
            querys = ast.literal_eval(querys_match.group(1))
        except Exception:
            querys = []

    start_date = start_match.group(1) if start_match else None
    end_date = end_match.group(1) if end_match else None

    return SearchQuery(querys=querys, start_date=start_date, end_date=end_date)

async def search_node(state: State) -> State:
    
    """搜索论文节点"""
    state_queue = None
    try:
        state_queue = state["state_queue"]
        current_state = state["value"]
        current_state.current_step = ExecutionState.SEARCHING
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="initializing",data=None))

        prompt = f"""
        请根据用户查询需求，生成检索查询条件。
        用户查询需求：{current_state.user_request}
        """
        response = await search_agent.run(task = prompt)
        search_query = response.messages[-1].content
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="user_review",data=f"{search_query}"))
        
        result = await userProxyAgent.on_messages(
            [TextMessage(content="请人工审核：查询条件是否符合？", source="AI")],
            cancellation_token=CancellationToken()
        )
        search_query = parse_search_query(result.content)

        # 调用检索服务
        paper_searcher = PaperSearcher()
        results = await paper_searcher.search_papers(
            querys = search_query.querys,
            start_date = search_query.start_date,
            end_date = search_query.end_date,
        )
        # [{'paper_id': '2411.11607v2', 'title': 'Performance evaluation of a ROS2 based Automated Driving System', 'authors': [...], 'summary': 'Automated driving is currently a prominent area of scientific work. In the\nfuture, highly automated driving and new Advanced Driver Assistance Systems\nwill become reality. While Advanced Driver Assistance Systems and automated\ndriving functions for certain domains are already commercially available,\nubiquitous automated driving in complex scenarios remains a subject of ongoing\nresearch. Contrarily to single-purpose Electronic Control Units, the software\nfor automated driving is often executed on high performance PCs. The Robot\nOperating System 2 (ROS2) is commonly used to connect components in an\nautomated driving system. Due to the time critical nature of automated driving\nsystems, the performance of the framework is especially important. In this\npaper, a thorough performance evaluation of ROS2 is conducted, both in terms of\ntimeliness and error rate. The results show that ROS2 is a suitable framework\nfor automated driving systems.', 'published': 2024, 'published_date': '2024-11-18T14:29:22+00:00', 'url': 'http://arxiv.org/abs/2411.11607v2', 'pdf_url': 'http://arxiv.org/pdf/2411.11607v2', 'primary_category': 'cs.RO', 'categories': [...], 'doi': '10.5220/0012556800003702'}, {'paper_id': '2307.06258v1', 'title': 'Connected Dependability Cage Approach for Safe Automated Driving', 'authors': [...], 'summary': "Automated driving systems can be helpful in a wide range of societal\nchallenges, e.g., mobility-on-demand and transportation logistics for last-mile\ndelivery, by aiding the vehicle driver or taking over the responsibility for\nthe dynamic driving task partially or completely. Ensuring the safety of\nautomated driving systems is no trivial task, even more so for those systems of\nSAE Level 3 or above. To achieve this, mechanisms are needed that can\ncontinuously monitor the system's operating conditions, also denoted as the\nsystem's operational design domain. This paper presents a safety concept for\nautomated driving systems which uses a combination of onboard runtime\nmonitoring via connected dependability cage and off-board runtime monitoring\nvia a remote command control center, to continuously monitor the system's ODD.\nOn one side, the connected dependability cage fulfills a double functionality:\n(1) to monitor continuously the operational design domain of the automated\ndriving system, and (2) to transfer the responsibility in a smooth and safe\nmanner between the automated driving system and the off-board remote safety\ndriver, who is present in the remote command control center. On the other side,\nthe remote command control center enables the remote safety driver the\nmonitoring and takeover of the vehicle's control. We evaluate our safety\nconcept for automated driving systems in a lab environment and on a test field\ntrack and report on results and lessons learned.", 'published': 2023, 'published_date': '2023-07-12T15:55:48+00:00', 'url': 'http://arxiv.org/abs/2307.06258v1', 'pdf_url': 'http://arxiv.org/pdf/2307.06258v1', 'primary_category': 'cs.RO', 'categories': [...], 'doi': None}]
        current_state.search_results = results
        if len(results) > 0:
            await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="completed",data=f"论文搜索完成，共找到 {len(results)} 篇论文"))
        else:
            await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="error",data="没有找到相关论文,请尝试其他查询条件"))
            current_state.error.search_node_error = "没有找到相关论文,请尝试其他查询条件"
        return {"value": current_state}
            
    except Exception as e:
        err_msg = f"Search failed: {str(e)}"
        state["value"].error.search_node_error = err_msg
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="error",data=err_msg))
        return state