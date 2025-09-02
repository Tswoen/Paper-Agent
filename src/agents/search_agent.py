from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console

from src.utils.log_utils import setup_logger
from src.tasks.paper_search import PaperSearcher

logger = setup_logger(__name__)


model_client = ModelClient.create_default_client()
paper_searcher = PaperSearcher()
search_papers = paper_searcher.search_papers

search_paper_tool = FunctionTool(search_papers, description="用于根据用户查询条件查询论文的工具，只接受英文查询条件")

search_agent = AssistantAgent(
    name="search_agent",
    model_client=model_client,
    tools=[search_paper_tool],
    system_message="你是一个论文查询助手，请根据用户的需求，进行语义分析，提取关键字作为查询条件，注意查询条件必须是英文，然后调用工具进行论文查询。",
)
