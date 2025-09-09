from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from src.core.prompts import retrieval_agent_prompt

from src.utils.log_utils import setup_logger

from src.core.model_client import create_default_client
from src.services.retrieval_tool import retrieval_tool

logger = setup_logger(__name__)


model_client = create_default_client()


retriever = FunctionTool(retrieval_tool, description="用于联网查询外部资料，来辅助写作的工具")

retrieval_agent = AssistantAgent(
    name="retrieval_agent",
    model_client=model_client,
    tools=[retriever],
    description="一个检索助手，负责根据条件联网查询外部资料。",
    system_message=retrieval_agent_prompt,
)
