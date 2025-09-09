from autogen_agentchat.agents import AssistantAgent
from src.core.prompts import writing_agent_prompt

from src.utils.log_utils import setup_logger

from src.core.model_client import create_default_client

logger = setup_logger(__name__)


model_client = create_default_client()

writing_agent = AssistantAgent(
    name="writing_agent",
    description="一个论文写作助手，负责根据指令写作。",
    model_client=model_client,
    system_message=writing_agent_prompt,
)
