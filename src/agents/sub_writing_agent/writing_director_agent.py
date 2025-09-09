from autogen_agentchat.agents import AssistantAgent
from src.core.prompts import writing_director_agent_prompt

from src.utils.log_utils import setup_logger

from src.core.model_client import create_default_client

logger = setup_logger(__name__)


model_client = create_default_client()


writing_director_agent = AssistantAgent(
    name="writing_director_agent",
    description="一个写作主管，你只负责告诉写作代理现在需要写那一部分",
    model_client=model_client,
    system_message=writing_director_agent_prompt,
)
