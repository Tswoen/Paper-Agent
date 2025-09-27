from autogen_agentchat.agents import AssistantAgent
from langchain_core.messages import content_blocks
from src.core.prompts import writing_agent_prompt
from src.agents.sub_writing_agent.writing_state_models import WritingState, SectionState
from typing import Dict, Any
from src.utils.log_utils import setup_logger

from src.core.model_client import create_default_client, create_subwriting_writing_model_client

logger = setup_logger(__name__)


model_client = create_subwriting_writing_model_client()

writing_agent = AssistantAgent(
    name="writing_agent",
    description="一个论文写作助手，负责根据指令写作。",
    model_client=model_client,
    system_message=writing_agent_prompt,
)


async def section_writing_node(state: WritingState) -> Dict[str, Any]:
    logger.info("开始执行内部写作节点")
    current_section_index = state["current_section_index"]
    sections = state["sections"]
    writted_sections = state["writted_sections"]
    retrieved_docs = state["retrieved_docs"]
    global_analyse = state["global_analysis"]
    
    # 第一次开始写作 或者 第一次写某一部分
    if (len(writted_sections)==0) or (len(writted_sections) == current_section_index+1 and writted_sections[current_section_index].completed):
        current_section_index += 1
        new_section = SectionState()
        writted_sections.append(new_section)

    writing_task = sections[state["current_section_index"]]
    prompt = f"""请根据以下内容完成写作任务：

            当前写作子任务: {writing_task}
            论文全局分析: {global_analyse}
            可用资料: 
            {retrieved_docs}

            请开始写作：
    """

    # response = await writing_agent.run(task = prompt)
    # content = response.messages[-1].content
    content = "写完了 APPROVED"
    writted_sections[-1].content = content
    if "APPROVED" in content:
        writted_sections[-1].completed = True
        retrieved_docs = []

    return {"writted_sections": writted_sections, "retrieved_docs": retrieved_docs, "current_section_index": current_section_index}