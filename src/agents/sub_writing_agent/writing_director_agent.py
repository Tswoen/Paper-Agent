from autogen_agentchat.agents import AssistantAgent
from src.core.prompts import writing_director_agent_prompt
from src.agents.sub_writing_agent.writing_state_models import WritingState
from src.core.state_models import BackToFrontData
from src.core.state_models import ExecutionState

from typing import Dict, Any, List

from src.utils.log_utils import setup_logger
from src.utils.tool_utils import handlerChunk

from src.core.model_client import create_default_client, create_subwriting_writing_director_model_client
from autogen_agentchat.base import TaskResult


logger = setup_logger(__name__)


model_client = create_subwriting_writing_director_model_client()


writing_director_agent = AssistantAgent(
    name="writing_director_agent",
    description="一个写作主管，你只负责拆分写作任务，并返回小节列表。",
    model_client=model_client,
    system_message=writing_director_agent_prompt,
    model_client_stream=True,
)

def parse_outline(outline_str: str) -> List[str]:
    """
    解析大纲字符串，提取每个带编号的小节
    
    Args:
        outline_str: 包含编号小节的字符串，每个小节以编号开头（如1.1, 2.3等）
        
    Returns:
        小节列表，每个元素是一个小节的完整内容
    """
    from typing import List
    import re
    
    # 使用正则表达式匹配小节编号（如1., 1.1, 2.3等）
    # 分割字符串并保留分隔符
    sections = re.split(r'(\d+\.\d+|\d+)\s', outline_str.strip())
    
    # 处理分割结果，组合成完整的小节
    result = []
    for i in range(1, len(sections), 2):
        # 组合编号和内容
        section = f"{sections[i].strip()} {sections[i+1].strip()}"
        result.append(section)
    
    return result

async def writing_director_node(state: WritingState) -> Dict[str, Any]:
    state_queue = state["state_queue"]
    await state_queue.put(BackToFrontData(step=ExecutionState.WRITING_DIRECTOR,state="initializing",data=None))
    try: 
        logger.info("开始执行写作主管节点")
        """写作主管节点：拆分大纲为小节"""
        user_request = state["user_request"]
        global_analysis = state["global_analysis"]
        prompt = f"""
        用户的需求:
        {user_request}
        该领域的分析:
        {global_analysis}
        请根据用户提供的需求和关于该领域的分析，生成结构清晰、逻辑连贯的写作子任务，每个子任务应该由一个小节组成：
        """
        # response = await writing_director_agent.run(task = prompt)
        is_thinking = None
        is_first = True
        async for chunk in writing_director_agent.run_stream(task = prompt):
            if is_first:
                is_first = False
                continue
            if not isinstance(chunk, TaskResult):
                if chunk.type == "ThoughtEvent":
                    continue
                if chunk.type == "TextMessage":
                    outline = chunk.content
                    break
                state,is_thinking = handlerChunk(is_thinking,chunk.content)
                if state is None:
                    continue
                await state_queue.put(BackToFrontData(step=ExecutionState.WRITING_DIRECTOR,state=state,data=chunk.content))
        sections = parse_outline(outline)
        await state_queue.put(BackToFrontData(step=ExecutionState.WRITING_DIRECTOR,state="completed",data=None))
        return {"sections": sections}
    except Exception as e:
        await state_queue.put(BackToFrontData(step=ExecutionState.WRITING_DIRECTOR,state="error",data=f"Writing director failed: {str(e)}"))
        return state
