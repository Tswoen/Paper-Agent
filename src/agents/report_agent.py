from autogen_agentchat.agents import AssistantAgent

from src.utils.log_utils import setup_logger
from src.utils.tool_utils import handlerChunk
from src.tasks.paper_search import PaperSearcher
from src.core.state_models import State,ExecutionState
from src.core.prompts import report_agent_prompt
from src.core.state_models import BackToFrontData
from autogen_agentchat.base import TaskResult

from src.core.model_client import create_default_client, create_report_model_client

logger = setup_logger(__name__)


model_client = create_report_model_client()


report_agent = AssistantAgent(
    name="report_agent",
    model_client=model_client,
    system_message=report_agent_prompt,
    model_client_stream=True
)

async def report_node(state: State) -> State:
    """报告生成节点"""
    state_queue = None
    try:
        state_queue = state["state_queue"]
        current_state = state["value"]
        current_state.current_step = ExecutionState.REPORTING
        await state_queue.put(BackToFrontData(step=ExecutionState.REPORTING,state="initializing",data=None))
        sections = current_state.writted_sections
        sections_text = "\n".join(sections) if sections else "无章节内容提供"
    
        prompt = f"""
        请将以下提供的章节内容组装成一份完整的调研报告，并以Markdown格式输出。

        【章节内容开始】
        {sections_text}
        【章节内容结束】

        【输出要求】
        1. 使用Markdown格式进行排版（标题、列表、加粗等）
        2. 自动补充必要的过渡语句使报告连贯
        3. 保持专业学术风格
        4. 直接输出完整报告，无需解释过程

        【额外说明】
        请确保章节逻辑顺序合理，如有需要可调整章节排列。
        """
        is_thinking = None
        is_First = True
        async for chunk in report_agent.run_stream(task = prompt):
            if is_First:
                is_First = False
                continue
            if not isinstance(chunk, TaskResult):
                if chunk.type == "ThoughtEvent":
                    continue
                if chunk.type == "TextMessage":
                    current_state.report_markdown = chunk.content
                    break

                state,is_thinking = handlerChunk(is_thinking,chunk.content)
                if state is None:
                    continue
                await state_queue.put(BackToFrontData(step=ExecutionState.REPORTING,state=state,data=chunk.content))
        
        await state_queue.put(BackToFrontData(step=ExecutionState.REPORTING,state="completed",data=None))
        return {"value": current_state}

    except Exception as e:
        err_msg = f"Report failed: {str(e)}"
        state["value"].error.report_node_error = err_msg
        await state_queue.put(BackToFrontData(step=ExecutionState.REPORTING,state="error",data=err_msg))
        return state