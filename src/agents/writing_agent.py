import sys
import os
# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import Dict, Any
from langgraph.graph import StateGraph
from sqlalchemy.sql.functions import current_date
from src.agents.sub_writing_agent.writing_state_models import WritingState
from src.core.state_models import State
from src.agents.sub_writing_agent import writing_director_agent, writing_agent, retrieval_agent
from src.agents.sub_writing_agent.writing_director_agent import writing_director_node
from src.agents.sub_writing_agent.writing_agent import section_writing_node
from src.agents.sub_writing_agent.retrieval_agent import retrieval_node
from src.core.state_models import ExecutionState
from src.core.state_models import BackToFrontData
from main import update_state

def condition_edge(state: WritingState) -> str:
    """判断是否继续下一个小节"""
    current_section_index = state["current_section_index"]
    writted_sections = state["writted_sections"]
    sections = state["sections"]

    if current_section_index+1 >= len(sections) and writted_sections[-1].completed:
        # 所有小节都已经完成
        return "end"
    elif current_section_index+1 == len(writted_sections) and writted_sections[-1].completed:
        # 移动到下一个小节
        current_section_index = 0
        return "section_writing_node"
    else:
        # 移动到检索节点
        return "retrieval_node"

class WritingWorkflow:
    def __init__(self):
        self.workflow = self.build_workflow()
        
    def build_workflow(self):
        """构建LangGraph工作流"""
        builder = StateGraph(WritingState)

        # 添加节点
        builder.add_node("writing_director_node", writing_director_node)
        builder.add_node("retrieval_node", retrieval_node)
        builder.add_node("section_writing_node", section_writing_node)

        # 设置入口点
        builder.set_entry_point("writing_director_node")

        # 添加边
        builder.add_edge("writing_director_node", "section_writing_node")
        builder.add_edge("retrieval_node", "section_writing_node")
        builder.add_conditional_edges("section_writing_node", condition_edge)

        # 编译图
        graph = builder.compile()
    
        return graph
    
async def writing_node(state: State) -> State:
    """运行写入工作流"""
    state_queue = None
    try:
        state_queue = state["state_queue"]
        current_state = state["value"]
        current_state.current_step = ExecutionState.WRITING
        await state_queue.put(BackToFrontData(step=ExecutionState.WRITING,state="processing",data=None))
        writing_state = WritingState()
        writing_state["user_request"] = current_state.user_request
        writing_state["global_analysis"] = current_state.analyse_results
        # writing_state["outline"] = outline
        writing_state["sections"] = []
        writing_state["writted_sections"] = []
        writing_state["current_section_index"] = -1
        writing_state["retrieved_docs"] = []
        writingWorkFlow = WritingWorkflow()
        writing_state = writingWorkFlow.workflow.invoke(writing_state)
        current_state.writted_sections = writing_state["writted_sections"]
        await state_queue.put(BackToFrontData(step=ExecutionState.WRITING,state="completed",data=writing_state["writted_sections"]))
        return {"value": current_state}
        
    except Exception as e:
        state["value"].error.writing_node_error = f"Writing failed: {str(e)}"
        await state_queue.put(BackToFrontData(step=ExecutionState.WRITING,state="error",data=str(e)))
        return state

async def main():
    global_analysis = """
    全局分析结果LangGraph 是一个基于图状态机架构的框架，专为编排复杂、有状态的 AI 智能体（Agent）工作流而设计。它通过引入“循环”概念，克服了传统链式结构无法处理循环和持续对话的局限，非常适合构建多步骤推理、工具调用和多智能体协作系统。其核心优势在于提供了极高的灵活性和清晰的状态管理，是开发高级AI应用的关键工具
    """
    writing_state = WritingState()
    writing_state["global_analysis"] = global_analysis 
    writing_state["outline"] = outline
    writing_state["sections"] = []
    writing_state["writted_sections"] = []
    writing_state["current_section_index"] = -1
    writing_state["retrieved_docs"] = []
    writingWorkFlow = WritingWorkflow()
    result = await writingWorkFlow.workflow.ainvoke(writing_state)
    # result是WritingState，而WritingState本质上就是一个字典
    print("result:")
    print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())