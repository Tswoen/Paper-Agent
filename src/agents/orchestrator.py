import sys
import os

from sqlalchemy import Null
from sqlalchemy.sql.functions import current_date
# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from src.core.state_models import PaperAgentState, ExecutionState,NodeError
from src.agents.search_agent import search_node
from src.agents.reading_agent import reading_node
from src.agents.analyse_agent import analyse_node
from src.agents.writing_agent import writing_node
from src.agents.report_agent import report_node
from typing import Dict, Any


import asyncio


class State(TypedDict):
    """LangGraph兼容的状态定义"""
    value: PaperAgentState

class ConfigSchema(TypedDict):
    """LangGraph兼容的配置定义"""
    value: Dict[str, Any]

class PaperAgentOrchestrator:
    def __init__(self):
        self.graph = self._build_graph()

    async def handle_error_node(self, state: State) -> str:
        """错误处理节点"""
        current_state = state["value"]
        current_state.current_step = ExecutionState.FAILED
        print(f"Workflow failed at {current_state.current_step}: {current_state.error}")
        return {"value": current_state}

    def condition_handler(self, state: State) -> bool:
        """条件处理函数"""
        # 如果state.get("error") is not None那么就返回到handle_error_node
        current_state = state["value"]
        err = current_state.error
        current_step = current_state.current_step
        if err.search_node_error is None and current_step == ExecutionState.SEARCHING:
            return "reading_node"
        elif err.reading_node_error is None and current_step == ExecutionState.READING:
            return "analyse_node"
        elif err.analyse_node_error is None and current_step == ExecutionState.ANALYZING:
            return "writing_node"
        elif err.writing_node_error is None and current_step == ExecutionState.WRITING:
            return "report_node"
        elif err.report_node_error is None and current_step == ExecutionState.REPORTING:
            return END
        else:
            return "handle_error_node"


    def _build_graph(self):
        """构建并编译LangGraph工作流"""
        builder = StateGraph(State, context_schema=ConfigSchema)
        
        # 添加节点
        builder.add_node("search_node", search_node)
        builder.add_node("reading_node", reading_node)
        builder.add_node("analyse_node", analyse_node)
        builder.add_node("writing_node", writing_node)
        builder.add_node("report_node", report_node)
        builder.add_node("handle_error_node", self.handle_error_node)

        builder.set_entry_point("search_node")
        
        # 定义工作流路径
        builder.add_edge(START, "search_node")
        builder.add_conditional_edges("search_node", self.condition_handler)
        builder.add_conditional_edges("reading_node", self.condition_handler)
        builder.add_conditional_edges("analyse_node", self.condition_handler)
        builder.add_conditional_edges("writing_node", self.condition_handler)
        builder.add_conditional_edges("report_node", self.condition_handler)
        builder.add_edge("handle_error_node", END)
        
        return builder.compile()
    

    
    async def run(self, user_request: str, max_papers: int = 2) -> PaperAgentState:
        """执行完整工作流"""
        # 初始化状态
        from src.core.state_models import BackToFrontData
        from main import update_state
        await update_state(BackToFrontData(step="front_orchestrator",state="test",data="nothing"))
        initial_state = PaperAgentState(
            user_request=user_request,
            max_papers=max_papers,
            error=NodeError(),
            config={}  # 可以传入各种配置
        )
        await update_state(BackToFrontData(step="back_orchestrator",state="test",data="nothing"))

        # 运行图
        final_state = await self.graph.ainvoke({"value": initial_state})
        return final_state["value"]
    
if __name__ == "__main__":
    # from IPython.display import Image, display

    # # Attempt to visualize the graph as a Mermaid diagram
    # try:
    #     display(Image(graph.get_graph().draw_mermaid_png()))
    # except Exception:
    #     # Handle cases where visualization fails (e.g., missing dependencies)
    #     pass
    orchestrator = PaperAgentOrchestrator()
    orchestrator.run("帮我写一篇有关llm在无人驾驶方面的调研报告。")

    