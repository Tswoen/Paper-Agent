from enum import Enum, auto
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, TypedDict
from typing_extensions import Annotated
from langgraph.graph.message import add_messages


class WritingStage(Enum):
    INIT = auto()
    OUTLINE = auto()
    SECTION_SPLIT = auto()
    WRITING = auto()
    RESEARCH = auto()
    COMPLETED = auto()

class SectionState(BaseModel):
    # title: str
    content: Optional[str] = None
    # research_materials: List[Dict[str, Any]] = []
    completed: bool = False

class WritingState(TypedDict):
    user_request: str
    # 全局分析结果
    global_analysis: Optional[str] = None
    # 输入的大纲内容
    # outline: str = ""
    # 写作任务列表
    sections: Optional[List[str]] = []
    writted_sections: Optional[List[SectionState]] = []
    # 当前正在处理的小节索引
    current_section_index: int = 0
    # 检索到的相关资料
    retrieved_docs: List[Dict[str, Any]] = []

