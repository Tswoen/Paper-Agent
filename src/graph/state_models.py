from __future__ import annotations

from typing import Any, TypedDict

from src.agents.contracts import ReviewRequest
from src.agents.searchAgent import SearchIntent
from src.paper_retrieval.models import PaperDocument


class State(TypedDict, total=False):
    """LangGraph 兼容的共享图状态。

    当前设计改为以 `State` 作为节点之间唯一共享载体，而不是额外包装一层
    Agent 输入输出对象。这样后续新增阅读、分析、规划、写作节点时，
    每个节点都可以只读取自己关心的字段，并把产出写回共享状态。
    """

    request: ReviewRequest
    search_intent: SearchIntent
    search_results: list[PaperDocument]
    search_scores: list[dict[str, Any]]
    current_step: str
    diagnostics: dict[str, Any]
    raw_model_output: str
