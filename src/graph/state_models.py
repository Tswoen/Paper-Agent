from __future__ import annotations

from typing import Any, TypedDict

from src.agents.contracts import ReviewRequest
from src.paper_retrieval.models import PaperDocument


JsonObject = dict[str, Any]


class State(TypedDict, total=False):
    """LangGraph 兼容的共享图状态。"""

    request: ReviewRequest
    search_results: list[PaperDocument]
    search_scores: list[JsonObject]
    search_summary: JsonObject
    search_artifact_refs: list[JsonObject]
    diagnostics: JsonObject
    current_step: str
