from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from src.agents.contracts import ReviewRequest
from src.paper_retrieval.models import PaperDocument

if TYPE_CHECKING:
    from src.graph.search_node import SearchNodeSink
    from src.llm import ProviderSnapshot
    from src.paper_retrieval import PaperSearchService
    from src.repositories.sessions.base import SessionRepository


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
    session_repo: SessionRepository
    session_key: str
    turn_id: str
    search_node_service: PaperSearchService
    search_node_llm: ProviderSnapshot | None | str
    search_node_sink: SearchNodeSink
