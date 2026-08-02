from __future__ import annotations

from typing import Any, Protocol, TypedDict

from src.agents.contracts import ReviewRequest
from src.graph.runtime import WorkflowRuntimeContext
from src.llm import ProviderSnapshot
from src.paper_retrieval import PaperSearchService
from src.paper_retrieval.models import PaperDocument
from src.repositories.sessions.base import SessionRepository


JsonObject = dict[str, Any]


class State(TypedDict, total=False):
    """LangGraph 兼容的共享图状态。"""

    request: ReviewRequest
    search_results: list[PaperDocument]
    search_scores: list[JsonObject]
    search_summary: JsonObject
    search_output: JsonObject
    search_artifact_refs: list[JsonObject]
    read_results: list[JsonObject]
    read_summary: JsonObject
    read_artifact_refs: list[JsonObject]
    read_resume_checkpoint: JsonObject
    analysis_report: JsonObject
    analysis_artifact_refs: list[JsonObject]
    writing_outline: JsonObject
    writing_outline_report: JsonObject
    writing_outline_artifact_refs: list[JsonObject]
    writing_sections: list[JsonObject]
    writing_report: JsonObject
    writing_artifact_refs: list[JsonObject]
    final_artifact_refs: list[JsonObject]
    diagnostics: JsonObject
    current_step: str
    assistant_message: str
    assistant_message_metadata: JsonObject
    session_repo: SessionRepository
    session_key: str
    turn_id: str
    runtime_context: WorkflowRuntimeContext
    search_node_service: PaperSearchService
    search_node_llm: ProviderSnapshot | None | str
    read_node_llm: ProviderSnapshot | None | str
    analysis_node_llm: ProviderSnapshot | None | str
    writing_outline_node_llm: ProviderSnapshot | None | str
    writing_node_llm: ProviderSnapshot | None | str
    # 中文注释：这里使用本地协议占位类型，避免在运行时导入 `search_node`
    # 造成循环依赖，同时确保 LangGraph 反射 `State` 注解时能找到合法类型名。
    search_node_sink: "SearchNodeSink"


class SearchNodeSink(Protocol):
    """描述搜索节点可选持久化 sink 的最小运行时协议。"""

    def persist(
        self,
        *,
        topic: str,
        intent: Any,
        raw_papers: list[PaperDocument],
        scored_papers: list[JsonObject],
        selected_papers: list[PaperDocument],
        search_summary: JsonObject,
        search_output: JsonObject,
        agent_diagnostics: JsonObject,
        search_halted: bool,
    ) -> Any:
        """持久化搜索产物并返回任意可转换为状态引用的结果对象。"""
