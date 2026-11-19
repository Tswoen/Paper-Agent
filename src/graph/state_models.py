from __future__ import annotations

from typing import Any, TypedDict


JsonObject = dict[str, Any]


class State(TypedDict, total=False):
    """LangGraph 兼容的共享图状态。

    中文说明：这里只放会被 Checkpointer 持久化的业务数据，且必须是纯 JSON 数据。
    会话仓储、事件通道、并发资源、模型快照等运行时依赖一律走
    `WorkflowRuntimeContext`（graph context）；领域对象（`ReviewRequest` /
    `PaperDocument`）在进入状态前转成 dict，避免把类身份写进 checkpoint。
    """

    request: JsonObject
    search_results: list[JsonObject]
    # DEPRECATED: 2026-08-22
    # 原因：关键词打分逻辑已下线，状态里不再携带打分明细。
    # 替代方案：无（打分明细没有下游业务读取）。
    # 计划移除：语义过滤稳定运行一段时间后清理。
    # search_scores: list[JsonObject]
    search_summary: JsonObject
    search_artifact_refs: list[JsonObject]
    read_results: list[JsonObject]
    read_summary: JsonObject
    read_artifact_refs: list[JsonObject]
    analysis_report: JsonObject
    analysis_artifact_refs: list[JsonObject]
    writing_outline: JsonObject
    writing_outline_report: JsonObject
    writing_outline_artifact_refs: list[JsonObject]
    writing_report: JsonObject
    writing_artifact_refs: list[JsonObject]
    diagnostics: JsonObject
    current_step: str
    session_key: str
    turn_id: str
