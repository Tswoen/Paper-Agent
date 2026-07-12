from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from src.models.sessions import utc_now


JsonObject = dict[str, Any]
WorkflowEventEmitter = Callable[[JsonObject], JsonObject]


class WorkflowSyncPort(Protocol):
    """定义工作流运行过程中统一发事件的最小接口。"""

    def emit(self, event: JsonObject) -> JsonObject:
        """发送一条标准事件，并返回已经补齐字段后的事件。"""

    def for_node(self, node_key: str, node_title: str) -> "WorkflowNodeReporter":
        """为某个节点创建一个带默认字段的轻量上报器。"""


@dataclass(slots=True)
class WorkflowRuntimeContext:
    """保存一次工作流运行时需要共享的上下文。"""

    session_key: str
    turn_id: str
    run_id: str | None = None
    workflow_name: str = "paper_graph"
    sync_port: WorkflowSyncPort | None = None


class InlineWorkflowSyncPort:
    """把工作流节点事件交给外层 emit 回调的简单实现。"""

    def __init__(
        self,
        emitter: WorkflowEventEmitter,
        *,
        session_key: str,
        turn_id: str,
        run_id: str | None = None,
        workflow_name: str = "paper_graph",
    ):
        """初始化一个可以直接被节点使用的同步端口。"""

        self._emitter = emitter
        self.session_key = session_key
        self.turn_id = turn_id
        self.run_id = run_id
        self.workflow_name = workflow_name

    def emit(self, event: JsonObject) -> JsonObject:
        """补齐运行期公共字段，然后立刻把事件交给外层统一处理。"""

        payload = copy.deepcopy(event)
        payload.setdefault("turn_id", self.turn_id)
        payload.setdefault("session_key", self.session_key)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("workflow_name", self.workflow_name)
        payload.setdefault("timestamp", utc_now())
        return self._emitter(payload)

    def for_node(self, node_key: str, node_title: str) -> "WorkflowNodeReporter":
        """返回一个已经绑定节点名字和标题的上报器。"""

        return WorkflowNodeReporter(sync_port=self, node_key=node_key, node_title=node_title)


@dataclass(frozen=True, slots=True)
class RuntimeStageDisplay:
    """描述某个阶段在前端该如何显示。"""

    event_key: str
    title: str
    show_content: str | None = None
    status: str | None = None
    updates_parent: bool = False


# 中文注释：这里不是做复杂配置系统，只是把“代码里的阶段名”翻译成“用户能看懂的事件名”。
# 如果以后新增节点，只需要在下面加少量映射；没有映射的阶段也会按原 stage 正常显示。
_STAGE_DISPLAY: dict[tuple[str, str], RuntimeStageDisplay] = {
    ("search", "plan_search"): RuntimeStageDisplay("plan_search", "生成检索条件"),
    ("search", "intent_ready"): RuntimeStageDisplay(
        "plan_search",
        "生成检索条件",
        show_content="检索条件生成完毕",
        status="completed",
    ),
    ("search", "fetch_results"): RuntimeStageDisplay("fetch_results", "拉取候选论文"),
    ("search", "raw_results_ready"): RuntimeStageDisplay(
        "fetch_results",
        "拉取候选论文",
        show_content="候选论文拉取完成",
        status="completed",
    ),
    ("search", "rank_completed"): RuntimeStageDisplay(
        "rank_results",
        "排序筛选论文",
        show_content="排序和筛选已完成",
        status="completed",
    ),
    ("search", "artifact_ready"): RuntimeStageDisplay(
        "save_search_artifact",
        "保存检索产物",
        show_content="检索产物已保存",
        status="completed",
    ),
    ("search", "search_done"): RuntimeStageDisplay(
        "search",
        "论文检索",
        show_content="论文检索已完成",
        status="completed",
        updates_parent=True,
    ),
    ("read", "read_start"): RuntimeStageDisplay("read", "论文阅读", updates_parent=True),
    ("read", "reading_abstract"): RuntimeStageDisplay("reading_abstract", "阅读论文摘要"),
    ("read", "downloading_full_text"): RuntimeStageDisplay("downloading_full_text", "下载论文全文"),
    ("read", "converting_markdown"): RuntimeStageDisplay("converting_markdown", "转换 Markdown"),
    ("read", "saving_chunks"): RuntimeStageDisplay("saving_chunks", "建立全文索引"),
    ("read", "paper_completed"): RuntimeStageDisplay(
        "paper_completed",
        "完成单篇论文",
        show_content="论文阅读完成",
        status="completed",
    ),
    ("read", "paper_artifact_ready"): RuntimeStageDisplay(
        "paper_artifact_ready",
        "保存单篇阅读结果",
        show_content="单篇阅读结果已保存",
        status="completed",
    ),
    ("read", "read_artifact_ready"): RuntimeStageDisplay(
        "read_artifact_ready",
        "保存阅读汇总",
        show_content="阅读汇总已保存",
        status="completed",
    ),
    ("read", "read_checkpoint_saved"): RuntimeStageDisplay(
        "read_checkpoint_saved",
        "保存恢复现场",
        show_content="恢复现场已保存",
        status="completed",
    ),
    ("read", "read_model_unavailable"): RuntimeStageDisplay(
        "read_model_unavailable",
        "等待阅读模型恢复",
        status="failed",
    ),
    ("read", "read_embedding_unavailable"): RuntimeStageDisplay(
        "read_embedding_unavailable",
        "等待向量服务恢复",
        status="failed",
    ),
    ("read", "read_done"): RuntimeStageDisplay(
        "read",
        "论文阅读",
        show_content="论文阅读已完成",
        status="completed",
        updates_parent=True,
    ),
    ("compose_reply", "compose_start"): RuntimeStageDisplay("compose_reply", "回复整理", updates_parent=True),
    ("compose_reply", "compose_reply"): RuntimeStageDisplay("compose_reply_step", "生成最终回复"),
    ("compose_reply", "compose_done"): RuntimeStageDisplay(
        "compose_reply",
        "回复整理",
        show_content="最终回复整理完成",
        status="completed",
        updates_parent=True,
    ),
}

_DONE_STATUSES = {"completed", "failed", "cancelled", "skipped"}


@dataclass(slots=True)
class WorkflowNodeReporter:
    """帮单个节点稳定地产生事件，避免节点里散落大量样板代码。"""

    sync_port: WorkflowSyncPort
    node_key: str
    node_title: str

    def started(self, message: str | None = None, **extra: Any) -> JsonObject:
        """告诉前端这个节点已经开始执行。"""

        show_content = message or f"{self.node_title}已开始执行"
        stage = _normalize_stage(extra.get("stage"))
        parent_event = self._emit_node_runtime_event("running", show_content, extra)
        if stage and not self._stage_updates_parent(stage):
            return self._emit_stage_runtime_event("running", show_content, extra)
        return parent_event

    def progress(self, message: str, **extra: Any) -> JsonObject:
        """告诉前端这个节点执行到了哪一步。"""

        stage = _normalize_stage(extra.get("stage"))
        if stage and self._stage_updates_parent(stage):
            return self._emit_node_runtime_event("running", message, extra)
        return self._emit_stage_runtime_event("running", message, extra)

    def completed(self, message: str | None = None, **extra: Any) -> JsonObject:
        """告诉前端这个节点已经顺利执行完。"""

        show_content = message or f"{self.node_title}已完成"
        stage = _normalize_stage(extra.get("stage"))
        if stage and not self._stage_updates_parent(stage):
            self._emit_stage_runtime_event("completed", show_content, extra)
        return self._emit_node_runtime_event("completed", show_content, extra)

    def failed(self, message: str, **extra: Any) -> JsonObject:
        """告诉前端这个节点执行失败，方便界面统一显示。"""

        stage = _normalize_stage(extra.get("stage"))
        if stage and not self._stage_updates_parent(stage):
            self._emit_stage_runtime_event("failed", message, extra)
        return self._emit_node_runtime_event("failed", message, extra)

    def reasoning_delta(self, content: str, **extra: Any) -> JsonObject:
        """把节点内部的思考说明实时往前端推。"""

        return self.sync_port.emit(
            {
                "event": "reasoning_delta",
                "content": content,
                "node_key": self.node_key,
                "node_title": self.node_title,
                **extra,
            }
        )

    def reasoning_end(self, **extra: Any) -> JsonObject:
        """告诉前端当前这段思考说明已经结束。"""

        return self.sync_port.emit(
            {
                "event": "reasoning_end",
                "node_key": self.node_key,
                "node_title": self.node_title,
                **extra,
            }
        )

    def message(
        self,
        *,
        role: str,
        content: str,
        kind: str = "message",
        metadata: JsonObject | None = None,
        media: list[JsonObject] | None = None,
        **extra: Any,
    ) -> JsonObject:
        """发送一条标准 message 事件，给会话消息列表消费。"""

        return self.sync_port.emit(
            {
                "event": "message",
                "role": role,
                "kind": kind,
                "content": content,
                "metadata": copy.deepcopy(metadata or {}),
                "media": copy.deepcopy(media or []),
                "node_key": self.node_key,
                "node_title": self.node_title,
                **extra,
            }
        )

    def delta(self, content: str, **extra: Any) -> JsonObject:
        """发送正文增量，适合未来逐段输出最终回答。"""

        return self.sync_port.emit(
            {
                "event": "delta",
                "content": content,
                "node_key": self.node_key,
                "node_title": self.node_title,
                **extra,
            }
        )

    def artifact(self, artifact: JsonObject, **extra: Any) -> JsonObject:
        """发送产物事件，同时把“产物已保存”同步到执行过程里。"""

        # 中文注释：产物本身仍然走 artifact 事件，方便右侧产物列表复用原来的展示逻辑。
        # 但执行过程里也需要看到“保存产物”这个步骤，所以这里先发一条 runtime_event。
        self._emit_stage_runtime_event(
            "completed",
            "产物已保存",
            {**extra, "artifact": copy.deepcopy(artifact)},
        )
        return self.sync_port.emit(
            {
                "event": "artifact",
                "artifact": copy.deepcopy(artifact),
                "node_key": self.node_key,
                "node_title": self.node_title,
                **extra,
            }
        )

    def _emit_node_runtime_event(self, status: str, show_content: str, extra: JsonObject) -> JsonObject:
        """发送或更新当前节点自己的根事件。"""

        return self._emit_runtime_event(
            event_id=self._node_event_id(),
            parent_id=None,
            event_type="workflow_node",
            title=self.node_title,
            status=status,
            show_content=show_content,
            extra=extra,
            stage_display=None,
        )

    def _emit_stage_runtime_event(self, status: str, show_content: str, extra: JsonObject) -> JsonObject:
        """发送或更新当前节点下面的某个步骤事件。"""

        stage = _normalize_stage(extra.get("stage")) or "step"
        stage_display = self._stage_display(stage)
        resolved_status = stage_display.status or status
        resolved_show_content = stage_display.show_content or show_content
        event_id = self._node_event_id() if stage_display.updates_parent else self._step_event_id(stage_display.event_key)
        parent_id = None if stage_display.updates_parent else self._node_event_id()
        event_type = "workflow_node" if stage_display.updates_parent else "workflow_step"
        return self._emit_runtime_event(
            event_id=event_id,
            parent_id=parent_id,
            event_type=event_type,
            title=stage_display.title,
            status=resolved_status,
            show_content=resolved_show_content,
            extra=extra,
            stage_display=stage_display,
        )

    def _emit_runtime_event(
        self,
        *,
        event_id: str,
        parent_id: str | None,
        event_type: str,
        title: str,
        status: str,
        show_content: str,
        extra: JsonObject,
        stage_display: RuntimeStageDisplay | None,
    ) -> JsonObject:
        """组装统一的 runtime_event，并交给外层 SSE 管道发送。"""

        now = utc_now()
        metadata = self._runtime_metadata(extra, stage_display)
        detail_content = _detail_content_from_extra(extra)
        # 中文注释：这里最重要的是 id。前端收到同一个 id 时会更新旧事件，
        # 而不是再新增一条，所以状态从“处理中”变为“完成”时界面不会重复。
        payload: JsonObject = {
            "event": "runtime_event",
            "id": event_id,
            "parent_id": parent_id,
            "type": event_type,
            "title": title,
            "status": status,
            "show_content": show_content,
            "detail_content": detail_content,
            "metadata": metadata,
            "created_at": now,
            "updated_at": now,
            "completed_at": now if status in _DONE_STATUSES else None,
            "content": show_content,
        }
        return self.sync_port.emit(payload)

    def _runtime_metadata(self, extra: JsonObject, stage_display: RuntimeStageDisplay | None) -> JsonObject:
        """把节点、阶段和业务字段整理成前端可展开查看的普通字典。"""

        metadata = {
            "node_key": self.node_key,
            "node_title": self.node_title,
            "stage": _normalize_stage(extra.get("stage")),
            "event_key": stage_display.event_key if stage_display is not None else self.node_key,
        }
        for key, value in extra.items():
            # 中文注释：stage 已经单独放过；其他字段原样放进 metadata，方便前端展示详情或恢复按钮使用。
            if key == "stage":
                continue
            metadata[key] = copy.deepcopy(value)
        return metadata

    def _node_event_id(self) -> str:
        """返回当前节点的稳定事件 id。"""

        turn_id = str(getattr(self.sync_port, "turn_id", "") or "session")
        return f"{turn_id}:{self.node_key}"

    def _step_event_id(self, event_key: str) -> str:
        """返回当前节点下某个步骤的稳定事件 id。"""

        return f"{self._node_event_id()}:{event_key}"

    def _stage_display(self, stage: str) -> RuntimeStageDisplay:
        """把内部 stage 转成前端更容易看懂的标题和更新目标。"""

        return _STAGE_DISPLAY.get(
            (self.node_key, stage),
            RuntimeStageDisplay(stage, _humanize_stage(stage)),
        )

    def _stage_updates_parent(self, stage: str) -> bool:
        """判断某个 stage 是不是应该直接更新节点根事件。"""

        return self._stage_display(stage).updates_parent


def _normalize_stage(value: Any) -> str | None:
    """把 stage 整理成非空字符串。"""

    text = str(value or "").strip()
    return text or None


def _humanize_stage(stage: str) -> str:
    """把没有配置的 stage 变成可读标题，避免界面直接显示下划线。"""

    return stage.replace("_", " ").strip() or "执行步骤"


def _detail_content_from_extra(extra: JsonObject) -> JsonObject | None:
    """从节点上报的额外字段里提取详情内容。"""

    detail: JsonObject = {}
    for key, value in extra.items():
        # 中文注释：stage 只是分类字段，标题里已经能看出来；详情里重复展示会显得啰嗦。
        if key == "stage":
            continue
        detail[key] = copy.deepcopy(value)
    return detail or None
