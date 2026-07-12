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
    """把工作流节点事件桥接到外层 emit 回调的简单实现。"""

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


@dataclass(slots=True)
class WorkflowNodeReporter:
    """帮单个节点稳定地产生事件，避免节点里散落大量样板代码。"""

    sync_port: WorkflowSyncPort
    node_key: str
    node_title: str

    def started(self, message: str | None = None, **extra: Any) -> JsonObject:
        """告诉外层这个节点已经开始执行。"""

        return self.sync_port.emit(
            {
                "event": "node_started",
                "node_key": self.node_key,
                "node_title": self.node_title,
                "message": message or f"{self.node_title}已开始执行",
                **extra,
            }
        )

    def progress(self, message: str, **extra: Any) -> JsonObject:
        """告诉外层这个节点执行到了哪一步。"""

        return self.sync_port.emit(
            {
                "event": "node_progress",
                "node_key": self.node_key,
                "node_title": self.node_title,
                "message": message,
                **extra,
            }
        )

    def completed(self, message: str | None = None, **extra: Any) -> JsonObject:
        """告诉外层这个节点已经顺利执行完。"""

        return self.sync_port.emit(
            {
                "event": "node_completed",
                 
                "node_key": self.node_key,
                "node_title": self.node_title,
                "message": message or f"{self.node_title}已完成",
                **extra,
            }
        )

    def failed(self, message: str, **extra: Any) -> JsonObject:
        """告诉外层这个节点执行失败，方便前端和日志统一显示。"""

        return self.sync_port.emit(
            {
                "event": "node_failed",
                "node_key": self.node_key,
                "node_title": self.node_title,
                "message": message,
                **extra,
            }
        )

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
        """发送一条标准 message 事件，给前端时间线直接消费。"""

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
        """发送产物事件，让前端在文件写入当下就能看到结果。"""

        return self.sync_port.emit(
            {
                "event": "artifact",
                "artifact": copy.deepcopy(artifact),
                "node_key": self.node_key,
                "node_title": self.node_title,
                **extra,
            }
        )

