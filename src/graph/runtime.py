from __future__ import annotations

import copy
import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol


if TYPE_CHECKING:
    from src.graph.runtime_resources import WorkflowRuntimeResources

from src.models.sessions import utc_now


JsonObject = dict[str, Any]
WorkflowEventEmitter = Callable[[JsonObject], JsonObject]


class WorkflowCancellation:
    """保存一次运行是否收到用户停止请求。"""

    def __init__(self) -> None:
        """创建一个还没有收到停止请求的控制对象。"""

        self._requested = False
        self._event = asyncio.Event()

    def request(self) -> None:
        """记录用户已经请求停止，并唤醒正在等待这个信号的代码。"""

        self._requested = True
        self._event.set()

    def is_requested(self) -> bool:
        """返回用户是否已经请求停止。"""

        return self._requested

    async def wait(self) -> None:
        """等待用户发出停止请求，供需要主动等待的任务使用。"""

        await self._event.wait()

    def raise_if_requested(self) -> None:
        """在安全边界检查停止请求，收到请求后让当前异步任务退出。"""

        if self._requested:
            raise asyncio.CancelledError()


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
    # 中文注释：这个对象只保存“用户是否点了停止”，不保存模型或数据库连接，
    # 节点可以在开始新阶段前检查它，后台服务也可以用它配合取消当前任务。
    cancellation: WorkflowCancellation | None = None
    # 中文注释：resources 里放的是单次 run 共用的并发控制对象和公共资源，
    # 例如下载限流、共用 AsyncClient、embedding 连接等。节点通过 runtime_context
    # 就能拿到这些资源，不需要再往 State 顶层散落很多字段。
    resources: WorkflowRuntimeResources | None = None


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
    # 决定某个阶段是“挂在节点下面的子步骤”，还是“直接更新节点本身”,updates_parent=True 的阶段会直接更新父节点，不会在前端多出一层重复卡片
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
    ("analyse", "analyse_start"): RuntimeStageDisplay("analyse", "论文分析", updates_parent=True),
    ("analyse", "analyse_subtopic"): RuntimeStageDisplay("analyse_subtopic", "分析子主题"),
    ("analyse", "analyse_overall"): RuntimeStageDisplay("analyse_overall", "综合子主题"),
    ("analyse", "analysis_artifact_ready"): RuntimeStageDisplay(
        "analysis_artifact_ready",
        "保存分析报告",
        show_content="分析报告已保存",
        status="completed",
    ),
    ("analyse", "analyse_done"): RuntimeStageDisplay(
        "analyse",
        "论文分析",
        show_content="论文分析已完成",
        status="completed",
        updates_parent=True,
    ),
    # 中文说明：写作大纲开始和结束时都直接更新“写作大纲”这一张主卡片，
    # 这样开始事件不会单独留下一个一直显示“处理中”的旧步骤。
    ("write_outline", "writing_outline_start"): RuntimeStageDisplay(
        "write_outline",
        "写作大纲",
        updates_parent=True,
    ),
    # 中文说明：模型用量回调也属于写作大纲主阶段，直接更新主卡片，
    # 避免额外生成一张一直停留在“处理中”的 writing outline 子卡片。
    ("write_outline", "writing_outline"): RuntimeStageDisplay(
        "write_outline",
        "写作大纲",
        updates_parent=True,
    ),
    # 中文说明：保存出来的 JSON 产物仍然单独显示，用户可以知道文件已经保存成功。
    ("write_outline", "writing_outline_artifact_ready"): RuntimeStageDisplay(
        "writing_outline_artifact_ready",
        "保存写作大纲",
        show_content="写作大纲产物已保存",
        status="completed",
    ),
    # 中文说明：结束事件继续更新同一张主卡片，并把它的状态改成已完成。
    ("write_outline", "writing_outline_done"): RuntimeStageDisplay(
        "write_outline",
        "写作大纲",
        show_content="写作大纲已完成",
        status="completed",
        updates_parent=True,
    ),
    # 中文说明：正文开始时只更新“正文写作”主卡，不再额外显示 writing start 子卡。
    ("write", "writing_start"): RuntimeStageDisplay(
        "write",
        "正文写作",
        updates_parent=True,
    ),
    # 中文说明：摘要和参考文献各自只保留一张卡，后续用同一个事件键更新处理状态。
    ("write", "writing_abstract"): RuntimeStageDisplay("writing_abstract", "摘要写作"),
    ("write", "writing_references"): RuntimeStageDisplay("writing_references", "参考文献生成"),
    ("compose_reply", "compose_start"): RuntimeStageDisplay("compose_reply", "回复整理", updates_parent=True),
    ("compose_reply", "compose_reply"): RuntimeStageDisplay("compose_reply_step", "生成最终回复"),
    ("compose_reply", "final_artifact_ready"): RuntimeStageDisplay(
        "final_artifact_ready",
        "保存最终 Markdown 论文",
        show_content="最终 Markdown 论文文件已保存",
        status="completed",
    ),
    ("compose_reply", "compose_done"): RuntimeStageDisplay(
        "compose_reply_step",
        "生成最终回复",
        show_content="最终回复整理完成",
        status="completed",
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
        # 中文注释：大多数进度都是 running，但单篇论文的某个阶段失败时，
        # 也需要更新同一张论文卡片为 failed，而不是另外新建一张失败卡片。
        runtime_status = _normalize_stage(extra.get("runtime_status")) or "running"
        if stage and self._stage_updates_parent(stage):
            return self._emit_node_runtime_event(runtime_status, message, extra)
        return self._emit_stage_runtime_event(runtime_status, message, extra)

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
        """发送产物事件，并按需把“产物已保存”同步到执行过程里。"""

        # 中文注释：产物本身仍然走 artifact 事件，方便右侧产物列表复用原来的展示逻辑。
        # 有些产物保存状态会由业务卡片自己更新，比如“某篇论文的阅读结果已保存”；
        # 这时可以传 emit_runtime_event=False，避免前端多出一张重复的保存卡片。
        emit_runtime_event = bool(extra.pop("emit_runtime_event", True))
        if emit_runtime_event:
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
        # 中文注释：阶段默认状态只适合成功场景。
        # 如果当前已经明确失败或取消，就保留真实状态，避免前端误显示成已完成。
        if status in {"failed", "cancelled", "skipped"}:
            resolved_status = status
        else:
            resolved_status = stage_display.status or status
        # 中文注释：有些阶段有默认文案，但单篇论文卡片会传入更具体的说明，
        # 例如“全文下载失败，已保留摘要阅读结果”。这种具体说明要优先展示。
        custom_show_content = str(extra.get("show_content") or "").strip()
        resolved_show_content = custom_show_content or stage_display.show_content or show_content
        custom_event_key = str(extra.get("event_key") or stage_display.event_key).strip() or stage_display.event_key
        custom_title = str(extra.get("stage_title") or stage_display.title).strip() or stage_display.title
        event_id = self._node_event_id() if stage_display.updates_parent else self._step_event_id(custom_event_key)
        parent_id = None if stage_display.updates_parent else self._node_event_id()
        event_type = "workflow_node" if stage_display.updates_parent else "workflow_step"
        return self._emit_runtime_event(
            event_id=event_id,
            parent_id=parent_id,
            event_type=event_type,
            title=custom_title,
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
        # 只有模型回调明确提供用量时才覆盖卡片数字，普通进度更新不会把已有用量清零。
        if "input_tokens" in extra:
            payload["input_tokens"] = _token_count(extra.get("input_tokens"))
        if "output_tokens" in extra:
            payload["output_tokens"] = _token_count(extra.get("output_tokens"))
        return self.sync_port.emit(payload)

    def _runtime_metadata(self, extra: JsonObject, stage_display: RuntimeStageDisplay | None) -> JsonObject:
        """把节点、阶段和业务字段整理成前端可展开查看的普通字典。"""

        # 中文说明：小节会在上报时传入带 section_id 的事件键，详情里的 event_key 也要保持一致，
        # 这样排查历史事件时能准确知道它对应哪一个小节。
        event_key = str(extra.get("event_key") or (stage_display.event_key if stage_display is not None else self.node_key)).strip()
        metadata = {
            "node_key": self.node_key,
            "node_title": self.node_title,
            "stage": _normalize_stage(extra.get("stage")),
            "event_key": event_key or self.node_key,
        }
        for key, value in extra.items():
            # 中文注释：stage 已经单独放过；其他字段原样放进 metadata，方便前端展示详情或恢复按钮使用。
            if key == "stage":
                continue
            if key in {"event_key", "stage_title"}:
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


def _token_count(value: Any) -> int:
    """把 token 数整理成非负整数，避免异常返回值影响运行事件。"""

    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


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
        if key in {"event_key", "stage_title"}:
            continue
        detail[key] = copy.deepcopy(value)
    return detail or None
