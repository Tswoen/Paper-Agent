from __future__ import annotations

import asyncio
import copy
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.models.sessions import utc_now
from src.repositories.sessions.base import SessionRepository
from src.services.sessions import AssistantMessageBuffer, MessageHandler, SessionError
from src.utils import get_logger, logging_context


JsonObject = dict[str, Any]
logger = get_logger(__name__)


@dataclass(slots=True)
class RunBrokerState:
    """描述一次运行在内存中的流式状态。

    中文说明：
    该结构只负责服务进程存活期间的 SSE 分发，不承担长期持久化职责。
    事件的长期保存仍由 `SessionRepository.append_event` 负责。
    """

    session_key: str
    run_id: str
    turn_id: str
    created_at: str
    events: list[JsonObject] = field(default_factory=list)
    subscribers: list[asyncio.Queue[JsonObject | None]] = field(default_factory=list)
    closed: bool = False


class SessionRunBroker:
    """为每个 run 提供内存事件通道与订阅能力。

    中文说明：
    后台工作流通过 broker 发布事件，SSE 接口通过 broker 订阅事件。
    这样可以把“实时推送”与“会话落库”解耦，避免路由层直接感知仓储细节。
    """

    def __init__(self):
        """初始化运行期事件 broker。"""

        self._runs: dict[str, RunBrokerState] = {}
        self._lock = asyncio.Lock()

    async def open_run(self, session_key: str, run_id: str, turn_id: str, created_at: str) -> None:
        """注册一条新的运行记录，使其可以被后续发布和订阅。"""

        async with self._lock:
            self._runs[run_id] = RunBrokerState(
                session_key=session_key,
                run_id=run_id,
                turn_id=turn_id,
                created_at=created_at,
            )

    async def publish(self, run_id: str, event: JsonObject) -> JsonObject:
        """向指定 run 发布一条事件，并广播给所有在线订阅者。"""

        async with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                raise SessionError(f"run not found: {run_id}", 404)
            event_copy = copy.deepcopy(event)
            event_copy.setdefault("stream_seq", len(state.events) + 1)
            state.events.append(event_copy)
            subscribers = list(state.subscribers)

        for queue in subscribers:
            await queue.put(copy.deepcopy(event_copy))
        return event_copy

    async def close_run(self, run_id: str) -> None:
        """标记指定 run 已结束，并唤醒所有订阅者退出。"""

        async with self._lock:
            state = self._runs.get(run_id)
            if state is None or state.closed:
                return
            state.closed = True
            subscribers = list(state.subscribers)

        for queue in subscribers:
            await queue.put(None)

    async def subscribe(self, run_id: str):
        """订阅指定 run 的事件流，并先补发历史缓存。"""

        queue: asyncio.Queue[JsonObject | None] = asyncio.Queue()
        async with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                raise SessionError(f"run not found: {run_id}", 404)
            history = [copy.deepcopy(item) for item in state.events]
            closed = state.closed
            if not closed:
                state.subscribers.append(queue)

        try:
            for event in history:
                yield event
            if closed:
                return
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield copy.deepcopy(item)
        finally:
            async with self._lock:
                state = self._runs.get(run_id)
                if state is not None and queue in state.subscribers:
                    state.subscribers.remove(queue)


class SessionRunService:
    """负责启动后台 run、发布事件并对接 SSE。

    中文说明：
    该服务是 `/runs + SSE` 的核心编排层，负责校验会话状态、创建 run 元数据、
    启动后台任务、把事件同时落库并推给 broker，以及在结束时收口状态。
    """

    def __init__(
        self,
        repo: SessionRepository,
        message_handler: MessageHandler | None = None,
        broker: SessionRunBroker | None = None,
    ):
        """初始化会话运行服务并注入仓储、事件处理器与 broker。"""

        self.repo = repo
        self.message_handler = message_handler
        self.broker = broker or SessionRunBroker()

    async def start_run(self, session_key: str, body: JsonObject | None = None) -> JsonObject:
        """创建一次新的后台运行并立即返回可订阅的流地址。"""

        body = body or {}
        content = str(body.get("content") or "")
        if not content.strip() and not body.get("media"):
            raise ValueError("content is required")

        record = self.repo.get(session_key)
        if record.run_started_at or record.status == "running":
            raise SessionError("session is already running", 409)

        turn_id = str(body.get("turn_id") or uuid.uuid4().hex)
        run_id = str(body.get("run_id") or f"run_{uuid.uuid4().hex}")
        media = list(body.get("media") or [])
        started_at = utc_now()

        self.repo.append_message(session_key, "user", content, media=media, turn_id=turn_id)
        self.repo.set_status(session_key, "running")
        self.repo.set_run_started_at(session_key, started_at)
        await self.broker.open_run(session_key, run_id, turn_id, started_at)

        await self._publish_runtime_event(
            session_key,
            run_id,
            turn_id,
            {
                "event": "message",
                "role": "user",
                "content": content,
                "media": media,
                "turn_id": turn_id,
                "timestamp": started_at,
            },
            persist_event=True,
        )
        await self._publish_runtime_event(
            session_key,
            run_id,
            turn_id,
            {
                "event": "status",
                "status": "running",
                "run_started_at": started_at,
                "turn_id": turn_id,
                "timestamp": started_at,
            },
            persist_event=True,
        )

        loop = asyncio.get_running_loop()
        asyncio.create_task(
            self._run_in_background(
                loop=loop,
                session_key=session_key,
                run_id=run_id,
                turn_id=turn_id,
                content=content,
                body=copy.deepcopy(body),
            )
        )
        return {
            "session_key": session_key,
            "run_id": run_id,
            "turn_id": turn_id,
            "status": "accepted",
            "stream_url": f"/api/sessions/{session_key}/runs/{run_id}/stream",
        }

    async def stream_events(self, session_key: str, run_id: str):
        """返回指定 run 的事件异步迭代器，并校验会话归属。"""

        async for event in self.broker.subscribe(run_id):
            if str(event.get("session_key") or "") != session_key:
                continue
            yield event

    async def _run_in_background(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        session_key: str,
        run_id: str,
        turn_id: str,
        content: str,
        body: JsonObject,
    ) -> None:
        """在线程池中执行阻塞工作流，并确保异常和收尾被正确处理。"""

        await asyncio.to_thread(
            self._run_sync,
            loop,
            session_key,
            run_id,
            turn_id,
            content,
            body,
        )

    def _run_sync(
        self,
        loop: asyncio.AbstractEventLoop,
        session_key: str,
        run_id: str,
        turn_id: str,
        content: str,
        body: JsonObject,
    ) -> None:
        """在工作线程中运行同步工作流，并通过线程安全桥接发布事件。"""

        resolved_handler = self.message_handler
        assistant_buffer = AssistantMessageBuffer()
        saw_turn_end = False

        def emit(event: JsonObject) -> None:
            """把工作线程中的事件安全地投递回主事件循环。"""

            nonlocal saw_turn_end
            event_copy = copy.deepcopy(event)
            assistant_buffer.apply(event_copy)
            if str(event_copy.get("event") or "") == "turn_end":
                saw_turn_end = True
            asyncio.run_coroutine_threadsafe(
                self._publish_runtime_event(
                    session_key,
                    run_id,
                    turn_id,
                    event_copy,
                    persist_event=True,
                ),
                loop,
            ).result()

        try:
            with logging_context(session_key=session_key, turn_id=turn_id, run_id=run_id):
                logger.info("后台 run 开始执行", extra={"content_length": len(content)})
                emit(
                    {
                        "event": "message",
                        "kind": "progress",
                        "role": "system",
                        "content": "正在启动论文工作流",
                        "step": "bootstrap",
                        "turn_id": turn_id,
                        "timestamp": utc_now(),
                    }
                )
                if resolved_handler is None:
                    raise SessionError("message handler is not configured", 500)
                resolved_handler(session_key, content, {"run_id": run_id, "turn_id": turn_id, **body}, emit)
                if not saw_turn_end:
                    emit(
                        {
                            "event": "turn_end",
                            "status": "completed",
                            "turn_id": turn_id,
                            "timestamp": utc_now(),
                        }
                    )
                asyncio.run_coroutine_threadsafe(
                    self._finalize_success(session_key, run_id, turn_id, assistant_buffer),
                    loop,
                ).result()
                logger.info("后台 run 执行完成")
        except Exception as error:
            logger.exception("后台 run 执行失败")
            asyncio.run_coroutine_threadsafe(
                self._finalize_failure(
                    session_key=session_key,
                    run_id=run_id,
                    turn_id=turn_id,
                    assistant_buffer=assistant_buffer,
                    error=error,
                    already_closed=saw_turn_end,
                ),
                loop,
            ).result()

    async def _finalize_success(
        self,
        session_key: str,
        run_id: str,
        turn_id: str,
        assistant_buffer: AssistantMessageBuffer,
    ) -> None:
        """在运行成功时写回助手消息、更新状态并关闭事件流。"""

        assistant_buffer.persist(self.repo, session_key, turn_id)
        self.repo.set_status(session_key, "completed")
        self.repo.set_run_started_at(session_key, None)
        await self.broker.close_run(run_id)

    async def _finalize_failure(
        self,
        *,
        session_key: str,
        run_id: str,
        turn_id: str,
        assistant_buffer: AssistantMessageBuffer,
        error: Exception,
        already_closed: bool,
    ) -> None:
        """在运行失败时补发错误与终止事件，并统一收口状态。"""

        await self._publish_runtime_event(
            session_key,
            run_id,
            turn_id,
            {
                "event": "error",
                "message": str(error),
                "content": str(error),
                "status": "failed",
                "turn_id": turn_id,
                "timestamp": utc_now(),
            },
            persist_event=True,
        )
        if not already_closed:
            await self._publish_runtime_event(
                session_key,
                run_id,
                turn_id,
                {
                    "event": "turn_end",
                    "status": "failed",
                    "turn_id": turn_id,
                    "timestamp": utc_now(),
                },
                persist_event=True,
            )
        assistant_buffer.persist(self.repo, session_key, turn_id)
        self.repo.set_status(session_key, "failed")
        self.repo.set_run_started_at(session_key, None)
        await self.broker.close_run(run_id)

    async def _publish_runtime_event(
        self,
        session_key: str,
        run_id: str,
        turn_id: str,
        event: JsonObject,
        *,
        persist_event: bool,
    ) -> JsonObject:
        """标准化单条运行事件，并按需同时落库和广播。"""

        normalized_event = self._normalize_event(session_key, run_id, turn_id, event)
        if persist_event:
            self.repo.append_event(
                session_key,
                str(normalized_event.get("event") or "unknown"),
                content=str(
                    normalized_event.get("content")
                    or normalized_event.get("delta")
                    or normalized_event.get("message")
                    or ""
                ),
                metadata={
                    key: copy.deepcopy(value)
                    for key, value in normalized_event.items()
                    if key != "content"
                },
                created_at=str(normalized_event["timestamp"]),
            )
        await self.broker.publish(run_id, normalized_event)
        return normalized_event

    def _normalize_event(self, session_key: str, run_id: str, turn_id: str, event: JsonObject) -> JsonObject:
        """补齐运行事件的统一字段，保证历史和实时口径一致。"""

        normalized_event = copy.deepcopy(event)
        normalized_event.setdefault("event", "message")
        normalized_event["session_key"] = session_key
        normalized_event["chat_id"] = session_key
        normalized_event["run_id"] = run_id
        normalized_event["turn_id"] = str(normalized_event.get("turn_id") or turn_id)
        normalized_event["timestamp"] = str(normalized_event.get("timestamp") or utc_now())
        return normalized_event


def encode_sse(event: JsonObject) -> str:
    """把结构化事件编码为 SSE 文本块。

    中文说明：
    这里显式输出 `event:` 和 `id:`，前端既能按事件名订阅，也能在需要时
    利用 `stream_seq` 做断线恢复的基础准备。
    """

    event_name = str(event.get("event") or "message")
    payload = json.dumps(event, ensure_ascii=False)
    event_id = str(event.get("stream_seq") or "")
    lines = []
    if event_id:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event_name}")
    lines.append(f"data: {payload}")
    return "\n".join(lines) + "\n\n"
