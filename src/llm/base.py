from __future__ import annotations

import asyncio
import inspect
import json
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Mapping as MappingABC
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping, Sequence

from src.utils import get_logger


Message = Mapping[str, Any]
JsonObject = dict[str, Any]

logger = get_logger(__name__)


@dataclass(slots=True)
class ToolCallRequest:
    # 内部统一成 OpenAI 风格的工具调用，便于上层不关心厂商协议。
    """统一描述一次模型发起的工具调用请求。

    Attributes:
        id: 工具调用唯一标识。某些厂商会返回，用于后续 tool_result 关联。
        name: 工具名称，通常对应 function name 或工具注册名。
        arguments: 工具参数，尽量保持结构化对象；若上游返回原始字符串，也允许保留。
        provider_specific_fields: 原始供应商字段，便于调试、透传或兼容特殊协议细节。

    这里刻意统一成接近 OpenAI `tool_calls` 的内部结构，这样上层 runner、
    agent 编排逻辑就不需要感知 Anthropic、OpenAI 或其他供应商的协议差异。
    """
    id: str | None
    name: str
    arguments: Any
    provider_specific_fields: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class GenerationSettings:
    # 只放通用生成参数，厂商私有参数统一走 ProviderConfig.extra_body。
    """描述一次生成请求中跨供应商共用的参数集合。

    Attributes:
        temperature: 采样温度，控制生成随机性。
        max_tokens: 最大生成 token 数。
        reasoning_effort: 推理强度或推理模式开关，供支持 reasoning/thinking 的模型使用。

    这里只承载“通用概念”，避免把厂商私有参数塞进基类。私有扩展统一通过
    `extra_body` 走原始协议透传，减少抽象层膨胀。
    """
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None


@dataclass(slots=True)
class LLMResponse:
    # 所有 provider 都返回这个结构，避免上层直接依赖 SDK 的响应/异常类型。
    """统一封装一次 LLM 调用结果。

    Attributes:
        content: 模型返回的主文本内容。
        tool_calls: 模型请求调用的工具列表。
        finish_reason: 结束原因，例如 `stop`、`length`、`tool_calls`、`error`。
        usage: 供应商返回的 token 使用量等统计信息。
        reasoning_content: 推理模型额外返回的 thinking/reasoning 内容。
        error_status_code: HTTP 状态码；仅失败时可能存在。
        error_kind: 归一化后的错误类别，例如 rate_limit、auth、server_error。
        error_type: 供应商错误体中的更细粒度错误类型。
        error_code: 供应商错误体中的业务错误码。
        error_retry_after_s: 建议等待多久再重试，单位秒。
        error_should_retry: 是否建议重试；若为空则由通用策略继续判断。

    这个对象把“成功响应”和“失败响应”都统一进同一个结构里，使得上层调用方
    不用分别处理 SDK 异常类型、HTTP 异常类型和正常返回类型。
    """
    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str | None = None
    usage: JsonObject | None = None
    reasoning_content: str | None = None
    error_status_code: int | None = None
    error_kind: str | None = None
    error_type: str | None = None
    error_code: str | None = None
    error_retry_after_s: float | None = None
    error_should_retry: bool | None = None

    @property
    def ok(self) -> bool:
        """判断当前响应是否可视为成功。

        Returns:
            只要 `finish_reason` 不是 `"error"` 就返回 True。

        这里的判断非常克制：长度截断、tool_calls、中止等情况虽然不一定是
        “完整回答”，但依然不是传输/协议级错误，因此统一视为成功响应。
        """
        return self.finish_reason != "error"


def normalize_token_usage(usage: Mapping[str, Any] | None) -> JsonObject:
    """把不同模型服务商返回的用量字段整理成统一的输入和输出数量。"""

    if not isinstance(usage, MappingABC):
        return {"input_tokens": 0, "output_tokens": 0}

    # OpenAI 常用 prompt/completion，Anthropic 常用 input/output，其他兼容服务还会使用带 count 的写法。
    input_keys = (
        "input_tokens",
        "prompt_tokens",
        "input_token_count",
        "prompt_token_count",
        "inputTokenCount",
        "promptTokenCount",
    )
    output_keys = (
        "output_tokens",
        "completion_tokens",
        "output_token_count",
        "completion_token_count",
        "outputTokenCount",
        "candidatesTokenCount",
        "completionTokenCount",
    )

    def read_count(keys: tuple[str, ...]) -> int:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, MappingABC):
                value = value.get("count")
            try:
                if value is not None:
                    return max(0, int(value))
            except (TypeError, ValueError):
                continue
        return 0

    return {
        "input_tokens": read_count(input_keys),
        "output_tokens": read_count(output_keys),
    }


@dataclass(slots=True)
class EmbeddingResponse:
    # embedding 的主结果是向量列表，所以单独定义响应对象，不和文本回复混在一起。
    """统一封装一次 embedding 调用结果。

    Attributes:
        embeddings: 与输入文本一一对应的向量列表。
        model: 实际完成向量化的模型名，便于前端和日志展示。
        usage: 供应商返回的 token 用量等统计信息。
        finish_reason: 结束原因；失败时统一为 `"error"`。
        error_status_code: HTTP 状态码；仅失败时可能存在。
        error_kind: 归一化后的错误类别，例如 rate_limit、auth、server_error。
        error_type: 供应商错误体中的更细粒度错误类型。
        error_code: 供应商错误体中的业务错误码。
        error_retry_after_s: 建议等待多久再重试，单位秒。
        error_should_retry: 是否建议重试；若为空则由通用策略继续判断。
        content: 错误说明或供应商原始错误文本。

    chat 返回的是自然语言文本，embedding 返回的是数字向量。这里单独建一个
    响应对象，让调用方不用在文本字段里猜测向量数据放在哪里；但错误字段保持和
    LLMResponse 一致，方便上层用同一套展示和重试逻辑。
    """
    embeddings: list[list[float]] = field(default_factory=list)
    model: str | None = None
    usage: JsonObject | None = None
    finish_reason: str | None = None
    error_status_code: int | None = None
    error_kind: str | None = None
    error_type: str | None = None
    error_code: str | None = None
    error_retry_after_s: float | None = None
    error_should_retry: bool | None = None
    content: str = ""

    @property
    def ok(self) -> bool:
        """判断 embedding 请求是否成功。"""

        return self.finish_reason != "error"


@dataclass(slots=True)
class StreamCallbacks:
    # 流式输出拆成文本、思考、工具调用三类增量，方便 runner 按需消费。
    """定义流式输出时可选的三类回调。

    Attributes:
        on_content_delta: 接收文本内容增量。
        on_thinking_delta: 接收 reasoning/thinking 内容增量。
        on_tool_call_delta: 接收工具调用相关增量，通常是参数片段或结构化事件。

    不同供应商在流式协议上差异很大，这里只统一“上层真正关心的事件类型”，
    由各 provider 把原始流式事件翻译成这些回调。
    """
    on_content_delta: Callable[[str], None] | None = None
    on_thinking_delta: Callable[[str], None] | None = None
    on_tool_call_delta: Callable[[JsonObject], None] | None = None


class ProviderHttpError(Exception):
    def __init__(self, status_code: int, body: str, headers: Mapping[str, str]):
        """表示供应商返回了明确的 HTTP 错误响应。

        Args:
            status_code: HTTP 状态码。
            body: 响应体文本，通常包含供应商返回的错误 JSON。
            headers: 响应头，用于提取 `retry-after` 等重试信息。

        这个异常主要给没有统一 SDK 错误类型的场景使用，便于在基类里按统一逻辑
        做错误归类和重试判断。
        """
        super().__init__(body)
        self.status_code = status_code
        self.body = body
        self.headers = headers


class ProviderConnectionError(Exception):
    """表示连接层异常，例如网络中断、DNS 失败、超时等非 HTTP 错误。"""
    pass


class LLMProvider(ABC):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        api_base: str,
        generation: GenerationSettings | None = None,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        client: Any | None = None,
        timeout_s: float = 60,
        max_retries: int | None = None,
        max_concurrency: int | None = None,
        include_stream_usage: bool | None = None,
    ):
        """初始化 LLM provider 的通用配置。

        Args:
            model: 当前 provider 使用的模型名，允许带内部路由前缀。
            api_key: 访问上游模型服务的 API Key，可为空以兼容本地代理或匿名模式。
            api_base: 上游服务地址，末尾斜杠会被统一去掉。
            generation: 默认生成参数 preset；单次调用可以覆盖。
            extra_headers: 额外请求头，例如供应商鉴权扩展字段。
            extra_body: 额外请求体字段，用于透传厂商私有参数。
            client: 可选外部注入 SDK client，常用于测试或自定义 transport。
            timeout_s: 请求超时时间，单位秒。
            max_retries: 最大重试次数；None 时使用默认 3 次。
            max_concurrency: 当前 provider 实例允许同时进行多少个真实请求。
            include_stream_usage: 流式请求是否尝试要求供应商返回 token 用量。

        这个基类只保存通用配置，不直接绑定任何具体协议；具体请求构造与响应解析
        由各个子类 provider 实现。
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.generation = generation or GenerationSettings()
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self.client = client
        self.timeout_s = float(max(1, timeout_s))
        # 运行控制统一放在基类，业务节点只负责调用 provider，不需要自己写重试和限流。
        self.max_retries = max(1, int(max_retries or 3))
        self.max_concurrency = max(1, int(max_concurrency or 2))
        self.include_stream_usage = True if include_stream_usage is None else bool(include_stream_usage)
        # 中文注释：provider 内部只保留一个“兜底并发保护”。真正的工作流并发上限，
        # 后续会优先由 WorkflowRuntimeResources 里的 read_model_semaphore / embedding_semaphore 控制。
        self._semaphore = asyncio.BoundedSemaphore(self.max_concurrency)
        # 中文注释：外部传进来的 client 通常由调用方负责关闭；provider 自己创建的 client 才在 aclose 里关闭。
        self._owns_client = client is None
        # 中文注释：旧同步 LangGraph 还没改完前，会短期通过这个 Runner 跑 async 主接口。
        # 它复用同一个事件循环，避免每次 asyncio.run 都新建/关闭循环导致 async client 连接池出问题。
        self._sync_runner: asyncio.Runner | None = None

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[JsonObject] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """执行一次非流式对话请求。

        子类需要把内部统一消息格式转换为目标厂商协议，并把原始响应解析成
        `LLMResponse`。这里定义的是 provider 必须遵守的最小异步接口契约。
        """
        raise NotImplementedError

    @abstractmethod
    async def chat_stream(
        self,
        messages: Sequence[Message],
        callbacks: StreamCallbacks,
        *,
        tools: Sequence[JsonObject] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """执行一次流式对话请求。

        子类除了返回最终聚合结果，还应在请求过程中通过 `callbacks` 把文本、
        thinking、工具调用等增量事件实时向上层透出。
        """
        raise NotImplementedError

    @abstractmethod
    async def embed(
        self,
        inputs: Sequence[str],
        *,
        dimensions: int | None = None,
    ) -> EmbeddingResponse:
        """执行一次文本向量化请求。

        Args:
            inputs: 需要转成向量的文本列表，返回结果需要与它一一对应。
            dimensions: 可选的目标向量维度，只传给支持该参数的 embedding 服务。

        Returns:
            统一的 EmbeddingResponse。支持 embedding 的 provider 需要返回向量；
            不支持 embedding 的 provider 应在自己的实现里抛出清楚错误，告诉用户
            为什么不能用它生成向量。
        """
        raise NotImplementedError

    def chat_with_retry(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        """兼容旧同步调用名，后续图执行改成 async 后会删除。"""

        # 中文注释：这里不再写第二套重试逻辑，只把旧同步调用临时桥接到 async 主接口。
        # 新代码请直接 `await provider.chat(...)`，不要继续扩大这个同步入口的使用范围。
        return self._run_async_compat(self.chat(messages, **kwargs))

    def chat_stream_with_retry(
        self,
        messages: Sequence[Message],
        callbacks: StreamCallbacks,
        **kwargs: Any,
    ) -> LLMResponse:
        """兼容旧同步流式调用名，后续图执行改成 async 后会删除。"""

        return self._run_async_compat(self.chat_stream(messages, callbacks, **kwargs))

    def embed_with_retry(self, inputs: Sequence[str], **kwargs: Any) -> EmbeddingResponse:
        """兼容旧同步 embedding 调用名，后续图执行改成 async 后会删除。"""

        return self._run_async_compat(self.embed(inputs, **kwargs))

    async def async_chat(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        """旧异步调用名；现在直接转到 async 主接口，不再用线程包装。"""

        return await self.chat(messages, **kwargs)

    async def async_chat_stream(self, messages: Sequence[Message], callbacks: StreamCallbacks, **kwargs: Any) -> LLMResponse:
        """旧异步流式调用名；现在直接转到 async 主接口，不再用线程包装。"""

        return await self.chat_stream(messages, callbacks, **kwargs)

    async def async_embed(self, inputs: Sequence[str], **kwargs: Any) -> EmbeddingResponse:
        """旧异步 embedding 调用名；现在直接转到 async 主接口，不再用线程包装。"""

        return await self.embed(inputs, **kwargs)

    async def list_models(self) -> list[JsonObject]:
        """列出 provider 可用模型；不支持的 provider 直接给出清楚提示。"""

        raise NotImplementedError("当前 provider 不支持读取模型目录")

    async def aclose(self) -> None:
        """关闭 provider 自己创建的异步客户端。"""

        # 中文注释：测试或调用方传进来的 client 可能还要继续复用，所以这里只关闭 provider 自己 new 的 client。
        if self._owns_client and self.client is not None:
            close = getattr(self.client, "aclose", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result
            else:
                close = getattr(self.client, "close", None)
                if close is not None:
                    close()
        self._close_sync_runner()

    def _run_async_compat(self, awaitable: Awaitable[Any]) -> Any:
        """给尚未迁移的同步图节点临时运行 async 主接口。"""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # 中文注释：同步线程里没有正在运行的事件循环，可以安全使用同一个 Runner 复用循环。
            if self._sync_runner is None:
                self._sync_runner = asyncio.Runner()
            return self._sync_runner.run(awaitable)
        # 中文注释：如果已经在 async 环境中，就不能再走同步桥，否则会把事件循环套住。
        # 这种场景说明调用方已经具备 await 条件，应直接 await provider.chat/embed。
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise RuntimeError("同步兼容接口不能在已有事件循环中调用，请改用 await provider.chat(...) / await provider.embed(...)")

    def _close_sync_runner(self) -> None:
        """关闭旧同步桥使用的 Runner。"""

        if self._sync_runner is None:
            return
        try:
            self._sync_runner.close()
        finally:
            self._sync_runner = None

    async def _run_llm_call(self, operation: str, call_once: Callable[[], Awaitable[LLMResponse]]) -> LLMResponse:
        """按统一规则执行一次文本模型调用，包含限流、重试和耗时日志。"""

        last = LLMResponse(finish_reason="error", error_kind="unknown", error_should_retry=True)
        started_at = time.perf_counter()
        attempts = 0
        async with self._semaphore:
            for attempt in range(self.max_retries):
                attempts = attempt + 1
                try:
                    # 中文注释：这里真正等待 provider 的异步 SDK 请求完成，不再把同步请求塞到线程里。
                    last = await call_once()
                except Exception as exc:
                    # 少数 provider 的底层方法可能直接抛异常，这里统一转成响应对象。
                    last = self._error_response(exc)
                if last.ok or not self._should_retry(last) or attempt == self.max_retries - 1:
                    self._log_call_result(operation, last, attempts, started_at)
                    return last
                await asyncio.sleep(self._retry_delay(attempts))
        self._log_call_result(operation, last, attempts, started_at)
        return last

    async def _run_embedding_call(self, operation: str, call_once: Callable[[], Awaitable[EmbeddingResponse]]) -> EmbeddingResponse:
        """按统一规则执行一次 embedding 调用，包含限流、重试和耗时日志。"""

        last = EmbeddingResponse(finish_reason="error", error_kind="unknown", error_should_retry=True)
        started_at = time.perf_counter()
        attempts = 0
        async with self._semaphore:
            for attempt in range(self.max_retries):
                attempts = attempt + 1
                try:
                    # 中文注释：embedding 也是网络 I/O，请直接 await 异步客户端，不再阻塞线程。
                    last = await call_once()
                except NotImplementedError:
                    # 不支持 embedding 是明确能力问题，不应吞掉或重试。
                    raise
                except Exception as exc:
                    last = self._embedding_error_response(exc)
                if last.ok or not self._should_retry(last) or attempt == self.max_retries - 1:
                    self._log_call_result(operation, last, attempts, started_at)
                    return last
                await asyncio.sleep(self._retry_delay(attempts))
        self._log_call_result(operation, last, attempts, started_at)
        return last

    def _retry_delay(self, attempt: int) -> float:
        """按固定公式计算第几次重试前的等待时间，返回值单位为秒。"""

        # 第 1 次重试从 500 毫秒开始，每次翻倍；超过 32000 毫秒后保持上限不再增长。
        base_delay_ms = min(500 * (2 ** (attempt - 1)), 32000)
        # 在基础等待时间上额外加入 0% 到 25% 的随机时间，避免多个请求同时再次发起。
        jitter_ms = random.uniform(0, base_delay_ms * 0.25)
        return (base_delay_ms + jitter_ms) / 1000

    def _log_call_result(self, operation: str, response: LLMResponse | EmbeddingResponse, attempts: int, started_at: float) -> None:
        """记录一次 provider 调用摘要，避免把完整模型内容写进日志。"""

        duration_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "模型调用完成",
            extra={
                "provider_model": self.model,
                "operation": operation,
                "duration_ms": duration_ms,
                "attempts": attempts,
                "finish_reason": response.finish_reason,
                "usage": response.usage,
                "error_kind": response.error_kind,
                "error_status_code": response.error_status_code,
            },
        )

    def _settings(
        self,
        temperature: float | None,
        max_tokens: int | None,
        reasoning_effort: str | None,
    ) -> GenerationSettings:
        """合并“单次调用参数”和“provider 默认参数”。

        Args:
            temperature: 本次调用显式传入的 temperature。
            max_tokens: 本次调用显式传入的 max_tokens。
            reasoning_effort: 本次调用显式传入的 reasoning_effort。

        Returns:
            最终生效的 GenerationSettings。

        规则是“调用参数优先，缺省时回退到 provider 预设值”。这样调用方既可以
        复用模型默认配置，也可以在单次请求中做轻量覆盖。
        """
        # 调用参数优先，其次回退到当前模型 preset 的默认生成参数。
        return GenerationSettings(
            temperature=self.generation.temperature if temperature is None else temperature,
            max_tokens=self.generation.max_tokens if max_tokens is None else max_tokens,
            reasoning_effort=self.generation.reasoning_effort if reasoning_effort is None else reasoning_effort,
        )

    def _error_response(self, exc: Exception) -> LLMResponse:
        """把不同来源的异常统一折叠为 `LLMResponse` 错误对象。

        Args:
            exc: provider 调用过程中抛出的异常，可能是自定义 HTTP 异常、SDK 异常，
                或更底层的连接异常。

        Returns:
            `finish_reason="error"` 的统一响应对象，附带状态码、错误分类、
            重试建议、错误码等尽可能多的结构化信息。

        这个函数的目标是把“异常控制流”改写成“数据返回值”，让上层调度器只面对
        一种返回形态，不需要针对不同 SDK 的异常类写大量分支。
        """
        error = self._error_fields(exc)
        return LLMResponse(
            finish_reason="error",
            error_status_code=error["error_status_code"],
            error_kind=error["error_kind"],
            error_retry_after_s=error["error_retry_after_s"],
            error_should_retry=error["error_should_retry"],
            error_type=error["error_type"],
            error_code=error["error_code"],
            content=error["content"],
        )

    def _embedding_error_response(self, exc: Exception) -> EmbeddingResponse:
        """把 embedding 调用异常统一折叠为 `EmbeddingResponse` 错误对象。

        Args:
            exc: embedding 调用过程中抛出的异常，可能是 HTTP 错误、SDK 错误或连接错误。

        Returns:
            `finish_reason="error"` 的统一 embedding 响应，方便上层按同一套字段展示。

        这个函数和 `_error_response` 的思路相同，只是返回对象不同。这样调用方可以
        清楚地区分“文本生成失败”和“向量生成失败”，不会把两类结果混在一起。
        """
        error = self._error_fields(exc)
        return EmbeddingResponse(
            finish_reason="error",
            error_status_code=error["error_status_code"],
            error_kind=error["error_kind"],
            error_retry_after_s=error["error_retry_after_s"],
            error_should_retry=error["error_should_retry"],
            error_type=error["error_type"],
            error_code=error["error_code"],
            content=error["content"],
        )

    def _error_fields(self, exc: Exception) -> JsonObject:
        """把各种异常提取成通用错误字段。"""

        # 官方 SDK 通常暴露 status_code/headers；自定义 HTTP 错误也按同一套字段处理。
        if isinstance(exc, ProviderHttpError):
            retry_after = _parse_retry_after(exc.headers.get("retry-after"))
            kind = _http_error_kind(exc.status_code)
            return {
                "error_status_code": exc.status_code,
                "error_kind": kind,
                "error_retry_after_s": retry_after,
                "error_should_retry": kind in {"rate_limit", "server_error"},
                "error_type": _json_error_field(exc.body, "type"),
                "error_code": _json_error_field(exc.body, "code"),
                "content": exc.body,
            }
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            # 兼容官方 SDK 自带异常类型：它们往往直接暴露 status_code 和 headers。
            headers = getattr(exc, "headers", {}) or {}
            retry_after = _parse_retry_after(headers.get("retry-after"))
            kind = _http_error_kind(status_code)
            body = str(exc)
            return {
                "error_status_code": status_code,
                "error_kind": kind,
                "error_retry_after_s": retry_after,
                "error_should_retry": kind in {"rate_limit", "server_error"},
                "error_type": _json_error_field(body, "type"),
                "error_code": _json_error_field(body, "code"),
                "content": body,
            }
        # 走到这里通常说明没有明确 HTTP 状态码，把它归类为连接层或未知异常。
        return {
            "error_status_code": None,
            "error_kind": "connection",
            "error_retry_after_s": None,
            "error_should_retry": True,
            "error_type": None,
            "error_code": None,
            "content": str(exc),
        }

    def _should_retry(self, response: LLMResponse | EmbeddingResponse) -> bool:
        """判断某个错误响应是否值得重试。

        Args:
            response: 已标准化的 LLMResponse 错误对象。

        Returns:
            是否应继续重试。

        若 provider 已经显式给出 `error_should_retry`，优先尊重该结论；
        否则按归一化错误类别做保守判断。
        """
        if response.error_should_retry is not None:
            return response.error_should_retry
        return response.error_kind in {"rate_limit", "server_error", "connection"}


def merge_body(base: Mapping[str, Any], extra: Mapping[str, Any]) -> JsonObject:
    """递归合并两个请求体字典。

    Args:
        base: 适配器生成的基础请求体。
        extra: 用户或配置额外注入的请求体字段。

    Returns:
        合并后的新字典，不会原地修改传入参数。

    与普通 `dict.update()` 不同，这里对嵌套字典执行递归合并，避免用户追加
    `extra_body` 时把适配器自动写入的整块嵌套结构直接覆盖掉。
    """
    # 递归合并避免用户 extra_body 覆盖掉适配器自动注入的嵌套字段。
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(merged.get(key), dict) and isinstance(value, MappingABC):
            # 双方都是映射对象时继续向下合并，保留两边的嵌套字段。
            merged[key] = merge_body(merged[key], value)
        else:
            # 标量值或类型不兼容时，以 extra 为准直接覆盖。
            merged[key] = value
    return merged


def _parse_retry_after(value: str | None) -> float | None:
    """解析 HTTP `Retry-After` 响应头。

    Args:
        value: `Retry-After` 原始字符串，可能是秒数，也可能是 HTTP 日期。

    Returns:
        建议等待秒数；无法解析时返回 None。

    该函数兼容两种标准格式：
    1. 纯数字秒数，例如 `"30"`。
    2. HTTP 日期，例如 `"Wed, 21 Oct 2015 07:28:00 GMT"`。
    """
    if not value:
        return None
    try:
        # 最常见情况是直接给秒数。
        return float(value)
    except ValueError:
        try:
            # 若是 HTTP 日期，则换算成“距离当前时刻还需等待多少秒”。
            return max((parsedate_to_datetime(value).timestamp() - time.time()), 0)
        except Exception:
            return None


def _http_error_kind(status_code: int) -> str:
    """把 HTTP 状态码归一化为内部错误类别。

    Args:
        status_code: HTTP 状态码。

    Returns:
        归一化错误类别字符串。

    这个映射故意保持简单，主要服务于重试策略与上层展示，而不试图覆盖
    每一家模型服务商的全部私有错误语义。
    """
    if status_code == 429:
        return "rate_limit"
    if status_code in {401, 403}:
        return "auth"
    if 500 <= status_code <= 599:
        return "server_error"
    return "invalid_request"


def _json_error_field(body: str, field: str) -> str | None:
    """从供应商错误响应 JSON 中提取指定字段。

    Args:
        body: 错误响应体文本，预期是 JSON 字符串。
        field: 希望读取的字段名，例如 `type` 或 `code`。

    Returns:
        对应字段值；若响应不是合法 JSON，或没有 `error` 对象，则返回 None。

    许多模型供应商会把错误包装为 `{ "error": { ... } }` 结构，这里做一个
    小工具函数，避免在多个 provider 中重复写 JSON 解析与容错逻辑。
    """
    try:
        error = json.loads(body).get("error", {})
        return error.get(field)
    except Exception:
        return None
