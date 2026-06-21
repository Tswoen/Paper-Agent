from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping, Sequence


Message = Mapping[str, Any]
JsonObject = dict[str, Any]


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
        self.timeout_s = timeout_s

    @abstractmethod
    def chat(
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
        `LLMResponse`。这里定义的是 provider 必须遵守的最小接口契约。
        """
        raise NotImplementedError

    @abstractmethod
    def chat_stream(
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

    def chat_with_retry(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        """执行带重试的非流式对话请求。

        Args:
            messages: 输入消息列表。
            **kwargs: 透传给 `chat` 的其余调用参数。

        Returns:
            成功响应，或重试耗尽后的最后一次错误响应。

        当前策略最多尝试 3 次。是否重试由 `_should_retry` 决定；等待时间优先使用
        服务端返回的 `retry-after`，否则使用指数退避 `1s/2s/4s` 风格的回退值。
        """
        # 重试逻辑放在基类，具体 provider 只负责一次请求和错误标准化。
        last = LLMResponse(finish_reason="error", error_kind="unknown", error_should_retry=True)
        for attempt in range(3):
            last = self.chat(messages, **kwargs)
            if last.ok or not self._should_retry(last) or attempt == 2:
                return last
            # 若服务端未给出明确等待时间，则使用简单指数退避降低连续打爆上游的风险。
            time.sleep(last.error_retry_after_s or 2**attempt)
        return last

    def chat_stream_with_retry(
        self,
        messages: Sequence[Message],
        callbacks: StreamCallbacks,
        **kwargs: Any,
    ) -> LLMResponse:
        """执行带重试的流式对话请求。

        Args:
            messages: 输入消息列表。
            callbacks: 流式输出回调集合。
            **kwargs: 透传给 `chat_stream` 的其余参数。

        Returns:
            成功响应，或重试耗尽后的最后一次错误响应。

        逻辑与 `chat_with_retry` 类似，只是底层调用改成流式接口。注意如果前几次
        流式请求已经向回调发出过部分增量，上层需要自行决定如何处理这些半成品输出。
        """
        last = LLMResponse(finish_reason="error", error_kind="unknown", error_should_retry=True)
        for attempt in range(3):
            last = self.chat_stream(messages, callbacks, **kwargs)
            if last.ok or not self._should_retry(last) or attempt == 2:
                return last
            time.sleep(last.error_retry_after_s or 2**attempt)
        return last

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
        # 官方 SDK 通常暴露 status_code/headers，这里统一沉淀为 LLMResponse。
        if isinstance(exc, ProviderHttpError):
            retry_after = _parse_retry_after(exc.headers.get("retry-after"))
            kind = _http_error_kind(exc.status_code)
            return LLMResponse(
                finish_reason="error",
                error_status_code=exc.status_code,
                error_kind=kind,
                error_retry_after_s=retry_after,
                error_should_retry=kind in {"rate_limit", "server_error"},
                error_type=_json_error_field(exc.body, "type"),
                error_code=_json_error_field(exc.body, "code"),
                content=exc.body,
            )
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            # 兼容官方 SDK 自带异常类型：它们往往直接暴露 status_code 和 headers。
            headers = getattr(exc, "headers", {}) or {}
            retry_after = _parse_retry_after(headers.get("retry-after"))
            kind = _http_error_kind(status_code)
            body = str(exc)
            return LLMResponse(
                finish_reason="error",
                error_status_code=status_code,
                error_kind=kind,
                error_retry_after_s=retry_after,
                error_should_retry=kind in {"rate_limit", "server_error"},
                error_type=_json_error_field(body, "type"),
                error_code=_json_error_field(body, "code"),
                content=body,
            )
        # 走到这里通常说明没有明确 HTTP 状态码，把它归类为连接层或未知异常。
        return LLMResponse(
            finish_reason="error",
            error_kind="connection",
            error_should_retry=True,
            content=str(exc),
        )

    def _should_retry(self, response: LLMResponse) -> bool:
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
