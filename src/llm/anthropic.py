from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from .base import EmbeddingResponse, JsonObject, LLMProvider, LLMResponse, Message, StreamCallbacks, ToolCallRequest
from .registry import ProviderSpec


class AnthropicProvider(LLMProvider):
    # Anthropic 原生协议和 Anthropic-compatible 网关都走官方 SDK。
    def __init__(self, *, spec: ProviderSpec, **kwargs: Any):
        """初始化 Anthropic 供应商客户端。

        Args:
            spec: 当前供应商的静态能力描述，用于统一承载兼容网关的协议差异。
            **kwargs: 传给 LLMProvider 基类的通用配置，例如 api_key、api_base、
                timeout、extra_headers、client 等。

        这里优先复用外部注入的 client，便于测试替身注入或接入自定义网关；
        如果没有传入，就使用官方 `anthropic` SDK 创建默认客户端。
        """
        super().__init__(**kwargs)
        self.spec = spec
        if self.client is None:
            # base_url 可指向官方 Anthropic，也可指向 Anthropic 协议兼容代理。
            from anthropic import AsyncAnthropic

            self.client = AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.api_base,
                default_headers=self.extra_headers or None,
                timeout=self.timeout_s,
                # 中文注释：项目基类已经统一处理重试和日志，SDK 这里关闭额外重试，避免请求次数变得不可控。
                max_retries=0,
            )

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[JsonObject] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """发起一次非流式 Anthropic Messages 请求，并统一处理重试、限流和耗时日志。"""

        return await self._run_llm_call(
            "chat",
            lambda: self._chat_once(messages, tools=tools, temperature=temperature, max_tokens=max_tokens, reasoning_effort=reasoning_effort),
        )

    async def _chat_once(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[JsonObject] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """只执行一次真实 Anthropic 请求；是否重试由基类统一决定。"""

        try:
            response = await _maybe_await(
                self.client.messages.create(
                    **self._build_kwargs(messages, tools, False, temperature, max_tokens, reasoning_effort)
                )
            )
            return self._parse_response(response)
        except Exception as exc:
            return self._error_response(exc)

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
        """发起一次流式 Anthropic Messages 请求，并统一处理重试、限流和耗时日志。"""

        return await self._run_llm_call(
            "chat_stream",
            lambda: self._chat_stream_once(messages, callbacks, tools=tools, temperature=temperature, max_tokens=max_tokens, reasoning_effort=reasoning_effort),
        )

    async def _chat_stream_once(
        self,
        messages: Sequence[Message],
        callbacks: StreamCallbacks,
        *,
        tools: Sequence[JsonObject] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """只执行一次真实 Anthropic 流式请求；是否重试由基类统一决定。"""

        content: list[str] = []
        usage: JsonObject | None = None
        finish_reason: str | None = None
        try:
            stream = await _maybe_await(
                self.client.messages.create(
                    **self._build_kwargs(messages, tools, True, temperature, max_tokens, reasoning_effort)
                )
            )
            async for event in _aiter(stream):
                item = _to_dict(event)
                event_type = item.get("type")
                delta = item.get("delta") or {}
                if isinstance(item.get("usage"), dict):
                    usage = item["usage"]
                if item.get("stop_reason"):
                    finish_reason = item.get("stop_reason")
                if event_type == "content_block_delta" and delta.get("type") == "text_delta":
                    text = delta.get("text") or ""
                    content.append(text)
                    if callbacks.on_content_delta:
                        callbacks.on_content_delta(text)
                if event_type == "content_block_delta" and delta.get("type") == "thinking_delta" and callbacks.on_thinking_delta:
                    callbacks.on_thinking_delta(delta.get("thinking") or "")
                if event_type == "content_block_delta" and delta.get("type") == "input_json_delta" and callbacks.on_tool_call_delta:
                    callbacks.on_tool_call_delta({"arguments_delta": delta.get("partial_json") or ""})
            return LLMResponse(content="".join(content), finish_reason=finish_reason or "stop", usage=usage)
        except Exception as exc:
            return self._error_response(exc)

    async def embed(
        self,
        inputs: Sequence[str],
        *,
        dimensions: int | None = None,
    ) -> EmbeddingResponse:
        """明确说明 Anthropic Messages provider 不支持 embedding。

        Args:
            inputs: 调用方希望生成向量的文本列表。
            dimensions: 调用方希望指定的向量维度。

        Raises:
            NotImplementedError: 当前 Anthropic Messages 接口不能生成文本向量。

        AnthropicProvider 主要负责对话生成。它没有 `/embeddings` 这类向量接口，
        所以这里直接抛出清楚错误，让设置页和阅读节点都能告诉用户该换 embedding provider。
        """
        raise NotImplementedError(
            "AnthropicProvider 不支持 embedding：Anthropic Messages 接口只用于对话生成，不能生成文本向量。"
            "请在 embedding profile 中选择 OpenAI 兼容 provider。"
        )

    def _build_kwargs(
        self,
        messages: Sequence[Message],
        tools: Sequence[JsonObject] | None,
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
        reasoning_effort: str | None,
    ) -> JsonObject:
        """构造发给 Anthropic SDK `messages.create` 的参数字典。

        Args:
            messages: 项目内部消息列表，需要转换为 Anthropic Messages 格式。
            tools: 可选工具定义，需要转换为 Anthropic 的 `input_schema` 格式。
            stream: 是否启用流式输出。
            temperature: 调用级采样温度。
            max_tokens: 调用级生成上限。
            reasoning_effort: 调用级推理强度。

        Returns:
            可直接传给 `self.client.messages.create(**kwargs)` 的参数字典。

        这里主要处理三类兼容差异：system 消息单独抽取、tool_use/tool_result
        结构转换，以及 thinking 模式与普通温度采样之间的参数约束。
        """
        # Anthropic 的 system、tool_use、tool_result 结构和 OpenAI 不同，需要单独转换。
        settings = self._settings(temperature, max_tokens, reasoning_effort)
        system, converted = _convert_messages(messages)
        kwargs: JsonObject = {
            "model": self._request_model_name(),
            "messages": converted,
            "max_tokens": settings.max_tokens or 4096,
            "stream": stream,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [_convert_tool(tool) for tool in tools]
        effort = _normalize_effort(settings.reasoning_effort)
        if settings.temperature is not None and effort is None and _anthropic_accepts_temperature(self._request_model_name()):
            # 中文注释：新版 Claude 模型会拒绝 temperature 等采样参数；只有确认可接收时才发送。
            kwargs["temperature"] = settings.temperature
        if effort is not None:
            # 中文注释：新版 Claude 使用 adaptive thinking，不再发送旧的 budget_tokens，避免 400 参数错误。
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": effort}
        if self.extra_body:
            # Anthropic 兼容网关的扩展字段也走 SDK extra_body。
            kwargs["extra_body"] = self.extra_body
        return kwargs

    def _request_model_name(self) -> str:
        """返回实际发送给 Anthropic 上游接口的模型名。

        Returns:
            发送给 SDK 的最终模型名。

        这里和 OpenAI 兼容适配器保持一致：是否裁剪 `provider/model` 这种内部
        路由前缀，由 `ProviderSpec.strip_model_prefix` 控制，而不是散落在各处
        直接写死字符串分割逻辑。
        """
        if self.spec.strip_model_prefix and "/" in self.model:
            return self.model.split("/", 1)[1]
        return self.model

    def _parse_response(self, data: Any) -> LLMResponse:
        """把 Anthropic SDK 的响应对象解析成项目统一的 LLMResponse。

        Args:
            data: SDK 原始响应对象，可能是 Pydantic 模型或普通 dict。

        Returns:
            统一响应对象，包括文本内容、thinking 内容、工具调用、结束原因和 usage。

        Anthropic 的响应由多个 content block 组成，不同 block type 分别代表
        文本、thinking、tool_use 等语义，这里会逐块拆解并归并成内部结构。
        """
        response = _to_dict(data)
        text: list[str] = []
        thinking: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        for block in response.get("content") or []:
            if block.get("type") == "text":
                text.append(block.get("text") or "")
            elif block.get("type") == "thinking":
                # thinking block 是 Anthropic 原生“推理过程”片段。
                thinking.append(block.get("thinking") or "")
            elif block.get("type") == "tool_use":
                # tool_use block 直接携带结构化 input，可原样作为工具调用参数使用。
                tool_calls.append(ToolCallRequest(block.get("id"), block.get("name", ""), block.get("input"), block))
        return LLMResponse(
            content="".join(text),
            tool_calls=tool_calls,
            finish_reason=response.get("stop_reason"),
            usage=response.get("usage"),
            reasoning_content="".join(thinking) or None,
        )


def _convert_messages(messages: Sequence[Message]) -> tuple[str | None, list[JsonObject]]:
    """把内部统一消息格式转换为 Anthropic Messages 协议格式。

    Args:
        messages: 项目内部的消息历史，整体更接近 OpenAI 风格表示。

    Returns:
        一个二元组：
        1. `system` 字符串或 None。Anthropic 要求 system 提示词单独传递。
        2. 转换后的 messages 列表，其中 tool_result/tool_use 会被改写成 Anthropic
           所要求的 content block 结构。

    这个函数是协议适配的核心入口，负责处理 system 拆分、tool 角色映射、以及
    assistant 中携带 tool_calls 的情况。
    """
    # 将项目内部的 OpenAI 风格历史转换成 Anthropic Messages 格式。
    system: list[str] = []
    converted: list[JsonObject] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            # Anthropic 的 system 不是普通消息，而是单独顶层参数，这里先收集起来。
            system.append(str(message.get("content") or ""))
            continue
        if role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": message.get("tool_call_id"), "content": str(message.get("content") or "")}],
                }
            )
            continue
        content = message.get("content") or ""
        blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
        if role == "assistant" and message.get("tool_calls"):
            # assistant 发起工具调用时，需要把 tool_calls 展开成 content 中的 tool_use block。
            blocks = list(blocks) + [_convert_tool_call(call) for call in message.get("tool_calls") or []]
        if role not in {"user", "assistant"}:
            raise ValueError(f"invalid message role: {role}")
        converted.append({"role": role, "content": blocks})
    return ("\n".join(system) or None), converted


def _convert_tool(tool: JsonObject) -> JsonObject:
    """把 OpenAI 风格的工具定义转换为 Anthropic tools 格式。

    Args:
        tool: 工具定义对象，可能已经是 function 结构，也可能直接是扁平对象。

    Returns:
        Anthropic 所需的工具描述，其中参数 schema 字段名为 `input_schema`。

    OpenAI/兼容协议通常使用 `function.parameters` 表示 JSON Schema，而 Anthropic
    期望字段名是 `input_schema`，这里做一层字段重命名与默认值补齐。
    """
    # OpenAI function schema 在 Anthropic 中对应 input_schema。
    function = tool.get("function") or tool
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
    }


def _convert_tool_call(call: JsonObject) -> JsonObject:
    """把内部工具调用记录转换为 Anthropic `tool_use` content block。

    Args:
        call: 内部统一格式的工具调用对象，通常兼容 OpenAI 的 `tool_calls` 结构。

    Returns:
        Anthropic `tool_use` block，可直接放进 assistant 消息的 content 列表。

    `function.arguments` 在很多实现里是 JSON 字符串；这里会尽量解析为对象。
    如果解析失败，则退化为 `{"value": 原始字符串}`，避免整条工具调用丢失。
    """
    function = call.get("function") or {}
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            # 遇到非标准 JSON 参数时保底包一层，至少保留原始信息供上层处理。
            arguments = {"value": arguments}
    return {"type": "tool_use", "id": call.get("id"), "name": function.get("name"), "input": arguments}


def _normalize_effort(value: str | None) -> str | None:
    """把配置里的 reasoning_effort 转成 Claude 当前支持的 effort 值。"""

    if not value:
        return None
    normalized = str(value).strip().lower().replace("-", "")
    if normalized in {"", "none", "off", "disabled"}:
        return None
    if normalized in {"low", "medium", "high", "xhigh", "max"}:
        return normalized
    # 中文注释：遇到旧配置或未知值时使用 high，既保留“开启推理”的意图，又避免传非法参数。
    return "high"


def _anthropic_accepts_temperature(model: str) -> bool:
    """判断当前 Claude 模型是否适合发送 temperature 参数。"""

    name = model.lower().split("/", 1)[-1]
    # 中文注释：Opus 4.7/4.8、Sonnet 5、Fable 5 等新模型会拒绝采样参数，
    # 设置页连通性测试常传 temperature=0，所以这里主动跳过，避免可用模型被误判失败。
    blocked_prefixes = (
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-mythos-5",
    )
    return not name.startswith(blocked_prefixes)


async def _maybe_await(value: Any) -> Any:
    """如果 SDK 或测试假对象返回 awaitable，就等待它；否则直接返回原值。"""

    if inspect.isawaitable(value):
        return await value
    return value


async def _aiter(stream: Any) -> AsyncIterator[Any]:
    """兼容真实异步流和测试里常见的同步假流。"""

    if hasattr(stream, "__aiter__"):
        async for item in stream:
            yield item
        return
    for item in stream:
        yield item


def _to_dict(value: Any) -> JsonObject:
    """把 SDK 返回对象统一转换成普通字典。

    Args:
        value: 可能是 SDK 的 Pydantic 对象、普通 dict，或其他可被 `dict()` 消费的对象。

    Returns:
        标准 JsonObject，便于后续用统一方式访问字段。

    Anthropic SDK 新版本同样可能返回带 `model_dump()` 的对象，这里按常见形态
    依次兼容，减少解析层对具体 SDK 类型的依赖。
    """
    if hasattr(value, "model_dump"):
        # 优先走 model_dump，保留嵌套结构并避免手动展开 SDK 对象。
        return value.model_dump()
    if isinstance(value, dict):
        return value
    # 兜底兼容键值对可迭代对象。
    return dict(value)
