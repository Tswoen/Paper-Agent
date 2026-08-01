from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from .base import EmbeddingResponse, JsonObject, LLMProvider, LLMResponse, Message, ProviderHttpError, StreamCallbacks, ToolCallRequest, merge_body
from .registry import ProviderSpec
from src.utils import get_logger

logger = get_logger(__name__)

class OpenAICompatProvider(LLMProvider):
    # 所有 OpenAI Chat Completions 兼容协议都走这一条 SDK 调用链。
    def __init__(self, *, spec: ProviderSpec, **kwargs: Any):
        """初始化 OpenAI 兼容供应商。

        Args:
            spec: 当前供应商的能力描述，例如是否需要裁剪模型名前缀、是否支持
                max_completion_tokens 等兼容性开关。
            **kwargs: 传给 LLMProvider 基类的通用配置，如 api_key、api_base、
                timeout、client、extra_headers 等。

        这里优先复用外部注入的 client，便于单元测试或调用方接入自定义 SDK；
        如果没有注入，则按 OpenAI Python SDK 的兼容接口创建默认客户端。
        """
        super().__init__(**kwargs)
        self.spec = spec
        if self.client is None:
            # 延迟导入便于测试注入 fake client，也避免未安装依赖时导入整个包失败。
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                default_headers=self.extra_headers or None,
                timeout=self.timeout_s,
                # 中文注释：项目基类已经统一做重试、计时和 Retry-After 处理，SDK 这里不再额外重试，避免实际请求次数翻倍。
                max_retries=5,
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
        """发起一次非流式 Chat Completions 请求，并统一处理重试、限流和耗时日志。"""

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
        """只执行一次真实请求；是否重试由基类统一决定。"""

        try:
            response = await _maybe_await(
                self.client.chat.completions.create(
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
        """发起一次流式 Chat Completions 请求，并统一处理重试、限流和耗时日志。"""

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
        """只执行一次真实流式请求；流结束前不会释放基类里的并发名额。"""

        content: list[str] = []
        usage: JsonObject | None = None
        finish_reason: str | None = None
        try:
            stream = await _maybe_await(
                self.client.chat.completions.create(
                    **self._build_kwargs(messages, tools, True, temperature, max_tokens, reasoning_effort)
                )
            )
            async for event in _aiter(stream):
                # OpenAI 兼容流式事件通常把增量内容放在 choices[0].delta 中。
                event_dict = _to_dict(event)
                if isinstance(event_dict.get("usage"), dict):
                    usage = event_dict["usage"]
                choice = event.choices[0] if getattr(event, "choices", None) else None
                if choice is not None and getattr(choice, "finish_reason", None):
                    finish_reason = getattr(choice, "finish_reason", None)
                delta = getattr(choice, "delta", None) if choice is not None else None
                text = getattr(delta, "content", None) or ""
                if text:
                    content.append(text)
                    if callbacks.on_content_delta:
                        callbacks.on_content_delta(text)  # 如果上层注册了文本回调，立刻把增量推送出去（前端实时打字效果靠这个）。
                reasoning = getattr(delta, "reasoning_content", None) or ""
                if reasoning and callbacks.on_thinking_delta:
                    callbacks.on_thinking_delta(reasoning)
                for item in getattr(delta, "tool_calls", None) or []:
                    if callbacks.on_tool_call_delta:
                        callbacks.on_tool_call_delta(_to_dict(item))
            return LLMResponse(content="".join(content), finish_reason=finish_reason or "stop", usage=usage)
        except Exception as exc:
            return self._error_response(exc)

    async def embed(
        self,
        inputs: Sequence[str],
        *,
        dimensions: int | None = None,
    ) -> EmbeddingResponse:
        """发起一次 OpenAI 兼容 embedding 请求，并统一处理重试、限流和耗时日志。"""

        return await self._run_embedding_call("embed", lambda: self._embed_once(inputs, dimensions=dimensions))

    async def _embed_once(
        self,
        inputs: Sequence[str],
        *,
        dimensions: int | None = None,
    ) -> EmbeddingResponse:
        """只执行一次真实 embedding 请求；是否重试由基类统一决定。"""

        texts = list(inputs)
        if not texts:
            return EmbeddingResponse(embeddings=[], model=self._request_model_name(), finish_reason="stop")
        endpoint = self.api_base.rstrip("/") + "/embeddings"
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        payload: JsonObject = {"model": self._request_model_name(), "input": texts}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if self.extra_body:
            # embedding 也允许复用 provider 上配置的额外请求体，便于兼容不同代理服务。
            payload = merge_body(payload, self.extra_body)
        try:
            body = await self._post_embedding(endpoint, headers, payload)
            return self._parse_embedding_response(body, len(texts))
        except Exception as exc:
            return self._embedding_error_response(exc)

    async def list_models(self) -> list[JsonObject]:
        """读取 OpenAI 兼容 provider 的模型目录，并转换成前端统一结构。"""

        response = await _maybe_await(self.client.models.list())
        raw = _to_dict(response) if hasattr(response, "model_dump") or isinstance(response, dict) else response
        data = raw.get("data", raw) if isinstance(raw, dict) else raw
        models: list[JsonObject] = []
        for item in data or []:
            model = _to_dict(item) if hasattr(item, "model_dump") or isinstance(item, dict) else item
            model_id = model.get("id") if isinstance(model, dict) else str(model)
            models.append(
                {
                    "id": model_id,
                    "label": model_id,
                    "owned_by": model.get("owned_by") if isinstance(model, dict) else None,
                    "context_window": model.get("context_window") if isinstance(model, dict) else None,
                }
            )
        return models

    def _build_kwargs(
        self,
        messages: Sequence[Message],
        tools: Sequence[JsonObject] | None,
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
        reasoning_effort: str | None,
    ) -> JsonObject:
        """构造传给 OpenAI Python SDK 的 Chat Completions 参数。

        Args:
            messages: 内部消息对象，需要先清洗成上游可接受的字段集合。
            tools: 可选工具定义。
            stream: 是否开启流式输出。
            temperature: 调用级采样温度，None 时使用基类默认设置。
            max_tokens: 调用级 token 上限，None 时使用基类默认设置。
            reasoning_effort: 调用级推理强度，None 时使用基类默认设置。

        Returns:
            可直接展开传给 client.chat.completions.create(**kwargs) 的参数字典。

        注意：
            这里不直接发送 HTTP 请求，只做协议层字段适配。不同兼容厂商常见差异
            包括模型名前缀、max_tokens 字段名、推理模型禁用 temperature 等。
        """
        # 这里只做协议参数适配，不直接拼 HTTP 请求。
        settings = self._settings(temperature, max_tokens, reasoning_effort)
        kwargs: JsonObject = {
            "model": self._request_model_name(),
            "messages": _sanitize_messages(messages),
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = list(tools)
        if settings.temperature is not None and not _is_reasoning_model(self.model):
            # o 系列和 GPT-5 等推理模型通常不接受 temperature，避免请求被上游拒绝。
            kwargs["temperature"] = settings.temperature
        if settings.max_tokens is not None:
            # 推理模型和新 OpenAI 模型使用 max_completion_tokens。
            token_key = "max_completion_tokens" if self.spec.supports_max_completion_tokens or _is_reasoning_model(self.model) else "max_tokens"
            kwargs[token_key] = settings.max_tokens
        if settings.reasoning_effort:
            kwargs["reasoning_effort"] = settings.reasoning_effort
        if stream and self.include_stream_usage:
            # 部分 OpenAI 兼容服务支持在最后一个流式片段返回 usage；不支持时由上游报错并走统一错误处理。
            kwargs["stream_options"] = {"include_usage": True}
        if self.extra_body:
            # 厂商兼容扩展交给 SDK 的 extra_body 注入，避免自己拼 HTTP 请求。
            kwargs["extra_body"] = self.extra_body
        return kwargs

    def _request_model_name(self) -> str:
        """返回实际发送给上游 API 的模型名。

        某些注册表中的模型名会带供应商前缀，例如 provider/model-name，方便
        应用内部路由；但上游 OpenAI 兼容接口只认识 model-name。是否裁剪
        前缀由 ProviderSpec.strip_model_prefix 控制。
        """
        if self.spec.strip_model_prefix and "/" in self.model:
            return self.model.split("/", 1)[1]
        return self.model

    async def _post_embedding(self, endpoint: str, headers: JsonObject, payload: JsonObject) -> JsonObject:
        """发送 embedding HTTP 请求并返回 JSON 响应体。

        Args:
            endpoint: 已拼好的 `/embeddings` 地址。
            headers: 请求头，包含鉴权信息和额外头。
            payload: 请求体，包含模型名、输入文本和可选维度。

        Returns:
            供应商返回的 JSON 字典。

        如果外部注入的 client 带有异步 post 方法，就优先复用它，方便设置页或测试传入
        假客户端；正式运行时没有这种假客户端，就临时创建 httpx.AsyncClient 发请求。
        """
        # 中文注释：AsyncOpenAI 自己也可能有内部 post 方法，但那不是普通 HTTP client。
        # 只有测试注入的轻量 HTTP 假客户端才走这里，正式 SDK client 走下面的 httpx.AsyncClient。
        http_client = self.client if hasattr(self.client, "post") and not hasattr(self.client, "chat") else None
        if http_client is not None:
            response = http_client.post(endpoint, headers=headers, json=payload)
            if inspect.isawaitable(response):
                response = await response
            response.raise_for_status()
            return response.json()
        async with httpx.AsyncClient(timeout=float(max(1, self.timeout_s))) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
            if response.status_code >= 400:
                raise ProviderHttpError(response.status_code, response.text, dict(response.headers))
            return response.json()

    def _parse_embedding_response(self, body: JsonObject, expected_count: int) -> EmbeddingResponse:
        """把 OpenAI 兼容 embedding 响应解析成统一结构。

        Args:
            body: 供应商返回的 JSON 响应体。
            expected_count: 本次请求发送的文本数量。

        Returns:
            解析后的 EmbeddingResponse。

        OpenAI 兼容接口通常把向量放在 data 列表中，并用 index 表示原始输入顺序。
        这里先按 index 排序，再逐条校验向量格式，避免错误数据写进本地向量库。
        """
        raw_items = body.get("data") if isinstance(body, dict) else None
        if not isinstance(raw_items, list):
            raise ValueError("embedding 服务没有返回 data 列表")
        ordered = sorted(raw_items, key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0)
        embeddings: list[list[float]] = []
        for item in ordered:
            vector = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vector, list) or not all(isinstance(value, int | float) for value in vector):
                raise ValueError("embedding 服务返回了无效向量")
            embeddings.append([float(value) for value in vector])
        if len(embeddings) != expected_count:
            raise ValueError("embedding 服务返回的向量数量与请求文本数量不一致")
        return EmbeddingResponse(
            embeddings=embeddings,
            model=body.get("model") or self._request_model_name(),
            usage=body.get("usage") if isinstance(body.get("usage"), dict) else None,
            finish_reason="stop",
        )

    def _parse_response(self, data: Any) -> LLMResponse:
        """把 SDK 原始响应转换成项目内部统一的 LLMResponse。

        Args:
            data: OpenAI SDK 返回的响应对象，可能是 Pydantic 模型、dict，
                或其他可转为 dict 的对象。

        Returns:
            统一响应对象，包括文本内容、工具调用、结束原因、用量统计和推理内容。

        工具调用的 function.arguments 在 OpenAI 协议中常以 JSON 字符串返回；
        这里会尽量反序列化成结构化对象，解析失败时保留原始字符串，交给上层兜底。
        """
        response = _to_dict(data)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        tool_calls = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments")
            try:
                # 兼容工具参数已经是 dict/list 的供应商，也兼容标准 JSON 字符串。
                args = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                # 非法 JSON 不在这一层丢弃，保留原值有助于调用方记录或修复。
                pass
            tool_calls.append(ToolCallRequest(call.get("id"), fn.get("name", ""), args, call))
        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            usage=response.get("usage"),
            reasoning_content=message.get("reasoning_content"),
        )


def _sanitize_messages(messages: Sequence[Message]) -> list[JsonObject]:
    """清洗并规范化发送给 Chat Completions 的消息列表。

    Args:
        messages: 项目内部的消息序列，可能包含上游不认识的内部元数据。

    Returns:
        只包含 OpenAI Chat Completions 安全字段的消息列表，并做必要的角色顺序修正。

    Raises:
        ValueError: 当消息 role 不是 system/user/assistant/tool 之一时抛出。

    这个函数的目标是把“内部可用”的消息转换为“协议安全”的消息，避免把
    trace id、调试字段或其他业务元数据意外发送给模型供应商。
    """
    # 只保留 Chat Completions 安全字段，避免把内部元数据传给上游。
    cleaned: list[JsonObject] = []
    for message in messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"invalid message role: {role}")
        item = {k: v for k, v in message.items() if k in {"role", "content", "tool_calls", "tool_call_id", "name"}}
        if role == "assistant" and item.get("tool_calls") and item.get("content") == "":
            # OpenAI 协议里，assistant 发起工具调用时 content 可为 null，而不是空字符串。
            item["content"] = None
        cleaned.append(item)
    return _enforce_role_alternation(cleaned)


def _enforce_role_alternation(messages: list[JsonObject]) -> list[JsonObject]:
    """尽量修复连续 user/assistant 消息导致的兼容问题。

    Args:
        messages: 已经过字段清洗的消息列表。

    Returns:
        经过最小合并处理后的消息列表。

    一些 OpenAI 兼容网关要求 user 和 assistant 消息严格交替。项目内部在拼接
    上下文时可能产生连续同角色消息，因此这里把连续 user 或 assistant 的文本
    内容合并到前一条消息，降低被上游拒绝的概率。
    """
    # 一些兼容网关对连续同角色消息很敏感，这里做最小修复。
    result: list[JsonObject] = []
    for message in messages:
        if result and message["role"] in {"user", "assistant"} and result[-1]["role"] == message["role"]:
            # 只合并文本内容，不跨 role 合并，也不改动 system/tool 的语义边界。
            result[-1]["content"] = f"{result[-1].get('content') or ''}\n{message.get('content') or ''}".strip()
            continue
        result.append(message)
    if result and result[-1]["role"] == "assistant" and not result[-1].get("tool_calls"):
        # 末尾 assistant 普通消息通常是历史回复，不应作为“待回答输入”的最后一条发给模型。
        result.pop()
    return result


def _is_reasoning_model(model: str) -> bool:
    """判断模型名是否属于需要特殊参数处理的推理模型。

    Args:
        model: 内部配置的模型名，可能带 provider/ 前缀。

    Returns:
        True 表示按推理模型处理，例如不发送 temperature，并优先使用
        max_completion_tokens；False 表示按普通 Chat Completions 模型处理。
    """
    # 只检查斜杠后的真实模型名，避免 provider 前缀影响判断。
    lower = model.lower().split("/", 1)[-1]
    return lower.startswith(("o1", "o3", "o4", "gpt-5"))


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
    """把 SDK 返回对象转换为普通字典。

    Args:
        value: 可能是 OpenAI SDK 的 Pydantic 对象、普通 dict，或其他可迭代键值对象。

    Returns:
        普通 JsonObject，便于后续用统一的 dict 访问方式解析响应。

    OpenAI Python SDK 新版本常返回带 model_dump() 的 Pydantic 模型；兼容厂商
    或测试 fake client 也可能直接返回 dict，因此这里按最常见形态依次兼容。
    """
    if hasattr(value, "model_dump"):
        # Pydantic v2 对象优先使用 model_dump，保留嵌套结构。
        return value.model_dump()
    if isinstance(value, dict):
        return value
    # 最后兜底给 dict()，兼容形如键值对迭代器的轻量对象。
    return dict(value)
