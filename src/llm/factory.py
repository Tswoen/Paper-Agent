from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .anthropic import AnthropicProvider
from .base import LLMProvider
from .config import AgentConfig, ModelConfig
from .openai_compat import OpenAICompatProvider
from .registry import match_provider_backend


@dataclass(slots=True)
class ProviderSnapshot:
    # 快照把 provider 实例和关键配置签名绑定，后续可用于热刷新判断。
    """描述一次 provider 装配后的运行时快照。

    Attributes:
        provider: 已实例化的 provider 对象，可直接发起请求。
        model: 当前快照绑定的模型名。
        context_window_tokens: 模型上下文窗口大小，用于上层做截断或预算判断。
        signature: 由关键配置计算出的稳定签名，可用于热刷新、缓存失效判断。

    这个结构把“可执行实例”和“影响执行行为的关键信息”打包在一起，方便上层在
    配置热更新时判断是否需要重建 provider。
    """
    provider: LLMProvider
    model: str
    context_window_tokens: int | None
    signature: str

    async def aclose(self) -> None:
        """关闭快照里 provider 自己创建的异步客户端。"""

        # 中文注释：短生命周期调用（例如设置页连通性测试）结束后可以直接关快照，
        # 不需要知道底层 provider 用的是 OpenAI、Anthropic 还是兼容网关。
        await self.provider.aclose()


def make_provider(
    config: ModelConfig,
    agent_name: str | None = None,
    *,
    embedding_profile_name: str | None = None,
    client: Any | None = None,
    timeout_s: float = 60,
) -> ProviderSnapshot:
    """根据模型配置装配并返回一个 provider 快照。

    Args:
        config: 全局模型配置对象，包含 provider、agent 和 embedding profile 配置。
        agent_name: 需要解析的 Agent 名称；为空时使用 default_agent。
        embedding_profile_name: 需要解析的 embedding profile 名称；为空时不走 embedding 装配。
        client: 可选外部注入 client，通常用于测试、mock 或复用自定义 SDK 实例。
        timeout_s: provider 请求超时时间，单位秒；embedding 和 chat 装配都会透传。

    Returns:
        `ProviderSnapshot`，其中包含实例化后的 provider 和配置签名。

    工厂函数只负责“装配成一个可用 provider 实例”，不关心后续调用 chat 还是
    embedding。默认仍按 agent 装配；当传入 embedding_profile_name 时，则按
    embedding profile 装配同样的 provider 实例，但不新增单独的 make_embedding_provider。
    """
    if agent_name is not None and embedding_profile_name is not None:
        raise ValueError("agent_name 和 embedding_profile_name 不能同时指定，请只选择一种模型配置")

    if embedding_profile_name is not None:
        # embedding profile 只决定“用哪个 provider 和哪个向量模型”，具体请求仍由 provider.embed 负责。
        profile_name = embedding_profile_name or config.default_embedding_profile
        profile, provider_config = config.resolve_embedding_provider_config(profile_name)
        provider_name = profile.provider
        spec = match_provider_backend(provider_config.backend)
        kwargs: dict[str, Any] = {
            "model": profile.model_name,
            "api_key": provider_config.api_key,
            "api_base": provider_config.api_base,
            "generation": None,
            "extra_headers": provider_config.extra_headers,
            "extra_body": provider_config.extra_body,
            "client": client,
            # provider 配置里显式写了超时时间时优先生效；调用方传入的 timeout_s 作为兜底。
            "timeout_s": provider_config.timeout_s or timeout_s,
            "max_retries": provider_config.max_retries,
            "max_concurrency": provider_config.max_concurrency,
            "include_stream_usage": provider_config.include_stream_usage,
        }
        provider = _instantiate_provider(spec, kwargs)
        signature_payload = {"target_type": "embedding_profile", "target_name": profile_name, "profile": asdict(profile)}
        return ProviderSnapshot(provider, profile.model_name, None, _signature(provider_name, signature_payload, asdict(provider_config)))

    # 默认路径保持原来的 Agent 装配方式，避免影响搜索节点、阅读摘要和设置页模型连通性测试。
    agent = config.resolve_agent(agent_name)
    provider_name, provider_config = config.resolve_provider_config(agent)
    spec = match_provider_backend(provider_config.backend)
    kwargs = {
        "model": agent.model_name,
        "api_key": provider_config.api_key,
        "api_base": provider_config.api_base,
        "generation": agent.generation,
        "extra_headers": provider_config.extra_headers,
        "extra_body": provider_config.extra_body,
        "client": client,
        # provider 配置里显式写了超时时间时优先生效；调用方传入的 timeout_s 作为兜底。
        "timeout_s": provider_config.timeout_s or timeout_s,
        "max_retries": provider_config.max_retries,
        "max_concurrency": provider_config.max_concurrency,
        "include_stream_usage": provider_config.include_stream_usage,
    }
    provider = _instantiate_provider(spec, kwargs)
    return ProviderSnapshot(provider, agent.model_name, agent.context_window_tokens, _signature(provider_name, _agent_signature(agent), asdict(provider_config)))


def _instantiate_provider(spec: Any, kwargs: dict[str, Any]) -> LLMProvider:
    """按 provider 后端类型创建具体适配器实例。"""

    if spec.backend == "openai_compat":
        # OpenAI 及大多数 OpenAI-compatible 网关都走统一适配器。
        return OpenAICompatProvider(spec=spec, **kwargs)
    if spec.backend == "anthropic":
        # Anthropic 原生协议与 Anthropic-compatible 网关共用同一适配器。
        return AnthropicProvider(spec=spec, **kwargs)
    raise ValueError(f"unsupported provider backend: {spec.backend}")


def _agent_signature(agent: AgentConfig) -> dict[str, Any]:
    return asdict(agent)


def _signature(provider_name: str, preset: dict[str, Any], provider_config: dict[str, Any]) -> str:
    """为当前 provider 装配结果生成稳定签名。

    Args:
        provider_name: 最终解析出的 provider 名称。
        preset: 序列化后的模型 preset 配置。
        provider_config: 序列化后的 provider 连接与鉴权配置。

    Returns:
        一个 SHA-256 十六进制摘要字符串。

    签名只覆盖会影响请求链路的关键字段。这样当模型、鉴权、base_url、生成参数
    或额外请求体发生变化时，外部系统可以快速识别“这个 provider 需要重建了”。
    """
    # 快照签名只包含影响请求链路的字段，用于之后做热刷新判断。
    raw = json.dumps(
        {"provider": provider_name, "preset": preset, "provider_config": provider_config},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
