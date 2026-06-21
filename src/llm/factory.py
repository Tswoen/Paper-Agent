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


def make_provider(
    config: ModelConfig,
    agent_name: str | None = None,
    *,
    client: Any | None = None,
) -> ProviderSnapshot:
    """根据 Agent 模型配置装配并返回一个 provider 快照。

    Args:
        config: 全局模型配置对象，包含 provider 配置和模型 preset。
        agent_name: 需要解析的 Agent 名称；为空时使用 default_agent。
        client: 可选外部注入 client，通常用于测试、mock 或复用自定义 SDK 实例。

    Returns:
        `ProviderSnapshot`，其中包含实例化后的 provider 和配置签名。

    工厂函数只负责“装配”而不负责请求执行。它会依次完成：
    1. 解析 Agent 配置。
    2. 解析最终 provider 实例配置。
    3. 根据该配置声明的 backend 选择协议规格。
    4. 选择具体适配器类并实例化。
    5. 生成稳定签名，供后续热刷新判断。
    """
    # 工厂只做装配：解析 Agent、加载 provider 实例配置、选择协议适配器。
    agent = config.resolve_agent(agent_name)
    provider_name, provider_config = config.resolve_provider_config(agent)
    spec = match_provider_backend(provider_config.backend)
    kwargs: dict[str, Any] = {
        "model": agent.model_name,
        "api_key": provider_config.api_key,
        "api_base": provider_config.api_base,
        "generation": agent.generation,
        "extra_headers": provider_config.extra_headers,
        "extra_body": provider_config.extra_body,
        "client": client,
    }
    if spec.backend == "openai_compat":
        # OpenAI 及大多数 OpenAI-compatible 网关都走统一适配器。
        provider = OpenAICompatProvider(spec=spec, **kwargs)
    elif spec.backend == "anthropic":
        # Anthropic 原生协议与 Anthropic-compatible 网关共用同一适配器。
        provider = AnthropicProvider(spec=spec, **kwargs)
    else:
        raise ValueError(f"unsupported provider backend: {spec.backend}")
    return ProviderSnapshot(provider, agent.model_name, agent.context_window_tokens, _signature(provider_name, _agent_signature(agent), asdict(provider_config)))


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
