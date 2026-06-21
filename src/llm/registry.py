from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    # 静态元数据承载厂商差异，避免为每个兼容网关新增一个 provider 类。
    """描述一个 provider 的静态能力与路由元数据。

    Attributes:
        name: provider 名称，也是配置表中的键。
        backend: 实际使用的适配器后端类型，例如 `openai_compat`、`anthropic`。
        default_api_base: 默认 API 基础地址；兼容网关通常留空，由用户显式提供。
        env_key: 默认读取 API Key 的环境变量名。
        keywords: 在 `auto` 模式下用于根据模型名猜测 provider 的关键字。
        strip_model_prefix: 是否在真正请求上游前移除 `provider/model` 前缀。
        supports_max_completion_tokens: 是否优先使用 `max_completion_tokens` 参数。

    这里存放的是“静态差异”，而不是运行时状态。这样系统可以用较少的 provider
    类支持多个兼容网关，把差异更多地收敛在元数据层。
    """
    name: str
    backend: str
    default_api_base: str
    env_key: str
    keywords: tuple[str, ...] = ()
    strip_model_prefix: bool = False
    supports_max_completion_tokens: bool = False


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        backend="openai_compat",
        default_api_base="https://api.openai.com/v1",
        env_key="OPENAI_API_KEY",
        keywords=("gpt-", "o1", "o3", "o4"),
        supports_max_completion_tokens=True,
    ),
    "openai_compat": ProviderSpec(
        name="openai_compat",
        backend="openai_compat",
        default_api_base="",
        env_key="OPENAI_API_KEY",
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        backend="anthropic",
        default_api_base="https://api.anthropic.com/v1/messages",
        env_key="ANTHROPIC_API_KEY",
        keywords=("claude-",),
        strip_model_prefix=True,
    ),
    "anthropic_compat": ProviderSpec(
        name="anthropic_compat",
        backend="anthropic",
        default_api_base="",
        env_key="ANTHROPIC_API_KEY",
        strip_model_prefix=True,
    ),
}


def match_provider_backend(backend: str) -> ProviderSpec:
    """根据 backend 名称返回对应的 provider 规格。

    Args:
        backend: 协议后端名称，例如 `openai_compat`、`anthropic`、`openai`。

    Returns:
        对应的 `ProviderSpec`。

    Raises:
        ValueError: 当 backend 不在当前系统支持范围内时抛出。

    这里匹配的是“底层协议实现类型”，不是前端选择的 provider 实例名称。
    """
    try:
        return PROVIDERS[backend]
    except KeyError as exc:
        raise ValueError(f"unknown provider backend: {backend}") from exc


def match_provider(provider: str | None, model: str) -> ProviderSpec:
    """根据显式 provider 或模型名自动匹配 provider 规格。

    Args:
        provider: 用户显式指定的 provider 名称；为 `None` 或 `auto` 时启用自动匹配。
        model: 模型名，可能带 provider 前缀，也可能只是裸模型名。

    Returns:
        匹配到的 `ProviderSpec`。

    Raises:
        ValueError: 当显式指定了未知 provider 时抛出。

    匹配优先级如下：
    1. 若显式指定 provider 且不是 `auto`，直接按名称查找。
    2. 若模型名带 `provider/model` 前缀，优先按前缀命中。
    3. 按各 provider 的关键字列表扫描模型名。
    4. 若仍无法确定，则抛出错误，要求调用方显式指定 provider。
    """
    # 显式 provider 优先，auto 模式才根据模型名前缀和关键字猜测。
    if provider and provider != "auto":
        try:
            return PROVIDERS[provider]
        except KeyError as exc:
            raise ValueError(f"unknown provider: {provider}") from exc

    model_lower = model.lower()
    if "/" in model_lower:
        # 例如 anthropic_compat/claude-test 可以直接命中对应兼容协议。
        prefix = model_lower.split("/", 1)[0]
        if prefix in PROVIDERS:
            return PROVIDERS[prefix]

    for spec in PROVIDERS.values():
        if any(keyword in model_lower for keyword in spec.keywords):
            return spec

    raise ValueError(f"unable to infer provider for model: {model}")
