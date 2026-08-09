from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import GenerationSettings
from .registry import match_provider_backend


JsonObject = dict[str, Any]


@dataclass(slots=True)
class ProviderConfig:
    """描述某个 provider 的连接、鉴权和调用控制配置。"""

    backend: str = "openai_compat"
    api_key: str | None = None
    api_key_env: str | None = None
    api_base: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    # 下面这些字段只控制“怎么调用模型”，不改变业务输入输出。
    # 统一放在 provider 配置里，业务节点就不用到处写超时、重试和并发限制。
    timeout_s: float | None = None
    max_retries: int | None = None
    max_concurrency: int | None = None
    include_stream_usage: bool | None = None


@dataclass(slots=True)
class LLMDefaults:
    """系统级 LLM 默认生成参数。"""

    temperature: float | None = 0.7
    max_tokens: int | None = 4000
    reasoning_effort: str | None = "none"
    context_window_tokens: int | None = 64000


@dataclass(slots=True)
class EmbeddingDefaults:
    """系统级 embedding 默认参数。"""

    dimensions: int | None = None
    batch_size: int | None = 32


@dataclass(slots=True)
class ReadDefaults:
    """保存阅读节点需要的本地处理参数。"""

    agent_name: str = "default_agent"
    paper_cache_dir: str = "data/paper_cache"
    connect_timeout_seconds: int = 10
    download_timeout_seconds: int = 60
    max_file_size_mb: int = 50
    chunk_size: int = 1200
    chunk_overlap: int = 150
    vector_store_path: str = "data/vector_store"
    vector_store_collection: str = "papers"


@dataclass(slots=True)
class PaperRetrievalConfig:
    """保存论文检索数据源需要的可选密钥。"""

    # 中文说明：这两个值只给论文数据源使用，和大模型服务的 API Key 完全分开，
    # 避免用户误把论文检索密钥填到模型设置里。
    openalex_api_key: str | None = None
    semantic_scholar_api_key: str | None = None


@dataclass(slots=True)
class SystemConfig:
    """从 config/system.yaml 读取的系统默认值。"""

    llm: LLMDefaults = field(default_factory=LLMDefaults)
    embedding: EmbeddingDefaults = field(default_factory=EmbeddingDefaults)
    paper_retrieval: PaperRetrievalConfig = field(default_factory=PaperRetrievalConfig)
    read: ReadDefaults = field(default_factory=ReadDefaults)

    @classmethod
    def load(cls, path: str | Path = "config/system.yaml") -> "SystemConfig":
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        return cls.from_dict(_parse_simple_yaml(config_path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SystemConfig":
        defaults = dict((data or {}).get("defaults") or {})
        llm = dict(defaults.get("llm") or {})
        embedding = dict(defaults.get("embedding") or {})
        paper_retrieval = dict((data or {}).get("paper_retrieval") or {})
        read = dict((data or {}).get("read") or {})
        return cls(
            llm=LLMDefaults(
                temperature=llm.get("temperature", 0.7),
                max_tokens=llm.get("max_tokens", 4000),
                reasoning_effort=llm.get("reasoning_effort", "none"),
                context_window_tokens=llm.get("context_window_tokens", 64000),
            ),
            embedding=EmbeddingDefaults(
                dimensions=embedding.get("dimensions"),
                batch_size=embedding.get("batch_size", 32),
            ),
            paper_retrieval=PaperRetrievalConfig(
                # 中文说明：配置里的 null 或空白内容都按“没有配置密钥”处理，
                # 这样用户不必为了关闭密钥专门删除配置行。
                openalex_api_key=_optional_text(paper_retrieval.get("openalex_api_key")),
                semantic_scholar_api_key=_optional_text(paper_retrieval.get("semantic_scholar_api_key")),
            ),
            read=ReadDefaults(
                agent_name=str(read.get("agent_name") or "default_agent"),
                paper_cache_dir=str(read.get("paper_cache_dir") or "data/paper_cache"),
                connect_timeout_seconds=_read_positive_int(read.get("connect_timeout_seconds"), 10),
                download_timeout_seconds=_read_positive_int(read.get("download_timeout_seconds"), 60),
                max_file_size_mb=_read_positive_int(read.get("max_file_size_mb"), 50),
                chunk_size=_read_positive_int(read.get("chunk_size"), 1200),
                chunk_overlap=_read_non_negative_int(read.get("chunk_overlap"), 150),
                vector_store_path=str(read.get("vector_store_path") or "data/vector_store"),
                vector_store_collection=str(read.get("vector_store_collection") or "papers"),
            ),
        )


@dataclass(slots=True)
class AgentConfig:
    """单个 Agent 的模型调用配置。"""

    provider: str
    model_name: str
    label: str | None = None
    # 面向使用者的角色说明，只描述这个 Agent 适合做什么。
    description: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    context_window_tokens: int | None = None

    @property
    def generation(self) -> GenerationSettings:
        return GenerationSettings(self.temperature, self.max_tokens, self.reasoning_effort)


@dataclass(slots=True)
class EmbeddingProfile:
    """单个 embedding profile 的模型配置。"""

    provider: str
    model_name: str
    label: str | None = None
    dimensions: int | None = None
    batch_size: int | None = None


@dataclass(slots=True)
class ModelConfig:
    """聚合 provider、agent 和 embedding profile 配置。"""

    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    agents: dict[str, AgentConfig] = field(default_factory=dict)
    embedding_profiles: dict[str, EmbeddingProfile] = field(default_factory=dict)
    system: SystemConfig = field(default_factory=SystemConfig)
    default_agent: str = "default_agent"
    default_embedding_profile: str = "default_embedding"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], system: SystemConfig | Mapping[str, Any] | None = None) -> "ModelConfig":
        system_config = system if isinstance(system, SystemConfig) else SystemConfig.from_dict(system)
        raw = _normalize_model_data(data)
        providers = {
            name: _provider_from_dict(name, value)
            for name, value in dict(raw.get("providers") or {}).items()
        }
        agents = _agents_from_dict(raw, system_config)
        embedding_profiles = _embedding_profiles_from_dict(raw, system_config)
        return cls(
            providers=providers,
            agents=agents,
            embedding_profiles=embedding_profiles,
            system=system_config,
            default_agent="default_agent",
            default_embedding_profile="default_embedding",
        )

    def resolve_agent(self, name: str | None = None) -> AgentConfig:
        agent_name = (name or self.default_agent).strip()
        if self.default_agent not in self.agents:
            raise ValueError("default_agent must be configured")
        if agent_name in self.agents:
            return self.agents[agent_name]
        # luna_agent、solar_agent 或其他不存在的 profile 都使用必配的默认模型，保证节点仍可运行。
        return self.agents[self.default_agent]

    def resolve_provider_config(self, agent: AgentConfig) -> tuple[str, ProviderConfig]:
        try:
            provider_config = self.providers[agent.provider]
        except KeyError as exc:
            raise ValueError(f"unknown provider: {agent.provider}") from exc

        spec = match_provider_backend(provider_config.backend)
        env_key = provider_config.api_key_env or spec.env_key
        api_key = provider_config.api_key or os.getenv(env_key)
        api_base = provider_config.api_base or spec.default_api_base
        if not api_base:
            raise ValueError(f"provider {agent.provider} requires api_base")
        # 返回补齐环境变量和默认 base_url 后的副本，避免把运行时值写回原始配置。
        return agent.provider, ProviderConfig(
            backend=spec.name,
            api_key=api_key,
            api_key_env=env_key,
            api_base=api_base,
            extra_headers=provider_config.extra_headers,
            extra_body=provider_config.extra_body,
            timeout_s=provider_config.timeout_s,
            max_retries=provider_config.max_retries,
            max_concurrency=provider_config.max_concurrency,
            include_stream_usage=provider_config.include_stream_usage,
        )

    def resolve_embedding_profile(self, name: str | None = None) -> EmbeddingProfile:
        profile_name = name or self.default_embedding_profile
        if profile_name in self.embedding_profiles:
            return self.embedding_profiles[profile_name]
        if self.default_embedding_profile in self.embedding_profiles:
            return self.embedding_profiles[self.default_embedding_profile]
        raise ValueError(f"unknown embedding profile: {profile_name}")

    def resolve_embedding_provider_config(self, name: str | None = None) -> tuple[EmbeddingProfile, ProviderConfig]:
        """解析 embedding 模型及其连接配置，并补齐环境变量中的密钥和默认地址。"""

        profile = self.resolve_embedding_profile(name)
        try:
            provider_config = self.providers[profile.provider]
        except KeyError as exc:
            raise ValueError(f"unknown embedding provider: {profile.provider}") from exc
        spec = match_provider_backend(provider_config.backend)
        env_key = provider_config.api_key_env or spec.env_key
        api_key = provider_config.api_key or os.getenv(env_key)
        api_base = provider_config.api_base or spec.default_api_base
        if not api_base:
            raise ValueError(f"embedding provider {profile.provider} requires api_base")
        return profile, ProviderConfig(
            backend=spec.name,
            api_key=api_key,
            api_key_env=env_key,
            api_base=api_base,
            extra_headers=provider_config.extra_headers,
            extra_body=provider_config.extra_body,
            timeout_s=provider_config.timeout_s,
            max_retries=provider_config.max_retries,
            max_concurrency=provider_config.max_concurrency,
            include_stream_usage=provider_config.include_stream_usage,
        )


def _normalize_model_data(data: Mapping[str, Any]) -> JsonObject:
    raw = dict(data or {})
    raw.setdefault("providers", {})
    raw.setdefault("agents", {})
    raw.setdefault("embedding_profiles", raw.pop("embeddingProfiles", {}))
    return raw


def _provider_from_dict(name: str, value: Mapping[str, Any]) -> ProviderConfig:
    return ProviderConfig(
        backend=value.get("backend") or value.get("provider_type") or value.get("providerType") or _infer_legacy_backend(name),
        api_key=value.get("api_key") or value.get("apiKey"),
        api_key_env=value.get("api_key_env") or value.get("apiKeyEnv"),
        api_base=value.get("api_base") or value.get("apiBase"),
        extra_headers=dict(value.get("extra_headers") or value.get("extraHeaders") or {}),
        extra_body=dict(value.get("extra_body") or value.get("extraBody") or {}),
        # 兼容前端常用的 camelCase，也兼容配置文件里的 snake_case。
        timeout_s=_optional_float(value.get("timeout_s", value.get("timeoutS"))),
        max_retries=_optional_int(value.get("max_retries", value.get("maxRetries"))),
        max_concurrency=_optional_int(value.get("max_concurrency", value.get("maxConcurrency"))),
        include_stream_usage=_optional_bool(value.get("include_stream_usage", value.get("includeStreamUsage"))),
    )


def _agents_from_dict(data: Mapping[str, Any], system: SystemConfig) -> dict[str, AgentConfig]:
    raw_agents = dict(data.get("agents") or {})
    agents: dict[str, AgentConfig] = {}
    for name, raw in raw_agents.items():
        agents[name] = _agent_from_dict(raw, system.llm)
    # default_agent 是所有节点回退时依赖的底线配置，不能拿其他 Agent 冒充。
    if "default_agent" not in agents:
        raise ValueError("default_agent must be configured")
    return agents


def _agent_from_dict(raw: Mapping[str, Any], defaults: LLMDefaults) -> AgentConfig:
    # Agent 只覆盖自己声明过的字段，未声明字段统一从 system.yaml 的 llm 默认值兜底。
    return AgentConfig(
        provider=str(raw.get("provider") or "auto"),
        model_name=str(raw.get("model_name") or raw.get("modelName") or raw.get("model") or ""),
        label=raw.get("label"),
        description=raw.get("description"),
        temperature=raw.get("temperature", defaults.temperature),
        max_tokens=raw.get("max_tokens", raw.get("maxTokens", defaults.max_tokens)),
        reasoning_effort=raw.get("reasoning_effort", raw.get("reasoningEffort", defaults.reasoning_effort)),
        context_window_tokens=raw.get("context_window_tokens", raw.get("contextWindowTokens", defaults.context_window_tokens)),
    )


def _embedding_profiles_from_dict(data: Mapping[str, Any], system: SystemConfig) -> dict[str, EmbeddingProfile]:
    raw_profiles = dict(data.get("embedding_profiles") or {})
    profiles: dict[str, EmbeddingProfile] = {}
    for name, raw in raw_profiles.items():
        profiles[name] = _embedding_from_dict(raw, system.embedding)
    return profiles


def _embedding_from_dict(raw: Mapping[str, Any], defaults: EmbeddingDefaults) -> EmbeddingProfile:
    return EmbeddingProfile(
        provider=str(raw.get("provider") or "auto"),
        model_name=str(raw.get("model_name") or raw.get("modelName") or raw.get("model") or ""),
        label=raw.get("label"),
        dimensions=raw.get("dimensions", defaults.dimensions),
        batch_size=raw.get("batch_size", raw.get("batchSize", defaults.batch_size)),
    )


def _infer_legacy_backend(provider_name: str) -> str:
    try:
        return match_provider_backend(provider_name).name
    except ValueError:
        return "openai_compat"


def _parse_simple_yaml(text: str) -> JsonObject:
    """解析当前 system.yaml 使用的二级缩进 YAML 子集。"""

    root: JsonObject = {}
    stack: list[tuple[int, JsonObject]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            node: JsonObject = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    if value == "null":
        return None
    if value in {"true", "false"}:
        return value == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"').strip("'")


def _optional_int(value: Any) -> int | None:
    """把可选整数配置安全读出来，读不到就保持 None。"""

    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    """把可选小数配置安全读出来，读不到就保持 None。"""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    """把可选布尔配置安全读出来，未配置时保持 None。"""

    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return None


def _optional_text(value: Any) -> str | None:
    """把配置中的可选文本整理成非空字符串或 None。"""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _read_positive_int(value: Any, default: int, *, maximum: int | None = None) -> int:
    """把阅读配置中的正整数安全转换出来，异常值回退到默认值。"""

    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    if resolved <= 0:
        return default
    return min(resolved, maximum) if maximum is not None else resolved


def _read_non_negative_int(value: Any, default: int) -> int:
    """把允许为零的阅读配置安全转换为整数，避免切片参数出现负数。"""

    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved >= 0 else default
