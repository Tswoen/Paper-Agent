from __future__ import annotations

import copy
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from src.llm import make_provider
from src.llm.config import AgentConfig, EmbeddingProfile, ModelConfig, ProviderConfig
from src.llm.registry import PROVIDERS, ProviderSpec, match_provider_backend
from src.repositories.settings.json import SettingsRepository


JsonObject = dict[str, Any]


class SettingsError(Exception):
    """设置接口的业务错误，HTTP 层会统一转换成错误响应。"""

    def __init__(self, message: str, status: int = 400):
        """初始化设置业务异常。"""

        super().__init__(message)
        self.status = status


def settings_payload(repo: SettingsRepository, agent_name: str | None = None) -> JsonObject:
    """返回给前端的完整设置快照。"""

    data = _normalized_config(repo.load())
    config = ModelConfig.from_dict(data, repo.system())
    resolved_agent_name = _resolve_agent_name(config, agent_name)
    active_agent = config.resolve_agent(resolved_agent_name)
    provider_config = config.providers.get(active_agent.provider, ProviderConfig())

    return {
        "agent": _agent_payload(resolved_agent_name, active_agent, provider_config),
        "active_agent": resolved_agent_name,
        "agents": _agent_items(config),
        "providers": _provider_items(config),
        "provider_types": _provider_type_items(),
        "embedding_profiles": _embedding_items(config),
        "defaults": {
            "llm": _dataclass_like(config.system.llm),
            "embedding": _dataclass_like(config.system.embedding),
        },
        # 中文注释：当前实现保存后下一轮请求即可生效，不需要进程重启。
        "requires_restart": False,
        "restart_required_sections": [],
        "apply_state": "applied_next_request",
        "runtime_capabilities": {
            "agent_model_settings": True,
            "embedding_profiles": True,
            "provider_model_catalog": True,
            "configured_model_connectivity_test": True,
        },
        "surface": "model_settings",
    }


def update_agent_settings(repo: SettingsRepository, patch: JsonObject) -> JsonObject:
    """更新默认 agent 的配置。"""

    data = _normalized_config(repo.load())
    name = str(patch.get("name") or patch.get("agent") or patch.get("agent_name") or patch.get("agentName") or "default_agent").strip()
    if not name:
        raise SettingsError("agent name is required")

    agents = _agents(data)
    raw_agent = agents.setdefault(name, {})
    _apply_agent_patch(raw_agent, patch)
    _validate_agent(data, raw_agent)
    repo.save(data)
    return settings_payload(repo, name)


def create_or_update_agent(repo: SettingsRepository, name: str, patch: JsonObject) -> JsonObject:
    """按名称创建或更新一个 agent 配置。"""

    body = dict(patch)
    body["name"] = name
    return update_agent_settings(repo, body)


def update_provider_settings(repo: SettingsRepository, name: str, patch: JsonObject) -> JsonObject:
    """更新 provider 配置。"""

    data = _normalized_config(repo.load())
    provider = _providers(data).setdefault(name, {})
    if "backend" in patch or "provider_type" in patch or "providerType" in patch:
        backend = str(patch.get("backend") or patch.get("provider_type") or patch.get("providerType") or "").strip()
        if not backend:
            raise SettingsError("provider backend is required")
        try:
            match_provider_backend(backend)
        except ValueError as exc:
            raise SettingsError(str(exc), 404) from exc
        provider["backend"] = backend
    if "api_key" in patch or "apiKey" in patch:
        provider["api_key"] = patch.get("api_key", patch.get("apiKey"))
    if "api_key_env" in patch or "apiKeyEnv" in patch:
        provider["api_key_env"] = patch.get("api_key_env", patch.get("apiKeyEnv"))
    if "api_base" in patch or "apiBase" in patch:
        provider["api_base"] = patch.get("api_base", patch.get("apiBase"))
    if "extra_headers" in patch or "extraHeaders" in patch:
        provider["extra_headers"] = dict(patch.get("extra_headers") or patch.get("extraHeaders") or {})
    if "extra_body" in patch or "extraBody" in patch:
        provider["extra_body"] = dict(patch.get("extra_body") or patch.get("extraBody") or {})
    repo.save(data)
    return settings_payload(repo)


def update_embedding_profile(repo: SettingsRepository, name: str, patch: JsonObject) -> JsonObject:
    """更新嵌入模型配置。"""

    data = _normalized_config(repo.load())
    profile = _embedding_profiles(data).setdefault(name, {})
    allowed = {
        "label": "label",
        "provider": "provider",
        "model": "model_name",
        "model_name": "model_name",
        "modelName": "model_name",
        "dimensions": "dimensions",
        "batch_size": "batch_size",
        "batchSize": "batch_size",
    }
    for incoming, target in allowed.items():
        if incoming in patch:
            profile[target] = patch[incoming]
    _validate_embedding(data, profile)
    repo.save(data)
    return settings_payload(repo)


def provider_models_payload(repo: SettingsRepository, provider: str, client: Any | None = None) -> JsonObject:
    """读取指定 provider 的模型目录载荷。"""

    data = _normalized_config(repo.load())
    config = ModelConfig.from_dict(data, repo.system())
    if provider not in config.providers:
        raise SettingsError(f"unknown provider: {provider}", 404)
    provider_config = config.providers.get(provider, ProviderConfig())
    spec = match_provider_backend(provider_config.backend)
    api_base = provider_config.api_base or spec.default_api_base
    if not api_base:
        return _models_status(provider, "missing_api_base", "provider requires api_base before model catalog can be fetched")
    if _provider_requires_key(spec) and not provider_config.api_key:
        return _models_status(provider, "not_configured", "provider api_key is not configured")

    try:
        model_client = client or _make_model_list_client(spec, provider_config, api_base)
        models = _parse_model_list(model_client.models.list())
    except NotImplementedError:
        return _models_status(provider, "unsupported", "provider model catalog is not supported")
    except Exception as exc:
        return _models_status(provider, "error", str(exc))

    return {
        "provider": provider,
        "label": _provider_label(provider),
        "status": "available",
        "catalog_kind": spec.backend,
        "models": models,
        "model_count": len(models),
        "message": "",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def model_connectivity_payload(
    repo: SettingsRepository,
    target_type: str,
    name: str,
    *,
    client: Any | None = None,
    embedding_client: Any | None = None,
) -> JsonObject:
    """按当前保存的模型配置做一次最小真实调用，用来判断这条配置能不能用。

    中文说明：
    1. 这里测试的是“这一行配置里的 provider + model_name 是否可调用”，不是 provider 是否能列出模型目录；
    2. agent 会发起一条最小对话请求；
    3. embedding_profile 会发起一次最小向量化请求；
    4. 返回值统一成前端容易展示的结构，按钮就可以直接显示“已通过 / 未通过 / 未配置”。
    """

    normalized_target = str(target_type or "").strip()
    target_name = str(name or "").strip()
    if not normalized_target:
        raise SettingsError("target_type is required")
    if not target_name:
        raise SettingsError("name is required")

    data = _normalized_config(repo.load())
    config = ModelConfig.from_dict(data, repo.system())

    if normalized_target == "agent":
        if target_name not in config.agents:
            raise SettingsError(f"unknown agent: {target_name}", 404)
        return _test_agent_connectivity(config, target_name, client=client)
    if normalized_target == "embedding_profile":
        if target_name not in config.embedding_profiles:
            raise SettingsError(f"unknown embedding profile: {target_name}", 404)
        return _test_embedding_connectivity(config, target_name, client=embedding_client)
    raise SettingsError(f"unsupported target_type: {normalized_target}")


def _normalized_config(data: JsonObject) -> JsonObject:
    """把原始配置标准化成统一内部结构。"""

    data = copy.deepcopy(data)
    data.setdefault("providers", {})
    data.setdefault("agents", {})
    data.setdefault("embedding_profiles", data.pop("embeddingProfiles", {}))
    return data


def _test_agent_connectivity(config: ModelConfig, name: str, *, client: Any | None = None) -> JsonObject:
    """对指定智能体配置发起一次最小对话请求。

    中文说明：
    1. 这里只问一句“请只回复 OK”，尽量减少 token 消耗；
    2. 只要模型能正常返回任意非空文本，就说明这条配置是可用的；
    3. 如果连 provider 都组装不起来，说明是配置问题，状态会标成 not_configured。
    """

    agent = config.resolve_agent(name)
    started_at = perf_counter()
    try:
        snapshot = make_provider(config, name, client=client)
    except Exception as exc:
        return _connectivity_payload(
            target_type="agent",
            name=name,
            provider=agent.provider,
            model=agent.model_name,
            status="not_configured",
            message=f"当前智能体配置还不能发起调用：{exc}",
            latency_ms=_elapsed_ms(started_at),
        )

    response = snapshot.provider.chat_with_retry(
        [{"role": "user", "content": "请只回复 OK"}],
        temperature=0,
        max_tokens=16,
    )
    if not response.ok:
        detail = response.content.strip() or "模型没有返回成功结果"
        return _connectivity_payload(
            target_type="agent",
            name=name,
            provider=agent.provider,
            model=agent.model_name,
            status="failed",
            message=detail,
            latency_ms=_elapsed_ms(started_at),
            error_kind=response.error_kind,
            error_status_code=response.error_status_code,
            finish_reason=response.finish_reason,
        )

    content = response.content.strip()
    if not content:
        return _connectivity_payload(
            target_type="agent",
            name=name,
            provider=agent.provider,
            model=agent.model_name,
            status="failed",
            message="模型接口已返回成功状态，但返回内容为空",
            latency_ms=_elapsed_ms(started_at),
            finish_reason=response.finish_reason,
        )

    return _connectivity_payload(
        target_type="agent",
        name=name,
        provider=agent.provider,
        model=agent.model_name,
        status="passed",
        message="模型已成功返回内容",
        latency_ms=_elapsed_ms(started_at),
        finish_reason=response.finish_reason,
    )


def _test_embedding_connectivity(config: ModelConfig, name: str, *, client: Any | None = None) -> JsonObject:
    """对指定嵌入模型配置发起一次最小 embedding 请求。

    中文说明：
    1. 这里不会去拉 provider 的模型目录；
    2. 而是直接调用当前 profile 绑定的 model_name；
    3. 只要返回了一条非空向量，就说明这条嵌入配置能真正参与索引和检索。
    """

    profile = config.resolve_embedding_profile(name)
    started_at = perf_counter()
    try:
        # 这里和阅读节点使用同一套 provider 装配方式，避免设置页通过、真实索引却失败。
        snapshot = make_provider(
            config,
            embedding_profile_name=name,
            client=client,
            timeout_s=float(max(1, config.system.read.download_timeout_seconds)),
        )
    except Exception as exc:
        return _connectivity_payload(
            target_type="embedding_profile",
            name=name,
            provider=profile.provider,
            model=profile.model_name,
            status="not_configured",
            message=f"当前嵌入配置还不能发起调用：{exc}",
            latency_ms=_elapsed_ms(started_at),
        )

    try:
        response = snapshot.provider.embed_with_retry(["连通性测试"], dimensions=profile.dimensions)
    except NotImplementedError as exc:
        return _connectivity_payload(
            target_type="embedding_profile",
            name=name,
            provider=profile.provider,
            model=profile.model_name,
            status="failed",
            message=f"当前 provider 不支持 embedding：{exc}",
            latency_ms=_elapsed_ms(started_at),
        )
    except Exception as exc:
        return _connectivity_payload(
            target_type="embedding_profile",
            name=name,
            provider=profile.provider,
            model=profile.model_name,
            status="failed",
            message=f"嵌入模型调用失败：{exc}",
            latency_ms=_elapsed_ms(started_at),
        )

    if not response.ok:
        detail = response.content.strip() or response.error_code or response.error_type or response.error_kind or "模型没有返回成功结果"
        return _connectivity_payload(
            target_type="embedding_profile",
            name=name,
            provider=profile.provider,
            model=profile.model_name,
            status="failed",
            message=f"嵌入模型调用失败：{detail}",
            latency_ms=_elapsed_ms(started_at),
            error_kind=response.error_kind,
            error_status_code=response.error_status_code,
            finish_reason=response.finish_reason,
        )

    vector = response.embeddings[0] if response.embeddings else None
    if not isinstance(vector, list) or not vector:
        return _connectivity_payload(
            target_type="embedding_profile",
            name=name,
            provider=profile.provider,
            model=profile.model_name,
            status="failed",
            message="嵌入模型返回的向量为空或格式不正确",
            latency_ms=_elapsed_ms(started_at),
            finish_reason=response.finish_reason,
        )

    return _connectivity_payload(
        target_type="embedding_profile",
        name=name,
        provider=profile.provider,
        model=profile.model_name,
        status="passed",
        message=f"嵌入模型已成功返回 {len(vector)} 维向量",
        latency_ms=_elapsed_ms(started_at),
        vector_dimensions=len(vector),
    )


def _providers(data: JsonObject) -> JsonObject:
    """返回 provider 配置字典，并在缺失时补空对象。"""

    return data.setdefault("providers", {})


def _agents(data: JsonObject) -> JsonObject:
    """返回 agent 配置字典，并在缺失时补空对象。"""

    return data.setdefault("agents", {})


def _embedding_profiles(data: JsonObject) -> JsonObject:
    """返回 embedding profile 配置字典，并在缺失时补空对象。"""

    return data.setdefault("embedding_profiles", {})


def _resolve_agent_name(config: ModelConfig, requested: str | None) -> str:
    """根据请求参数与默认值解析当前生效的 agent 名称。"""

    if requested and requested in config.agents:
        return requested
    if config.default_agent in config.agents:
        return config.default_agent
    try:
        return next(iter(config.agents))
    except StopIteration as exc:
        raise SettingsError("no agent configured") from exc


def _agent_payload(name: str, agent: AgentConfig, provider_config: ProviderConfig) -> JsonObject:
    """构造当前活动 agent 的前端响应结构。"""

    return {
        "name": name,
        "label": agent.label or name,
        "model": agent.model_name,
        "model_name": agent.model_name,
        "provider": agent.provider,
        "resolved_provider": agent.provider,
        "has_api_key": bool(provider_config.api_key),
        "max_tokens": agent.max_tokens,
        "context_window_tokens": agent.context_window_tokens,
        "temperature": agent.temperature,
        "reasoning_effort": agent.reasoning_effort,
    }


def _agent_items(config: ModelConfig) -> list[JsonObject]:
    """构造全部 agent 列表响应。"""

    return [
        _agent_item(name, agent, name == config.default_agent)
        for name, agent in config.agents.items()
    ]


def _agent_item(name: str, agent: AgentConfig, is_default: bool) -> JsonObject:
    """构造单个 agent 的列表项结构。"""

    return {
        "name": name,
        "label": agent.label or name,
        "is_default": is_default,
        "model": agent.model_name,
        "model_name": agent.model_name,
        "provider": agent.provider,
        "resolved_provider": agent.provider,
        "max_tokens": agent.max_tokens,
        "context_window_tokens": agent.context_window_tokens,
        "temperature": agent.temperature,
        "reasoning_effort": agent.reasoning_effort,
        "reasoning_effort_values": ["none", "low", "medium", "high"],
    }


def _embedding_items(config: ModelConfig) -> list[JsonObject]:
    """构造全部 embedding profile 列表响应。"""

    return [
        _embedding_item(name, profile, name == config.default_embedding_profile)
        for name, profile in config.embedding_profiles.items()
    ]


def _embedding_item(name: str, profile: EmbeddingProfile, is_default: bool) -> JsonObject:
    """构造单个 embedding profile 的列表项结构。"""

    return {
        "name": name,
        "label": profile.label or name,
        "is_default": is_default,
        "provider": profile.provider,
        "model": profile.model_name,
        "model_name": profile.model_name,
        "dimensions": profile.dimensions,
        "batch_size": profile.batch_size,
    }


def _provider_items(config: ModelConfig) -> list[JsonObject]:
    """构造全部 provider 的前端响应列表。"""

    items = []
    for name, provider_config in sorted(config.providers.items()):
        spec = match_provider_backend(provider_config.backend)
        configured = _provider_configured(spec, provider_config)
        # 中文注释：管理端需要“可直接回填到表单”的原始配置值，避免前端只能看到脱敏摘要。
        editable_config = {
            "backend": provider_config.backend,
            "api_key": provider_config.api_key,
            "api_key_env": provider_config.api_key_env,
            "api_base": provider_config.api_base,
            "extra_headers": provider_config.extra_headers,
            "extra_body": provider_config.extra_body,
        }
        items.append(
            {
                "name": name,
                "label": _provider_label(name),
                "configured": configured,
                "auth_type": "api_key",
                "api_key_required": _provider_requires_key(spec),
                "api_key_hint": _api_key_hint(provider_config.api_key),
                "api_key_env": provider_config.api_key_env,
                "api_base": provider_config.api_base,
                "default_api_base": spec.default_api_base,
                "model_selectable": True,
                "provider_type": provider_config.backend,
                "backend": spec.backend,
                "oauth_login_supported": False,
                "editable_config": editable_config,
            }
        )
    return items


def _apply_agent_patch(agent: JsonObject, patch: JsonObject) -> None:
    """把请求补丁映射到内部 agent 配置字段。"""

    allowed = {
        "label": "label",
        "provider": "provider",
        "model": "model_name",
        "model_name": "model_name",
        "modelName": "model_name",
        "max_tokens": "max_tokens",
        "maxTokens": "max_tokens",
        "context_window_tokens": "context_window_tokens",
        "contextWindowTokens": "context_window_tokens",
        "temperature": "temperature",
        "reasoning_effort": "reasoning_effort",
        "reasoningEffort": "reasoning_effort",
    }
    for incoming, target in allowed.items():
        if incoming in patch:
            agent[target] = patch[incoming]


def _validate_agent(data: JsonObject, agent: JsonObject) -> None:
    """校验 agent 配置是否完整且引用了合法 provider。"""

    provider = agent.get("provider")
    model_name = agent.get("model_name") or agent.get("model")
    if not provider:
        raise SettingsError("agent provider is required")
    if not model_name:
        raise SettingsError("agent model_name is required")
    _validate_provider_for_save(data, str(provider))


def _validate_embedding(data: JsonObject, profile: JsonObject) -> None:
    """校验 embedding profile 配置是否完整。"""

    provider = profile.get("provider")
    model_name = profile.get("model_name") or profile.get("model")
    if not provider:
        raise SettingsError("embedding provider is required")
    if not model_name:
        raise SettingsError("embedding model_name is required")
    _validate_provider_for_save(data, str(provider))


def _validate_provider_for_save(data: JsonObject, provider: str | None) -> None:
    """校验引用的 provider 是否存在且已完成基础配置。"""

    if not provider or provider == "auto":
        return
    config = ModelConfig.from_dict(data)
    if provider not in config.providers:
        raise SettingsError(f"unknown provider: {provider}", 404)
    provider_config = config.providers.get(provider, ProviderConfig())
    spec = match_provider_backend(provider_config.backend)
    if not _provider_configured(spec, provider_config):
        raise SettingsError(f"provider is not configured: {provider}")


def _provider_configured(spec: ProviderSpec, config: ProviderConfig) -> bool:
    """判断 provider 是否已具备最基本的可用条件。"""

    if spec.default_api_base or config.api_base:
        return bool(config.api_key) if _provider_requires_key(spec) else True
    return False


def _provider_requires_key(spec: ProviderSpec) -> bool:
    """判断指定 provider backend 是否需要 API Key。"""

    return spec.backend in {"openai_compat", "anthropic"}


def _api_key_hint(api_key: str | None) -> str | None:
    """把 API Key 转成脱敏后的提示文本。"""

    if not api_key:
        return None
    return f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "configured"


def _provider_label(name: str) -> str:
    """把 provider 名称转换成更友好的展示标签。"""

    return name.replace("_", " ").title()


def _provider_type_items() -> list[JsonObject]:
    """返回全部 provider 类型说明。"""

    items = []
    for name, spec in PROVIDERS.items():
        items.append(
            {
                "name": name,
                "label": _provider_label(name),
                "backend": spec.backend,
                "default_api_base": spec.default_api_base,
                "api_key_required": _provider_requires_key(spec),
            }
        )
    return items


def _connectivity_payload(
    *,
    target_type: str,
    name: str,
    provider: str,
    model: str,
    status: str,
    message: str,
    latency_ms: int,
    error_kind: str | None = None,
    error_status_code: int | None = None,
    finish_reason: str | None = None,
    vector_dimensions: int | None = None,
) -> JsonObject:
    """把连通性测试结果整理成统一结构，方便前端直接展示。"""

    return {
        "target_type": target_type,
        "name": name,
        "provider": provider,
        "model": model,
        "status": status,
        "message": message,
        "latency_ms": latency_ms,
        "error_kind": error_kind,
        "error_status_code": error_status_code,
        "finish_reason": finish_reason,
        "vector_dimensions": vector_dimensions,
        "tested_at": datetime.now(timezone.utc).isoformat(),
    }


def _elapsed_ms(started_at: float) -> int:
    """把一次测试的耗时转换成毫秒，前端展示更直观。"""

    return max(0, int((perf_counter() - started_at) * 1000))


def _models_status(provider: str, status: str, message: str) -> JsonObject:
    """构造模型目录拉取失败或受限时的统一响应结构。"""

    return {
        "provider": provider,
        "label": _provider_label(provider),
        "status": status,
        "catalog_kind": "unknown",
        "models": [],
        "model_count": 0,
        "message": message,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _make_model_list_client(spec: ProviderSpec, config: ProviderConfig, api_base: str) -> Any:
    """按 provider backend 创建模型目录客户端。"""

    if spec.backend == "openai_compat":
        from openai import OpenAI

        return OpenAI(api_key=config.api_key, base_url=api_base, default_headers=config.extra_headers or None)
    raise NotImplementedError


def _parse_model_list(response: Any) -> list[JsonObject]:
    """把 provider 返回的模型目录解析成统一列表结构。"""

    raw = response.model_dump() if hasattr(response, "model_dump") else response
    data = raw.get("data", raw) if isinstance(raw, dict) else raw
    models = []
    for item in data or []:
        model = item.model_dump() if hasattr(item, "model_dump") else item
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


def _dataclass_like(value: Any) -> JsonObject:
    """把 dataclass 风格对象转成普通字典。"""

    return {
        key: getattr(value, key)
        for key in getattr(value, "__dataclass_fields__", {})
    }
