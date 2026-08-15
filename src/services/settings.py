from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from src.llm import make_provider
from src.llm.config import (
    AgentConfig,
    EmbeddingProfile,
    ModelConfig,
    ProviderConfig,
    _embedding_profiles_from_dict,
    _provider_from_dict,
)
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
    """返回给前端的完整设置快照。

    中文说明：
    配置不完整时（例如只保存了 Provider、还没有 default_agent），页面仍然要展示
    已经保存的部分：Provider 和嵌入模型照常返回，智能体列表留空。这样设置页
    永远不会因为“还没配完”而打不开，用户可以在网页里逐步补全配置。
    """

    data = _normalized_config(repo.load())
    system = repo.system()
    try:
        config = ModelConfig.from_dict(data, system)
    except ValueError:
        # default_agent 缺失时 from_dict 会拒绝构建；但 Provider 和嵌入模型
        # 不依赖 default_agent，单独解析出来用于展示，智能体部分暂时留空。
        config = ModelConfig(
            providers={
                name: _provider_from_dict(name, value)
                for name, value in (data.get("providers") or {}).items()
            },
            agents={},
            embedding_profiles=_embedding_profiles_from_dict(data, system),
            system=system,
        )

    try:
        resolved_agent_name = _resolve_agent_name(config, agent_name)
        active_agent = config.resolve_agent(resolved_agent_name)
        provider_config = config.providers.get(active_agent.provider, ProviderConfig())
        agent_payload = _agent_payload(resolved_agent_name, active_agent, provider_config)
        agent_items = _agent_items(config)
    except (ValueError, SettingsError):
        # 还没有 default_agent 时智能体没有可展示内容，先留空。
        agent_payload = None
        resolved_agent_name = ""
        agent_items = []

    return {
        "agent": agent_payload,
        "active_agent": resolved_agent_name,
        "agents": agent_items,
        "providers": _provider_items(config),
        "provider_types": _provider_type_items(),
        "embedding_profiles": _embedding_items(config),
        "defaults": {
            "llm": _dataclass_like(system.llm),
            "embedding": _dataclass_like(system.embedding),
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
    # 这些字段只影响模型调用方式，不改变 provider 的基础鉴权信息。
    # 支持两种命名，是为了让配置文件和前端表单都能自然传值。
    _apply_optional_provider_field(provider, patch, "timeout_s", "timeoutS")
    _apply_optional_provider_field(provider, patch, "max_retries", "maxRetries")
    _apply_optional_provider_field(provider, patch, "max_concurrency", "maxConcurrency")
    _apply_optional_provider_field(provider, patch, "include_stream_usage", "includeStreamUsage")
    repo.save(data)
    return settings_payload(repo)


def delete_provider_settings(repo: SettingsRepository, name: str) -> JsonObject:
    """删除一个 provider 配置，并清理引用它的智能体和嵌入模型。

    中文说明：
    如果某个智能体或嵌入模型还引用着被删的 Provider，配置里会留下悬空引用，
    之后跑任务时会报“找不到 Provider”的怪错，所以删除时一起清掉。
    """

    data = _normalized_config(repo.load())
    if name not in _providers(data):
        raise SettingsError(f"unknown provider: {name}", 404)

    del _providers(data)[name]
    agents = _agents(data)
    for agent_name in [n for n, a in agents.items() if a.get("provider") == name]:
        del agents[agent_name]
    profiles = _embedding_profiles(data)
    for profile_name in [n for n, p in profiles.items() if p.get("provider") == name]:
        del profiles[profile_name]

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

    # 中文注释：这是旧同步入口，只给还没迁移的测试或调用方过渡使用；新路由会直接 await async_provider_models_payload。
    return _run_async_for_legacy(async_provider_models_payload(repo, provider, client=client))


async def async_provider_models_payload(repo: SettingsRepository, provider: str, client: Any | None = None) -> JsonObject:
    """异步读取指定 provider 的模型目录载荷。"""

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

    snapshot = None
    try:
        if client is not None:
            # 测试或特殊调用方可以继续传假 client；正式路径统一走 provider.list_models。
            raw_models = client.models.list()
            if hasattr(raw_models, "__await__"):
                raw_models = await raw_models
            models = _parse_model_list(raw_models)
        else:
            config.agents.setdefault(
                "__model_catalog__",
                AgentConfig(provider=provider, model_name="__model_catalog__"),
            )
            snapshot = make_provider(config, "__model_catalog__")
            models = await snapshot.provider.list_models()
    except NotImplementedError:
        return _models_status(provider, "unsupported", "provider model catalog is not supported")
    except Exception as exc:
        return _models_status(provider, "error", str(exc))
    finally:
        if snapshot is not None:
            await snapshot.aclose()

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
    """按当前保存的模型配置做一次最小真实调用，用来判断这条配置能不能用。"""

    # 中文注释：这是旧同步入口，只做临时兼容；FastAPI 路由已经改用 async_model_connectivity_payload。
    return _run_async_for_legacy(
        async_model_connectivity_payload(
            repo,
            target_type,
            name,
            client=client,
            embedding_client=embedding_client,
        )
    )


async def async_model_connectivity_payload(
    repo: SettingsRepository,
    target_type: str,
    name: str,
    *,
    client: Any | None = None,
    embedding_client: Any | None = None,
) -> JsonObject:
    """按当前保存的模型配置做一次最小真实异步调用。"""

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
        return await _test_agent_connectivity(config, target_name, client=client)
    if normalized_target == "embedding_profile":
        if target_name not in config.embedding_profiles:
            raise SettingsError(f"unknown embedding profile: {target_name}", 404)
        return await _test_embedding_connectivity(config, target_name, client=embedding_client)
    raise SettingsError(f"unsupported target_type: {normalized_target}")


def _run_async_for_legacy(awaitable: Any) -> Any:
    """让旧同步入口临时运行新的 async 实现。"""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 中文注释：旧同步测试或旧同步服务调用时通常没有事件循环，可以安全地临时跑一次。
        return asyncio.run(awaitable)
    # 中文注释：如果已经在 FastAPI 这类 async 环境中，必须直接 await async_* 函数，不能再走同步桥。
    if hasattr(awaitable, "close"):
        awaitable.close()
    raise RuntimeError("当前环境已经有事件循环，请改用 async_provider_models_payload / async_model_connectivity_payload")


def _normalized_config(data: JsonObject) -> JsonObject:
    """把原始配置标准化成统一内部结构。"""

    data = copy.deepcopy(data)
    data.setdefault("providers", {})
    data.setdefault("agents", {})
    data.setdefault("embedding_profiles", data.pop("embeddingProfiles", {}))
    return data


async def _test_agent_connectivity(config: ModelConfig, name: str, *, client: Any | None = None) -> JsonObject:
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

    try:
        response = await snapshot.provider.chat(
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
    finally:
        # 中文注释：设置页测试是短生命周期调用，完成后立即关闭 provider 自己创建的连接池。
        await snapshot.aclose()


async def _test_embedding_connectivity(config: ModelConfig, name: str, *, client: Any | None = None) -> JsonObject:
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
        try:
            response = await snapshot.provider.embed(["连通性测试"], dimensions=profile.dimensions)
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
    finally:
        # 中文注释：设置页测试结束后释放 provider 自己创建的异步连接，避免出现未关闭连接警告。
        await snapshot.aclose()


def _apply_optional_provider_field(provider: JsonObject, patch: JsonObject, snake_name: str, camel_name: str) -> None:
    """把 provider 的可选运行控制字段写回配置。"""

    if snake_name in patch:
        provider[snake_name] = patch[snake_name]
    elif camel_name in patch:
        provider[snake_name] = patch[camel_name]


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

    if config.default_agent not in config.agents:
        raise SettingsError("default_agent must be configured")
    if requested and requested in config.agents:
        return requested
    return config.default_agent


def _agent_payload(name: str, agent: AgentConfig, provider_config: ProviderConfig) -> JsonObject:
    """构造当前活动 agent 的前端响应结构。"""

    return {
        "name": name,
        "label": agent.label or name,
        "description": _agent_description(name, agent),
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
        "description": _agent_description(name, agent),
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


def _agent_description(name: str, agent: AgentConfig) -> str:
    """返回列表中展示的 Agent 作用说明。"""

    if agent.description and agent.description.strip():
        return agent.description.strip()

    # 没有单独写说明时，根据内置的三个角色给出清晰的默认提示。
    descriptions = {
        "luna_agent": "基础能力，适合简单任务，价格最低。",
        "default_agent": "中等能力，适合大多数任务，默认使用，价格适中。",
        "solar_agent": "最高能力，适合最难任务，价格最高。",
    }
    return descriptions.get(name.lower(), "通用模型能力，适合按需分配任务。")


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
            "timeout_s": provider_config.timeout_s,
            "max_retries": provider_config.max_retries,
            "max_concurrency": provider_config.max_concurrency,
            "include_stream_usage": provider_config.include_stream_usage,
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
    if "default_agent" not in (data.get("agents") or {}):
        # 中文注释：default_agent 是所有节点运行时的默认兜底档位，必须先存在。
        # 这里提前给一个能看懂的中文提示，避免用户拿到让人摸不着头脑的英文报错。
        raise SettingsError(
            "系统要求先存在名为 default_agent 的智能体（所有节点运行必需的默认档位），"
            "请先把智能体名称填成 default_agent 再保存"
        )
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
