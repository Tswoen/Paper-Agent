from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm.config import AgentConfig, EmbeddingProfile, ModelConfig, ProviderConfig, SystemConfig
from src.llm.registry import PROVIDERS, ProviderSpec, match_provider_backend


JsonObject = dict[str, Any]


class SettingsError(Exception):
    """设置接口的业务错误，router 层会统一转换成 HTTP 响应。"""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class SettingsRepository:
    """极简配置仓库，只负责 JSON 文件或内存配置的 load/save。"""

    def __init__(
        self,
        path: str | Path | None = None,
        initial: JsonObject | None = None,
        system_path: str | Path | None = None,
        system: SystemConfig | JsonObject | None = None,
    ):
        self.path = Path(path) if path else None
        self.system_path = Path(system_path) if system_path else Path("config/system.yaml")
        self._memory = copy.deepcopy(initial or {})
        self._system = system if isinstance(system, SystemConfig) else SystemConfig.from_dict(system)

    def load(self) -> JsonObject:
        if self.path and self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return copy.deepcopy(self._memory)

    def save(self, data: JsonObject) -> None:
        normalized = copy.deepcopy(data)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        self._memory = normalized

    def system(self) -> SystemConfig:
        if self.path:
            return SystemConfig.load(self.system_path)
        return self._system


def settings_payload(repo: SettingsRepository, agent_name: str | None = None) -> JsonObject:
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
        # 当前实现保存后下一轮请求即可生效，不需要进程重启。
        "requires_restart": False,
        "restart_required_sections": [],
        "apply_state": "applied_next_request",
        "runtime_capabilities": {
            "agent_model_settings": True,
            "embedding_profiles": True,
            "provider_model_catalog": True,
        },
        "surface": "model_settings",
    }


def update_agent_settings(repo: SettingsRepository, patch: JsonObject) -> JsonObject:
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
    body = dict(patch)
    body["name"] = name
    return update_agent_settings(repo, body)


def update_provider_settings(repo: SettingsRepository, name: str, patch: JsonObject) -> JsonObject:
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


def _normalized_config(data: JsonObject) -> JsonObject:
    data = copy.deepcopy(data)
    data.setdefault("providers", {})
    data.setdefault("agents", {})
    data.setdefault("embedding_profiles", data.pop("embeddingProfiles", {}))
    return data


def _providers(data: JsonObject) -> JsonObject:
    return data.setdefault("providers", {})


def _agents(data: JsonObject) -> JsonObject:
    return data.setdefault("agents", {})


def _embedding_profiles(data: JsonObject) -> JsonObject:
    return data.setdefault("embedding_profiles", {})


def _resolve_agent_name(config: ModelConfig, requested: str | None) -> str:
    if requested and requested in config.agents:
        return requested
    if config.default_agent in config.agents:
        return config.default_agent
    try:
        return next(iter(config.agents))
    except StopIteration as exc:
        raise SettingsError("no agent configured") from exc


def _agent_payload(name: str, agent: AgentConfig, provider_config: ProviderConfig) -> JsonObject:
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
    return [
        _agent_item(name, agent, name == config.default_agent)
        for name, agent in config.agents.items()
    ]


def _agent_item(name: str, agent: AgentConfig, is_default: bool) -> JsonObject:
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
    return [
        _embedding_item(name, profile, name == config.default_embedding_profile)
        for name, profile in config.embedding_profiles.items()
    ]


def _embedding_item(name: str, profile: EmbeddingProfile, is_default: bool) -> JsonObject:
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
    items = []
    for name, provider_config in sorted(config.providers.items()):
        spec = match_provider_backend(provider_config.backend)
        configured = _provider_configured(spec, provider_config)
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
            }
        )
    return items


def _apply_agent_patch(agent: JsonObject, patch: JsonObject) -> None:
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
    provider = agent.get("provider")
    model_name = agent.get("model_name") or agent.get("model")
    if not provider:
        raise SettingsError("agent provider is required")
    if not model_name:
        raise SettingsError("agent model_name is required")
    _validate_provider_for_save(data, str(provider))


def _validate_embedding(data: JsonObject, profile: JsonObject) -> None:
    provider = profile.get("provider")
    model_name = profile.get("model_name") or profile.get("model")
    if not provider:
        raise SettingsError("embedding provider is required")
    if not model_name:
        raise SettingsError("embedding model_name is required")
    _validate_provider_for_save(data, str(provider))


def _validate_provider_for_save(data: JsonObject, provider: str | None) -> None:
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
    if spec.default_api_base or config.api_base:
        return bool(config.api_key) if _provider_requires_key(spec) else True
    return False


def _provider_requires_key(spec: ProviderSpec) -> bool:
    return spec.backend in {"openai_compat", "anthropic"}


def _api_key_hint(api_key: str | None) -> str | None:
    if not api_key:
        return None
    return f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "configured"


def _provider_label(name: str) -> str:
    return name.replace("_", " ").title()


def _provider_type_items() -> list[JsonObject]:
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


def _models_status(provider: str, status: str, message: str) -> JsonObject:
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
    if spec.backend == "openai_compat":
        from openai import OpenAI

        return OpenAI(api_key=config.api_key, base_url=api_base, default_headers=config.extra_headers or None)
    raise NotImplementedError


def _parse_model_list(response: Any) -> list[JsonObject]:
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
    return {
        key: getattr(value, key)
        for key in getattr(value, "__dataclass_fields__", {})
    }
