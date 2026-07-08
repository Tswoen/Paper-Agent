import os
import re
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values
from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from src.core.config import config
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

configuration = APIRouter(prefix="/config", tags=["configuration"])

MODELS_CONFIG_PATH = Path(__file__).parent / "models.yaml"
SYSTEM_PARAMS_PATH = Path(__file__).parent / "system_params.yaml"
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_CONFIG_PATH = PROJECT_ROOT / ".env"
ENV_LINE_PATTERN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
ENV_VAR_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
SAFE_ENV_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9_./:@%+=,\-]*$")
ARXIV_API_URL = "http://export.arxiv.org/api/query"

PROVIDER_TYPE_OPTIONS = [
    {
        "value": "openai",
        "label": "OpenAI",
        "default_base_url": "https://api.openai.com/v1",
    },
    {
        "value": "siliconflow",
        "label": "SiliconFlow",
        "default_base_url": "https://api.siliconflow.cn/v1",
    },
    {
        "value": "dashscope",
        "label": "DashScope",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    {
        "value": "ark",
        "label": "Ark",
        "default_base_url": "https://ark.cn-beijing.volces.com/api/v3",
    },
    {
        "value": "custom",
        "label": "OpenAI Compatible",
        "default_base_url": "",
    },
]

KNOWN_PROVIDER_TYPES = {item["value"] for item in PROVIDER_TYPE_OPTIONS}

DEFAULT_MODEL_DEFINITIONS = [
    {
        "key": "default-model",
        "label": "默认 LLM",
        "description": "业务模块未指定专用模型时回退使用的大语言模型。",
        "kind": "llm",
    },
    {
        "key": "default-embedding-model",
        "label": "默认 Embedding",
        "description": "向量化模块未指定专用模型时回退使用的嵌入模型。",
        "kind": "embedding",
    },
]

AGENT_MODEL_DEFINITIONS = [
    {
        "key": "search-model",
        "label": "检索规划",
        "description": "根据用户主题生成论文检索条件。",
        "kind": "llm",
    },
    {
        "key": "reading-model",
        "label": "论文阅读",
        "description": "读取论文并抽取核心问题、方法、结果与贡献。",
        "kind": "llm",
    },
    {
        "key": "subanalyse-cluster-model",
        "label": "分析聚类",
        "description": "为论文聚类结果生成主题描述和关键词。",
        "kind": "llm",
    },
    {
        "key": "subanalyse-deep-analyse-model",
        "label": "深度分析",
        "description": "针对单个论文聚类进行深入学术分析。",
        "kind": "llm",
    },
    {
        "key": "subanalyse-global-analyse-model",
        "label": "全局分析",
        "description": "汇总多主题分析，生成趋势、热点、局限与展望。",
        "kind": "llm",
    },
    {
        "key": "subwriting-writing-director-model",
        "label": "写作规划",
        "description": "根据分析结果拆分报告写作小节和大纲。",
        "kind": "llm",
    },
    {
        "key": "subwriting-writing-model",
        "label": "小节写作",
        "description": "负责具体报告小节内容生成。",
        "kind": "llm",
    },
    {
        "key": "subwriting_retrieval_model",
        "label": "小节检索",
        "description": "负责具体报告小节相关论文检索。",
        "kind": "llm",
    },
    {
        "key": "report-model",
        "label": "最终报告",
        "description": "将各小节组装为完整 Markdown 调研报告。",
        "kind": "llm",
    },
]

EMBEDDING_MODEL_DEFINITIONS = [
    {
        "key": "chroma-embedding-model",
        "label": "知识库默认嵌入",
        "description": "创建知识库和通用知识库问答时使用的嵌入模型。",
        "kind": "embedding",
    },
    {
        "key": "cluster-embedding-model",
        "label": "聚类分析嵌入",
        "description": "论文聚类相似度计算使用的嵌入模型。",
        "kind": "embedding",
    },
]

ALL_MODEL_DEFINITIONS = (
    DEFAULT_MODEL_DEFINITIONS
    + AGENT_MODEL_DEFINITIONS
    + EMBEDDING_MODEL_DEFINITIONS
)
MODEL_DEFINITION_BY_KEY = {item["key"]: item for item in ALL_MODEL_DEFINITIONS}
DEFAULT_MODEL_KEYS = {item["key"] for item in DEFAULT_MODEL_DEFINITIONS}


class ProviderPayload(BaseModel):
    id: str
    type: str = "custom"
    base_url: str = ""
    api_key: str = ""


class ModelConfigPayload(BaseModel):
    key: str
    provider: str = ""
    model: str = ""
    dimension: int | None = Field(default=None, ge=1)


class ModelSettingsPayload(BaseModel):
    providers: list[ProviderPayload] = Field(default_factory=list)
    default_models: list[ModelConfigPayload] = Field(default_factory=list)
    agent_models: list[ModelConfigPayload] = Field(default_factory=list)
    embedding_models: list[ModelConfigPayload] = Field(default_factory=list)


class TestModelPayload(BaseModel):
    provider: ProviderPayload
    model: ModelConfigPayload
    model_type: str = "llm"


def _load_models_yaml() -> dict[str, Any]:
    if not MODELS_CONFIG_PATH.exists():
        return {}

    try:
        with open(MODELS_CONFIG_PATH, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
            if not isinstance(data, dict):
                raise ValueError("models.yaml 顶层结构必须是对象")
            return data
    except Exception as exc:
        logger.error(f"读取模型配置失败: {exc}")
        raise HTTPException(status_code=500, detail=f"读取模型配置失败: {exc}") from exc


def _save_models_yaml(data: dict[str, Any]) -> None:
    tmp_path = MODELS_CONFIG_PATH.with_suffix(".yaml.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(
                data,
                file,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
        os.replace(tmp_path, MODELS_CONFIG_PATH)
    except Exception as exc:
        logger.error(f"保存模型配置失败: {exc}")
        raise HTTPException(status_code=500, detail=f"保存模型配置失败: {exc}") from exc
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _provider_api_key_env_name(provider_id: str) -> str:
    env_prefix = re.sub(r"[^A-Za-z0-9_]", "_", provider_id.strip()).upper()
    return f"{env_prefix}_API_KEY"


def _provider_api_key_env_ref(provider_id: str, provider_data: dict[str, Any]) -> str:
    configured_ref = str(provider_data.get("api_key", "")).strip()
    if ENV_VAR_NAME_PATTERN.fullmatch(configured_ref):
        return configured_ref
    return _provider_api_key_env_name(provider_id)


def _load_env_file_values() -> dict[str, str]:
    if not ENV_CONFIG_PATH.exists():
        return {}

    values = dotenv_values(ENV_CONFIG_PATH)
    return {key: str(value) for key, value in values.items() if value is not None}


def _format_env_assignment(key: str, value: str) -> str:
    if SAFE_ENV_VALUE_PATTERN.fullmatch(value):
        return f"{key}={value}"
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"{key}='{escaped}'"


def _update_env_file(updates: dict[str, str], remove_keys: set[str] | None = None) -> None:
    remove_keys = set(remove_keys or set())
    lines = ENV_CONFIG_PATH.read_text(encoding="utf-8").splitlines() if ENV_CONFIG_PATH.exists() else []
    next_lines: list[str] = []
    seen_updates: set[str] = set()

    for line in lines:
        match = ENV_LINE_PATTERN.match(line)
        key = match.group(1) if match else ""

        if key in remove_keys and key not in updates:
            continue

        if key in updates:
            next_lines.append(_format_env_assignment(key, updates[key]))
            seen_updates.add(key)
        else:
            next_lines.append(line)

    missing_updates = [key for key in updates if key not in seen_updates]
    if missing_updates and next_lines and next_lines[-1].strip():
        next_lines.append("")

    for key in missing_updates:
        next_lines.append(_format_env_assignment(key, updates[key]))

    ENV_CONFIG_PATH.write_text("\n".join(next_lines).rstrip() + "\n", encoding="utf-8")

    for key, value in updates.items():
        os.environ[key] = value
    for key in remove_keys - set(updates):
        os.environ.pop(key, None)


def _ensure_arxiv_api_url_config() -> None:
    raw: dict[str, Any] = {}
    if SYSTEM_PARAMS_PATH.exists():
        try:
            with open(SYSTEM_PARAMS_PATH, "r", encoding="utf-8") as file:
                loaded = yaml.safe_load(file) or {}
                if isinstance(loaded, dict):
                    raw = loaded
        except Exception as exc:
            logger.warning(f"读取 system_params.yaml 失败，跳过 ARXIV_API_URL 迁移: {exc}")
            return

    if raw.get("ARXIV_API_URL"):
        return

    raw["ARXIV_API_URL"] = ARXIV_API_URL
    tmp_path = SYSTEM_PARAMS_PATH.with_suffix(".yaml.tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            raw,
            file,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
    os.replace(tmp_path, SYSTEM_PARAMS_PATH)


def _save_provider_api_keys(payload: ModelSettingsPayload, existing: dict[str, Any]) -> None:
    old_provider_ids = existing.get("model-provider", [])
    if not isinstance(old_provider_ids, list):
        old_provider_ids = []

    old_env_names = set()
    for provider_id in old_provider_ids:
        provider_data = existing.get(str(provider_id), {})
        if not isinstance(provider_data, dict):
            provider_data = {}
        old_env_names.add(_provider_api_key_env_ref(str(provider_id), provider_data))

    updates = {
        _provider_api_key_env_name(provider.id): provider.api_key
        for provider in payload.providers
        if provider.api_key
    }
    current_env_names = {_provider_api_key_env_name(provider.id) for provider in payload.providers}
    remove_keys = old_env_names.union(current_env_names - set(updates)) - set(updates)
    remove_keys.add("ARXIV_API_URL")

    _ensure_arxiv_api_url_config()
    _update_env_file(updates, remove_keys)


def _provider_type(provider_id: str, provider_data: dict[str, Any]) -> str:
    configured_type = str(provider_data.get("type", "")).strip()
    if configured_type:
        return configured_type
    return provider_id if provider_id in KNOWN_PROVIDER_TYPES else "custom"


def _normalize_provider(provider_id: str, raw: dict[str, Any], env_values: dict[str, str]) -> dict[str, str]:
    provider_data = raw.get(provider_id, {})
    if not isinstance(provider_data, dict):
        provider_data = {}

    api_key_env = _provider_api_key_env_ref(provider_id, provider_data)
    return {
        "id": provider_id,
        "type": _provider_type(provider_id, provider_data),
        "base_url": str(provider_data.get("base_url", "")),
        "api_key": env_values.get(api_key_env, ""),
        "api_key_env": api_key_env,
    }


def _normalize_providers(raw: dict[str, Any], env_values: dict[str, str]) -> list[dict[str, str]]:
    provider_ids = raw.get("model-provider", [])
    if not isinstance(provider_ids, list):
        provider_ids = []

    providers = [_normalize_provider(str(provider_id), raw, env_values) for provider_id in provider_ids]
    if providers:
        return providers

    inferred = []
    for key, value in raw.items():
        if isinstance(value, dict) and {"api_key", "base_url"} <= set(value.keys()):
            inferred.append(_normalize_provider(str(key), raw, env_values))
    return inferred


def _model_fallback(raw: dict[str, Any], kind: str) -> dict[str, Any]:
    fallback_key = "default-embedding-model" if kind == "embedding" else "default-model"
    fallback = raw.get(fallback_key, {})
    return fallback if isinstance(fallback, dict) else {}


def _is_default_model_key(key: str) -> bool:
    return key in DEFAULT_MODEL_KEYS


def _normalize_model_config(raw: dict[str, Any], definition: dict[str, str]) -> dict[str, Any]:
    key = definition["key"]
    kind = definition["kind"]
    model_data = raw.get(key, {})
    if not isinstance(model_data, dict):
        model_data = {}

    providers = raw.get("model-provider", [])
    first_provider = providers[0] if isinstance(providers, list) and providers else ""

    has_model_provider = bool(model_data.get("model-provider"))
    has_model_name = bool(model_data.get("model"))

    if not _is_default_model_key(key) and (not has_model_provider or not has_model_name):
        item = {
            "key": key,
            "provider": "",
            "model": "",
        }

        if kind == "embedding":
            item["dimension"] = None

        return item

    fallback = _model_fallback(raw, kind) if _is_default_model_key(key) else {}

    item = {
        "key": key,
        "provider": str(
            model_data.get("model-provider")
            or fallback.get("model-provider")
            or first_provider
        ),
        "model": str(model_data.get("model") or fallback.get("model") or ""),
    }

    if kind == "embedding":
        dimension = model_data.get("dimension", fallback.get("dimension"))
        item["dimension"] = dimension if dimension is not None else None

    return item


def _model_items(raw: dict[str, Any], definitions: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            **definition,
            "config": _normalize_model_config(raw, definition),
        }
        for definition in definitions
    ]


def _settings_response(raw: dict[str, Any]) -> dict[str, Any]:
    env_values = _load_env_file_values()
    return {
        "providers": _normalize_providers(raw, env_values),
        "default_models": _model_items(raw, DEFAULT_MODEL_DEFINITIONS),
        "agent_models": _model_items(raw, AGENT_MODEL_DEFINITIONS),
        "embedding_models": _model_items(raw, EMBEDDING_MODEL_DEFINITIONS),
        "provider_types": PROVIDER_TYPE_OPTIONS,
        "models_path": str(MODELS_CONFIG_PATH),
    }


def _validate_provider_id(provider_id: str) -> None:
    if not provider_id:
        raise HTTPException(status_code=400, detail="Provider ID 不能为空")
    if not all(char.isalnum() or char in {"_", "-"} for char in provider_id):
        raise HTTPException(status_code=400, detail=f"Provider ID '{provider_id}' 只能包含字母、数字、下划线和短横线")


def _validate_settings(payload: ModelSettingsPayload) -> None:
    if not payload.providers:
        raise HTTPException(status_code=400, detail="至少需要配置一个模型 Provider")

    provider_ids = []
    for provider in payload.providers:
        provider.id = provider.id.strip()
        provider.type = provider.type.strip() or "custom"
        provider.base_url = provider.base_url.strip()
        provider.api_key = provider.api_key.strip()

        _validate_provider_id(provider.id)
        if provider.id in provider_ids:
            raise HTTPException(status_code=400, detail=f"Provider ID '{provider.id}' 重复")
        if not provider.base_url:
            raise HTTPException(status_code=400, detail=f"Provider '{provider.id}' 缺少 API URL")
        if provider.api_key and ENV_VAR_NAME_PATTERN.fullmatch(provider.api_key):
            api_key_env = _provider_api_key_env_name(provider.id)
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Provider '{provider.id}' 的 API Key 请填写真实密钥，不要填写环境变量名。"
                    f"保存后后端会自动写入 .env 的 {api_key_env}。"
                ),
            )
        provider_ids.append(provider.id)

    valid_provider_ids = set(provider_ids)
    all_models = payload.default_models + payload.agent_models + payload.embedding_models
    expected_keys = set(MODEL_DEFINITION_BY_KEY)

    for item in all_models:
        item.key = item.key.strip()
        item.provider = item.provider.strip()
        item.model = item.model.strip()

        if item.key not in expected_keys:
            raise HTTPException(status_code=400, detail=f"未知模型配置键: {item.key}")

        uses_default_model = not _is_default_model_key(item.key) and not item.provider and not item.model
        if uses_default_model:
            item.dimension = None
            continue

        if not _is_default_model_key(item.key) and (not item.provider or not item.model):
            raise HTTPException(status_code=400, detail=f"模型配置 '{item.key}' 请完整填写 Provider 和模型名称，或选择默认配置")

        if item.provider not in valid_provider_ids:
            raise HTTPException(status_code=400, detail=f"模型配置 '{item.key}' 使用了不存在的 Provider: {item.provider}")
        if not item.model:
            raise HTTPException(status_code=400, detail=f"模型配置 '{item.key}' 缺少模型名称")

        definition = MODEL_DEFINITION_BY_KEY[item.key]
        if definition["kind"] == "embedding" and item.dimension is not None and item.dimension <= 0:
            raise HTTPException(status_code=400, detail=f"嵌入模型 '{item.key}' 的维度必须大于 0")


def _model_payload_to_yaml(item: ModelConfigPayload) -> dict[str, Any]:
    definition = MODEL_DEFINITION_BY_KEY[item.key]
    data: dict[str, Any] = {
        "model-provider": item.provider,
        "model": item.model,
    }

    if definition["kind"] == "embedding" and item.dimension is not None:
        data["dimension"] = item.dimension

    return data


def _uses_default_model_payload(item: ModelConfigPayload) -> bool:
    return not _is_default_model_key(item.key) and not item.provider and not item.model


def _payload_to_yaml(payload: ModelSettingsPayload, existing: dict[str, Any]) -> dict[str, Any]:
    next_config = dict(existing)
    old_provider_ids = set(existing.get("model-provider", [])) if isinstance(existing.get("model-provider"), list) else set()
    next_provider_ids = [provider.id for provider in payload.providers]

    next_config["model-provider"] = next_provider_ids

    for provider_id in old_provider_ids - set(next_provider_ids):
        next_config.pop(provider_id, None)

    for provider in payload.providers:
        next_config[provider.id] = {
            "type": provider.type,
            "api_key": _provider_api_key_env_name(provider.id),
            "base_url": provider.base_url,
        }

    for item in payload.default_models:
        next_config[item.key] = _model_payload_to_yaml(item)

    for item in payload.agent_models + payload.embedding_models:
        if _uses_default_model_payload(item):
            next_config.pop(item.key, None)
        else:
            next_config[item.key] = _model_payload_to_yaml(item)

    return next_config


@configuration.get("/model-settings")
async def get_model_settings():
    """获取模型配置页面所需的规范化配置。"""
    raw = _load_models_yaml()
    return _settings_response(raw)


@configuration.put("/model-settings")
async def save_model_settings(payload: ModelSettingsPayload):
    """保存模型配置到 models.yaml，并刷新运行时配置。"""
    _validate_settings(payload)
    existing = _load_models_yaml()
    next_config = _payload_to_yaml(payload, existing)
    _save_provider_api_keys(payload, existing)
    _save_models_yaml(next_config)
    config.reload()
    logger.info("模型配置已通过配置页面保存并重新加载")
    return {
        "message": "模型配置已写入 models.yaml，并已重新加载运行时配置",
        "settings": _settings_response(next_config),
    }


@configuration.post("/model-settings/test")
async def test_model_connectivity(payload: TestModelPayload):
    """测试一张模型配置卡片的 OpenAI-compatible 连通性。"""
    provider = payload.provider
    model = payload.model
    model_type = payload.model_type.strip().lower()

    if model_type not in {"llm", "embedding"}:
        raise HTTPException(status_code=400, detail="model_type 只能是 llm 或 embedding")

    base_url = provider.base_url.strip()
    model_name = model.model.strip()
    api_key = provider.api_key.strip()

    if not base_url:
        return {"status": "failed", "message": "API URL 为空，无法测试连通性"}
    if not api_key:
        return {"status": "failed", "message": "API Key 为空，无法测试连通性"}
    if ENV_VAR_NAME_PATTERN.fullmatch(api_key):
        return {
            "status": "failed",
            "message": "API Key 请填写真实密钥，不要填写环境变量名。",
        }
    if not model_name:
        return {"status": "failed", "message": "模型名称为空，无法测试连通性"}

    started_at = time.perf_counter()
    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=20.0)
        if model_type == "embedding":
            params: dict[str, Any] = {
                "model": model_name,
                "input": "Paper-Agent connectivity test",
            }
            if model.dimension:
                params["dimensions"] = model.dimension
            response = client.embeddings.create(**params)
            dimension = len(response.data[0].embedding) if response.data else 0
            latency_ms = round((time.perf_counter() - started_at) * 1000)
            return {
                "status": "success",
                "message": f"Embedding 连通成功，返回向量维度 {dimension}",
                "latency_ms": latency_ms,
                "dimension": dimension,
            }

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with OK for a Paper-Agent connectivity test.",
                }
            ],
            temperature=0,
            max_tokens=12,
        )
        content = response.choices[0].message.content if response.choices else ""
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        return {
            "status": "success",
            "message": f"LLM 连通成功，响应: {content or 'OK'}",
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        logger.warning(f"模型连通性测试失败: {exc}")
        return {
            "status": "failed",
            "message": str(exc),
            "latency_ms": latency_ms,
        }
