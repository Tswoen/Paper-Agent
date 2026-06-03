import os
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from src.core.config import config
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

configuration = APIRouter(prefix="/config", tags=["configuration"])

MODELS_CONFIG_PATH = Path(__file__).parent / "models.yaml"

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
        "key": "subwriting-retrieval-model",
        "label": "写作检索",
        "description": "写作过程中调用知识库检索辅助资料。",
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
        "key": "embedding-model",
        "label": "知识库默认嵌入",
        "description": "创建知识库和通用知识库问答时使用的嵌入模型。",
        "kind": "embedding",
    },
    {
        "key": "chroma-embedding-model",
        "label": "Chroma 向量库",
        "description": "ChromaDB 文档入库和相似度检索使用的嵌入模型。",
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


def _provider_type(provider_id: str, provider_data: dict[str, Any]) -> str:
    configured_type = str(provider_data.get("type", "")).strip()
    if configured_type:
        return configured_type
    return provider_id if provider_id in KNOWN_PROVIDER_TYPES else "custom"


def _normalize_provider(provider_id: str, raw: dict[str, Any]) -> dict[str, str]:
    provider_data = raw.get(provider_id, {})
    if not isinstance(provider_data, dict):
        provider_data = {}

    return {
        "id": provider_id,
        "type": _provider_type(provider_id, provider_data),
        "base_url": str(provider_data.get("base_url", "")),
        "api_key": str(provider_data.get("api_key", "")),
    }


def _normalize_providers(raw: dict[str, Any]) -> list[dict[str, str]]:
    provider_ids = raw.get("model-provider", [])
    if not isinstance(provider_ids, list):
        provider_ids = []

    providers = [_normalize_provider(str(provider_id), raw) for provider_id in provider_ids]
    if providers:
        return providers

    inferred = []
    for key, value in raw.items():
        if isinstance(value, dict) and {"api_key", "base_url"} <= set(value.keys()):
            inferred.append(_normalize_provider(str(key), raw))
    return inferred


def _model_fallback(raw: dict[str, Any], kind: str) -> dict[str, Any]:
    fallback_key = "default-embedding-model" if kind == "embedding" else "default-model"
    fallback = raw.get(fallback_key, {})
    return fallback if isinstance(fallback, dict) else {}


def _normalize_model_config(raw: dict[str, Any], definition: dict[str, str]) -> dict[str, Any]:
    key = definition["key"]
    kind = definition["kind"]
    model_data = raw.get(key, {})
    if not isinstance(model_data, dict):
        model_data = {}

    fallback = _model_fallback(raw, kind)
    providers = raw.get("model-provider", [])
    first_provider = providers[0] if isinstance(providers, list) and providers else ""

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
    return {
        "providers": _normalize_providers(raw),
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
        if not provider.api_key:
            raise HTTPException(status_code=400, detail=f"Provider '{provider.id}' 缺少 API Key 或环境变量名")
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
            "api_key": provider.api_key,
            "base_url": provider.base_url,
        }

    for item in payload.default_models + payload.agent_models + payload.embedding_models:
        next_config[item.key] = _model_payload_to_yaml(item)

    return next_config


def _resolve_api_key(api_key: str) -> str:
    return os.environ.get(api_key, api_key)


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
    _save_models_yaml(next_config)
    config.reload()
    logger.info("模型配置已通过配置页面保存并重新加载")
    return {
        "message": "模型配置已保存",
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

    api_key = _resolve_api_key(provider.api_key.strip())
    base_url = provider.base_url.strip()
    model_name = model.model.strip()

    if not api_key:
        return {"status": "failed", "message": "API Key 为空，无法测试连通性"}
    if not base_url:
        return {"status": "failed", "message": "API URL 为空，无法测试连通性"}
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
