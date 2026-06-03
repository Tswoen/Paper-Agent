import os
import re

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from .config import config
from src.utils.log_utils import setup_logger
from openai import OpenAI


logger = setup_logger(__name__)

ENV_VAR_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def resolve_api_key(provider: str, api_key: str | None) -> str:
    """Resolve real API keys and fail early when an env-var reference is missing."""
    api_key = str(api_key or "").strip()
    if not api_key:
        raise ValueError(f"Provider '{provider}' 未配置 API Key 或环境变量名")

    resolved_key = os.environ.get(api_key) or config.get(api_key)
    if resolved_key:
        return str(resolved_key).strip()

    if ENV_VAR_NAME_PATTERN.fullmatch(api_key):
        raise ValueError(
            f"Provider '{provider}' 的 API Key 配置为环境变量 '{api_key}'，"
            "但当前后端进程未读取到该环境变量。请在 .env 中设置该变量，或在系统配置中填写真实 API Key。"
        )

    return api_key

class ModelClient:
    """OpenAIChatCompletionClient的封装类，简化模型客户端的创建和配置"""
    
    @staticmethod
    def create_client(
        provider: str = None,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        vision: bool = True,
        function_calling: bool = True,
        json_output: bool = True,
        structured_output: bool = True,
        family: str = "Qwen"
    ) -> OpenAIChatCompletionClient:
        """
        创建并返回一个配置好的OpenAIChatCompletionClient实例
        
        参数:
            provider: 模型提供商，如'siliconflow', 'openai'等
            model: 模型名称，如果为None则从配置中获取
            api_key: API密钥，如果为None则从配置中获取
            base_url: API基础URL，如果为None则从配置中获取
            vision: 是否支持视觉功能
            function_calling: 是否支持函数调用
            json_output: 是否支持JSON输出
            structured_output: 是否支持结构化输出
            family: 模型家族名称，默认根据provider设置
            
        返回:
            配置好的OpenAIChatCompletionClient实例
        """
        # 从配置中加载默认值
        provider_config = config.get(provider)
        if not isinstance(provider_config, dict):
            raise ValueError(f"未找到模型 Provider 配置: {provider}")

        # 如果未提供参数，则使用配置中的默认值
        api_key = api_key or provider_config.get("api_key")
        base_url = base_url or provider_config.get("base_url")
        api_key = resolve_api_key(provider, api_key)
        
        # 根据provider设置默认family
        if family == "Qwen" and provider != "siliconflow":
            family = "GPT" if provider == "openai" else provider.capitalize()
        
        # 验证必要参数
        if not model:
            raise ValueError(f"未指定模型名称，请在参数中提供或在配置文件中设置{provider}.model")
        if not base_url:
            raise ValueError(f"未指定API基础URL，请在参数中提供或在配置文件中设置{provider}.base_url")
        
        # 创建ModelInfo
        model_info = ModelInfo(
            vision=vision,
            function_calling=function_calling,
            json_output=json_output,
            family=family,
            structured_output=structured_output
        )
        
        # 创建并返回客户端实例
        return OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            model_info=model_info
        )

    @staticmethod
    def create_embedding_client(
        provider: str = None,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
    ) -> OpenAI:
        provider_config = config.get(provider)
        if not isinstance(provider_config, dict):
            raise ValueError(f"未找到模型 Provider 配置: {provider}")

        # 如果未提供参数，则使用配置中的默认值
        api_key = api_key or provider_config.get("api_key")
        base_url = base_url or provider_config.get("base_url")
        api_key = resolve_api_key(provider, api_key)
        
        # 验证必要参数
        if not model:
            raise ValueError(f"未指定模型名称，请在参数中提供或在配置文件中设置{provider}.model")
        if not base_url:
            raise ValueError(f"未指定API基础URL，请在参数中提供或在配置文件中设置{provider}.base_url")

        client = OpenAI(
                api_key= api_key,
                base_url=base_url,
                default_headers={
                    "X-Model": model  # 设置默认模型
                }
        )
        return client


def create_model_client(client_type: str) -> OpenAIChatCompletionClient:
    try:
        model_config = config.get(client_type, {})
        provider = model_config.get("model-provider")
        model = model_config.get("model")

        # 检查是否配置了阅读模型
        if not provider or not model:
            logger.warning(f"警告：未配置{client_type}模型，使用默认模型代替")
            return create_default_client()
        
        return ModelClient.create_client(
                provider=provider,
                model=model
        )
    except Exception as e:
        print(f"创建阅读模型客户端失败: {e}，使用默认模型代替")
        return create_default_client()

def create_embedding_client(client_type: str) -> OpenAI:
    try:
        model_config = config.get(client_type, {})
        provider = model_config.get("model-provider")
        model = model_config.get("model")

        # 检查是否配置了阅读模型
        if not provider or not model:
            logger.warning(f"警告：未配置{client_type}模型，使用默认模型代替")
            return create_default_embedding_client()
        
        return ModelClient.create_embedding_client(
                provider=provider,
                model=model
        )
    except Exception as e:
        print(f"创建{client_type}模型客户端失败: {e}，使用默认模型代替")
        return create_default_embedding_client()

def create_default_client() -> OpenAIChatCompletionClient:
    """创建默认的OpenAIChatCompletionClient实例，使用配置中指定的默认模型"""
    default_model_config = config.get("default-model", {})
    provider = default_model_config.get("model-provider", "siliconflow")
    model = default_model_config.get("model", "Qwen/Qwen3-32B")
    
    return ModelClient.create_client(
        provider=provider,
        model=model
    )

def create_default_embedding_client() -> OpenAI:
    """创建默认的OpenAIEmbeddingClient实例，使用配置中指定的默认模型"""
    default_model_config = config.get("default-embedding-model", {})
    provider = default_model_config.get("model-provider", "siliconflow")
    model = default_model_config.get("model", "Qwen/Qwen3-Embedding-8B")
    
    return ModelClient.create_embedding_client(
        provider=provider,
        model=model
    )

def create_search_model_client() -> OpenAIChatCompletionClient:
    """创建用于搜索的模型客户端实例"""
    return create_model_client("search-model")

def create_reading_model_client() -> OpenAIChatCompletionClient:
    """创建用于阅读论文的模型客户端实例"""
    return create_model_client("reading-model")

def create_subanalyse_cluster_model_client() -> OpenAIChatCompletionClient:
    """创建用于分析聚类的模型客户端实例"""
    return create_model_client("subanalyse-cluster-model")

def create_subanalyse_deep_analyse_model_client() -> OpenAIChatCompletionClient:
    """创建用于深度分析的模型客户端实例"""
    return create_model_client("subanalyse-deep-analyse-model")

def create_subanalyse_global_analyse_model_client() -> OpenAIChatCompletionClient:
    """创建用于全局分析的模型客户端实例"""
    return create_model_client("subanalyse-global-analyse-model") 

def create_subwriting_writing_director_model_client() -> OpenAIChatCompletionClient:
    """创建用于写作主管的模型客户端实例"""
    return create_model_client("subwriting-writing-director-model") 

def create_subwriting_writing_model_client() -> OpenAIChatCompletionClient:
    """创建用于写作的模型客户端实例"""
    return create_model_client("subwriting-writing-model") 

def create_subwriting_retrieval_model_client() -> OpenAIChatCompletionClient:
    """创建用于检索的模型客户端实例"""
    return create_model_client("subwriting-retrieval-model") 

def create_report_model_client() -> OpenAIChatCompletionClient:
    """创建用于写作报告的模型客户端实例"""
    return create_model_client("report-model")

def create_cluster_embedding_client() -> OpenAI:
    """创建用于聚类嵌入的模型客户端实例"""
    return create_embedding_client("cluster-embedding-model")


# ===================重排序模型===================
# import json

# import numpy as np
# import requests

# from src.utils import get_docker_safe_url

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


# class OnlineReranker:
#     def __init__(self, model_name, api_key, base_url, **kwargs):
#         self.url = get_docker_safe_url(base_url)
#         self.model = model_name
#         self.api_key = api_key
#         self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

#     def compute_score(self, sentence_pairs, batch_size=256, max_length=512, normalize=False):
#         # TODO 还没实现 batch_size
#         query, sentences = sentence_pairs[0], sentence_pairs[1]
#         payload = self.build_payload(query, sentences, max_length)
#         response = requests.request("POST", self.url, json=payload, headers=self.headers)
#         response = json.loads(response.text)
#         # logger.debug(f"SiliconFlow Reranker response: {response}")

#         results = sorted(response["results"], key=lambda x: x["index"])
#         all_scores = [result["relevance_score"] for result in results]

#         if normalize:
#             all_scores = [sigmoid(score) for score in all_scores]

#         return all_scores

#     def build_payload(self, query, sentences, max_length=512):
#         return {
#             "model": self.model,
#             "query": query,
#             "documents": sentences,
#             "max_chunks_per_doc": max_length,
#         }


# def get_reranker(model_id, **kwargs):
#     support_rerankers = config.reranker_names.keys()
#     assert model_id in support_rerankers, f"Unsupported Reranker: {model_id}, only support {support_rerankers}"

#     model_info = config.reranker_names[model_id]
#     base_url = model_info["base_url"]
#     api_key = os.getenv(model_info["api_key"], model_info["api_key"])
#     assert api_key, f"{model_info['name']} api_key is required"
#     return OnlineReranker(model_name=model_info["name"], api_key=api_key, base_url=base_url, **kwargs)



if __name__ == "__main__":
    client = create_report_model_client()
    print(client)
    
