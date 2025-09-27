from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from .config import config
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

class ModelClient:
    """OpenAIChatCompletionClient的封装类，简化模型客户端的创建和配置"""
    
    @staticmethod
    def create_client(
        provider: str = "siliconflow",
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

        # 如果未提供参数，则使用配置中的默认值
        api_key = api_key or provider_config.get("api_key")
        base_url = base_url or provider_config.get("base_url")
        
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

def create_model_client(client_type: str) -> OpenAIChatCompletionClient:
    try:
        """创建用于阅读论文的客户端实例"""
        model_config = config.get("reading-model", {})
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

def create_default_client() -> OpenAIChatCompletionClient:
    """创建默认的OpenAIChatCompletionClient实例，使用配置中指定的默认模型"""
    default_model_config = config.get("default-model", {})
    provider = default_model_config.get("model-provider", "siliconflow")
    model = default_model_config.get("model", "Qwen/Qwen3-32B")
    
    return ModelClient.create_client(
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

if __name__ == "__main__":
    client = create_report_model_client()
    print(client)
    
