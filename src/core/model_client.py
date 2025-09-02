from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from .config import Config
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

class ModelClient:
    """OpenAIChatCompletionClient的封装类，简化模型客户端的创建和配置"""
    
    @staticmethod
    def create_client(
        provider: str = "siliconflow",
        model: str = "Qwen/Qwen3-32B",
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
            family: 模型家族名称
            
        返回:
            配置好的OpenAIChatCompletionClient实例
        """
        # 从配置中加载默认值
        config = Config()
        provider_config = config.get(provider)

        # 如果未提供参数，则使用配置中的默认值
        model = model or provider_config.get("model")
        api_key = api_key or provider_config.get("api_key")
        base_url = base_url or provider_config.get("base_url")
        
        # 验证必要参数
        if not model:
            raise ValueError(f"未指定模型名称，请在参数中提供或在配置文件中设置{provider.upper()}_MODEL")
        if not base_url:
            raise ValueError(f"未指定API基础URL，请在参数中提供或在配置文件中设置{provider.upper()}_BASE_URL")
        
        # 创建ModelInfo
        model_info = ModelInfo(
            vision=vision,
            function_calling=function_calling,
            json_output=json_output,
            family=family,
            structured_output=structured_output
        )
        
        logger.info(f"创建OpenAIChatCompletionClient - 提供商: {provider}, 模型: {model}")
        logger.info(f"模型api_key: {api_key}")
        logger.info(f"模型base_url: {base_url}")
        # 创建并返回客户端实例
        return OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            model_info=model_info
        )


# 工厂函数：快速创建默认客户端

def create_default_client() -> OpenAIChatCompletionClient:
    """创建默认的OpenAIChatCompletionClient实例，使用siliconflow提供商和Qwen3-32B模型"""
    return ModelClient.create_client(
        provider="siliconflow",
        model="Qwen/Qwen3-32B",
        base_url="https://api.siliconflow.cn/v1",
        family="Qwen"
    )
