import os
from typing import Any, List, Dict, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pathlib import Path
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from src.core.config import config


class ChromaClient:
    """
    ChromaDB客户端封装类，提供文档嵌入、存储和查询功能
    """
    
    def __init__(self, 
                 collection_name: str = "default_collection",
                 embedding_model: str = "Qwen/Qwen3-Embedding-8B"):
        """
        初始化Chroma客户端
        
        :param collection_name: 集合名称
        :param persist_directory: 数据持久化目录
        :param embedding_model: 嵌入模型名称
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # 创建Chroma客户端
        self.client = chromadb.PersistentClient(path=Path(__file__).parent.parent.parent / "data" / "chromadb")
        
        # 初始化嵌入函数
        self.embedding_function = self.create_embedding_client()
    
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def create_embedding_client(self) -> OpenAIEmbeddingFunction:
        try:
            model_config = config.get("chroma-embedding-model", {})
            provider = model_config.get("model-provider")
            model = model_config.get("model")
            
            # 检查是否配置了阅读模型
            if not provider or not model:
                return self.create_default_embedding_client()

            provider_config = config.get(provider)

            # 如果未提供参数，则使用配置中的默认值
            api_key = provider_config.get("api_key")
            base_url = provider_config.get("base_url")

            return OpenAIEmbeddingFunction(
                    model_name=model,
                    api_key=api_key,
                    api_base=base_url,
                )
            
            
        except Exception as e:
            print(f"创建嵌入模型客户端失败: {e}，使用默认模型代替")
            return self.create_default_embedding_client()

    def create_default_embedding_client() -> OpenAIEmbeddingFunction:
        default_model_config = config.get("default-embedding-model", {})
        provider = default_model_config.get("model-provider", "siliconflow")
        model = default_model_config.get("model", "Qwen/Qwen3-Embedding-8B")
        provider_config = config.get(provider)

        # 如果未提供参数，则使用配置中的默认值
        api_key = provider_config.get("api_key")
        base_url = provider_config.get("base_url")

        return OpenAIEmbeddingFunction(
                model_name=model,
                api_key=api_key,
                api_base=base_url,
            )
    def add_documents(self, 
                     documents: List[str], 
                     metadatas: Optional[List[dict]] = None, 
                     ids: Optional[List[str]] = None) -> None:
        """
        添加文档到集合
        
        :param documents: 文档内容列表
        :param metadatas: 元数据列表(可选)
        :param ids: 文档ID列表(可选)
        """
        if not ids:
            ids = [str(i) for i in range(len(documents))]
            
        if not metadatas:
            metadatas = [{} for _ in documents]

        metadatas = [self.safe_metadata_conversion(metadata) for metadata in metadatas]

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
    
    def query(self, 
              query_texts: List[str], 
              n_results: int = 5, 
              where: Optional[Dict] = None) -> Dict:
        """
        查询相似文档
        
        :param query_texts: 查询文本列表
        :param n_results: 返回结果数量
        :param where: 过滤条件(可选)
        :return: 查询结果字典
        """
        return self.collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            include=["metadatas"]
        )
    
    def delete_collection(self) -> None:
        """删除当前集合"""
        self.client.delete_collection(name=self.collection_name)
    
    def reset(self) -> None:
        """重置客户端(删除所有数据)"""
        self.client.reset()
    
    def get_collection_stats(self) -> Dict:
        """
        获取集合统计信息
        
        :return: 包含集合统计信息的字典
        """
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata
        }
    
    def safe_metadata_conversion(self, data):
        """安全地将数据转换为 ChromaDB 兼容的元数据"""
        if hasattr(data, 'model_dump'):
            data = data.model_dump()
        
        metadata = {}
        
        for key, value in data.items():
            # 处理 None 值，转换为空字符串
            if value is None:
                metadata[key] = ""
                continue
                
            # 跳过复杂的嵌套结构
            if isinstance(value, (dict, list, tuple, set)):
                if isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
                    # 简单列表转换为字符串
                    metadata[key] = ", ".join(str(item) for item in value)
                else:
                    # 复杂结构跳过或转换为JSON字符串
                    try:
                        metadata[key] = json.dumps(value, ensure_ascii=False)
                    except:
                        metadata[key] = str(value)
            elif isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                # 其他类型转换为字符串
                metadata[key] = str(value)
        
        return metadata