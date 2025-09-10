import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


class ChromaClient:
    """
    ChromaDB客户端封装类，提供文档嵌入、存储和查询功能
    """
    
    def __init__(self, 
                 collection_name: str = "default_collection",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        初始化Chroma客户端
        
        :param collection_name: 集合名称
        :param persist_directory: 数据持久化目录
        :param embedding_model: 嵌入模型名称
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        # 创建Chroma客户端
        self.client = chromadb.PersistentClient(path=Path(__file__).parent.parent.parent / "data" / "chromadb")
        
        # 初始化嵌入函数
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
    
    def add_documents(self, 
                     documents: List[str], 
                     metadatas: Optional[List[Dict]] = None, 
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
            
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        # 持久化数据
        self.client.persist()
    
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
            where=where
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
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.persist()
        self.client = None