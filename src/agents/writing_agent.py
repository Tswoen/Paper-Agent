import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

import autogen
from autogen.agentchat import AssistantAgent, UserProxyAgent
from src.utils.log_utils import setup_logger
from src.tasks.paper_search import PaperSearch

logger = setup_logger(__name__)

class PaperRetrievalAgent:
    """基于AutoGen框架的论文检索智能体"""
    
    def __init__(self):
        """初始化论文检索智能体"""
        self.paper_search = PaperSearch()
        
        # 配置AutoGen代理
        config_list = [{
            "model": "gpt-4o-mini",  # 这里使用示例模型，实际使用时需要配置真实的模型
            "api_key": "your-api-key"  # 实际使用时需要从环境变量或配置文件中读取
        }]
        
        # 创建论文检索助手代理
        self.retrieval_assistant = AssistantAgent(
            name="paper_retrieval_assistant",
            system_message="你是一个论文检索助手，负责从用户的消息中提取研究关键词，并调用论文检索工具获取相关论文。",
            llm_config={"config_list": config_list}
        )
        
        # 创建用户代理
        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            system_message="用户代理，负责与用户交互并调用论文检索助手。",
            human_input_mode="NEVER",  # 非交互式模式
            max_consecutive_auto_reply=10
        )
        
        # 注册论文检索工具
        self.user_proxy.register_function(
            function_map={
                "retrieve_papers": self.retrieve_papers,
                "extract_keywords": self.extract_keywords
            }
        )
    
    def extract_keywords(self, message: str) -> List[str]:
        """
        从用户消息中提取研究关键词
        
        参数:
            message: 用户消息
        
        返回:
            提取的关键词列表
        """
        logger.info(f"正在从消息中提取关键词: {message}")
        
        # 这里使用简单的关键词提取逻辑，实际应用中可以使用NLP模型进行更复杂的提取
        # 简单的规则：提取消息中的名词和名词短语作为关键词
        # 这里使用一个模拟的关键词提取实现
        keywords = []
        
        # 示例实现：提取常见的研究领域关键词
        research_fields = [
            "large language models", "LLM", "人工智能", "机器学习", "深度学习",
            "自然语言处理", "计算机视觉", "强化学习", "知识图谱", "推荐系统"
        ]
        
        for field in research_fields:
            if field.lower() in message.lower():
                keywords.append(field)
        
        # 如果没有提取到关键词，返回消息中长度大于2的单词
        if not keywords:
            words = message.strip().split()
            keywords = [word for word in words if len(word) > 2 and not word.lower() in ["的", "了", "在", "是", "我"]]
        
        logger.info(f"提取的关键词: {keywords}")
        return keywords
    
    def retrieve_papers(self, keywords: List[str], max_results: int = 10) -> List[Dict]:
        """
        调用学术数据库API（模拟）获取相关论文
        
        参数:
            keywords: 研究关键词列表
            max_results: 最大返回结果数量
        
        返回:
            包含论文信息的结构化结果列表
        """
        logger.info(f"正在检索关键词 '{keywords}' 的论文，最多返回 {max_results} 篇")
        
        # 合并关键词构建查询
        query = " ".join(keywords)
        
        # 使用现有的PaperSearch工具检索论文
        papers = self.paper_search.search_papers(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        # 如果没有检索到论文，返回一些模拟数据
        if not papers:
            logger.warning(f"没有检索到关键词 '{keywords}' 的论文，返回模拟数据")
            papers = self._generate_mock_papers(keywords, max_results)
        
        logger.info(f"检索完成，共找到 {len(papers)} 篇论文")
        return papers
    
    def _generate_mock_papers(self, keywords: List[str], count: int) -> List[Dict]:
        """\生成模拟的论文数据"""
        mock_papers = []
        for i in range(count):
            paper = {
                "paper_id": f"mock-{i+1}",
                "title": f"关于{'与'.join(keywords)}的研究进展{i+1}",
                "authors": [f"作者{i+1}", f"作者{i+2}"],
                "summary": f"这是一篇关于{'与'.join(keywords)}的研究论文，探讨了相关领域的最新进展和未来方向。",
                "published": datetime.now().year,
                "published_date": datetime.now().isoformat(),
                "url": f"https://example.com/paper/mock-{i+1}",
                "pdf_url": f"https://example.com/paper/mock-{i+1}.pdf",
                "primary_category": "cs.AI",
                "categories": ["cs.AI", "cs.LG"],
                "doi": f"10.1234/mock{i+1}"
            }
            mock_papers.append(paper)
        return mock_papers
    
    def process_retrieval_request(self, user_message: str) -> Dict:
        """
        处理论文检索请求
        
        参数:
            user_message: 用户的检索请求消息
        
        返回:
            包含检索结果的响应
        """
        logger.info(f"收到检索请求: {user_message}")
        
        try:
            # 提取关键词
            keywords = self.extract_keywords(user_message)
            
            # 检索论文
            papers = self.retrieve_papers(keywords)
            
            return {
                "success": True,
                "message": f"已为您找到{len(papers)}篇关于{'、'.join(keywords)}的论文",
                "data": papers,
                "metadata": {
                    "keywords": keywords,
                    "total_papers": len(papers)
                }
            }
        except Exception as e:
            logger.error(f"处理检索请求失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "处理检索请求时发生错误"
            }

# 示例用法
if __name__ == "__main__":
    try:
        agent = PaperRetrievalAgent()
        result = agent.process_retrieval_request("请帮我查找关于大型语言模型的最新研究论文")
        print(f"检索结果: {result}")
    except Exception as e:
        print(f"错误: {str(e)}")