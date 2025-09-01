import logging
from typing import Dict, List, Optional

import autogen
from autogen.agentchat import AssistantAgent, UserProxyAgent
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

class PaperAnalysisAgent:
    """基于AutoGen框架的论文分析智能体"""
    
    def __init__(self):
        """初始化论文分析智能体"""
        # 配置AutoGen代理
        config_list = [{
            "model": "gpt-4o-mini",  # 示例模型
            "api_key": "your-api-key"  # 实际使用时需要配置
        }]
        
        # 创建论文分析助手代理
        self.analysis_assistant = AssistantAgent(
            name="paper_analysis_assistant",
            system_message="你是一个论文分析专家，负责分析论文的内容、结构和贡献，提取关键信息和发现。",
            llm_config={"config_list": config_list}
        )
        
        # 创建用户代理
        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            system_message="用户代理，负责与用户交互并调用论文分析助手。",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10
        )
        
        # 注册论文分析工具
        self.user_proxy.register_function(
            function_map={
                "analyze_paper": self.analyze_paper,
                "summarize_paper": self.summarize_paper,
                "extract_key_findings": self.extract_key_findings
            }
        )
    
    def analyze_paper(self, paper: Dict) -> Dict:
        """
        分析论文的详细内容
        
        参数:
            paper: 论文信息字典
        
        返回:
            包含论文分析结果的字典
        """
        logger.info(f"正在分析论文: {paper.get('title', '未知标题')}")
        
        # 模拟论文分析过程
        # 实际应用中可以使用更复杂的NLP模型进行深入分析
        analysis = {
            "paper_id": paper.get('paper_id'),
            "title": paper.get('title'),
            "authors": paper.get('authors', []),
            "summary": paper.get('summary', ''),
            "content_structure": "摘要、引言、方法、实验、结论",
            "contributions": ["提出了新的方法", "取得了更好的实验结果"],
            "limitations": ["数据集规模有限", "计算资源要求较高"],
            "keywords": self._extract_keywords_from_paper(paper),
            "technical_depth": "中等",
            "novelty": "较高",
            "relevance": "与当前研究趋势高度相关"
        }
        
        logger.info(f"论文分析完成: {paper.get('title', '未知标题')}")
        return analysis
    
    def summarize_paper(self, paper: Dict, max_length: int = 300) -> Dict:
        """
        生成论文摘要
        
        参数:
            paper: 论文信息字典
            max_length: 摘要最大长度
        
        返回:
            包含论文摘要的字典
        """
        logger.info(f"正在生成论文摘要: {paper.get('title', '未知标题')}")
        
        # 模拟生成摘要
        original_summary = paper.get('summary', '')
        summary = original_summary[:max_length] + '...' if len(original_summary) > max_length else original_summary
        
        result = {
            "paper_id": paper.get('paper_id'),
            "title": paper.get('title'),
            "summary": summary,
            "length": len(summary)
        }
        
        logger.info(f"论文摘要生成完成: {paper.get('title', '未知标题')}")
        return result
    
    def extract_key_findings(self, paper: Dict) -> List[Dict]:
        """
        提取论文的关键发现
        
        参数:
            paper: 论文信息字典
        
        返回:
            关键发现列表
        """
        logger.info(f"正在提取论文关键发现: {paper.get('title', '未知标题')}")
        
        # 模拟提取关键发现
        # 实际应用中可以使用NLP技术从论文摘要和内容中提取
        findings = [
            {
                "id": 1,
                "description": "提出了一种新的方法来解决当前问题",
                "relevance": "高"
            },
            {
                "id": 2,
                "description": "在多个数据集上取得了SOTA性能",
                "relevance": "高"
            },
            {
                "id": 3,
                "description": "发现了一些有趣的现象或规律",
                "relevance": "中等"
            }
        ]
        
        logger.info(f"论文关键发现提取完成: {paper.get('title', '未知标题')}")
        return findings
    
    def _extract_keywords_from_paper(self, paper: Dict) -> List[str]:
        """从论文中提取关键词"""
        # 简单实现：从标题和摘要中提取关键词
        title = paper.get('title', '').lower()
        summary = paper.get('summary', '').lower()
        
        # 常见研究领域关键词
        research_keywords = [
            "large language models", "llm", "人工智能", "机器学习", "深度学习",
            "自然语言处理", "计算机视觉", "强化学习", "知识图谱", "推荐系统",
            "transformer", "bert", "gpt", "生成模型", "预训练模型"
        ]
        
        keywords = []
        for keyword in research_keywords:
            if keyword in title or keyword in summary:
                keywords.append(keyword)
        
        return list(set(keywords))  # 去重
    
    def process_analysis_request(self, paper: Dict, analysis_type: str = "full") -> Dict:
        """
        处理论文分析请求
        
        参数:
            paper: 论文信息字典
            analysis_type: 分析类型 (full, summary, findings)
        
        返回:
            包含分析结果的响应
        """
        logger.info(f"收到论文分析请求: {paper.get('title', '未知标题')}, 分析类型: {analysis_type}")
        
        try:
            if analysis_type == "full":
                result = self.analyze_paper(paper)
            elif analysis_type == "summary":
                result = self.summarize_paper(paper)
            elif analysis_type == "findings":
                result = self.extract_key_findings(paper)
            else:
                result = self.analyze_paper(paper)  # 默认进行全面分析
            
            return {
                "success": True,
                "message": f"论文{analysis_type}分析完成",
                "data": result,
                "metadata": {
                    "paper_id": paper.get('paper_id'),
                    "analysis_type": analysis_type
                }
            }
        except Exception as e:
            logger.error(f"处理论文分析请求失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "处理论文分析请求时发生错误"
            }

# 示例用法
if __name__ == "__main__":
    try:
        agent = PaperAnalysisAgent()
        # 创建一个模拟论文对象
        mock_paper = {
            "paper_id": "12345",
            "title": "大型语言模型在自然语言处理中的应用研究",
            "authors": ["张三", "李四"],
            "summary": "本文探讨了大型语言模型在自然语言处理任务中的应用，包括文本分类、情感分析、机器翻译等多个任务。实验结果表明，大型语言模型在这些任务上都取得了显著的性能提升。",
            "published": 2024,
            "published_date": "2024-01-01T00:00:00",
            "url": "https://example.com/paper/12345",
            "pdf_url": "https://example.com/paper/12345.pdf",
            "primary_category": "cs.CL",
            "categories": ["cs.CL", "cs.AI"],
            "doi": "10.1234/example.12345"
        }
        result = agent.process_analysis_request(mock_paper, analysis_type="full")
        print(f"分析结果: {result}")
    except Exception as e:
        print(f"错误: {str(e)}")