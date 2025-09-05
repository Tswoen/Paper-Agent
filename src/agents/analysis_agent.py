import logging
from typing import Dict, List, Optional
from autogen_agentchat.agents import BaseChatAgent
import autogen
from autogen.agentchat import AssistantAgent, UserProxyAgent
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

class PaperAnalysisAgent(BaseChatAgent):
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
    
   