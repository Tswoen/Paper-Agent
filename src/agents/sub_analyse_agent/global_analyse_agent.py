"""
测试分析智能体 - 全局分析功能
"""

import asyncio
import sys
import os
import json
from typing import Dict, Any, List
from src.core.prompts import global_analyse_agent_prompt
from src.core.model_client import create_default_client
from autogen_agentchat.agents import AssistantAgent
from src.agents.sub_analyse_agent.deep_analyse_agent import DeepAnalyseResult
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

class GlobalanalyseAgent:
    async def run(self, cluster_results: List[DeepAnalyseResult]):
        """统一接口方法"""
        return await self.generate_global_analyse(cluster_results)
        
    def __init__(self, model_client=None):
        """初始化聚类智能体"""
        self.model_client = create_default_client()
        self.global_analyse_agent = AssistantAgent(
            name="global_analyse_agent",
            model_client= self.model_client,
            system_message = global_analyse_agent_prompt
        )
    
    async def generate_global_analyse(self, analyse_results: List[DeepAnalyseResult]) -> Dict[str, Any]:
        """生成全局分析草稿 - 汇总各主题分析结果"""
        try:
            # 准备所有聚类的分析内容
            cluster_summaries = []
            for result in analyse_results:
                cluster_summaries.append({
                    "cluster_id": result.cluster_id,
                    "theme": result.theme,
                    "keywords": result.keywords,
                    "paper_count": result.paper_count,
                    "analyse_summary": result.deep_analyse[:1000] + "..." if len(result.deep_analyse) > 10000 else result.deep_analyse
                })
            
            prompt = f"""
基于以下多个主题的深入分析结果，生成一份完整的全局分析草稿：

{json.dumps(cluster_summaries, ensure_ascii=False, indent=2)}

# 全局分析要求
1. 技术趋势总结：分析各主题之间的技术关联和发展趋势
2. 方法对比：对比不同主题采用的核心方法和技术路线
3. 应用领域分析：总结各主题的主要应用场景和领域
4. 研究热点识别：识别当前的研究热点和未来发展方向
5. 局限性总结：总结各技术路线的共同局限性
6. 建议与展望：提出进一步研究的建议和未来展望

请生成结构清晰、内容完整的全局分析草稿。
"""
            
            response = await self.global_analyse_agent.run(task=prompt)
            global_analyse = response.messages[-1].content
            
            return {
                "total_clusters": len(analyse_results),
                "total_papers": sum(result.paper_count for result in analyse_results),
                "cluster_themes": [result.theme for result in analyse_results],
                "global_analyse": global_analyse,
                "cluster_summaries": cluster_summaries
            }
            
        except Exception as e:
            logger.error(f"生成全局分析时出错: \n{e}")
            return {
                "total_clusters": len(cluster_results),
                "total_papers": sum(result.paper_count for result in cluster_results),
                "cluster_themes": [result.theme for result in cluster_results],
                "global_analyse": f"全局分析失败: {str(e)}",
                "cluster_summaries": []
            }