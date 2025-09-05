"""
测试分析智能体 - 全局分析功能
"""

import asyncio
import sys
import os
import json
from typing import Dict, Any
from src.core.prompts import global_analysis_agent_prompt
from src.core.model_client import create_default_client
from autogen_agentchat.agents import AssistantAgent
from src.agents.sub_analysis_agent.deep_analyse_agent import ClusterAnalysisResult


class GlobalAnalysisAgent:
    def __init__(self, model_client=None):
        """初始化聚类智能体"""
        self.model_client = create_default_client()
        self.deep_analysis_agent = AssistantAgent(
            name="global_analysis_agent",
            model_client= self.model_client,
            system_message = global_analysis_agent_prompt
        )
    
    async def generate_global_analysis(self, cluster_results: List[ClusterAnalysisResult]) -> Dict[str, Any]:
        """生成全局分析草稿 - 汇总各主题分析结果"""
        try:
            # 准备所有聚类的分析内容
            cluster_summaries = []
            for result in cluster_results:
                cluster_summaries.append({
                    "cluster_id": result.cluster_id,
                    "theme": result.theme,
                    "keywords": result.keywords,
                    "paper_count": result.paper_count,
                    "analysis_summary": result.deep_analysis[:5000] + "..." if len(result.deep_analysis) > 5000 else result.deep_analysis
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
            
            response = await self.deep_analysis_agent.run(task=prompt)
            global_analysis = response.messages[-1].content
            
            return {
                "total_clusters": len(cluster_results),
                "total_papers": sum(result.paper_count for result in cluster_results),
                "cluster_themes": [result.theme for result in cluster_results],
                "global_analysis": global_analysis,
                "cluster_summaries": cluster_summaries
            }
            
        except Exception as e:
            print(f"生成全局分析时出错: {e}")
            return {
                "total_clusters": len(cluster_results),
                "total_papers": sum(result.paper_count for result in cluster_results),
                "cluster_themes": [result.theme for result in cluster_results],
                "global_analysis": f"全局分析失败: {str(e)}",
                "cluster_summaries": []
            }