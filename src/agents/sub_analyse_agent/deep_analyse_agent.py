
#!/usr/bin/env python3
"""
测试分析智能体 - 单主题深入分析功能演示
"""

import asyncio
import sys
import os
import json
from typing import Dict, Any, List
from dataclasses import dataclass
from src.core.prompts import deep_analyse_agent_prompt
from src.core.model_client import create_default_client, create_subanalyse_deep_analyse_model_client
from autogen_agentchat.agents import AssistantAgent
from src.agents.sub_analyse_agent.cluster_agent import PaperCluster
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class DeepAnalyseResult:
    """聚类分析结果封装类"""
    cluster_id: int
    theme: str
    keywords: List[str]
    paper_count: int
    deep_analyse: str
    papers: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "cluster_id": self.cluster_id,
            "theme": self.theme,
            "keywords": self.keywords,
            "paper_count": self.paper_count,
            "deep_analyse": self.deep_analyse,
            "papers": self.papers
        }

class DeepAnalyseAgent:
    async def run(self, cluster_data):
        """统一接口方法"""
        return await self.deep_analyze_cluster(cluster_data)
        
    def __init__(self, model_client=None):
        """初始化聚类智能体"""
        self.model_client = create_subanalyse_deep_analyse_model_client()
        self.deep_analyse_agent = AssistantAgent(
            name="deep_analyse_agent",
            model_client= self.model_client,
            system_message = deep_analyse_agent_prompt
        )
    async def deep_analyze_cluster(self, cluster: PaperCluster) -> DeepAnalyseResult:
        """对单个聚类进行深入分析"""
        try:
            
            prompt = f"""
                基于以下聚类信息和详细的论文内容，进行深入的学术分析：

                ## 基本信息
                - **聚类主题**：{cluster.theme_description}
                - **核心关键词**：{', '.join(cluster.keywords)}
                - **论文数量**：{len(cluster.papers)}

                ## 详细论文数据
                {json.dumps(cluster.papers, ensure_ascii=False, indent=2)}

                请以结构化的方式组织你的分析结果。
"""
            
            response = await self.deep_analyse_agent.run(task=prompt)
            analyse_content = response.messages[-1].content
            
            return DeepAnalyseResult(
                cluster_id=cluster.cluster_id,
                theme=cluster.theme_description,
                keywords=cluster.keywords,
                paper_count=len(cluster.papers),
                deep_analyse=analyse_content,
                papers=cluster.papers
            )
                
        except Exception as e:
            logger.error(f"深入分析聚类时出错: \n{e}")
            return DeepAnalyseResult(
                cluster_id=cluster.cluster_id,
                theme=cluster.theme_description,
                keywords=cluster.keywords,
                paper_count=len(cluster.papers),
                deep_analyse=f"分析失败: {str(e)}",
                papers=cluster.papers
            )
    

async def main():
    """主测试函数 - 测试并行深入分析功能"""
    logger.info("=== 开始测试并行深入分析功能 ===")
    
    # 创建分析智能体
    analyse_agent = DeepanalyseAgent()
    
    # 创建多个模拟的聚类对象
    clusters = [
        PaperCluster(
            cluster_id=1,
            theme_description="聚焦大模型在自动驾驶知识评估与参数优化中的关键技术应用",
            keywords=['自动驾驶基准测试', 'QR分解', '交通规则评估', '参数优化方法', '数值推理'],
            papers=paperslist
        ),
        PaperCluster(
            cluster_id=2,
            theme_description="大模型在自然语言处理中的优化技术研究",
            keywords=['大模型优化', '参数效率', 'LoRA方法', '微调技术', '计算效率'],
            papers=paperslist[:1]  # 只包含第一篇论文
        ),
        PaperCluster(
            cluster_id=3,
            theme_description="深度学习模型在计算机视觉领域的应用",
            keywords=['计算机视觉', '图像识别', '目标检测', '深度学习', '神经网络'],
            papers=paperslist[1:]  # 只包含第二篇论文
        )
    ]
    
    print(f"共有 {len(clusters)} 个聚类需要分析")
    for i, cluster in enumerate(clusters, 1):
        print(f"聚类 {i}: {cluster.theme_description} ({len(cluster.papers)}篇论文)")
    
    print("\n开始并行深入分析...")
    
    try:
        # 执行并行深入分析
        analyse_results = await analyse_agent.deep_analyze_clusters(clusters)
        
        print("\n=== 并行深入分析结果 ===")
        for result in analyse_results:
            print(f"\n--- 聚类 {result.cluster_id} 分析结果 ---")
            print(f"主题: {result.theme}")
            print(f"关键词: {', '.join(result.keywords)}")
            print(f"论文数量: {result.paper_count}")
            print(f"分析摘要: {result.deep_analyse[:200]}...")
        
        print(f"\n=== 分析完成 ===")
        print(f"成功分析 {len([r for r in analyse_results if not r.deep_analyse.startswith('分析失败')])} 个聚类")
        print(f"失败 {len([r for r in analyse_results if r.deep_analyse.startswith('分析失败')])} 个聚类")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())