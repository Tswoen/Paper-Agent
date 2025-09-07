
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
from src.core.model_client import create_default_client
from autogen_agentchat.agents import AssistantAgent
from src.agents.sub_analyse_agent.cluster_agent import PaperCluster
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)




# papers = {
#     "papers": 
# }
paperslist = [
        {
            "paper_id": "2508.21824v1",
            "title": "DriveQA: Benchmarking LLMs for Driving Knowledge",
            "core_problem": "DriveQA aims to determine if an LLM can pass a driving knowledge test by thoroughly testing its understanding of traffic rules and edge cases that standard autonomous driving benchmarks do not cover",
            "key_methodology": {
                "name": "Benchmark Design",
                "principle": "Combine text and vision-based evaluations to test traffic understanding in LLMs and MLLMs.",
                "novelty": "we present DriveQA, an extensive open-source text and vision-based benchmark"
            },
            "datasets_used": [
                "DriveQA (an extensive traffic regulations and scenarios benchmark)",
                "SST-2 (67k sentences)"
            ],
            "evaluation_metrics": [
                "Accuracy",
                "Performance on real-world datasets such as nuScenes and BDD"
            ],
            "main_results": "state-of-the-art LLMs and MLLMs perform well on basic traffic rules but exhibit significant weaknesses in numerical reasoning, traffic sign variations, and spatial layouts; pretraining on DriveQA improves downstream driving task performance and generalization across real-world datasets",
            "limitations": "the experiments focus on commercial models without extensive comparison with specific alternatives",
            "contributions": [
                "we present DriveQA, an extensive open-source text and vision-based benchmark that exhaustively covers traffic regulations and scenarios",
                "we show that state-of-the-art LLMs and MLLMs perform well on basic traffic rules but exhibit significant weaknesses in numerical reasoning and complex right-of-way scenarios",
                "we show that fine-tuning on DriveQA improves accuracy across multiple categories, particularly in regulatory sign recognition and intersection decision-making",
                "we show that pretraining on DriveQA enhances downstream driving task performance, leading to improved results on real-world datasets"
            ],
            "author_institutions": None,
            "extract_source": {
                "author_institution": None,
                "contribution": [
                    "Abstract line 2-3",
                    "Abstract line 4-5",
                    "Abstract line 5-6",
                    "Abstract line 7-8"
                ],
                "core_problem": "Introduction line 1-3",
                "datasets_used": [
                    "Abstract line 8",
                    "Introduction line 5"
                ],
                "evaluation_metrics": [
                    "Abstract line 5",
                    "Abstract line 8"
                ],
                "key_methodology.name": "Benchmark Design",
                "key_methodology.novelty": "we present DriveQA, an extensive open-source text and vision-based benchmark",
                "key_methodology.principle": "DriveQA combines text and vision-based evaluations to test traffic understanding in LLMs and MLLMs",
                "limitations": "the experiments focus on commercial models without extensive comparison with specific alternatives",
                "main_results": [
                    "state-of-the-art LLMs and MLLMs perform well on basic traffic rules but fail in numerical reasoning, complex right-of-way scenarios, and spatial layouts",
                    "pretraining on DriveQA leads to improved results on real-world datasets like nuScenes and BDD"
                ]
            }
        },
        {
            "paper_id": "2508.21810v1",
            "title": "QR-LoRA: Efficient Parameter Initialization for LoRA",
            "core_problem": "QR-LoRA addresses the inefficiency of parameter initialization in LoRA methods by replacing SVD with QR decomposition, reducing the costs and improving interpretability",
            "key_methodology": {
                "name": "QR-LoRA",
                "principle": "QR-LoRA expresses the LoRA update as a linear combination of basis vectors extracted via QR decomposition and trains only scalar coefficients",
                "novelty": "we extract an orthonormal basis from the pretrained weight matrix using QR decomposition with column pivoting"
            },
            "datasets_used": ["GLUE tasks"],
            "evaluation_metrics": [
                "Performance compared to full fine-tuning and SVD-LoRA",
                "Parameter count reduction"
            ],
            "main_results": "QR-LoRA achieves performance comparable to or better than full fine-tuning and SVD-LoRA with only 601 parameters, a 1000x reduction in trainable parameters compared to full fine-tuning",
            "limitations": "the method's generalization to tasks beyond GLUE is not evaluated",
            "contributions": [
                "proposing QR-LoRA, which extracts an orthonormal basis from pretrained weights using QR decomposition",
                "QR-LoRA can match or exceed full fine-tuning, standard LoRA, and SVD-LoRA with minimal parameters"
            ]
        }
    ]

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
        self.model_client = create_default_client()
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

                聚类主题：{cluster.theme_description}
                聚类关键词：{', '.join(cluster.keywords)}
                论文数量：{len(cluster.papers)}

                详细论文信息：
                {json.dumps(cluster.papers, ensure_ascii=False, indent=2)}

                # 分析维度
                1. 趋势分析：按时间顺序分析该技术的发展脉络
                2. 方法对比：对比不同论文提出的方法优缺点
                3. 性能分析：在共同数据集上的表现对比（如有）
                4. 局限性：该技术路线的共同局限性

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