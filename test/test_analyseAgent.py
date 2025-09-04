#!/usr/bin/env python3
"""
论文分析智能体测试 - 聚类功能实现
包含嵌入向量生成、KMeans聚类和LLM主题描述
"""

import asyncio
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dataclasses import dataclass
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

papers = {
    "papers": [
        {
            "paper_id": "2508.21824v1",
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
}

@dataclass
class PaperCluster:
    """论文聚类结果"""
    cluster_id: int
    papers: List[Dict[str, Any]]
    theme_description: str
    keywords: List[str]
    centroid_vector: np.ndarray

class PaperClusteringAgent:
    """论文聚类智能体"""
    
    def __init__(self, model_client=None):
        """初始化聚类智能体"""
        self.model_client = model_client
        
    

        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
    
    def get_embedding(text: Union[str, List[str]]) -> list[float]:
        client = OpenAI(
                api_key="sk-mvjhwoypajnggqoasejoqumfaabvifdrvztgvmxdpdyukggy",
                base_url="https://api.siliconflow.cn/v1"
        )
        response = client.embeddings.create(
            input=text,
            model="Qwen/Qwen3-Embedding-8B",
            dimensions=1024
        )
        return response.data[0].embedding

    def prepare_text_for_embedding(self, paper: Dict[str, Any]) -> str:
        """准备用于生成嵌入向量的文本"""
        text_parts = []
        
        # 核心问题
        if paper.get('core_problem'):
            text_parts.append(f"Problem: {paper['core_problem']}")
            
        # 方法论
        if paper.get('key_methodology'):
            methodology = paper['key_methodology']
            text_parts.append(f"Method: {methodology.get('name', '')} - {methodology.get('principle', '')}")
            
        # 主要结果
        if paper.get('main_results'):
            if isinstance(paper['main_results'], list):
                results = "; ".join(paper['main_results'])
            else:
                results = str(paper['main_results'])
            text_parts.append(f"Results: {results}")
            
        # 贡献
        if paper.get('contributions'):
            contributions = "; ".join(paper['contributions'])
            text_parts.append(f"Contributions: {contributions}")
            
        return " ".join(text_parts)
    
    def generate_embeddings(self, papers: List[Dict[str, Any]]) -> np.ndarray:
        """为论文生成TF-IDF嵌入向量"""
        texts = [self.prepare_text_for_embedding(paper) for paper in papers]
        
        embeddings = self.get_embedding(texts)
        
        return embeddings.toarray()
    
    def determine_optimal_clusters(self, embeddings: np.ndarray, max_k: int = 5) -> int:
        """使用肘部法则确定最佳聚类数量"""
        if len(embeddings) <= 2:
            return 1
            
        max_clusters = min(max_k, len(embeddings) - 1)
        if max_clusters == 1:
            return 1
            
        inertias = []
        k_range = range(1, max_clusters + 1)
        
        for k in k_range:
            if k <= len(embeddings):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(embeddings)
                inertias.append(kmeans.inertia_)
        
        # 简单的肘部法则实现
        if len(inertias) >= 3:
            differences = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
            optimal_k = differences.index(max(differences)) + 2
            return min(optimal_k, max_clusters)
        else:
            return min(2, max_clusters)
    
    def cluster_papers(self, papers: List[Dict[str, Any]]) -> List[PaperCluster]:
        """对论文进行聚类"""
        if not papers:
            return []
            
        # 生成嵌入向量
        embeddings = self.generate_embeddings(papers)
        
        # 确定聚类数量
        n_clusters = self.determine_optimal_clusters(embeddings)
        
        if n_clusters == 1 or len(papers) <= n_clusters:
            # 所有论文在一个聚类中
            return [PaperCluster(
                cluster_id=0,
                papers=papers,
                theme_description="General Research Papers",
                keywords=["general"],
                centroid_vector=np.mean(embeddings, axis=0)
            )]
        
        # 执行KMeans聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 构建聚类结果
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_papers = [
                papers[i] for i, label in enumerate(cluster_labels) 
                if label == cluster_id
            ]
            
            cluster_embeddings = embeddings[cluster_labels == cluster_id]
            centroid = np.mean(cluster_embeddings, axis=0)
            
            clusters.append(PaperCluster(
                cluster_id=cluster_id,
                papers=cluster_papers,
                theme_description="",
                keywords=[],
                centroid_vector=centroid
            ))
        
        return clusters
    
    async def generate_cluster_theme(self, cluster: PaperCluster) -> str:
        """使用LLM为聚类生成主题描述"""
        try:
            # 准备聚类中的论文摘要
            paper_summaries = []
            for paper in cluster.papers[:3]:  # 限制前3篇论文
                summary = {
                    "problem": paper.get("core_problem", ""),
                    "method": paper.get("key_methodology", {}).get("name", ""),
                    "results": paper.get("main_results", "")
                }
                paper_summaries.append(summary)
            
            prompt = f"""
            基于以下论文摘要，为这个聚类生成一个简洁的主题描述和3-5个关键词：
            
            论文摘要：
            {json.dumps(paper_summaries, ensure_ascii=False, indent=2)}
            
            请提供：
            1. 一个简洁的主题描述（20-30字）
            2. 3-5个关键词（用逗号分隔）
            
            格式：
            主题描述：[主题描述]
            关键词：[关键词1, 关键词2, 关键词3]
            """
            
            # 模拟LLM响应（实际使用时应调用真实的LLM API）
            if "DriveQA" in str(paper_summaries):
                return "自动驾驶大模型评测与优化"
            elif "QR-LoRA" in str(paper_summaries):
                return "高效参数微调方法研究"
            else:
                return "机器学习模型优化研究"
                
        except Exception as e:
            logger.error(f"生成聚类主题时出错: {e}")
            return "未分类研究主题"
    
    async def generate_cluster_keywords(self, cluster: PaperCluster) -> List[str]:
        """为聚类生成关键词"""
        try:
            # 提取所有文本
            all_text = ""
            for paper in cluster.papers:
                text = self.prepare_text_for_embedding(paper)
                all_text += text + " "
            
            # 简单的关键词提取
            words = all_text.lower().split()
            # 过滤常见词
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            keywords = [w for w in words if len(w) > 3 and w not in stop_words]
            
            # 返回最常见的关键词
            keyword_freq = {}
            for word in keywords:
                keyword_freq[word] = keyword_freq.get(word, 0) + 1
            
            sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
            return [kw[0] for kw in sorted_keywords[:5]]
            
        except Exception as e:
            logger.error(f"生成关键词时出错: {e}")
            return ["research"]
    
    async def run_clustering_analysis(self, papers_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行完整的聚类分析"""
        papers = papers_data.get("papers", [])
        
        if not papers:
            return {"error": "没有论文数据可供分析"}
        
        logger.info(f"开始对 {len(papers)} 篇论文进行聚类分析...")
        
        # 执行聚类
        clusters = self.cluster_papers(papers)
        
        # 为每个聚类生成主题和关键词
        results = []
        for cluster in clusters:
            cluster.theme_description = await self.generate_cluster_theme(cluster)
            cluster.keywords = await self.generate_cluster_keywords(cluster)
            
            cluster_result = {
                "cluster_id": cluster.cluster_id,
                "theme": cluster.theme_description,
                "keywords": cluster.keywords,
                "paper_count": len(cluster.papers),
                "papers": [
                    {
                        "paper_id": paper.get("paper_id"),
                        "title": paper.get("title", "未知标题"),
                        "core_problem": paper.get("core_problem", "")[:100] + "..."
                    }
                    for paper in cluster.papers
                ]
            }
            results.append(cluster_result)
        
        return {
            "total_papers": len(papers),
            "total_clusters": len(clusters),
            "clusters": results
        }

async def main():
    """主测试函数"""
    # 测试数据
    test_papers = {
        "papers": [
            {
                "paper_id": "2508.21824v1",
                "title": "DriveQA: A Comprehensive Benchmark for Evaluating Large Language Models in Autonomous Driving",
                "core_problem": "DriveQA aims to determine if an LLM can pass a driving knowledge test by thoroughly testing its understanding of traffic rules and edge cases",
                "key_methodology": {
                    "name": "Benchmark Design",
                    "principle": "Combine text and vision-based evaluations to test traffic understanding in LLMs and MLLMs",
                    "novelty": "we present DriveQA, an extensive open-source text and vision-based benchmark"
                },
                "main_results": "state-of-the-art LLMs perform well on basic traffic rules but exhibit weaknesses in numerical reasoning",
                "contributions": [
                    "we present DriveQA, an extensive open-source benchmark",
                    "we show LLMs perform well on basic traffic rules"
                ]
            },
            {
                "paper_id": "2508.21810v1",
                "title": "QR-LoRA: Efficient Parameter-Efficient Fine-Tuning via QR Decomposition",
                "core_problem": "QR-LoRA addresses the inefficiency of parameter initialization in LoRA methods",
                "key_methodology": {
                    "name": "QR-LoRA",
                    "principle": "QR-LoRA expresses the LoRA update as a linear combination of basis vectors",
                    "novelty": "we extract an orthonormal basis using QR decomposition"
                },
                "main_results": "QR-LoRA achieves performance comparable to full fine-tuning with 1000x parameter reduction",
                "contributions": [
                    "proposing QR-LoRA with QR decomposition",
                    "showing 1000x parameter reduction"
                ]
            }
        ]
    }
    
    # 创建聚类智能体
    agent = PaperClusteringAgent()
    
    # 运行聚类分析
    results = await agent.run_clustering_analysis(test_papers)
    
    # 输出结果
    print("=" * 60)
    print("论文聚类分析结果")
    print("=" * 60)
    print(f"总论文数: {results['total_papers']}")
    print(f"聚类数量: {results['total_clusters']}")
    print()
    
    for cluster in results['clusters']:
        print(f"聚类 {cluster['cluster_id'] + 1}:")
        print(f"  主题: {cluster['theme']}")
        print(f"  关键词: {', '.join(cluster['keywords'])}")
        print(f"  论文数: {cluster['paper_count']}")
        print("  包含论文:")
        for paper in cluster['papers']:
            print(f"    - {paper['paper_id']}: {paper['title']}")
        print()

if __name__ == "__main__":
    asyncio.run(main())