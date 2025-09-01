from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from simhash import Simhash

@dataclass
class PaperMetadata:
    """论文元数据类"""
    title: str
    authors: List[str]
    venue: str
    year: int
    doi: Optional[str] = None

def calculate_simhash(text: str) -> int:
    """计算文本的SimHash值"""
    return Simhash(text).value

def deduplicate_papers(papers: List[PaperMetadata], threshold: int = 3) -> List[PaperMetadata]:
    """
    基于标题相似度去重论文
    
    参数:
        papers: 论文元数据列表
        threshold: SimHash海明距离阈值，默认为3
        
    返回:
        去重后的论文列表
    """
    if not papers:
        return []
        
    # 计算所有论文标题的SimHash
    hashes = [calculate_simhash(paper.title) for paper in papers]
    
    # 相似度比较
    unique_indices = []
    for i in range(len(papers)):
        is_duplicate = False
        for j in unique_indices:
            # 计算海明距离
            distance = bin(hashes[i] ^ hashes[j]).count('1')
            if distance <= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_indices.append(i)
    
    return [papers[i] for i in unique_indices]