import logging
import re
from typing import List, Dict, Optional, Callable
from datetime import datetime

from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

class PaperFilter:
    """论文筛选器，用于筛选高质量的论文"""
    
    def __init__(self):
        """初始化论文筛选器"""
        pass
    
    def filter_papers(self, 
                     papers: List[Dict], 
                     min_citation_count: Optional[int] = None, 
                     min_published_year: Optional[int] = None, 
                     required_keywords: Optional[List[str]] = None, 
                     excluded_keywords: Optional[List[str]] = None, 
                     category_whitelist: Optional[List[str]] = None, 
                     category_blacklist: Optional[List[str]] = None, 
                     custom_filter: Optional[Callable[[Dict], bool]] = None) -> List[Dict]:
        """
        综合筛选论文
        
        参数:
            papers: 论文列表
            min_citation_count: 最小引用次数
            min_published_year: 最小发布年份
            required_keywords: 必须包含的关键词列表
            excluded_keywords: 必须排除的关键词列表
            category_whitelist: 允许的论文类别列表
            category_blacklist: 排除的论文类别列表
            custom_filter: 自定义筛选函数
        
        返回:
            筛选后的论文列表
        """
        logger.info(f"开始筛选论文，原始数量: {len(papers)}")
        
        filtered_papers = papers.copy()
        
        # 按发布年份筛选
        if min_published_year:
            filtered_papers = self._filter_by_year(filtered_papers, min_published_year)
            logger.info(f"按年份筛选后数量: {len(filtered_papers)}")
        
        # 按关键词包含筛选
        if required_keywords:
            filtered_papers = self._filter_by_required_keywords(filtered_papers, required_keywords)
            logger.info(f"按必须包含关键词筛选后数量: {len(filtered_papers)}")
        
        # 按关键词排除筛选
        if excluded_keywords:
            filtered_papers = self._filter_by_excluded_keywords(filtered_papers, excluded_keywords)
            logger.info(f"按排除关键词筛选后数量: {len(filtered_papers)}")
        
        # 按论文类别筛选
        if category_whitelist:
            filtered_papers = self._filter_by_category_whitelist(filtered_papers, category_whitelist)
            logger.info(f"按类别白名单筛选后数量: {len(filtered_papers)}")
        
        if category_blacklist:
            filtered_papers = self._filter_by_category_blacklist(filtered_papers, category_blacklist)
            logger.info(f"按类别黑名单筛选后数量: {len(filtered_papers)}")
        
        # 按引用次数筛选（如果有引用数据）
        if min_citation_count:
            filtered_papers = self._filter_by_citations(filtered_papers, min_citation_count)
            logger.info(f"按引用次数筛选后数量: {len(filtered_papers)}")
        
        # 自定义筛选
        if custom_filter:
            filtered_papers = self._filter_by_custom(filtered_papers, custom_filter)
            logger.info(f"按自定义条件筛选后数量: {len(filtered_papers)}")
        
        logger.info(f"论文筛选完成，最终保留数量: {len(filtered_papers)}")
        return filtered_papers
    
    def filter_high_quality_papers(self, papers: List[Dict], year_threshold: int = 2020) -> List[Dict]:
        """
        筛选高质量论文的快捷方法
        
        参数:
            papers: 论文列表
            year_threshold: 年份阈值
        
        返回:
            高质量论文列表
        """
        logger.info(f"开始筛选高质量论文，应用默认高质量筛选条件")
        
        return self.filter_papers(
            papers=papers,
            min_published_year=year_threshold,
            required_keywords=["state-of-the-art", "novel", "significant", "breakthrough", "improved"],
            category_blacklist=["cs.CY", "cs.GT", "cs.HC", "cs.LG"],  # 示例：排除某些类别
            custom_filter=self._is_high_quality_paper
        )
    
    def _filter_by_year(self, papers: List[Dict], min_year: int) -> List[Dict]:
        """按发布年份筛选"""
        return [paper for paper in papers if paper.get("published", 0) >= min_year]
    
    def _filter_by_required_keywords(self, papers: List[Dict], keywords: List[str]) -> List[Dict]:
        """筛选包含指定关键词的论文"""
        filtered = []
        for paper in papers:
            # 检查标题和摘要
            text = (paper.get("title", "") + " " + paper.get("summary", "")).lower()
            if any(keyword.lower() in text for keyword in keywords):
                filtered.append(paper)
        return filtered
    
    def _filter_by_excluded_keywords(self, papers: List[Dict], keywords: List[str]) -> List[Dict]:
        """筛选不包含指定关键词的论文"""
        filtered = []
        for paper in papers:
            # 检查标题和摘要
            text = (paper.get("title", "") + " " + paper.get("summary", "")).lower()
            if not any(keyword.lower() in text for keyword in keywords):
                filtered.append(paper)
        return filtered
    
    def _filter_by_category_whitelist(self, papers: List[Dict], categories: List[str]) -> List[Dict]:
        """筛选属于指定类别的论文"""
        filtered = []
        for paper in papers:
            paper_categories = paper.get("categories", [])
            # 检查是否有交集
            if any(cat in paper_categories for cat in categories):
                filtered.append(paper)
        return filtered
    
    def _filter_by_category_blacklist(self, papers: List[Dict], categories: List[str]) -> List[Dict]:
        """筛选不属于指定类别的论文"""
        filtered = []
        for paper in papers:
            paper_categories = paper.get("categories", [])
            # 检查是否没有交集
            if not any(cat in paper_categories for cat in categories):
                filtered.append(paper)
        return filtered
    
    def _filter_by_citations(self, papers: List[Dict], min_citations: int) -> List[Dict]:
        """按引用次数筛选"""
        # 注意：arxiv API本身不直接提供引用次数
        # 这里只是一个框架，实际使用时可能需要集成其他API获取引用数据
        filtered = []
        for paper in papers:
            # 如果论文有引用数据，则进行筛选
            if "citations" in paper and paper["citations"] >= min_citations:
                filtered.append(paper)
            # 否则默认保留
            elif "citations" not in paper:
                filtered.append(paper)
        return filtered
    
    def _filter_by_custom(self, papers: List[Dict], custom_filter: Callable[[Dict], bool]) -> List[Dict]:
        """应用自定义筛选函数"""
        return [paper for paper in papers if custom_filter(paper)]
    
    def _is_high_quality_paper(self, paper: Dict) -> bool:
        """判断是否为高质量论文的默认函数"""
        # 检查标题长度（太短可能不够深入）
        title = paper.get("title", "")
        if len(title) < 10:
            return False
        
        # 检查摘要长度（太短可能内容不够丰富）
        summary = paper.get("summary", "")
        if len(summary) < 100:
            return False
        
        # 检查作者数量（合作研究通常质量更高）
        authors = paper.get("authors", [])
        # 不过分限制作者数量，单作者也可能有高质量论文
        
        # 检查是否有DOI（有DOI通常更正式）
        if "doi" in paper and paper["doi"]:
            return True
        
        # 默认认为是高质量论文
        return True
    
    def deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        去重论文列表
        
        参数:
            papers: 论文列表
        
        返回:
            去重后的论文列表
        """
        logger.info(f"开始去重论文，原始数量: {len(papers)}")
        
        seen_titles = set()
        deduplicated = []
        
        for paper in papers:
            title = paper.get("title", "").strip().lower()
            # 简单的去重方法，基于标题
            if title and title not in seen_titles:
                seen_titles.add(title)
                deduplicated.append(paper)
        
        logger.info(f"论文去重完成，去重后数量: {len(deduplicated)}")
        return deduplicated

# 示例用法
if __name__ == "__main__":
    # 创建模拟论文数据
    mock_papers = [
        {
            "paper_id": "1",
            "title": "Advanced Large Language Models",
            "authors": ["Author A", "Author B"],
            "summary": "This paper presents a novel approach to large language models...",
            "published": 2023,
            "categories": ["cs.CL", "cs.AI"],
            "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf"
        },
        {
            "paper_id": "2",
            "title": "Old Research Paper",
            "authors": ["Author C"],
            "summary": "An older approach to language modeling...",
            "published": 2018,
            "categories": ["cs.CL"],
            "pdf_url": "https://arxiv.org/pdf/1801.00001.pdf"
        },
        {
            "paper_id": "3",
            "title": "State-of-the-Art NLP Systems",
            "authors": ["Author D", "Author E", "Author F"],
            "summary": "We introduce significant improvements to existing NLP systems...",
            "published": 2022,
            "categories": ["cs.CL", "stat.ML"],
            "doi": "10.1234/example.2022",
            "pdf_url": "https://arxiv.org/pdf/2201.00001.pdf"
        }
    ]
    
    try:
        filterer = PaperFilter()
        
        # 筛选高质量论文
        high_quality_papers = filterer.filter_high_quality_papers(mock_papers, year_threshold=2020)
        
        print(f"原始论文数量: {len(mock_papers)}")
        print(f"高质量论文数量: {len(high_quality_papers)}")
        
        for i, paper in enumerate(high_quality_papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   作者: {', '.join(paper['authors'])}")
            print(f"   发布年份: {paper['published']}")
            print(f"   类别: {', '.join(paper['categories'])}")
    except Exception as e:
        print(f"筛选失败: {e}")