import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from src.utils.log_utils import setup_logger
from src.tasks.paper_search import PaperSearch
import arxiv

logger = setup_logger(__name__)

class SearchAgent:
    """搜索代理，处理前端请求并调用论文搜索工具"""
    
    def __init__(self):
        """初始化搜索代理"""
        self.paper_search = PaperSearch()
    
    def process_search_request(self, 
                              query: str, 
                              start_date: Optional[str] = None, 
                              end_date: Optional[str] = None, 
                              max_results: int = 20) -> Dict:
        """
        处理前端发送的搜索请求
        
        参数:
            query: 搜索查询字符串
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            max_results: 最大返回结果数量
        
        返回:
            包含搜索结果和处理信息的字典
        """
        logger.info(f"处理搜索请求: query='{query}', start_date={start_date}, end_date={end_date}")
        
        try:
            # 解析查询，提取主题和作者
            topics, authors = self._parse_query(query)
            
            # 根据解析结果构建搜索查询
            all_results = self._search_papers(topics, authors, start_date, end_date, max_results)
            
            return {
                "success": True,
                "data": all_results,
                "metadata": {
                    "query": query,
                    "topics": topics,
                    "authors": authors,
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_papers": len(all_results)
                }
            }
        except Exception as e:
            logger.error(f"处理搜索请求失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "query": query,
                    "start_date": start_date,
                    "end_date": end_date
                }
            }
    
    def _parse_query(self, query: str) -> Tuple[List[str], List[str]]:
        """
        解析查询字符串，提取主题和作者信息
        
        参数:
            query: 搜索查询字符串
        
        返回:
            (主题列表, 作者列表)的元组
        """
        topics = []
        authors = []
        
        # 简单的规则来提取作者信息
        # 1. 查找形如 "author:张三" 或 "作者:李四" 的模式
        author_patterns = [
            r'author:([^\s,]+)',  # author:xxx 格式
            r'作者:([^\s,]+)'      # 作者:xxx 格式
        ]
        
        for pattern in author_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                author_name = match.group(1)
                if author_name and author_name not in authors:
                    authors.append(author_name)
            # 从原始查询中移除作者信息
            query = re.sub(pattern, '', query, flags=re.IGNORECASE)
        
        # 提取主题信息
        # 1. 查找引号中的短语
        quoted_topics = re.findall(r'"([^"]+)"', query)
        topics.extend(quoted_topics)
        # 从查询中移除引号中的短语
        query = re.sub(r'"[^"]+"', '', query)
        
        # 2. 使用空格分割剩余的查询，提取关键词作为主题
        remaining_terms = query.strip().split()
        if remaining_terms and remaining_terms != ['AND', 'OR', 'NOT']:
            topics.extend(remaining_terms)
        
        # 去重并清理空字符串
        topics = [topic.strip() for topic in topics if topic.strip()]
        topics = list(set(topics))
        
        logger.info(f"解析查询结果 - 主题: {topics}, 作者: {authors}")
        return topics, authors
    
    def _search_papers(self, 
                      topics: List[str], 
                      authors: List[str], 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None, 
                      max_results: int = 20) -> List[Dict]:
        """
        根据主题和作者调用论文搜索工具
        
        参数:
            topics: 主题列表
            authors: 作者列表
            start_date: 开始日期
            end_date: 结束日期
            max_results: 最大返回结果数量
        
        返回:
            论文列表
        """
        all_papers = []
        
        # 如果既有主题又有作者，优先使用组合查询
        if topics and authors:
            # 为每个作者和主题的组合构建查询
            for author in authors:
                for topic in topics:
                    combined_query = f"{topic} AND au:{author}"
                    papers = self.paper_search.search_papers(
                        query=combined_query,
                        max_results=max_results // (len(topics) * len(authors)),
                        sort_by=arxiv.SortCriterion.SubmittedDate,
                        sort_order=arxiv.SortOrder.Descending,
                        start_date=start_date,
                        end_date=end_date
                    )
                    all_papers.extend(papers)
        elif topics:
            # 只搜索主题
            for topic in topics:
                papers = self.paper_search.search_by_topic(
                    topic=topic,
                    limit=max_results // len(topics) if topics else max_results,
                    recent_days=None  # 使用传入的start_date和end_date
                )
                all_papers.extend(papers)
        elif authors:
            # 只搜索作者
            for author in authors:
                papers = self.paper_search.search_by_author(
                    author_name=author,
                    limit=max_results // len(authors) if authors else max_results
                )
                all_papers.extend(papers)
        else:
            # 没有有效查询，返回空列表
            logger.warning("没有有效的搜索查询，返回空结果")
            return []
        
        # 去重，根据论文ID
        seen_ids = set()
        unique_papers = []
        for paper in all_papers:
            if paper['paper_id'] not in seen_ids:
                seen_ids.add(paper['paper_id'])
                unique_papers.append(paper)
        
        # 按提交日期排序并限制结果数量
        unique_papers.sort(key=lambda x: x['published_date'] or '', reverse=True)
        return unique_papers[:max_results]

if __name__ == "__main__":
    # 测试代码
    try:
        agent = SearchAgent()
        result = agent.process_search_request(
            query="large language models author:Bengio",
            start_date="2024-01-01",
            end_date="2024-12-31",
            max_results=10
        )
        print(f"搜索结果: {result}")
        print(f"找到 {len(result.get('data', []))} 篇论文")
    except Exception as e:
        print(f"错误: {str(e)}")