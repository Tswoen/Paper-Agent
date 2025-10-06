import arxiv
import logging
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta

from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

class PaperSearcher:
    """论文搜索器，使用arxiv库搜索论文"""
    
    def __init__(self):
        """初始化论文搜索器"""
        pass
    
    async def search_papers(self, 
                      querys: List[str], 
                      max_results: int = 50, 
                      sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance, 
                      sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending, 
                      start_date: Optional[Union[str, datetime]] = None, 
                      end_date: Optional[Union[str, datetime]] = None) -> List[Dict]:
        """
        搜索arXiv论文
        
        参数:
            querys: 搜索关键词
            max_results: 最大返回结果数量
            sort_by: 排序方式 (Relevance, LastUpdatedDate, SubmittedDate)
            sort_order: 排序顺序 (Ascending, Descending)
            start_date: 开始日期，可以是字符串(YYYY-MM-DD)或datetime对象
            end_date: 结束日期，可以是字符串(YYYY-MM-DD)或datetime对象
        
        返回:
            论文列表，每项包含论文的详细信息
        """
        querys = ['artificial intelligence', 'AI', 'llm', 'machine learning', 'deep learning']
        try:
            # 构建搜索查询
            search_query = ""
            for query in querys:
                search_query += "all:%22"+query+"%22 OR "
            search_query = search_query[:-4]
            # 添加日期范围过滤
            if start_date or end_date:
                start_date_str = self._format_date(start_date) if start_date else "190001010000"
                end_date_str = self._format_date(end_date) if end_date else datetime.now().strftime("%Y%m%d2359")
                date_filter = f"submittedDate:[{start_date_str} TO {end_date_str}]"
                search_query = f"{search_query} AND {date_filter}"

            logger.info(f"开始搜索论文: query='{search_query}', max_results={max_results}, sort_by={sort_by}")


            logger.info(f"论文搜索查询条件: {search_query}")

            # 创建搜索对象
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=sort_by,
                sort_order=sort_order
            )
            
            logger.info(f"论文搜索结果为：{search.results()}")
            # 执行搜索并解析结果
            # 使用新方法格式化论文列表
            papers = self.format_papers_list(search.results())
            
            logger.info(f"论文搜索完成，共找到 {len(papers)} 篇论文")
            return papers
        except Exception as e:
            logger.error(f"论文搜索失败: {str(e)}")
            raise
    
    async def search_by_topic(self, 
                       topic: str, 
                       limit: int = 10, 
                       recent_days: Optional[int] = None) -> List[Dict]:
        """
        按主题搜索最近的论文
        
        参数:
            topic: 主题关键词
            limit: 返回结果数量限制
            recent_days: 搜索最近多少天的论文，None表示不限制
        
        返回:
            论文列表
        """
        logger.info(f"按主题搜索论文: topic='{topic}', limit={limit}, recent_days={recent_days}")
        
        # 计算开始日期
        start_date = None
        if recent_days:
            start_date = datetime.now() - timedelta(days=recent_days)
        
        # 调用搜索方法
        return self.search_papers(
            query=topic,
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
            start_date=start_date
        )
    
    def format_papers_list(self, search_results) -> List[Dict]:
        """
        将搜索结果（迭代器或列表）格式化为论文信息字典列表
        
        参数:
            search_results: arxiv搜索结果对象（可能是迭代器）
        
        返回:
            格式化后的论文信息字典列表
        """
        # 将迭代器转换为列表以便后续处理
        results_list = list(search_results)
        
        # 格式化论文列表
        formatted_papers = [self._parse_paper_result(result) for result in results_list]
        
        logger.info(f"开始格式化论文列表，共 {len(results_list)} 篇论文")
        return formatted_papers

    def search_by_author(self, 
                        author_name: str, 
                        limit: int = 10) -> List[Dict]:
        """
        按作者搜索论文
        
        参数:
            author_name: 作者姓名
            limit: 返回结果数量限制
        
        返回:
            论文列表
        """
        logger.info(f"按作者搜索论文: author='{author_name}', limit={limit}")
        
        # 使用作者字段搜索
        query = f"au:{author_name}"
        return self.search_papers(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
    
    def _parse_paper_result(self, result: arxiv.Result) -> Dict:
        """
        解析arXiv搜索结果
        
        参数:
            result: arxiv.Result对象
        
        返回:
            包含论文信息的字典
        """
        # 从结果URL中提取论文ID
        paper_id = result.get_short_id()
        
        # 提取发布年份
        published_year = result.published.year if result.published else None
        
        return {
            "paper_id": paper_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published": published_year,
            "published_date": result.published.isoformat() if result.published else None,
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
            "primary_category": result.primary_category,
            "categories": result.categories,
            "doi": result.doi if hasattr(result, 'doi') else None
        }
    
    def _format_date(self, date: Union[str, datetime]) -> str:
        """
        
        格式化日期为arXiv API支持的格式YYYYMMDDTTTT
        
        参数:
            date: 日期字符串或datetime对象
        
        返回:
            格式化后的日期字符串YYYYMMDD0000
        """
        if isinstance(date, datetime):
            return date.strftime("%Y%m%d0000")
        elif isinstance(date, str):
            # 定义多种可能的日期格式
            date_formats = [
                "%Y-%m-%d",      # YYYY-MM-DD
                "%Y/%m/%d",      # YYYY/MM/DD
                "%Y.%m.%d",      # YYYY.MM.DD
                "%Y-%m",         # YYYY-MM
                "%Y/%m",         # YYYY/MM
                "%Y",            # YYYY
                "%Y年%m月%d日",  # 中文格式
                "%Y年%m月",      # 中文格式（年月）
                "%Y年",          # 中文格式（年）
            ]
            
            for fmt in date_formats:
                try:
                    if fmt == "%Y":  # 单独处理只有年份的情况
                        if len(date) == 4 and date.isdigit():
                            parsed_date = datetime(int(date), 1, 1)
                            return parsed_date.strftime("%Y%m%d0000")
                    elif fmt in ["%Y-%m", "%Y/%m", "%Y年%m月"]:  # 处理年月格式
                        try:
                            parsed_date = datetime.strptime(date, fmt)
                            return parsed_date.strftime("%Y%m%d0000")
                        except:
                            continue
                    else:
                        parsed_date = datetime.strptime(date, fmt)
                        return parsed_date.strftime("%Y%m%d0000")
                except ValueError:
                    continue
            
            # 如果所有格式都失败，尝试使用dateutil或返回默认值
            try:
                from dateutil import parser
                parsed_date = parser.parse(date)
                return parsed_date.strftime("%Y%m%d0000")
            except:
                # 最终fallback：当前日期
                return datetime.now().strftime("%Y%m%d0000")
        
        # 默认返回当前日期
        return datetime.now().strftime("%Y%m%d0000")

# 示例用法
if __name__ == "__main__":
    data = PaperSearcher()._format_date("2023")
    print(data)