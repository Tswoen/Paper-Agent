from langchain_community.utilities import ArxivAPIWrapper
import logging
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta

from src.utils.log_utils import setup_logger
from src.tasks.deduplicator import PaperMetadata, deduplicate_papers

logger = setup_logger(__name__)

class ArxivFetcher:
    """ArXiv论文获取器，用于从ArXiv批量获取论文元数据"""
    
    def __init__(self, max_results: int = 100, top_k_results: int = 50):
        """
        初始化ArXiv论文获取器
        
        参数:
            max_results: 最大获取结果数
            top_k_results: 返回的前k篇结果
        """
        self.max_results = max_results
        self.top_k_results = top_k_results
        self.arxiv = ArxivAPIWrapper(
            top_k_results=top_k_results,
            load_max_docs=max_results,
            doc_content_chars_max=0  # 不限制内容长度
        )
    
    def _format_query(self, keywords: Union[str, List[str]], 
                     start_date: Optional[str] = None, 
                     end_date: Optional[str] = None, 
                     excluded_keywords: Optional[List[str]] = None) -> str:
        """
        格式化ArXiv查询字符串
        
        参数:
            keywords: 关键词或关键词列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            excluded_keywords: 排除的关键词列表
        
        返回:
            格式化的查询字符串
        """
        # 处理关键词
        if isinstance(keywords, list):
            query_parts = [f"{keyword}" for keyword in keywords]
        else:
            query_parts = [keywords]
        
        # 添加日期范围
        if start_date:
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            # 使用正确的ArXiv日期查询格式
            date_query = f"submittedDate:[{start_date} TO {end_date}]"
            query_parts.append(date_query)
        
        # 添加排除关键词
        if excluded_keywords:
            for keyword in excluded_keywords:
                query_parts.append(f"NOT {keyword}")
        
        # 构建完整查询
        query = " AND ".join(query_parts)
        
        logger.debug(f"生成的ArXiv查询: {query}")
        return query
    
    def fetch_papers(self, keywords: Union[str, List[str]], 
                    start_date: Optional[str] = None, 
                    end_date: Optional[str] = None, 
                    excluded_keywords: Optional[List[str]] = None, 
                    deduplicate: bool = True) -> List[Dict]:
        """
        批量获取ArXiv论文
        
        参数:
            keywords: 搜索关键词或关键词列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            excluded_keywords: 排除的关键词列表
            deduplicate: 是否对结果进行去重
        
        返回:
            论文元数据列表
        """
        try:
            logger.info(f"开始从ArXiv获取论文，关键词: {keywords}")
            
            # 格式化查询
            query = self._format_query(keywords, start_date, end_date, excluded_keywords)
            
            # 执行查询
            # results = self.arxiv.run(query)
            results = self.arxiv.run("1102.5509")

            logger.info(f"从ArXiv获取论文完成，results:{results} ")
            # 解析结果
            papers = self._parse_results(results)
            
            # 确保解析结果不为空且是列表
            if not papers:
                logger.warning(f"查询结果为空或格式不匹配: {results}")
            
            logger.info(f"从ArXiv获取论文完成，papers:{papers} ")
            # 过滤不完整的论文数据
            valid_papers = []
            for paper in papers:
                # 确保论文有必要的字段
                if all(key in paper for key in ["title", "authors", "published"]):
                    valid_papers.append(paper)
                else:
                    logger.warning(f"跳过不完整的论文数据: {paper.keys()}")
            
            logger.info(f"过滤后有效论文数: {len(valid_papers)}")
            
            # 去重处理
            if deduplicate and valid_papers:
                papers_metadata = [
                    PaperMetadata(
                        title=paper["title"],
                        authors=paper["authors"],
                        venue="arXiv",
                        year=int(paper["published"]),
                        doi=paper.get("doi")
                    ) for paper in valid_papers
                ]
                
                unique_metadata = deduplicate_papers(papers_metadata)
                
                # 根据去重后的元数据过滤原始论文列表
                unique_titles = {metadata.title for metadata in unique_metadata}
                valid_papers = [paper for paper in valid_papers if paper["title"] in unique_titles]
                
                logger.info(f"去重后剩余论文数: {len(valid_papers)}")
            
            papers = valid_papers
            
            logger.info(f"从ArXiv获取论文完成，共获取 {len(papers)} 篇")
            return papers
            
        except Exception as e:
            logger.error(f"从ArXiv获取论文失败: {str(e)}")
            raise
    
    def _parse_results(self, results: str) -> List[Dict]:
        """
        解析ArXiv API返回的结果
        
        参数:
            results: API返回的原始结果字符串
        
        返回:
            解析后的论文列表
        """
        papers = []
        
        # 分割多篇论文
        raw_papers = results.strip().split("\n\n")
        
        for raw_paper in raw_papers:
            if not raw_paper.strip():
                continue
                
            paper_data = {}
            lines = raw_paper.strip().split("\n")
            
            # 解析标题
            if lines and lines[0].startswith("Title:"):
                paper_data["title"] = lines[0][6:].strip()
            
            # 解析作者
            authors_line = next((line for line in lines if line.startswith("Authors:")), None)
            if authors_line:
                paper_data["authors"] = [author.strip() for author in authors_line[8:].split(",")]
            
            # 解析发布日期
            date_line = next((line for line in lines if line.startswith("Published:")), None)
            if date_line:
                date_str = date_line[10:].strip()
                try:
                    # 尝试解析日期格式
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    paper_data["published"] = date_obj.year
                    paper_data["publish_date"] = date_str
                except ValueError:
                    # 如果日期格式不符合预期，记录原始日期字符串
                    paper_data["published"] = date_str[:4]  # 尝试提取年份
                    paper_data["publish_date"] = date_str
            
            # 解析摘要
            abstract_line = next((i for i, line in enumerate(lines) if line.startswith("Abstract:")), None)
            if abstract_line is not None:
                abstract_lines = lines[abstract_line+1:]
                paper_data["abstract"] = "\n".join(abstract_lines).strip()
            
            # 解析论文URL
            url_line = next((line for line in lines if line.startswith("URL:")), None)
            if url_line:
                paper_data["url"] = url_line[4:].strip()
                # 生成PDF URL
                arxiv_id = paper_data["url"].split("/")[-1]
                paper_data["pdf_url"] = f"http://arxiv.org/pdf/{arxiv_id}"
                paper_data["paper_id"] = arxiv_id
            
            # 添加来源信息
            paper_data["source"] = "arxiv"
            
            papers.append(paper_data)
            
        return papers
    
    def fetch_papers_by_topic(self, topic: str, limit: int = 50, 
                             recent_days: Optional[int] = None) -> List[Dict]:
        """
        根据主题获取论文
        
        参数:
            topic: 研究主题
            limit: 结果数量限制
            recent_days: 只获取最近多少天的论文
        
        返回:
            论文元数据列表
        """
        start_date = None
        
        if recent_days:
            # 计算开始日期
            end_date = datetime.now()
            start_date = end_date - timedelta(days=recent_days)
            start_date_str = start_date.strftime("%Y-%m-%d")
            logger.info(f"获取 {topic} 主题最近 {recent_days} 天的论文")
        
        # 创建临时API实例以调整结果数量
        temp_arxiv = ArxivAPIWrapper(
            max_results=limit,
            top_k_results=min(limit, self.top_k_results),
            load_max_docs=limit,
            doc_content_chars_max=0
        )
        
        # 保存原始API实例
        original_arxiv = self.arxiv
        
        try:
            # 临时替换API实例
            self.arxiv = temp_arxiv
            papers = self.fetch_papers(
                keywords=topic,
                start_date=start_date_str if recent_days else None,
                deduplicate=True
            )
            return papers[:limit]  # 确保不超过限制数量
        finally:
            # 恢复原始API实例
            self.arxiv = original_arxiv

# 示例用法
if __name__ == "__main__":
    # fetcher = ArxivFetcher(max_results=10, top_k_results=10)
    
    # try:
    #     # 获取最近30天的LLM相关论文
    #     papers = fetcher.fetch_papers_by_topic("1102.5509", limit=5, recent_days=30)
        
    #     print(f"获取到 {len(papers)} 篇论文")
    #     for i, paper in enumerate(papers, 1):
    #         print(f"\n{i}. {paper['title']}")
    #         print(f"   作者: {', '.join(paper['authors'])}")
    #         print(f"   发布年份: {paper['published']}")
    #         print(f"   URL: {paper['url']}")
    #         print(f"   PDF URL: {paper['pdf_url']}")
    # except Exception as e:
    #     print(f"获取论文失败: {e}")
    from langchain_community.retrievers import ArxivRetriever

    # 初始化检索器
    retriever = ArxivRetriever(
        load_max_docs=2,
        get_full_documents=True  # 获取完整的文档内容
    )

    # 按文章标识符检索
    docs = retriever.invoke("1102.5509")

    # 输出文档的元数据和内容
    print(docs[0].metadata)  # 显示文献的元信息
    print("\n\n")
    print(docs[0].page_content[:2000])  # 显示文献的部分内容

