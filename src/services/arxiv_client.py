import arxiv
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.core.config import config
from src.utils.log_utils import setup_logger

# 设置日志
def get_logger():
    return setup_logger(name='arxiv_fetcher', log_file='arxiv_fetcher.log')

logger = get_logger()

class ArxivFetcher:
    """ArXiv论文数据抓取器"""
    
    def __init__(self):
        # 从配置中获取参数
        self.max_papers = int(config.get('MAX_PAPERS', 50))
        self.batch_size = int(config.get('ARXIV_BATCH_SIZE', 100))
        
        logger.info(f"ArXiv抓取器初始化完成: 最大论文数={self.max_papers}, 批大小={self.batch_size}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def fetch_papers(self, query=None, sort_by='submittedDate', sort_order='descending'):
        """
        从ArXiv抓取论文数据
        
        参数:
            query: 搜索查询字符串，默认为None（抓取最新论文）
            sort_by: 排序字段，默认为'submittedDate'
            sort_order: 排序顺序，默认为'descending'
        
        返回:
            论文元数据列表
        """
        # 默认查询 - 如果没有提供查询字符串
        if query is None:
            # query = 'submittedDate:[202301010600 TO 202401010600]'
            query = 'cat:cs.* OR cat:stat.ML OR cat:q-bio.* OR cat:physics.comp-ph'
        
        logger.info(f"开始抓取ArXiv论文，查询: {query}")
        
        try:
            # 创建arxiv客户端
            search = arxiv.Search(
                query=query,
                max_results=self.max_papers,
                sort_by=arxiv.SortCriterion.SubmittedDate if sort_by == 'submittedDate' else arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending if sort_order == 'descending' else arxiv.SortOrder.Ascending
            )
            
            # 获取论文迭代器
            results = arxiv.Client().results(search)
            
            # 转换为我们需要的格式
            papers = []
            for i, paper in enumerate(results):
                # 提取作者信息
                authors = [author.name for author in paper.authors]
                
                # 提取分类标签
                categories = paper.categories
                
                # 构建论文对象
                paper_data = {
                    'arxiv_id': paper.get_short_id(),
                    'title': paper.title.strip(),
                    'abstract': paper.summary.strip(),
                    'authors': authors,
                    'published': paper.published.isoformat(),
                    'categories': categories,
                    'link': paper.entry_id
                }
                papers.append(paper_data)

                logger.info(f"第{i+1}篇论文是：{paper_data}")

                # 记录进度
                if (i + 1) % 10 == 0 or (i + 1) == self.max_papers:
                    logger.info(f"已抓取 {i + 1}/{self.max_papers} 篇论文")
                    

                # 避免请求过于频繁，添加延迟
                if i < self.max_papers - 1:
                    time.sleep(1)  # 减少请求频率，避免触发API限制
            
            logger.info(f"ArXiv论文抓取完成，共获取 {len(papers)} 篇论文")
            return papers
        except Exception as e:
            logger.error(f"抓取ArXiv论文时出错: {e}")
            raise

# 创建全局实例
arxiv_fetcher = ArxivFetcher()

# 方便直接调用的函数
def fetch_arxiv_papers(query=None, max_papers=None, sort_by='relevance', sort_order='descending'):
    """
    方便直接调用的函数，从ArXiv抓取论文数据
    
    参数:
        query: 搜索查询字符串
        max_papers: 最大抓取论文数，优先级高于配置中的MAX_PAPERS
        sort_by: 排序字段
        sort_order: 排序顺序
    
    返回:
        论文元数据列表
    """
    # 如果提供了max_papers，临时修改配置
    original_max_papers = None
    if max_papers is not None:
        original_max_papers = arxiv_fetcher.max_papers
        arxiv_fetcher.max_papers = max_papers
    
    try:
        return arxiv_fetcher.fetch_papers(query, sort_by, sort_order)
    finally:
        # 恢复原始配置
        if original_max_papers is not None:
            arxiv_fetcher.max_papers = original_max_papers

if __name__ == '__main__':
    # 测试抓取功能
    try:
        papers = fetch_arxiv_papers(max_papers=10)
        logger.info(f"测试抓取成功，获取了 {len(papers)} 篇论文")
        if papers:
            logger.info(f"第一篇论文标题: {papers[0]['title']}")
    except Exception as e:
        logger.error(f"测试抓取失败: {e}")