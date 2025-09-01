from data_collection.data_fetcher import fetch_arxiv_papers
from utils.log_utils import setup_logger

# 设置日志
def get_logger():
    return setup_logger(name='main', log_file='project.log')

logger = get_logger()

papers = fetch_arxiv_papers(max_papers=10)

logger.info(f"测试抓取成功，获取了 {len(papers)} 篇论文")