import asyncio
import os
import hashlib
import uuid
from pathlib import Path
from typing import List, Dict, Optional
import arxiv

from src.utils.log_utils import setup_logger
from src.tasks.paper_search import PaperSearch
from src.tasks.paper_filter import PaperFilter

logger = setup_logger(__name__)

class PaperDownloader:
    """论文下载器，支持异步并发下载PDF文件到本地缓存目录"""
    
    def __init__(self, cache_base_dir: str = "data/cache", max_concurrency: int = 5):
        """
        初始化论文下载器
        
        参数:
            cache_base_dir: 缓存基础目录
            max_concurrency: 最大并发数
        """
        self.cache_base_dir = Path(cache_base_dir)
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        
        # 确保基础缓存目录存在
        self.cache_base_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化搜索器和筛选器
        self.searcher = PaperSearch()
        self.filterer = PaperFilter()
    
    def _get_user_cache_dir(self, user_id: str) -> Path:
        """
        获取用户的缓存目录，实现用户隔离
        
        参数:
            user_id: 用户唯一标识符
        
        返回:
            用户缓存目录路径
        """
        # 使用用户ID的哈希值作为目录名，增加安全性
        user_hash = hashlib.md5(user_id.encode()).hexdigest()[:8]
        user_cache_dir = self.cache_base_dir / user_hash
        user_cache_dir.mkdir(parents=True, exist_ok=True)
        return user_cache_dir
    
    def _get_paper_file_path(self, user_id: str, paper_id: str, file_ext: str = "pdf") -> Path:
        """
        获取论文文件的保存路径
        
        参数:
            user_id: 用户唯一标识符
            paper_id: 论文唯一标识符
            file_ext: 文件扩展名
        
        返回:
            论文文件保存路径
        """
        user_cache_dir = self._get_user_cache_dir(user_id)
        return user_cache_dir / f"{paper_id}.{file_ext}"
    
    def is_paper_cached(self, user_id: str, paper_id: str) -> bool:
        """
        检查论文是否已经缓存
        
        参数:
            user_id: 用户唯一标识符
            paper_id: 论文唯一标识符
        
        返回:
            是否已缓存
        """
        file_path = self._get_paper_file_path(user_id, paper_id)
        return file_path.exists()
    
    async def _download_single_paper(self, 
                                    user_id: str, paper_id: str) -> Dict[str, str]:
        """
        下载单篇论文（使用arxiv库）
        
        参数:
            user_id: 用户唯一标识符
            paper_id: 论文唯一标识符
        
        返回:
            包含下载状态的字典
        """
        # 检查是否已缓存
        if self.is_paper_cached(user_id, paper_id):
            logger.info(f"论文已缓存: {paper_id}")
            return {
                "paper_id": paper_id,
                "status": "cached",
                "path": str(self._get_paper_file_path(user_id, paper_id))
            }
        
        # 使用信号量限制并发
        async with self.semaphore:
            try:
                logger.info(f"开始下载论文: {paper_id}")
                # 使用arxiv库的论文ID直接搜索
                search = arxiv.Search(id_list=[paper_id])
                paper = next(search.results())
                
                # 获取用户缓存目录
                user_cache_dir = self._get_user_cache_dir(user_id)
                
                # 使用arxiv库的download_pdf方法下载
                pdf_path = paper.download_pdf(dirpath=str(user_cache_dir))
                
                logger.info(f"论文下载成功: {paper_id}")
                return {
                    "paper_id": paper_id,
                    "status": "success",
                    "path": str(pdf_path)
                }
            except StopIteration:
                logger.error(f"未找到论文: {paper_id}")
                return {
                    "paper_id": paper_id,
                    "status": "failed",
                    "error": "论文不存在"
                }
            except Exception as e:
                logger.error(f"论文下载异常: {paper_id}, 错误: {str(e)}")
                return {
                    "paper_id": paper_id,
                    "status": "failed",
                    "error": str(e)
                }
    
    async def search_filter_and_download(self, 
                                        user_id: str, 
                                        query: str, 
                                        max_results: int = 20, 
                                        recent_days: Optional[int] = None, 
                                        year_threshold: int = 2020) -> List[Dict[str, str]]:
        """
        一站式搜索、筛选和下载高质量论文
        
        参数:
            user_id: 用户唯一标识符
            query: 搜索关键词
            max_results: 最大搜索结果数量
            recent_days: 搜索最近多少天的论文，None表示不限制
            year_threshold: 高质量论文的年份阈值
        
        返回:
            下载结果列表
        """
        try:
            # 1. 搜索论文
            logger.info(f"开始搜索论文: query='{query}', max_results={max_results}")
            papers = self.searcher.search_by_topic(query, limit=max_results, recent_days=recent_days)
            logger.info(f"搜索到的论文数量：'{len(papers)}篇")
            
            if not papers:
                logger.warning(f"未找到匹配的论文: {query}")
                return []
            
            # 2. 去重
            papers = self.filterer.deduplicate_papers(papers)
            logger.info(f"去重后的论文数量：'{len(papers)}篇")
            
            # 3. 筛选高质量论文
            high_quality_papers = self.filterer.filter_high_quality_papers(papers, year_threshold=year_threshold)
            logger.info(f"筛选后的论文数量：'{len(papers)}篇")

            if not high_quality_papers:
                logger.warning(f"未找到符合高质量标准的论文")
                return []
            
            # 4. 准备下载列表（只需要paper_id）
            paper_ids = [paper["paper_id"] for paper in high_quality_papers]
            
            logger.info(f"准备下载 {len(paper_ids)} 篇高质量论文")
            
            # 5. 下载论文
            results = await self.download_papers(user_id, paper_ids)
            
            # 6. 将下载结果与论文信息合并
            for i, result in enumerate(results):
                if result["status"] in ["success", "cached"]:
                    results[i]["paper_info"] = high_quality_papers[i]
            
            return results
        except Exception as e:
            logger.error(f"搜索、筛选和下载论文过程中发生错误: {str(e)}")
            raise
    
    async def download_papers(self, user_id: str, paper_ids: List[str]) -> List[Dict[str, str]]:
        """
        异步并发下载多篇论文（使用arxiv库）
        
        参数:
            user_id: 用户唯一标识符
            paper_ids: 论文ID列表
        
        返回:
            下载结果列表
        """
        if not paper_ids:
            return []
            
        logger.info(f"开始批量下载论文，共 {len(paper_ids)} 篇，最大并发数: {self.max_concurrency}")
        
        # 创建下载任务列表
        tasks = [
            self._download_single_paper(user_id, paper_id)
            for paper_id in paper_ids
        ]
        
        # 并发执行所有下载任务
        results = await asyncio.gather(*tasks)
        
        # 统计下载结果
        success_count = sum(1 for r in results if r["status"] in ["success", "cached"])
        logger.info(f"论文下载完成，成功/缓存: {success_count}, 失败: {len(results) - success_count}")
        
        return results

# 辅助函数：生成临时用户ID
def generate_temp_user_id() -> str:
    """生成临时用户ID"""
    return str(uuid.uuid4())


if __name__ == "__main__":
    async def demo_search_filter_and_download():
        downloader = PaperDownloader(max_concurrency=3)
        
        # 生成临时用户ID
        user_id = generate_temp_user_id()
        
        # 一站式搜索、筛选和下载高质量论文
        try:
            results = await downloader.search_filter_and_download(
                user_id=user_id,
                query="large language models",
                max_results=10,
                recent_days=30,
                year_threshold=2020
            )
            
            # 打印下载结果
            print(f"\n下载完成，共处理 {len(results)} 篇论文")
            success_count = sum(1 for r in results if r["status"] in ["success", "cached"])
            print(f"成功/缓存: {success_count}, 失败: {len(results) - success_count}")
            
            # 打印成功下载的论文信息
            for result in results:
                if result["status"] in ["success", "cached"]:
                    print(f"\n论文ID: {result['paper_id']}")
                    print(f"标题: {result['paper_info']['title']}")
                    print(f"作者: {', '.join(result['paper_info']['authors'])}")
                    print(f"状态: {result['status']}")
                    print(f"保存路径: {result['path']}")
        except Exception as e:
            print(f"演示过程中发生错误: {e}")
    
    # 运行演示
    asyncio.run(demo_search_filter_and_download())