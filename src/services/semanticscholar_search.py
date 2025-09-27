from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

import os
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from semanticscholar import SemanticScholar
import json
import httpx

# 设置API密钥
os.environ["SERP_API_KEY"] = "4a0d7556aaa9bdf806083661d38e205c838fdae709e38a714b84e22315d3f1fa"
proxies = {
    "http://": "http://127.0.0.1:7890",
    "https://": "http://127.0.0.1:7890",
}
# 创建工具实例
def create_google_scholar_tool():
    api_wrapper=GoogleScholarAPIWrapper()
    query = "LLM Models author: \'E Ferrara\'"
    results = api_wrapper.run(query)  # 直接返回结构化数据列表
    print(results)
    print("=================="*5)
    # 验证结果结构
    if results:
        print("\n第一篇论文的结构化数据:")
        structured_results = json.dumps(results, indent=2, ensure_ascii=False)
        print(structured_results)

def semantic_scholar_tool():
    from semanticscholar import SemanticScholar

    sch = SemanticScholar()

    # 搜索
    results = sch.search_paper("Graph Neural Network", limit=2)

    for paper in results:
        paper_id = paper['paperId']   # 每篇文章的唯一ID
        detail = sch.get_paper(paper_id)  # 拉取详情，包括摘要
        print("标题:", detail['title'])
        print("摘要:", detail.get('abstract', "无摘要"))
        print("="*60)
    
def scholar_search():
    from scholarly import scholarly

# 搜索关键词
    search_query = scholarly.search_pubs("Graph Neural Network")

    # 获取前 5 篇
    for i in range(5):
        paper = next(search_query)
        print("标题:", paper['bib']['title'])
        print("作者:", paper['bib'].get('author'))
        print("摘要:", paper['bib'].get('abstract', "无摘要"))
        print("年份:", paper['bib'].get('pub_year'))
        print("引用数:", paper.get('num_citations'))
        print("="*50)

def habanero_search():
    from habanero import Crossref

    cr = Crossref()

    # 查询关键词
    res = cr.works(query="Graph Neural Network", limit=5)

    # 输出结果
    for item in res['message']['items']:
        print("标题:", item.get('title', ["无标题"])[0])
        authors = item.get('author', [])
        author_names = [f"{a.get('given', '')} {a.get('family', '')}" for a in authors]
        print("作者:", ", ".join(author_names))
        print("出版年:", item.get('published-print', item.get('published-online', {})).get('date-parts', [["未知"]])[0][0])
        print("DOI:", item.get('DOI'))
        # CrossRef API 一般没有完整摘要，部分期刊可能有
        print("摘要:", item.get('abstract', "无摘要"))
        print("="*50)


if __name__ == "__main__":
    habanero_search()

