# from semanticscholar import SemanticScholar
# import os

# 设置代理（替换为你的代理地址）
# os.environ["http_proxy"] = "http://127.0.0.1:7897"
# os.environ["https_proxy"] = "http://127.0.0.1:7897"
# scholar = SemanticScholar()
# paper = scholar.get_paper('2509.15218')
# results = scholar.search_paper('machine learning',limit=1)
# print(paper.title)
# all_results = [item for item in results]
# print(all_results)
# ARXIV:

import os
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper

# 设置API密钥
os.environ["SERP_API_KEY"] = "4a0d7556aaa9bdf806083661d38e205c838fdae709e38a714b84e22315d3f1fa"

# 创建工具实例
tool = GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper())

results = tool.run("LLM Models author:\'E Ferrara\'")
print(results)
# for paper in results:
#     print(f"Title: {paper['title']}")
#     print(f"Authors: {paper['authors']}")
#     print(f"Summary: {paper['summary']}")
#     print(f"Total-Citations: {paper['total_citations']}")
#     print("\n")
