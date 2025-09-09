serapi_key =  "4a0d7556aaa9bdf806083661d38e205c838fdae709e38a714b84e22315d3f1fa"

from langchain_community.utilities import SerpAPIWrapper

# 初始化SerpAPI的包装器
search = SerpAPIWrapper(serpapi_api_key=serapi_key)

def retrieval_tool(query: str) -> str:
    """
    使用SerpAPI进行网络检索并返回结果。
    """
    return search.run(query)
