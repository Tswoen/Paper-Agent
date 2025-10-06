from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from src.core.prompts import retrieval_agent_prompt
from typing import Dict, Any
from src.utils.log_utils import setup_logger

from src.core.model_client import create_default_client, create_subwriting_retrieval_model_client
from src.services.retrieval_tool import retrieval_tool
from src.agents.sub_writing_agent.writing_state_models import WritingState, SectionState
import asyncio

logger = setup_logger(__name__)


model_client = create_subwriting_retrieval_model_client()


retriever = FunctionTool(retrieval_tool, description="用于联网查询外部资料，来辅助写作的工具")

retrieval_agent = AssistantAgent(
    name="retrieval_agent",
    model_client=model_client,
    # tools=[retriever],
    description="一个检索助手，负责根据条件联网查询外部资料。",
    system_message=retrieval_agent_prompt,
)

def parse_to_list(s: str) -> list[str]:
    # 使用正则表达式提取[]之间的内容
    # 只保留第一个匹配到的[]对中的内容
    import re
    # 去除首尾的换行符和空格
    s = s.strip()

    match = re.search(r'\[(.*?)\]', s, re.DOTALL)
    if not match:
        return []
    
    content = match.group(1).strip()
    if not content:
        return []

    content = content.replace('，', ',')  # 中文逗号替换为英文逗号
    # 按逗号分割并过滤空字符串
    items = [item.strip() for item in content.split(',') if item.strip()]
    return items
    
async def retrieval_node(state: WritingState) -> Dict[str, Any]:
    logger.info("开始写作检索节点")

    try:
        # writted_sections = state["writted_sections"]
        retrieved_docs = state["retrieved_docs"]

        # query = writted_sections[-1].content
        # logger.info(f"检索条件: {query}")
        # querys = parse_to_list(query)
        querys  = ['语言模型', '大模型','语言模型原理']
        # retrieved_docs = []
        # 将querys并行交给retrieval_tool去执行，并将结果合并
        results = retrieval_tool(querys)
        # results = await asyncio.gather(*[retrieval_tool(query) for query in querys])
        # 去重
        for result in results:
            for paper in result:
                flag = True
                for doc in retrieved_docs:
                    if paper["paper_id"] == doc["paper_id"]:
                        flag = False
                if flag:
                    retrieved_docs.append(paper)


        # 生成检索条件
        # response = await retrieval_agent.run(task = prompt)
        # content = response.messages[-1].content

        # 调用检索服务
        # retrieved_docs = retrieval_tool(content)
        # retrieved_docs = [{"paper_id": "1", "title": "langraph介绍", "abstract": "是一个AI框架", "content": "LangGraph 是一个专为构建复杂、有状态的 AI 智能体（Agent）工作流设计的框架，基于图状态机（Graph State Machine）架构，由 LangChain 团队开发，可看作是对 LangChain 的扩展与增强。"}]
        return {"retrieved_docs": retrieved_docs}
    except Exception as e:
        logger.error(f"Retrieval failed: {str(e)}")
        return state

def main(state: WritingState):
    """主函数"""
    asyncio.run(retrieval_node(state))

if __name__ == "__main__":
    from src.agents.sub_writing_agent.writing_state_models import WritingState
    state_queue = asyncio.Queue()
    state = WritingState(
            user_request="帮我写一篇关于人工智能的调研报告",
            retrieved_docs=[],
        )
    main(state)