import unittest

from src.agents import ReviewRequest
from src.agents.base import AgentContext
from src.agents.searchAgent import SearchAgent, build_search_agent
from src.llm.base import LLMResponse


class _FakeProvider:
    """用于验证 SearchAgent 是否真的调用了大模型。"""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls = []

    def chat_with_retry(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return LLMResponse(content=self.response_text, finish_reason="stop")


class _FakeSnapshot:
    """最小化模拟 ProviderSnapshot，只暴露 SearchAgent 需要的 provider。"""

    def __init__(self, provider):
        self.provider = provider


class SearchAgentToolIntegrationTest(unittest.TestCase):
    def test_build_search_agent_returns_search_agent(self):
        """验证搜索 Agent 可以被正常构建。"""

        agent = build_search_agent(llm=None)

        self.assertEqual(agent.spec.name, "search_agent")
        self.assertEqual(agent.spec.role, "search")

    def test_search_agent_uses_llm_to_generate_keywords(self):
        """验证 SearchAgent 会调用大模型生成检索关键词。"""

        provider = _FakeProvider('{"keywords":["multi-agent systems","literature review automation","academic retrieval"]}')
        agent = SearchAgent(AgentContext(spec=SearchAgent.spec, llm=_FakeSnapshot(provider)))
        state = {
            "request": ReviewRequest(
                topic="多智能体文献综述自动化",
                constraints={"sources": ["openalex"], "max_results": 5, "year_from": 2022},
            )
        }

        output = agent.run(state)

        self.assertEqual(len(provider.calls), 1)
        self.assertTrue(output["diagnostics"]["used_llm"])
        self.assertIn("multi-agent systems", output["search_intent"].keywords)
        self.assertFalse(output["search_halted"])
        self.assertEqual(output["search_intent"].topic, "多智能体文献综述自动化")
        self.assertIn("academic retrieval", output["search_intent"].keywords)

    def test_search_agent_halts_when_llm_is_missing(self):
        """验证没有 LLM 时直接终止搜索，不再启用规则关键词兜底。"""

        agent = build_search_agent(llm=None)
        state = {
            "request": ReviewRequest(
                topic="literature review automation",
                constraints={"sources": ["openalex"], "max_results": 5, "year_from": 2022},
            )
        }

        output = agent.run(state)

        self.assertFalse(output["diagnostics"]["used_llm"])
        self.assertTrue(output["search_halted"])
        self.assertEqual(output["search_intent"].keywords, [])


if __name__ == "__main__":
    unittest.main()
