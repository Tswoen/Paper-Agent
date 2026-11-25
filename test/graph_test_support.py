"""图测试共用的替身对象与 hermetic 环境。

中文说明：这里的对象只服务于测试，不进入生产代码。`hermetic_graph_llms` 的作用是
把图测试固定在“本地没有模型配置”的状态，避免测试结果随开发者本地的
`config/model.json` 漂移：一旦该文件存在，写作和大纲节点会自己去加载真实
provider，让端到端用例变成慢、花钱、结果不确定的模型调用。
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest import mock

from src.agents.analyseAgent import AnalyseModelResult


class StubAnalyseAgent:
    """让分析节点在没有模型配置时也能走完的替身。

    中文说明：子主题分析返回 `parsed=None`，节点会改用 `_fallback_subtopic_analysis`；
    全局分析返回空的 `parsed`，`_normalize_overall_analysis` 会给八个字段全部补上
    兜底文案。这样测试既不依赖模型的输出格式，也不需要维护任何 JSON fixture。
    """

    async def async_analyse_subtopic(self, *, topic, group, usage_callback=None):
        """返回“模型不可用”的结果，让节点走子主题兜底。"""

        return AnalyseModelResult(parsed=None, reason="stub: 子主题分析走兜底")

    async def async_analyse_overall(self, *, topic, subtopic_analyses, usage_callback=None):
        """返回空 JSON，让节点用服务端兜底补全八部分综合分析。"""

        return AnalyseModelResult(parsed={}, raw_model_output="{}")


@contextmanager
def hermetic_graph_llms():
    """把图执行期间的模型装配钉死，使测试与本地 model.json 解耦。

    中文说明：分析节点换替身，三个 `load_*_agent_llm` 一律返回 None。
    后三个不是可选项——写作和大纲节点在本地存在 model.json 时会加载真实 provider，
    那会让这组测试既慢又不可复现。
    """

    with (
        mock.patch("src.graph.analyse_node.build_analyse_agent", lambda *args, **kwargs: StubAnalyseAgent()),
        mock.patch("src.graph.analyse_node.load_analyse_agent_llm", lambda *args, **kwargs: None),
        mock.patch("src.graph.writing_outline_node.load_writing_outline_agent_llm", lambda *args, **kwargs: None),
        mock.patch("src.graph.writing_node.load_writing_agent_llm", lambda *args, **kwargs: None),
    ):
        yield
