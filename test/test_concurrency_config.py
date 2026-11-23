"""5a-4 并发配置测试：system.yaml 顶层 concurrency → 运行时资源 → 图运行参数。

覆盖三段注入链：
1. 配置层：ConcurrencySettings 的默认值、解析与非法值回退、YAML 加载；
2. 组合根：SessionRunService 把配置翻译成 RuntimeConcurrencyLimits；
3. 图入口：_build_run_config 把论文级并发写进 RunnableConfig 的 max_concurrency。

全部使用临时目录，不触碰真实 data/。
"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.graph.graph import _build_run_config
from src.graph.runtime import WorkflowRuntimeContext
from src.graph.runtime_resources import RuntimeConcurrencyLimits, WorkflowRuntimeResources
from src.llm.config import ConcurrencySettings, SystemConfig
from src.services.session_runs import _concurrency_limits_from_system


class ConcurrencySettingsTests(unittest.TestCase):
    """配置层：默认值、解析与容错。"""

    def test_defaults_match_legacy_hardcoded_values(self):
        """缺省配置时五个上限必须与断代前的硬编码默认一致（4/3/3/2/1）。"""

        settings = SystemConfig.from_dict({}).concurrency
        self.assertEqual(settings, ConcurrencySettings())
        self.assertEqual(settings.search_source, 4)
        self.assertEqual(settings.paper_task, 3)
        self.assertEqual(settings.download, 3)
        self.assertEqual(settings.read_model, 2)
        self.assertEqual(settings.embedding, 1)
        # 配置默认值与运行时 limits 的兜底默认值也要对齐，避免两条回退路径不一致。
        limits = RuntimeConcurrencyLimits()
        self.assertEqual(limits.search_source_concurrency, settings.search_source)
        self.assertEqual(limits.paper_task_concurrency, settings.paper_task)
        self.assertEqual(limits.download_concurrency, settings.download)
        self.assertEqual(limits.read_model_concurrency, settings.read_model)
        self.assertEqual(limits.embedding_concurrency, settings.embedding)

    def test_from_dict_parses_concurrency_section(self):
        config = SystemConfig.from_dict(
            {
                "concurrency": {
                    "search_source": 6,
                    "paper_task": 5,
                    "download": 4,
                    "read_model": 3,
                    "embedding": 2,
                }
            }
        )

        self.assertEqual(config.concurrency.search_source, 6)
        self.assertEqual(config.concurrency.paper_task, 5)
        self.assertEqual(config.concurrency.download, 4)
        self.assertEqual(config.concurrency.read_model, 3)
        self.assertEqual(config.concurrency.embedding, 2)

    def test_invalid_values_fall_back_per_key(self):
        """缺省、null、非正数、非数字都逐键回退默认，不影响其他键。"""

        config = SystemConfig.from_dict(
            {
                "concurrency": {
                    "search_source": 0,
                    "paper_task": -2,
                    "download": "abc",
                    "read_model": None,
                    "embedding": "7",
                }
            }
        )

        self.assertEqual(config.concurrency.search_source, 4)
        self.assertEqual(config.concurrency.paper_task, 3)
        self.assertEqual(config.concurrency.download, 3)
        self.assertEqual(config.concurrency.read_model, 2)
        self.assertEqual(config.concurrency.embedding, 7)

    def test_yaml_load_reads_concurrency_block(self):
        """手写的二级 YAML 解析要能读到顶层 concurrency 段（含注释行）。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "system.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "# 并发上限",
                        "concurrency:",
                        "  search_source: 8",
                        "  paper_task: 6",
                        "  download: 5",
                        "  read_model: 4",
                        "  embedding: 2",
                    ]
                ),
                encoding="utf-8",
            )

            config = SystemConfig.load(config_path)

        self.assertEqual(config.concurrency.search_source, 8)
        self.assertEqual(config.concurrency.paper_task, 6)
        self.assertEqual(config.concurrency.download, 5)
        self.assertEqual(config.concurrency.read_model, 4)
        self.assertEqual(config.concurrency.embedding, 2)

    def test_repository_system_yaml_has_concurrency_defaults(self):
        config = SystemConfig.load("config/system.yaml")

        self.assertEqual(config.concurrency.search_source, 4)
        self.assertEqual(config.concurrency.paper_task, 3)
        self.assertEqual(config.concurrency.download, 3)
        self.assertEqual(config.concurrency.read_model, 2)
        self.assertEqual(config.concurrency.embedding, 1)


class ConcurrencyInjectionTests(unittest.TestCase):
    """注入链：组合根映射、资源信号量与图运行参数。"""

    def test_session_run_limits_mapping(self):
        """组合根把配置五个键逐一映射到 RuntimeConcurrencyLimits。"""

        custom = SystemConfig.from_dict(
            {
                "concurrency": {
                    "search_source": 6,
                    "paper_task": 5,
                    "download": 4,
                    "read_model": 3,
                    "embedding": 2,
                }
            }
        )
        with mock.patch(
            "src.services.session_runs.SystemConfig.load",
            return_value=custom,
        ):
            limits = _concurrency_limits_from_system()

        self.assertEqual(limits.search_source_concurrency, 6)
        self.assertEqual(limits.paper_task_concurrency, 5)
        self.assertEqual(limits.download_concurrency, 4)
        self.assertEqual(limits.read_model_concurrency, 3)
        self.assertEqual(limits.embedding_concurrency, 2)

    def test_limits_flow_into_semaphores(self):
        """注入的 limits 要真实决定五个资源级信号量的初值。"""

        limits = RuntimeConcurrencyLimits(
            search_source_concurrency=6,
            paper_task_concurrency=5,
            download_concurrency=4,
            read_model_concurrency=3,
            embedding_concurrency=2,
        )
        resources = WorkflowRuntimeResources(limits=limits)
        try:
            # 中文注释：Semaphore 没有公开的余量读取接口，测试直接看内部计数。
            self.assertEqual(resources.search_source_semaphore._value, 6)
            self.assertEqual(resources.paper_task_semaphore._value, 5)
            self.assertEqual(resources.download_semaphore._value, 4)
            self.assertEqual(resources.read_model_semaphore._value, 3)
            self.assertEqual(resources.embedding_semaphore._value, 2)
        finally:
            asyncio.run(resources.aclose())

    def test_build_run_config_injects_max_concurrency(self):
        """图运行参数带上论文级并发；thread_id 与 metadata 行为保持不变。"""

        resources = WorkflowRuntimeResources(
            limits=RuntimeConcurrencyLimits(paper_task_concurrency=5)
        )
        try:
            runtime = WorkflowRuntimeContext(
                session_key="s1",
                turn_id="t1",
                run_id="run-1",
                resources=resources,
            )
            config = _build_run_config(runtime, thread_id="s1:t1")
        finally:
            asyncio.run(resources.aclose())

        self.assertEqual(config["max_concurrency"], 5)
        self.assertEqual(config["configurable"], {"thread_id": "s1:t1"})
        self.assertEqual(config["tags"], ["paper_graph"])
        self.assertEqual(config["metadata"]["session_key"], "s1")
        self.assertEqual(config["metadata"]["turn_id"], "t1")
        self.assertEqual(config["metadata"]["run_id"], "run-1")

    def test_build_run_config_falls_back_without_resources(self):
        """没有 runtime 或没有 resources 时回退默认论文并发 3，而不是缺键。"""

        config_without_runtime = _build_run_config(None, thread_id="s1:t1")
        self.assertEqual(config_without_runtime["max_concurrency"], 3)

        runtime_without_resources = WorkflowRuntimeContext(session_key="s1", turn_id="t1")
        config_without_resources = _build_run_config(runtime_without_resources)
        self.assertEqual(config_without_resources["max_concurrency"], 3)
        self.assertNotIn("configurable", config_without_resources)


if __name__ == "__main__":
    unittest.main()
