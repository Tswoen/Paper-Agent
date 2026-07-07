import json
import os
import unittest
from pathlib import Path

from src.llm import ModelConfig, StreamCallbacks, SystemConfig, make_provider


def _live_connectivity_test_enabled() -> bool:
    """判断是否显式开启真实模型连通性测试。

    中文说明：
    1. 该测试会真实请求外部模型服务；
    2. 可能消耗额度，并受到账号限流、余额、网络环境影响；
    3. 因此默认关闭，只有在手动设置环境变量后才执行。
    """

    return True


def _resolve_agent_name(config: ModelConfig) -> str:
    """从模型配置中解析要用于连通性测试的 Agent 名称。

    中文说明：
    1. 优先读取环境变量 `MODEL_TEST_AGENT`，便于手动指定目标 Agent；
    2. 若未指定，则优先使用 `default_agent`；
    3. 如果配置中没有 `default_agent`，则回退到第一个可用 Agent。
    """

    configured_agent_name = os.getenv("MODEL_TEST_AGENT", "").strip()
    if configured_agent_name:
        return configured_agent_name
    if "default_agent" in config.agents:
        return "default_agent"
    first_agent_name = next(iter(config.agents), "")
    if first_agent_name:
        return first_agent_name
    raise AssertionError("config/model.json 中未找到可用的 agent 配置")


@unittest.skipUnless(
    _live_connectivity_test_enabled(),
    "set RUN_MODEL_CONNECTIVITY_TEST=1 to run live model connectivity test",
)
class ModelConnectivityTest(unittest.TestCase):
    """真实模型连通性测试。

    中文说明：
    该测试类用于验证当前 `config/model.json` 与 `config/system.yaml`
    对应的模型配置是否可真实连通，并返回有效响应。
    """

    def _load_snapshot(self):
        """加载当前项目配置并创建可用于连通性测试的 Provider 快照。

        中文说明：
        该函数会复用项目正式运行时的配置装配逻辑，避免测试和生产环境
        使用两套不同的模型初始化方式，确保测试结果更可信。
        """

        config_path = Path("config/model.json")
        self.assertTrue(config_path.exists(), "config/model.json 不存在，无法执行模型连通性测试")

        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        system_config = SystemConfig.load()
        model_config = ModelConfig.from_dict(config_data, system_config)
        agent_name = _resolve_agent_name(model_config)
        provider_snapshot = make_provider(model_config, agent_name)
        return agent_name, provider_snapshot

    def test_model_chat_connectivity(self):
        """验证非流式对话接口是否可以成功连通并返回文本。

        中文说明：
        这是最直接的连通性检查：
        1. 能否正确鉴权；
        2. 能否访问目标 base URL；
        3. 能否让目标模型返回有效文本；
        4. 出错时是否能得到清晰的错误信息。
        """

        agent_name, snapshot = self._load_snapshot()
        response = snapshot.provider.chat_with_retry(
            [{"role": "user", "content": "请只回复 OK"}],
            max_tokens=16,
        )

        self.assertTrue(
            response.ok,
            (
                f"模型连通性测试失败 | agent={agent_name} | model={snapshot.model} | "
                f"status_code={response.error_status_code} | error_kind={response.error_kind} | "
                f"error_type={response.error_type} | error_code={response.error_code} | "
                f"message={response.content}"
            ),
        )
        self.assertTrue(
            (response.content or "").strip(),
            f"模型已连通，但返回内容为空 | agent={agent_name} | model={snapshot.model}",
        )

    def test_model_stream_connectivity(self):
        """验证流式对话接口是否可以成功连通并收到增量输出。

        中文说明：
        很多时候非流式接口正常，但流式接口会因为网关转发、SSE 支持、
        厂商兼容层差异而失败，因此这里额外补一条流式连通性验证。
        """

        agent_name, snapshot = self._load_snapshot()
        deltas: list[str] = []
        response = snapshot.provider.chat_stream_with_retry(
            [{"role": "user", "content": "请只回复 OK"}],
            StreamCallbacks(on_content_delta=deltas.append),
            max_tokens=16,
        )

        self.assertTrue(
            response.ok,
            (
                f"模型流式连通性测试失败 | agent={agent_name} | model={snapshot.model} | "
                f"status_code={response.error_status_code} | error_kind={response.error_kind} | "
                f"error_type={response.error_type} | error_code={response.error_code} | "
                f"message={response.content}"
            ),
        )
        self.assertTrue(
            (response.content or "").strip(),
            f"模型流式调用已连通，但最终聚合内容为空 | agent={agent_name} | model={snapshot.model}",
        )
        self.assertTrue(
            "".join(deltas).strip(),
            f"模型流式调用已连通，但没有收到任何增量内容 | agent={agent_name} | model={snapshot.model}",
        )


if __name__ == "__main__":
    unittest.main()
