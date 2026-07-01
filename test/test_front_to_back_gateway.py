import unittest

from fastapi.testclient import TestClient

from src.api import GatewayConfig, create_app
from src.presentation.stream_aggregator import ChatStreamAggregator
from src.repositories.sessions.sqlite import SQLiteSessionRepository
from src.repositories.settings.json import SettingsRepository

# 中文注释：兼容 unittest 的两种入口方式，既支持 discover，也支持直接模块运行。
try:
    from .frontend_api_client import FrontendApiClient
except ImportError:
    from frontend_api_client import FrontendApiClient


class FrontToBackFastApiTest(unittest.TestCase):
    """前后端网关联调测试。

    中文说明：
    这一组测试主要覆盖单机版 FastAPI 网关的核心链路，包括：
    1. bootstrap 启动信息。
    2. session 的创建、查询与消息提交流程。
    3. 事件流聚合成前端时间线的行为。
    """

    def _client(self):
        """构造测试用应用与仓储对象。

        中文说明：
        这里显式注入 settings 仓储和 sessions 仓储，确保测试只依赖当前
        进程内的固定初始数据，不受外部真实配置文件影响。
        """

        settings = SettingsRepository(
            initial={
                "providers": {"openai": {"api_key": "sk-test", "api_base": "https://api.openai.com/v1"}},
                "agents": {"default_agent": {"model_name": "gpt-5-mini", "provider": "openai"}},
                "embedding_profiles": {},
            }
        )
        sessions = SQLiteSessionRepository()
        app = create_app(
            settings_repo=settings,
            sessions_repo=sessions,
            config=GatewayConfig(),
        )
        return TestClient(app), sessions

    def _api(self, client: TestClient) -> FrontendApiClient:
        """把 FastAPI TestClient 包装成前端 API 客户端约定的 transport。

        中文说明：
        测试中统一走 `FrontendApiClient`，这样可以尽量模拟真实前端访问后端
        的方式，而不是在测试里手写大量重复请求细节。
        """

        def transport(method, url, body, timeout):
            """执行单次 HTTP 调用并返回客户端约定的结果格式。"""

            response = client.request(method, url, json=body)
            return response.status_code, response.json()

        return FrontendApiClient(transport)

    def test_bootstrap_exposes_local_runtime_capabilities(self):
        """验证 bootstrap 暴露的本地运行能力和 session 基础能力。"""

        client, _ = self._client()
        api = self._api(client)

        bootstrap = api.bootstrap()
        created = api.create_session("Paper reading")
        sessions = api.list_sessions()

        self.assertFalse(bootstrap["runtime_capabilities"]["websocket_stream"])
        self.assertTrue(bootstrap["runtime_capabilities"]["fastapi_rest"])
        self.assertFalse(bootstrap["runtime_capabilities"]["auth_required"])
        self.assertEqual(created["session"]["title"], "Paper reading")
        self.assertEqual(sessions["sessions"][0]["key"], created["session"]["key"])

    def test_api_is_directly_available_without_token(self):
        """验证单机场景下 API 可直接调用，不需要 token。"""

        client, _ = self._client()
        api = self._api(client)

        created = api.create_session("No Auth")
        sessions = api.list_sessions()

        self.assertEqual(created["session"]["title"], "No Auth")
        self.assertEqual(sessions["sessions"][0]["key"], created["session"]["key"])

    def test_http_message_submit_updates_thread(self):
        """验证消息提交完成后线程快照会同步更新。"""

        client, _ = self._client()
        api = self._api(client)
        created = api.create_session("Thread")["session"]

        result = api.send_message(created["key"], "总结这篇论文")
        thread = api.fetch_thread(created["key"])

        self.assertEqual(result["events"][-1]["event"], "turn_end")
        self.assertEqual(thread["messages"][0]["role"], "user")
        self.assertEqual(thread["messages"][0]["content"], "总结这篇论文")

    def test_http_events_can_be_aggregated_into_ui_timeline(self):
        """验证 HTTP 返回的事件流能够被聚合成前端时间线。"""

        client, _ = self._client()
        api = self._api(client)
        created = api.create_session("A")["session"]
        aggregator = ChatStreamAggregator()

        aggregator.add_optimistic_user_message("hello", turn_id="turn-1")
        result = api.send_message(created["key"], "hello", turn_id="turn-1")
        for event in result["events"]:
            aggregator.apply(event)
        snapshot = aggregator.snapshot()
        assistant = snapshot["messages"][-1]

        self.assertFalse(snapshot["is_streaming"])
        self.assertEqual(assistant["role"], "assistant")
        self.assertIn("已收到：hello", assistant["content"])
        self.assertIn("收到问题", assistant["reasoning"])


if __name__ == "__main__":
    unittest.main()
