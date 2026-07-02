import tempfile
import unittest

from fastapi.testclient import TestClient

from src.api import GatewayConfig, create_app
from src.presentation.stream_aggregator import ChatStreamAggregator
from src.repositories.sessions.sqlite import SQLiteSessionRepository
from src.repositories.settings.json import SettingsRepository

try:
    from .frontend_api_client import FrontendApiClient
except ImportError:
    from frontend_api_client import FrontendApiClient


class FrontToBackFastApiTest(unittest.TestCase):
    """前后端网关联调测试。"""

    @staticmethod
    def _fake_message_handler_factory(repo: SQLiteSessionRepository):
        """构造一个稳定的测试消息处理器，避免测试依赖外部检索服务。"""

        def _handler(chat_id, content, frame):
            turn_id = frame.get("turn_id")
            repo.write_artifact(
                chat_id,
                artifact_type="paper_search_manifest",
                name="search_manifest.json",
                content='{"status":"ok"}',
                relative_path=f"artifacts/search/{turn_id}/search_manifest.json",
                metadata={"turn_id": turn_id},
            )
            return [
                {
                    "event": "reasoning_delta",
                    "chat_id": chat_id,
                    "content": "已完成论文检索：原始候选 2 篇，最终保留 1 篇。",
                    "turn_id": turn_id,
                },
                {"event": "reasoning_end", "chat_id": chat_id, "turn_id": turn_id},
                {
                    "event": "message",
                    "chat_id": chat_id,
                    "role": "assistant",
                    "content": f"已完成论文检索，主题：{content}",
                    "turn_id": turn_id,
                },
                {"event": "stream_end", "chat_id": chat_id, "turn_id": turn_id},
            ]

        return _handler

    def _client(self):
        """构造测试用应用与仓储对象。"""

        settings = SettingsRepository(
            initial={
                "providers": {"openai": {"api_key": "sk-test", "api_base": "https://api.openai.com/v1"}},
                "agents": {"default_agent": {"model_name": "gpt-5-mini", "provider": "openai"}},
                "embedding_profiles": {},
            }
        )
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        sessions = SQLiteSessionRepository(storage_root=temp_dir.name)
        app = create_app(
            settings_repo=settings,
            sessions_repo=sessions,
            config=GatewayConfig(),
            message_handler=self._fake_message_handler_factory(sessions),
        )
        return TestClient(app), sessions

    def _api(self, client: TestClient) -> FrontendApiClient:
        """把 FastAPI TestClient 包装成前端 API 客户端约定的 transport。"""

        def transport(method, url, body, timeout):
            """执行单次 HTTP 调用并返回客户端约定的数据结构。"""

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
        self.assertEqual(thread["messages"][-1]["role"], "assistant")

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
        self.assertIn("已完成论文检索", assistant["content"])
        self.assertIn("已完成论文检索", assistant["reasoning"])

    def test_http_message_submit_persists_search_artifacts(self):
        """验证通过 HTTP 提交消息后，会话线程中能看到检索产物。"""

        client, sessions = self._client()
        api = self._api(client)
        created = api.create_session("Artifacts")["session"]

        api.send_message(created["key"], "multi-agent literature review")
        thread = sessions.get(created["key"]).thread()

        self.assertGreaterEqual(len(thread["artifacts"]), 1)
        self.assertIn("search_manifest.json", [artifact["name"] for artifact in thread["artifacts"]])


if __name__ == "__main__":
    unittest.main()
