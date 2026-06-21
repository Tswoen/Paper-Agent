import unittest

from fastapi.testclient import TestClient

# 兼容 unittest 的两种入口：discover -s test 与 python -m unittest test.xxx
try:
    from .frontend_api_client import FrontendApiClient
except ImportError:
    from frontend_api_client import FrontendApiClient

from src.router import (
    ChatStreamAggregator,
    GatewayConfig,
    SessionRepository,
    SettingsRepository,
    create_app,
)


class FrontToBackFastApiTest(unittest.TestCase):
    def _client(self):
        settings = SettingsRepository(
            initial={
                "providers": {"openai": {"api_key": "sk-test", "api_base": "https://api.openai.com/v1"}},
                "agents": {"default_agent": {"model_name": "gpt-5-mini", "provider": "openai"}},
                "embedding_profiles": {},
            }
        )
        sessions = SessionRepository()
        app = create_app(
            settings_repo=settings,
            sessions_repo=sessions,
            config=GatewayConfig(),
        )
        return TestClient(app), sessions

    def _api(self, client: TestClient) -> FrontendApiClient:
        def transport(method, url, body, timeout):
            response = client.request(method, url, json=body)
            return response.status_code, response.json()

        return FrontendApiClient(transport)

    def test_bootstrap_exposes_local_runtime_capabilities(self):
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
        client, _ = self._client()
        api = self._api(client)

        created = api.create_session("No Auth")
        sessions = api.list_sessions()

        self.assertEqual(created["session"]["title"], "No Auth")
        self.assertEqual(sessions["sessions"][0]["key"], created["session"]["key"])

    def test_http_message_submit_updates_thread(self):
        client, _ = self._client()
        api = self._api(client)
        created = api.create_session("Thread")["session"]

        result = api.send_message(created["key"], "总结这篇论文")
        thread = api.fetch_thread(created["key"])

        self.assertEqual(result["events"][-1]["event"], "turn_end")
        self.assertEqual(thread["messages"][0]["role"], "user")
        self.assertEqual(thread["messages"][0]["content"], "总结这篇论文")

    def test_http_events_can_be_aggregated_into_ui_timeline(self):
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
