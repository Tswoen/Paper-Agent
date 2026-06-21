import unittest

from fastapi.testclient import TestClient

from src.router import GatewayConfig, SettingsRepository, create_app


class _FakeModels:
    def list(self):
        return {"data": [{"id": "gpt-test", "owned_by": "test"}]}


class _FakeClient:
    models = _FakeModels()


class SettingsFastApiTest(unittest.TestCase):
    def _client(self):
        repo = SettingsRepository(
            initial={
                "providers": {
                    "openai": {"api_key": "sk-test", "api_base": "https://api.openai.com/v1"},
                    "anthropic_compat": {"api_key": "ak-test", "api_base": "https://proxy.example/v1"},
                },
                "agents": {
                    "default_agent": {
                        "model_name": "gpt-5-mini",
                        "provider": "openai",
                        "max_tokens": 100,
                        "context_window_tokens": 8192,
                    }
                },
                "embedding_profiles": {
                    "default_embedding": {
                        "model_name": "text-embedding-3-small",
                        "provider": "openai",
                    }
                },
            }
        )
        app = create_app(settings_repo=repo, config=GatewayConfig())
        return repo, TestClient(app)

    def test_get_settings_returns_full_snapshot(self):
        _, client = self._client()

        response = client.get("/api/settings")
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["agent"]["resolved_provider"], "openai")
        self.assertTrue(payload["providers"][0]["name"])
        self.assertTrue(payload["agents"][0]["is_default"])
        self.assertTrue(payload["embedding_profiles"][0]["is_default"])

    def test_create_agent_configuration_returns_named_agent(self):
        _, client = self._client()

        response = client.post(
            "/api/settings/agents/proxy-claude",
            json={"label": "Proxy Claude", "provider": "anthropic_compat", "model_name": "anthropic_compat/claude-test"},
        )
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["active_agent"], "proxy-claude")
        self.assertEqual(payload["agent"]["resolved_provider"], "anthropic_compat")

    def test_missing_agent_model_is_rejected(self):
        _, client = self._client()

        response = client.put("/api/settings/agents/broken", json={"provider": "openai"})
        payload = response.json()

        self.assertEqual(response.status_code, 400)
        self.assertIn("model_name", payload["error"]["message"])

    def test_provider_models_payload_is_read_only(self):
        repo, _ = self._client()
        before = repo.load()

        from src.router.settings_api import provider_models_payload

        payload = provider_models_payload(repo, "openai", client=_FakeClient())

        self.assertEqual(payload["status"], "available")
        self.assertEqual(payload["models"][0]["id"], "gpt-test")
        self.assertEqual(repo.load(), before)
