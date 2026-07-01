import unittest

from fastapi.testclient import TestClient

from src.api import GatewayConfig, create_app
from src.repositories.settings.json import SettingsRepository
from src.services.settings import provider_models_payload


class _FakeModels:
    """模拟 provider 模型目录接口。"""

    def list(self):
        """返回固定的模型目录数据。

        中文说明：
        这里不访问任何真实远端服务，只返回稳定的假数据，确保测试专注于
        payload 解析和只读行为本身。
        """

        return {"data": [{"id": "gpt-test", "owned_by": "test"}]}


class _FakeClient:
    """模拟模型目录客户端。"""

    models = _FakeModels()


class SettingsFastApiTest(unittest.TestCase):
    """设置接口测试。

    中文说明：
    这一组测试覆盖设置快照读取、agent 保存、校验错误以及只读模型目录查询，
    用于验证新的 settings 分层结构没有破坏原有 API 行为。
    """

    def _client(self):
        """构造测试用设置仓储与 FastAPI 客户端。"""

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
        """验证读取设置接口时会返回完整快照。"""

        _, client = self._client()

        response = client.get("/api/settings")
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["agent"]["resolved_provider"], "openai")
        self.assertTrue(payload["providers"][0]["name"])
        self.assertTrue(payload["agents"][0]["is_default"])
        self.assertTrue(payload["embedding_profiles"][0]["is_default"])

    def test_create_agent_configuration_returns_named_agent(self):
        """验证创建命名 agent 后响应中会切换到该 agent。"""

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
        """验证缺少模型名的 agent 配置会被拒绝。"""

        _, client = self._client()

        response = client.put("/api/settings/agents/broken", json={"provider": "openai"})
        payload = response.json()

        self.assertEqual(response.status_code, 400)
        self.assertIn("model_name", payload["error"]["message"])

    def test_provider_models_payload_is_read_only(self):
        """验证 provider 模型目录查询不会修改仓储数据。"""

        repo, _ = self._client()
        before = repo.load()

        payload = provider_models_payload(repo, "openai", client=_FakeClient())

        self.assertEqual(payload["status"], "available")
        self.assertEqual(payload["models"][0]["id"], "gpt-test")
        self.assertEqual(repo.load(), before)
