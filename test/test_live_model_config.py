import json
import unittest
from pathlib import Path

from src.llm import ModelConfig, StreamCallbacks, SystemConfig, make_provider


def _live_test_enabled() -> bool:
    return True
    # return os.getenv("RUN_LIVE_MODEL_TEST") == "1"


@unittest.skipUnless(_live_test_enabled(), "set RUN_LIVE_MODEL_TEST=1 to run live model integration tests")
class LiveModelConfigTest(unittest.IsolatedAsyncioTestCase):
    def _load_snapshot(self):
        config_path = Path("config/model.json")
        self.assertTrue(config_path.exists(), "config/model.json does not exist")

        data = json.loads(config_path.read_text(encoding="utf-8"))
        config = ModelConfig.from_dict(data, SystemConfig.load())

        agent_name = "default_agent" if "default_agent" in config.agents else next(iter(config.agents), None)
        self.assertIsNotNone(agent_name, "no agent found in config/model.json")

        snapshot = make_provider(config, agent_name)
        return agent_name, snapshot

    async def test_can_create_agent_from_model_json_and_chat(self):
        agent_name, snapshot = self._load_snapshot()

        response = await snapshot.provider.chat(
            [{"role": "user", "content": "Reply with exactly OK."}],
            max_tokens=16,
        )
        print(response)

        self.assertTrue(response.ok, f"chat failed for agent {agent_name}: {response.content}")
        self.assertTrue((response.content or "").strip(), f"empty response for agent {agent_name}")

    async def test_can_create_agent_from_model_json_and_chat_stream(self):
        agent_name, snapshot = self._load_snapshot()
        chunks: list[str] = []

        response = await snapshot.provider.chat_stream(
            [{"role": "user", "content": "Reply with exactly OK."}],
            StreamCallbacks(on_content_delta=chunks.append),
            max_tokens=16,
        )

        self.assertTrue(response.ok, f"stream chat failed for agent {agent_name}: {response.content}")
        self.assertTrue((response.content or "").strip(), f"empty stream response for agent {agent_name}")
        self.assertTrue("".join(chunks).strip(), f"no stream delta received for agent {agent_name}")


if __name__ == "__main__":
    unittest.main()
