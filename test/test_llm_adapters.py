import unittest

from src.llm import ModelConfig, StreamCallbacks, make_provider


class _Create:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeOpenAIClient:
    def __init__(self, response):
        create = _Create(response)
        self.calls = create.calls
        self.chat = type("Chat", (), {})()
        self.chat.completions = type("Completions", (), {"create": create.create})()


class FakeAnthropicClient:
    def __init__(self, response):
        create = _Create(response)
        self.calls = create.calls
        self.messages = type("Messages", (), {"create": create.create})()


class LLMAdaptersTest(unittest.IsolatedAsyncioTestCase):
    async def test_openai_compat_uses_sdk_chat_completion_request(self):
        client = FakeOpenAIClient({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
        config = ModelConfig.from_dict(
            {
                "providers": {"openai": {"api_key": "k"}},
                "agents": {"default_agent": {"model_name": "gpt-5-mini", "provider": "openai", "max_tokens": 100, "temperature": 0.7}},
            }
        )

        snapshot = make_provider(config, client=client)
        response = await snapshot.provider.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(response.content, "ok")
        kwargs = client.calls[0]
        self.assertEqual(kwargs["max_completion_tokens"], 100)
        self.assertNotIn("temperature", kwargs)

    async def test_anthropic_converts_tools_and_system_message(self):
        client = FakeAnthropicClient(
            {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1},
            }
        )
        config = ModelConfig.from_dict(
            {
                "providers": {"anthropic": {"api_key": "k"}},
                "agents": {"default_agent": {"model_name": "claude-3-5-sonnet-latest", "provider": "anthropic", "max_tokens": 50}},
            }
        )

        snapshot = make_provider(config, client=client)
        response = await snapshot.provider.chat(
            [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
        )

        self.assertEqual(response.content, "ok")
        kwargs = client.calls[0]
        self.assertEqual(kwargs["system"], "be brief")
        self.assertEqual(kwargs["tools"][0]["input_schema"], {"type": "object"})

    def test_anthropic_compat_accepts_custom_base(self):
        config = ModelConfig.from_dict(
            {
                "providers": {"anthropic_compat": {"api_key": "k", "apiBase": "https://proxy.example/v1"}},
                "agents": {"default_agent": {"model_name": "anthropic_compat/claude-test", "provider": "anthropic_compat"}},
            }
        )

        snapshot = make_provider(config, client=FakeAnthropicClient({"content": []}))

        self.assertEqual(snapshot.provider.api_base, "https://proxy.example/v1")

    def test_stream_callback_contract_is_stable(self):
        seen = []
        callbacks = StreamCallbacks(on_content_delta=seen.append)
        callbacks.on_content_delta("x")
        self.assertEqual(seen, ["x"])


if __name__ == "__main__":
    unittest.main()
