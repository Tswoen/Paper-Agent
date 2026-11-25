import asyncio
import unittest

from src.llm import ModelConfig, StreamCallbacks, make_provider
from src.llm.base import EmbeddingResponse, ProviderHttpError
from src.trace import bind_trace_context, current_trace_context


def _openai_response(content="ok", usage=None):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 11, "completion_tokens": 7},
    }


class ScriptedOpenAIClient:
    """按脚本依次返回响应或抛出异常的假 OpenAI 客户端。"""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                outcome = outer.outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions()


class FakeAnthropicClient:
    def __init__(self, response):
        outer = self
        self.calls = []

        class _Messages:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return response

        self.messages = _Messages()


class _Collector:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)
        return event


def _make_openai_provider(client):
    config = ModelConfig.from_dict(
        {
            "providers": {"openai": {"api_key": "k"}},
            "agents": {"default_agent": {"model_name": "gpt-5-mini", "provider": "openai"}},
        }
    )
    return make_provider(config, client=client).provider


def _make_anthropic_provider(client):
    config = ModelConfig.from_dict(
        {
            "providers": {"anthropic": {"api_key": "k"}},
            "agents": {"default_agent": {"model_name": "claude-3-5-sonnet-latest", "provider": "anthropic"}},
        }
    )
    return make_provider(config, client=client).provider


class LLMTraceTest(unittest.IsolatedAsyncioTestCase):
    async def test_chat_emits_full_fidelity_llm_call_event(self):
        # 中文注释：全量保真是 V2 的核心契约——超长 prompt 一字不砍。
        long_content = "论文全文" * 8000
        messages = [
            {"role": "system", "content": "你是论文阅读助手"},
            {"role": "user", "content": long_content},
        ]
        provider = _make_openai_provider(ScriptedOpenAIClient([_openai_response("阅读完成")]))
        collector = _Collector()

        with bind_trace_context(collector.emit, node_key="read", stage="reading_abstract", parent_id="turn-1:read"):
            response = await provider.chat(messages)

        self.assertTrue(response.ok)
        self.assertEqual(len(collector.events), 1)
        event = collector.events[0]
        self.assertEqual(event["event"], "llm_call")
        self.assertEqual(event["type"], "llm_call")
        self.assertEqual(event["status"], "completed")
        self.assertEqual(event["node_key"], "read")
        self.assertEqual(event["stage"], "reading_abstract")
        self.assertEqual(event["parent_id"], "turn-1:read")
        self.assertEqual(len(event["id"]), 32)
        self.assertIsNone(event["error"])
        self.assertGreaterEqual(event["duration_ms"], 0)
        self.assertTrue(event["started_at"])
        self.assertEqual(event["tokens"], {"input_tokens": 11, "output_tokens": 7})
        # 新事件不带 trace_version 字段（已决定清理）。
        self.assertNotIn("trace_version", event)
        payload = event["payload"]
        self.assertEqual(payload["operation"], "chat")
        self.assertEqual(payload["model"], "gpt-5-mini")
        self.assertEqual(payload["attempts"], 1)
        self.assertEqual(payload["finish_reason"], "stop")
        self.assertEqual(payload["prompt"], messages)
        self.assertEqual(payload["prompt"][1]["content"], long_content)
        self.assertEqual(payload["response"], "阅读完成")
        self.assertEqual(payload["tool_calls"], [])

    async def test_without_context_nothing_is_emitted(self):
        provider = _make_openai_provider(ScriptedOpenAIClient([_openai_response()]))
        self.assertIsNone(current_trace_context())
        response = await provider.chat([{"role": "user", "content": "hi"}])
        self.assertTrue(response.ok)

    async def test_retries_are_recorded_in_attempts(self):
        provider = _make_openai_provider(
            ScriptedOpenAIClient(
                [
                    ProviderHttpError(500, '{"error": {"type": "server_error"}}', {}),
                    ProviderHttpError(500, '{"error": {"type": "server_error"}}', {}),
                    _openai_response(),
                ]
            )
        )
        provider._retry_delay = lambda attempt: 0.0
        collector = _Collector()

        with bind_trace_context(collector.emit):
            response = await provider.chat([{"role": "user", "content": "hi"}])

        self.assertTrue(response.ok)
        self.assertEqual(len(collector.events), 1)
        self.assertEqual(collector.events[0]["payload"]["attempts"], 3)
        self.assertEqual(collector.events[0]["status"], "completed")

    async def test_non_retryable_failure_records_error_and_body(self):
        error_body = '{"error": {"type": "authentication_error", "code": "bad_key"}}'
        provider = _make_openai_provider(ScriptedOpenAIClient([ProviderHttpError(401, error_body, {})]))
        collector = _Collector()

        with bind_trace_context(collector.emit):
            response = await provider.chat([{"role": "user", "content": "hi"}])

        self.assertFalse(response.ok)
        self.assertEqual(len(collector.events), 1)
        event = collector.events[0]
        self.assertEqual(event["status"], "failed")
        self.assertEqual(event["payload"]["attempts"], 1)
        self.assertEqual(event["error"]["kind"], "auth")
        self.assertEqual(event["error"]["status_code"], 401)
        self.assertEqual(event["error"]["code"], "bad_key")
        # 错误体也是排错信息，全量保留在 response 里。
        self.assertEqual(event["payload"]["response"], error_body)

    async def test_emit_failure_never_breaks_the_call(self):
        def broken_emit(event):
            raise RuntimeError("sink down")

        provider = _make_openai_provider(ScriptedOpenAIClient([_openai_response()]))
        with bind_trace_context(broken_emit):
            response = await provider.chat([{"role": "user", "content": "hi"}])
        self.assertTrue(response.ok)

    async def test_sync_bridge_propagates_context_per_call(self):
        # 中文注释：Runner 的 context 默认首次冻结；修复后每次同步桥调用
        # 都必须拿到调用方当前的上下文，两个 run 不能串账。
        provider = _make_openai_provider(ScriptedOpenAIClient([_openai_response("one"), _openai_response("two")]))
        first, second = _Collector(), _Collector()

        with bind_trace_context(first.emit, node_key="read"):
            await asyncio.to_thread(provider.chat_with_retry, [{"role": "user", "content": "one"}])
        with bind_trace_context(second.emit, node_key="analyse"):
            await asyncio.to_thread(provider.chat_with_retry, [{"role": "user", "content": "two"}])

        self.assertEqual(len(first.events), 1)
        self.assertEqual(len(second.events), 1)
        self.assertEqual(first.events[0]["node_key"], "read")
        self.assertEqual(first.events[0]["payload"]["response"], "one")
        self.assertEqual(second.events[0]["node_key"], "analyse")
        self.assertEqual(second.events[0]["payload"]["response"], "two")

    async def test_chat_stream_emits_with_stream_operation(self):
        class _Delta:
            content = "ok"
            reasoning_content = None
            tool_calls = None

        class _Choice:
            delta = _Delta()
            finish_reason = "stop"

        class _Chunk:
            choices = [_Choice()]

            def model_dump(self):
                return {"usage": {"prompt_tokens": 3, "completion_tokens": 2}}

        provider = _make_openai_provider(ScriptedOpenAIClient([[_Chunk()]]))
        collector = _Collector()

        with bind_trace_context(collector.emit):
            response = await provider.chat_stream([{"role": "user", "content": "hi"}], StreamCallbacks())

        self.assertTrue(response.ok)
        self.assertEqual(len(collector.events), 1)
        payload = collector.events[0]["payload"]
        self.assertEqual(payload["operation"], "chat_stream")
        self.assertEqual(payload["response"], "ok")
        self.assertEqual(collector.events[0]["tokens"], {"input_tokens": 3, "output_tokens": 2})

    async def test_anthropic_chat_is_covered_by_the_same_seam(self):
        client = FakeAnthropicClient(
            {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 4, "output_tokens": 2},
            }
        )
        provider = _make_anthropic_provider(client)
        collector = _Collector()

        with bind_trace_context(collector.emit, node_key="analyse"):
            response = await provider.chat([{"role": "user", "content": "hi"}])

        self.assertTrue(response.ok)
        self.assertEqual(len(collector.events), 1)
        event = collector.events[0]
        self.assertEqual(event["payload"]["model"], "claude-3-5-sonnet-latest")
        self.assertEqual(event["tokens"], {"input_tokens": 4, "output_tokens": 2})
        self.assertEqual(event["node_key"], "analyse")

    async def test_embedding_call_is_not_instrumented(self):
        # 中文注释：向量化属后期开发，V2 明确不埋 _run_embedding_call。
        provider = _make_openai_provider(ScriptedOpenAIClient([]))
        collector = _Collector()

        async def _once():
            return EmbeddingResponse(embeddings=[[0.1]], model="emb", finish_reason="stop")

        with bind_trace_context(collector.emit):
            result = await provider._run_embedding_call("embed", _once)

        self.assertTrue(result.ok)
        self.assertEqual(collector.events, [])

    async def test_each_call_gets_a_unique_event_id(self):
        provider = _make_openai_provider(ScriptedOpenAIClient([_openai_response(), _openai_response()]))
        collector = _Collector()

        with bind_trace_context(collector.emit):
            await provider.chat([{"role": "user", "content": "one"}])
            await provider.chat([{"role": "user", "content": "two"}])

        self.assertEqual(len(collector.events), 2)
        self.assertNotEqual(collector.events[0]["id"], collector.events[1]["id"])


if __name__ == "__main__":
    unittest.main()
