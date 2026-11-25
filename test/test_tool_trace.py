import asyncio
import unittest
from types import SimpleNamespace

from src.graph.search_node import _search_one_subtopic
from src.models.sessions import utc_now
from src.trace import bind_trace_context, current_trace_context, emit_trace_event, trace_tool


class _Collector:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)
        return event


def _tool_events(collector):
    return [event for event in collector.events if event["event"] in {"tool_call", "tool_result"}]


class ToolTraceTest(unittest.IsolatedAsyncioTestCase):
    async def test_pair_fields_and_parent_chain(self):
        collector = _Collector()
        with bind_trace_context(collector.emit, node_key="read", parent_id="turn-1:read"):
            with trace_tool("download_fulltext", args={"paper_id": "P3"}, stage="downloading_full_text") as call:
                call.set_result({"status": "ok", "bytes": 123})

        start, end = _tool_events(collector)
        self.assertEqual(start["event"], "tool_call")
        self.assertEqual(start["status"], "running")
        self.assertEqual(start["parent_id"], "turn-1:read")
        self.assertEqual(start["node_key"], "read")
        self.assertEqual(start["stage"], "downloading_full_text")
        self.assertEqual(start["payload"], {"name": "download_fulltext", "args": {"paper_id": "P3"}})
        self.assertEqual(end["event"], "tool_result")
        self.assertEqual(end["parent_id"], start["id"])
        self.assertEqual(end["status"], "completed")
        self.assertEqual(end["payload"]["result"], {"status": "ok", "bytes": 123})
        self.assertGreaterEqual(end["duration_ms"], 0)
        self.assertIsNone(end["error"])

    async def test_nested_llm_call_parented_to_tool(self):
        collector = _Collector()
        with bind_trace_context(collector.emit, node_key="read", parent_id="turn-1:read"):
            with trace_tool("extract_fulltext") as call:
                emit_trace_event("llm_call", status="completed", started_at=utc_now(), payload={})
                call_id = call.call_id

        llm_event = next(event for event in collector.events if event["event"] == "llm_call")
        self.assertEqual(llm_event["parent_id"], call_id)

    async def test_parent_restored_after_tool(self):
        collector = _Collector()
        with bind_trace_context(collector.emit, parent_id="turn-1:read"):
            with trace_tool("build_chunks"):
                pass
            emit_trace_event("llm_call", status="completed", started_at=utc_now(), payload={})

        llm_event = next(event for event in collector.events if event["event"] == "llm_call")
        self.assertEqual(llm_event["parent_id"], "turn-1:read")

    async def test_exception_records_failed_and_reraises(self):
        collector = _Collector()
        with bind_trace_context(collector.emit):
            with self.assertRaises(ValueError) as ctx:
                with trace_tool("convert_markdown"):
                    raise ValueError("bad pdf")

        self.assertEqual(str(ctx.exception), "bad pdf")
        end = _tool_events(collector)[1]
        self.assertEqual(end["status"], "failed")
        self.assertEqual(end["error"], {"kind": "ValueError", "message": "bad pdf"})

    async def test_cancelled_records_cancelled_and_reraises(self):
        collector = _Collector()
        with bind_trace_context(collector.emit):
            with self.assertRaises(asyncio.CancelledError):
                with trace_tool("download_fulltext"):
                    raise asyncio.CancelledError()

        end = _tool_events(collector)[1]
        self.assertEqual(end["status"], "cancelled")

    async def test_graceful_failure_marked_via_set_result(self):
        collector = _Collector()
        with bind_trace_context(collector.emit):
            with trace_tool("download_fulltext") as call:
                call.set_result(
                    {"status": "download_failed", "reason": "连接超时"},
                    status="failed",
                    error={"message": "连接超时"},
                )

        end = _tool_events(collector)[1]
        self.assertEqual(end["status"], "failed")
        self.assertEqual(end["error"], {"message": "连接超时"})
        self.assertEqual(end["payload"]["result"]["reason"], "连接超时")

    async def test_no_context_no_events(self):
        self.assertIsNone(current_trace_context())
        with trace_tool("paper_search", args={"keyword": "gnn"}) as call:
            call.set_result({"paper_count": 5})
        self.assertEqual(call.status, "completed")

    async def test_emit_failure_never_breaks_tool(self):
        class _BrokenCollector:
            def emit(self, event):
                raise RuntimeError("sink down")

        with bind_trace_context(_BrokenCollector().emit):
            with trace_tool("build_chunks") as call:
                call.set_result({"chunk_count": 42})
        self.assertEqual(call.result, {"chunk_count": 42})

    async def test_nested_tools_restore_chain(self):
        collector = _Collector()
        with bind_trace_context(collector.emit, parent_id="turn-1:read"):
            with trace_tool("outer") as outer:
                with trace_tool("inner") as inner:
                    emit_trace_event("llm_call", status="completed", started_at=utc_now(), payload={})
                emit_trace_event("llm_call", status="completed", started_at=utc_now(), payload={})

        llm_events = [event for event in collector.events if event["event"] == "llm_call"]
        self.assertEqual(llm_events[0]["parent_id"], inner.call_id)
        self.assertEqual(llm_events[1]["parent_id"], outer.call_id)
        outer_result = next(
            event for event in collector.events
            if event["event"] == "tool_result" and event["parent_id"] == outer.call_id
        )
        self.assertEqual(outer_result["status"], "completed")

    async def test_context_fully_restored_after_failure(self):
        collector = _Collector()
        with bind_trace_context(collector.emit, parent_id="turn-1:read") as context:
            with self.assertRaises(ValueError):
                with trace_tool("outer"):
                    raise ValueError("boom")
            self.assertEqual(context.parent_id, "turn-1:read")


class SearchSiteTraceTest(unittest.IsolatedAsyncioTestCase):
    """验证检索子主题包点真实产出事件，且失败传播语义不变。"""

    def _inputs(self, service):
        intent = SimpleNamespace(sources=["arxiv"], year_from=2020, year_to=2026, excluded_terms=[])
        subtopic = SimpleNamespace(subtopic="图神经网络", keyword="graph neural network")
        return service, intent, subtopic

    async def test_search_subtopic_emits_tool_events(self):
        class _Service:
            async def async_search(self, **kwargs):
                return SimpleNamespace(
                    papers=[object(), object()],
                    sources_used=["arxiv"],
                    source_results={"arxiv": 2},
                    errors={},
                )

        collector = _Collector()
        service, intent, subtopic = self._inputs(_Service())
        with bind_trace_context(collector.emit, node_key="search", parent_id="turn-1:search"):
            result_subtopic, response = await _search_one_subtopic(
                service, intent, subtopic, runtime_resources=None, candidate_limit=10
            )

        self.assertIs(result_subtopic, subtopic)
        self.assertEqual(len(response.papers), 2)
        start, end = _tool_events(collector)
        self.assertEqual(start["payload"]["name"], "paper_search")
        self.assertEqual(start["payload"]["args"]["subtopic"], "图神经网络")
        self.assertEqual(start["payload"]["args"]["limit"], 10)
        self.assertEqual(end["status"], "completed")
        self.assertEqual(end["payload"]["result"]["paper_count"], 2)
        self.assertEqual(end["payload"]["result"]["source_results"], {"arxiv": 2})

    async def test_search_failure_propagates_unchanged(self):
        class _Service:
            async def async_search(self, **kwargs):
                raise RuntimeError("source down")

        collector = _Collector()
        service, intent, subtopic = self._inputs(_Service())
        with bind_trace_context(collector.emit, node_key="search"):
            with self.assertRaises(RuntimeError) as ctx:
                await _search_one_subtopic(service, intent, subtopic, runtime_resources=None, candidate_limit=10)

        self.assertEqual(str(ctx.exception), "source down")
        end = _tool_events(collector)[1]
        self.assertEqual(end["status"], "failed")
        self.assertEqual(end["error"]["kind"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
