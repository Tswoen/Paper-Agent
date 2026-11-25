import asyncio
import unittest

from langgraph.runtime import Runtime

from src.graph.graph import _with_cancellation_boundary
from src.graph.runtime import (
    InlineWorkflowSyncPort,
    WorkflowCancellation,
    WorkflowRuntimeContext,
)
from src.models.sessions import utc_now
from src.trace import current_trace_context, emit_trace_event


class _Collector:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)
        return event


def _runtime(collector, *, cancellation=None, turn_id="turn-1"):
    return Runtime(
        context=WorkflowRuntimeContext(
            session_key="session-1",
            turn_id=turn_id,
            cancellation=cancellation,
            sync_port=InlineWorkflowSyncPort(
                collector.emit,
                session_key="session-1",
                turn_id=turn_id,
            ),
        )
    )


class NodeTraceTest(unittest.IsolatedAsyncioTestCase):
    async def test_start_end_pair_and_diff_stats(self):
        async def _node(state, runtime):
            return {"current_step": "read", "read_results": []}

        collector = _Collector()
        guarded = _with_cancellation_boundary("test_node", _node)
        result = await guarded({}, _runtime(collector))

        self.assertEqual(result, {"current_step": "read", "read_results": []})
        self.assertEqual([event["event"] for event in collector.events], ["node_start", "node_end"])
        start, end = collector.events
        self.assertEqual(start["status"], "running")
        self.assertEqual(end["status"], "completed")
        for event in (start, end):
            self.assertEqual(event["node_key"], "test_node")
            self.assertEqual(event["parent_id"], "turn-1:test_node")
            self.assertTrue(event["started_at"])
            self.assertGreaterEqual(event["duration_ms"], 0)
        self.assertEqual(end["payload"]["changed_keys"], ["current_step", "read_results"])
        self.assertGreater(end["payload"]["result_bytes"], 0)
        self.assertIsNone(end["error"])

    async def test_llm_call_inside_node_inherits_context(self):
        # 中文注释：C2 的核心效果——节点内任意深度发 trace 事件都自动带上归属。
        async def _node(state, runtime):
            emit_trace_event(
                "llm_call",
                status="completed",
                started_at=utc_now(),
                payload={"model": "fake"},
            )
            return {"current_step": "read"}

        collector = _Collector()
        guarded = _with_cancellation_boundary("read", _node)
        await guarded({}, _runtime(collector))

        llm_events = [event for event in collector.events if event["event"] == "llm_call"]
        self.assertEqual(len(llm_events), 1)
        self.assertEqual(llm_events[0]["node_key"], "read")
        self.assertEqual(llm_events[0]["parent_id"], "turn-1:read")

    async def test_failure_records_and_reraises_unchanged(self):
        async def _node(state, runtime):
            raise ValueError("boom")

        collector = _Collector()
        guarded = _with_cancellation_boundary("test_node", _node)

        with self.assertRaises(ValueError) as ctx:
            await guarded({}, _runtime(collector))

        self.assertEqual(str(ctx.exception), "boom")
        self.assertEqual([event["event"] for event in collector.events], ["node_start", "node_end"])
        end = collector.events[1]
        self.assertEqual(end["status"], "failed")
        self.assertEqual(end["error"], {"kind": "ValueError", "message": "boom"})

    async def test_undeclared_key_gate_still_raises(self):
        async def _node(state, runtime):
            return {"current_step": "read", "not_declared_anywhere": True}

        collector = _Collector()
        guarded = _with_cancellation_boundary("test_node", _node)

        with self.assertRaises(KeyError):
            await guarded({}, _runtime(collector))

        end = collector.events[-1]
        self.assertEqual(end["event"], "node_end")
        self.assertEqual(end["status"], "failed")
        self.assertEqual(end["error"]["kind"], "KeyError")

    async def test_cancel_during_node_discards_result(self):
        cancellation = WorkflowCancellation()

        async def _node(state, runtime):
            cancellation.request()
            return {"current_step": "read"}

        collector = _Collector()
        guarded = _with_cancellation_boundary("test_node", _node)

        with self.assertRaises(asyncio.CancelledError):
            await guarded({}, _runtime(collector, cancellation=cancellation))

        end = collector.events[-1]
        self.assertEqual(end["event"], "node_end")
        self.assertEqual(end["status"], "cancelled")
        self.assertNotIn("changed_keys", end["payload"])

    async def test_cancel_before_node_emits_nothing(self):
        # 中文注释：停止检查①在绑定之前，时序不变——已取消的节点不留任何牌。
        cancellation = WorkflowCancellation()
        cancellation.request()
        calls = []

        async def _node(state, runtime):
            calls.append(state)
            return {"current_step": "read"}

        collector = _Collector()
        guarded = _with_cancellation_boundary("test_node", _node)

        with self.assertRaises(asyncio.CancelledError):
            await guarded({}, _runtime(collector, cancellation=cancellation))

        self.assertEqual(calls, [])
        self.assertEqual(collector.events, [])

    async def test_bare_runtime_emits_nothing_and_keeps_gate(self):
        async def _good_node(state, runtime):
            return {"current_step": "read"}

        async def _bad_node(state, runtime):
            return {"not_declared_anywhere": True}

        guarded = _with_cancellation_boundary("test_node", _good_node)
        self.assertEqual(await guarded({}, None), {"current_step": "read"})

        guarded_bad = _with_cancellation_boundary("test_node", _bad_node)
        with self.assertRaises(KeyError):
            await guarded_bad({}, None)

    async def test_emit_failure_never_breaks_node(self):
        class _BrokenCollector:
            def emit(self, event):
                raise RuntimeError("sink down")

        async def _node(state, runtime):
            return {"current_step": "read"}

        guarded = _with_cancellation_boundary("test_node", _node)
        result = await guarded({}, _runtime(_BrokenCollector()))
        self.assertEqual(result, {"current_step": "read"})

    async def test_context_restored_after_node(self):
        async def _node(state, runtime):
            self.assertIsNotNone(current_trace_context())
            return {"current_step": "read"}

        collector = _Collector()
        guarded = _with_cancellation_boundary("test_node", _node)
        await guarded({}, _runtime(collector))
        self.assertIsNone(current_trace_context())

        async def _failing_node(state, runtime):
            raise ValueError("boom")

        guarded_fail = _with_cancellation_boundary("test_node", _failing_node)
        with self.assertRaises(ValueError):
            await guarded_fail({}, _runtime(collector))
        self.assertIsNone(current_trace_context())

    async def test_result_bytes_tolerates_non_serializable_values(self):
        # 中文注释：量体积是防御性的——混进不可序列化对象时转字符串，不炸节点。
        async def _node(state, runtime):
            return {"current_step": "read", "diagnostics": {"odd": object()}}

        collector = _Collector()
        guarded = _with_cancellation_boundary("test_node", _node)
        await guarded({}, _runtime(collector))

        end = collector.events[-1]
        self.assertEqual(end["status"], "completed")
        self.assertIsInstance(end["payload"]["result_bytes"], int)


if __name__ == "__main__":
    unittest.main()
