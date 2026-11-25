import asyncio
import unittest

from langgraph.runtime import Runtime

from src.agents import ReviewRequest
from src.graph.graph import _check_declared_state_keys, _with_cancellation_boundary
from src.graph.reply_node import run_compose_reply_node
from src.graph.runtime import WorkflowCancellation, WorkflowRuntimeContext
from src.graph.state_models import State


class StateContractTest(unittest.TestCase):
    """验证共享状态的字段门禁会拦住未声明字段。"""

    def test_declared_keys_pass(self):
        """已声明字段构成的更新应当原样通过。"""

        _check_declared_state_keys("node", {"current_step": "read", "read_results": []})

    def test_unknown_key_raises(self):
        """未声明字段必须直接报错，不能悄悄进入图状态。"""

        with self.assertRaises(KeyError) as ctx:
            _check_declared_state_keys("node", {"current_step": "read", "read_paper_statuses": []})

        self.assertIn("read_paper_statuses", str(ctx.exception))

    def test_non_dict_payload_is_ignored(self):
        """节点返回非字典时交给 LangGraph 自己报错，门禁不重复判断。"""

        _check_declared_state_keys("node", None)

    def test_node_wrapper_rejects_undeclared_key(self):
        """门禁必须真正挂在节点包装器上，而不是只存在于工具函数里。"""

        async def _bad_node(state, runtime):
            """故意返回一个未声明的字段。"""

            return {"current_step": "read", "not_declared_anywhere": True}

        guarded = _with_cancellation_boundary("bad_node", _bad_node)
        with self.assertRaises(KeyError):
            asyncio.run(guarded({}, None))

    def test_node_wrapper_passes_declared_update(self):
        """合法节点返回值必须原样透传。"""

        async def _good_node(state, runtime):
            """返回一个合法的增量更新。"""

            return {"current_step": "read"}

        guarded = _with_cancellation_boundary("good_node", _good_node)
        self.assertEqual(asyncio.run(guarded({}, None)), {"current_step": "read"})

    def test_removed_fields_stay_undeclared(self):
        """阶段 1 删掉的字段不允许被重新加回 State。"""

        removed = {
            "search_output",
            "assistant_message",
            "assistant_message_metadata",
            "final_artifact_refs",
            "search_node_sink",
            "read_paper_statuses",
            # 阶段 3.1：与 writing_report["sections"] 重复的平铺副本，已删除。
            "writing_sections",
            # 版本断代：旧手写阅读恢复现场字段，恢复改由官方 Checkpointer + note 负责。
            "read_resume_checkpoint",
        }
        self.assertEqual(removed & set(State.__annotations__), set())

    def test_reply_node_only_returns_declared_keys(self):
        """回复节点删掉三个字段后，仍然只能返回已声明的键。"""

        node = run_compose_reply_node()
        result = asyncio.run(node({"request": ReviewRequest(topic="state contract")}, None))

        self.assertEqual(set(result) - set(State.__annotations__), set())
        for name in ("assistant_message", "assistant_message_metadata", "final_artifact_refs", "search_node_sink"):
            self.assertNotIn(name, result)


class CancellationBoundaryTest(unittest.TestCase):
    """验证节点包装器的停止语义：既不启动已取消的节点，也不提交半成品结果。"""

    def test_cancelled_before_node_skips_execution(self):
        """进入节点前已经请求停止时，节点函数不应被执行。"""

        calls: list[dict] = []

        async def _node(state, runtime):
            """记录节点被调用过，用于证明取消发生在执行之前。"""

            calls.append(state)
            return {"current_step": "read"}

        cancellation = WorkflowCancellation()
        cancellation.request()
        guarded = _with_cancellation_boundary("cancelled_node", _node)
        runtime = Runtime(
            context=WorkflowRuntimeContext(
                session_key="session",
                turn_id="turn",
                cancellation=cancellation,
            )
        )

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(guarded({}, runtime))

        self.assertEqual(calls, [])

    def test_cancelled_during_node_discards_result(self):
        """节点执行期间请求停止时，该节点的返回值应被丢弃，不能写回状态。"""

        cancellation = WorkflowCancellation()

        async def _node(state, runtime):
            """模拟节点执行到一半用户点了停止，然后仍然返回一个结果。"""

            cancellation.request()
            return {"current_step": "read"}

        guarded = _with_cancellation_boundary("cancelled_node", _node)
        runtime = Runtime(
            context=WorkflowRuntimeContext(
                session_key="session",
                turn_id="turn",
                cancellation=cancellation,
            )
        )

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(guarded({}, runtime))


if __name__ == "__main__":
    unittest.main()
