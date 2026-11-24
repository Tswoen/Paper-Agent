import unittest
from types import SimpleNamespace
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.graph.checkpointing import build_checkpoint_bundle
from src.llm.config import CheckpointSettings
from src.services.trace_state_history import build_state_history


class _ProbeState(TypedDict, total=False):
    counter: int
    items: list


def _run_probe_graph(thread_id: str):
    """跑一个两节点小图，返回持有真实快照的内存 bundle。"""

    bundle = build_checkpoint_bundle(CheckpointSettings(backend="memory"))

    def node_a(state):
        return {"counter": 1, "items": ["a"]}

    def node_b(state):
        return {"counter": 2}

    graph = StateGraph(_ProbeState)
    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    app = graph.compile(checkpointer=bundle.saver)
    app.invoke({"counter": 0}, config={"configurable": {"thread_id": thread_id}})
    return bundle


class StateHistoryTest(unittest.TestCase):
    def test_chronological_rows_diffs_and_internal_filtering(self):
        bundle = _run_probe_graph("history-1")
        self.addCleanup(bundle.close)

        rows = build_state_history(bundle.saver, "history-1")

        # input + step0 + 两个节点 = 4 张快照，时间正序。
        self.assertEqual(len(rows), 4)
        self.assertEqual([row["seq"] for row in rows], [1, 2, 3, 4])
        self.assertEqual(rows[0]["source"], "input")
        self.assertEqual(rows[0]["changed"], {})
        self.assertEqual(rows[1]["changed"], {"counter": 0})
        self.assertEqual(rows[2]["changed"], {"counter": 1, "items": ["a"]})
        self.assertEqual(rows[3]["changed"], {"counter": 2})
        self.assertTrue(all(row["removed_keys"] == [] for row in rows))
        # 框架内部通道（__start__ / branch:to:*）不得出现在账本里。
        for row in rows:
            for key in row["changed"]:
                self.assertFalse(key.startswith("__"))
                self.assertFalse(key.startswith("branch:"))
        # parent 链首尾相接。
        self.assertIsNone(rows[0]["parent_checkpoint_id"])
        for prev, curr in zip(rows, rows[1:]):
            self.assertEqual(curr["parent_checkpoint_id"], prev["checkpoint_id"])
        self.assertTrue(all(row["created_at"] for row in rows))

    def test_removed_keys_and_full_values(self):
        saver = SimpleNamespace(
            list=lambda config: [
                SimpleNamespace(
                    checkpoint={"id": "c2", "ts": "t2", "channel_values": {"items": [{"note": "全文"}]}},
                    metadata={"source": "loop", "step": 2},
                    parent_config={"configurable": {"checkpoint_id": "c1"}},
                ),
                SimpleNamespace(
                    checkpoint={"id": "c1", "ts": "t1", "channel_values": {"items": [], "current_step": "read"}},
                    metadata={"source": "loop", "step": 1},
                    parent_config=None,
                ),
            ]
        )

        rows = build_state_history(saver, "fake-thread")

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["changed"], {"items": [], "current_step": "read"})
        self.assertEqual(rows[1]["changed"], {"items": [{"note": "全文"}]})
        self.assertEqual(rows[1]["removed_keys"], ["current_step"])

    def test_unknown_thread_is_empty(self):
        bundle = _run_probe_graph("history-2")
        self.addCleanup(bundle.close)
        self.assertEqual(build_state_history(bundle.saver, "no-such-thread"), [])

    def test_missing_saver_or_thread_is_empty(self):
        self.assertEqual(build_state_history(None, "t"), [])
        self.assertEqual(build_state_history(object(), ""), [])
        self.assertEqual(build_state_history(object(), None), [])

    def test_read_failure_degrades_to_empty(self):
        def _broken_list(config):
            raise RuntimeError("db locked")

        saver = SimpleNamespace(list=_broken_list)
        self.assertEqual(build_state_history(saver, "t"), [])


if __name__ == "__main__":
    unittest.main()
