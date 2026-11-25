import unittest

from src.models.artifacts import to_artifact_ref


class ArtifactRefTest(unittest.TestCase):
    """验证 State / 事件使用的产物引用只保留标识字段。"""

    def test_drops_local_path(self):
        """仓储记录里的本地路径不能进入引用。"""

        ref = to_artifact_ref(
            {
                "id": "artifact-1",
                "artifact_type": "paper_search_intent",
                "name": "search_intent.json",
                "path": "C:\\Users\\someone\\data\\sessions\\key\\artifacts\\search_intent.json",
                "size": 3,
                "created_at": "2026-11-09T00:00:00+00:00",
                "metadata": {"turn_id": "turn-1"},
            }
        )

        self.assertEqual(
            set(ref),
            {"artifact_id", "artifact_type", "name", "size", "created_at", "metadata"},
        )
        self.assertNotIn("path", ref)

    def test_maps_record_id_to_artifact_id(self):
        """仓储记录的 id 字段在引用里叫 artifact_id。"""

        ref = to_artifact_ref({"id": "artifact-1"})

        self.assertEqual(ref["artifact_id"], "artifact-1")
        self.assertEqual(ref["size"], 0)
        self.assertEqual(ref["metadata"], {})


if __name__ == "__main__":
    unittest.main()