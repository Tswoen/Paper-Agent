"""阅读节点恢复字段的回归测试。"""

from __future__ import annotations

import unittest

from src.graph.read_node import _full_text_from_payload


class FullTextPayloadTest(unittest.TestCase):
    """锁定恢复 checkpoint 时各字段的类型和范围。"""

    def test_chunk_count_above_100_is_preserved(self):
        """chunk_count 是实际分块数量，不能被相关性评分的 0-100 上限截断。"""

        status = _full_text_from_payload(
            {
                "status": "chunks_saved",
                "chunk_count": 345,
            }
        )

        self.assertEqual(status.chunk_count, 345)

    def test_invalid_chunk_count_falls_back_to_zero(self):
        """异常值不应让恢复流程失败。"""

        self.assertEqual(_full_text_from_payload({"chunk_count": "many"}).chunk_count, 0)
        self.assertEqual(_full_text_from_payload({"chunk_count": -3}).chunk_count, 0)


if __name__ == "__main__":
    unittest.main()
