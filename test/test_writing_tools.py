"""写作 Agent 全文检索工具的回归测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.agents.writingAgent import (
    _vector_result_payload,
    build_writing_agent,
    execute_writing_tool,
    get_chunk_by_embed,
)


def _write_paper_cache(root: Path, paper_id: str, title: str, chunks: list[dict]) -> None:
    """在临时论文缓存中写入 metadata.json 和 chunk.json。"""

    paper_dir = root / paper_id
    paper_dir.mkdir()
    (paper_dir / "metadata.json").write_text(
        json.dumps({"paper": {"title": title}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (paper_dir / "chunk.json").write_text(
        json.dumps(
            {
                "paperId": paper_id,
                "chunker": "page",
                "chunks": chunks,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _chunk(paper_id: str, chunk_id: str, content: str) -> dict:
    return {
        "chunkId": chunk_id,
        "paperId": paper_id,
        "chunk_index": 0,
        "content": content,
        "page_start": 1,
        "page_end": 1,
        "section": "page_1",
    }


class WritingChunkSearchTest(unittest.IsolatedAsyncioTestCase):
    """锁定 get_chunk_by_embed 的向量优先、关键词兜底行为。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.cache_dir = Path(self.temp_dir.name)
        _write_paper_cache(
            self.cache_dir,
            "paper_federated",
            "Federated Learning Privacy",
            [
                _chunk(
                    "paper_federated",
                    "paper_federated:p0001",
                    "联邦学习中的隐私攻击与差分隐私防御是本文的核心问题。",
                ),
                _chunk(
                    "paper_federated",
                    "paper_federated:p0002",
                    "本文还讨论了模型聚合和通信开销。",
                ),
            ],
        )
        _write_paper_cache(
            self.cache_dir,
            "paper_other",
            "Unrelated Topic",
            [
                _chunk(
                    "paper_other",
                    "paper_other:p0001",
                    "This paper studies graph neural networks.",
                ),
            ],
        )

    async def test_keyword_fallback_returns_relevant_chunk(self):
        """向量不可用时，应该从本地 chunk.json 找到最相关的片段。"""

        with patch(
            "src.agents.writingAgent._vector_search_chunks",
            AsyncMock(return_value=[]),
        ):
            result = await get_chunk_by_embed("联邦学习隐私攻击", cache_dir=self.cache_dir)

        self.assertEqual(result["mode"], "keyword")
        self.assertGreaterEqual(len(result["results"]), 1)
        best = result["results"][0]
        self.assertEqual(best["paperId"], "paper_federated")
        self.assertIn("隐私攻击", best["content"])
        self.assertEqual(best["title"], "Federated Learning Privacy")

    async def test_execute_writing_tool_uses_async_search(self):
        """写作循环的工具入口也要走新的异步检索实现。"""

        with patch(
            "src.agents.writingAgent._vector_search_chunks",
            AsyncMock(return_value=[]),
        ):
            result = await execute_writing_tool(
                tool_name="get_chunk_by_embed",
                arguments={"query": "联邦学习隐私攻击"},
                read_results=[],
                session_read_results=[],
                cache_dir=self.cache_dir,
            )

        self.assertEqual(result["tool"], "get_chunk_by_embed")
        self.assertEqual(result["result"]["mode"], "keyword")
        self.assertGreaterEqual(len(result["result"]["results"]), 1)
        self.assertNotIn("暂未实现", str(result))

    async def test_section_loop_run_tool_awaits_chunk_search(self):
        """小节写作循环里的 run_tool 必须等待新的异步检索工具。"""

        agent = build_writing_agent(None)
        with patch(
            "src.agents.writingAgent._vector_search_chunks",
            AsyncMock(return_value=[]),
        ):
            state = await agent.run_tool(
                {
                    "tool_name": "get_chunk_by_embed",
                    "tool_arguments": {"query": "联邦学习隐私攻击"},
                    "read_results": [],
                    "session_read_results": [],
                    "cache_dir": str(self.cache_dir),
                    "tool_results": [],
                    "tool_call_count": 0,
                }
            )

        tool_result = state["tool_results"][-1]
        self.assertEqual(tool_result["result"]["mode"], "keyword")
        self.assertEqual(state["tool_call_count"], 1)

    async def test_empty_query_returns_none_mode(self):
        """空查询不应该扫描全库。"""

        result = await get_chunk_by_embed("", cache_dir=self.cache_dir)

        self.assertEqual(result["mode"], "none")
        self.assertEqual(result["results"], [])

    def test_vector_result_payload_keeps_citation_fields(self):
        """向量结果必须保留正文引用需要的 paperId 和 chunkId。"""

        class Row:
            id = "paper_federated:p0001"
            distance = 0.12
            metadata = {"paperId": "paper_federated", "title": "Federated Learning Privacy"}
            document = "联邦学习中的隐私攻击与差分隐私防御。"

        payload = _vector_result_payload(Row())

        self.assertIsNotNone(payload)
        self.assertEqual(payload["paperId"], "paper_federated")
        self.assertEqual(payload["chunkId"], "paper_federated:p0001")
        self.assertEqual(payload["source"], "vector")
        self.assertEqual(payload["distance"], 0.12)


if __name__ == "__main__":
    unittest.main()
