"""全文提取校验器与精读重试路由的回归测试。

背景：修复"证据核查覆盖率仅 39%"问题（见 EVAL_NOTES.md 第六节）。根因是提取校验器
有两个假阳性 bug（引用正则过宽、irrelevant:false 被当多余字段），且格式违规绕过了重试。
本测试锁定修复后的行为，防止回归。

运行：uv run python -m unittest test.test_extraction_validation -v
"""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

from src.graph import read_node
from src.utils.read_utils.extraction import (
    _validate_extraction,
    _citation_ids_from_extraction,
    async_extract_paper_from_chunks,
)

# 真实 chunkId 形如 arXiv id + :p + 4 位页码
VALID_IDS = {"1901.00383v1:p0001", "1901.00383v1:p0002"}


def _six_fields(suffix: str = "1901.00383v1:p0001") -> dict:
    """构造六个字段全部带有效 chunkId 引用的合法提取结果。"""

    cite = f"[{suffix}]"
    return {
        "research_topic": f"研究了多智能体检索{cite}",
        "research_object": f"若干基准数据集{cite}",
        "methods": f"提出对比学习方法{cite}",
        "conclusions": f"方法显著优于基线{cite}",
        "contributions": f"新框架；新评测{cite}",
        "limitations": f"仅在单一场景验证{cite}",
    }


class ValidateExtractionTest(unittest.TestCase):
    """_validate_extraction 的引用识别修复（改动一 1a）。"""

    def test_valid_citation_passes(self):
        result = _validate_extraction(_six_fields(), valid_chunk_ids=VALID_IDS)
        self.assertIn("1901.00383v1:p0001", result["methods"])

    def test_only_non_chunkid_brackets_raises_missing(self):
        """字段只含 [2024]/[Transformer]/Markdown 链接 → 报"缺少有效引用"，不再报"引用不存在"。"""
        payload = _six_fields()
        payload["methods"] = "提出 Transformer 方法 [2024] [see](http://x)"
        with self.assertRaises(ValueError) as ctx:
            _validate_extraction(payload, valid_chunk_ids=VALID_IDS)
        self.assertIn("缺少有效 chunkId 引用", str(ctx.exception))
        self.assertNotIn("不存在的 chunkId", str(ctx.exception))

    def test_real_citation_plus_year_passes(self):
        """真实 chunkId + 年份混在一起 → 通过（年份被忽略，真实引用计数）。"""
        payload = _six_fields()
        payload["methods"] = "提出方法 [1901.00383v1:p0001] 于 [2024] 年"
        result = _validate_extraction(payload, valid_chunk_ids=VALID_IDS)
        self.assertIn("1901.00383v1:p0001", result["methods"])

    def test_missing_field_raises(self):
        payload = _six_fields()
        del payload["methods"]
        with self.assertRaises(ValueError) as ctx:
            _validate_extraction(payload, valid_chunk_ids=VALID_IDS)
        self.assertIn("缺少字段", str(ctx.exception))

    def test_non_string_field_raises(self):
        payload = _six_fields()
        payload["methods"] = 123
        with self.assertRaises(ValueError) as ctx:
            _validate_extraction(payload, valid_chunk_ids=VALID_IDS)
        self.assertIn("必须是字符串", str(ctx.exception))

    def test_empty_field_exempt(self):
        """空字符串字段豁免引用要求（全文未提及某字段时合法）。"""
        payload = _six_fields()
        payload["limitations"] = ""
        result = _validate_extraction(payload, valid_chunk_ids=VALID_IDS)
        self.assertEqual(result["limitations"], "")


class CitationIdsFilterTest(unittest.TestCase):
    """_citation_ids_from_extraction 过滤假引用（防止污染 chunks_used）。"""

    def test_filters_non_chunkid_brackets(self):
        extraction = {"methods": "方法 [1901.00383v1:p0001] 于 [2024] 提出 [Transformer]"}
        ids = _citation_ids_from_extraction(extraction, valid_chunk_ids=VALID_IDS)
        self.assertEqual(ids, ["1901.00383v1:p0001"])

    def test_no_filter_when_valid_none(self):
        """不传 valid_chunk_ids 时保持旧行为（全部收集），向后兼容。"""
        extraction = {"methods": "方法 [a] 于 [2024]"}
        ids = _citation_ids_from_extraction(extraction)
        self.assertEqual(ids, ["a", "2024"])


class IrrelevantFalseStripTest(unittest.IsolatedAsyncioTestCase):
    """async_extract_paper_from_chunks 剥离 irrelevant:false / reason（改动一 1b）。"""

    async def test_irrelevant_false_does_not_break(self):
        """模型返回 {"irrelevant": false, ...六字段合法...} 应正常提取，而非被判多余字段打回。"""
        payload = {"irrelevant": False, "reason": "", **_six_fields()}
        fake_chunks = [types.SimpleNamespace(chunk_id="1901.00383v1:p0001", content="正文")]

        async def fake_call_model(llm, messages, *, runtime_resources):
            return types.SimpleNamespace(
                ok=True, content=__import__("json").dumps(payload), error_code="", error_type=""
            )

        import src.utils.read_utils.extraction as ext

        orig_call = ext._call_model
        orig_load_chunks = ext.load_chunks_file
        ext._call_model = fake_call_model
        ext.load_chunks_file = lambda path: fake_chunks
        try:
            with tempfile.TemporaryDirectory() as d:
                chunks_path = Path(d) / "chunk.json"
                record = await async_extract_paper_from_chunks(
                    types.SimpleNamespace(paperId="1901.00383v1", id="1901.00383v1"),
                    chunks_path=chunks_path,
                    llm=None,
                    topic="多智能体检索",
                    force=True,
                )
            # 未被 irrelevant 多余字段打死，正常返回六字段提取记录
            self.assertNotEqual(record.get("status"), "irrelevant")
            self.assertIn("extraction", record)
            self.assertIn("1901.00383v1:p0001", record["extraction"]["methods"])
        finally:
            ext._call_model = orig_call
            ext.load_chunks_file = orig_load_chunks


def _fake_config(max_retries: int = 1) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        verify_enabled=True,
        verify_max_retries=max_retries,
        verify_optional_fields=("limitations",),
        verify_max_chars=120000,
    )


class ExtractAndVerifyRetryTest(unittest.IsolatedAsyncioTestCase):
    """_extract_and_verify 把格式违规纳入重试（改动二）。"""

    async def _run(self, fake_extract, fake_verify):
        orig_extract = read_node.async_extract_paper_from_chunks
        orig_verify = read_node.async_verify_extraction
        read_node.async_extract_paper_from_chunks = fake_extract
        read_node.async_verify_extraction = fake_verify
        try:
            return await read_node._extract_and_verify(
                types.SimpleNamespace(paperId="x", id="x"),
                topic="t",
                chunks_path=Path("/tmp/chunk.json"),
                markdown_path=Path("/tmp/paper.md"),
                llm=None,
                runtime_resources=None,
                config=_fake_config(max_retries=1),
            )
        finally:
            read_node.async_extract_paper_from_chunks = orig_extract
            read_node.async_verify_extraction = orig_verify

    async def test_format_error_retries_then_succeeds(self):
        """attempt0 抛 ValueError、attempt1 成功 → 最终返回有效记录（证明格式错误现在会重试）。"""
        calls = {"n": 0}

        async def fake_extract(paper, **kw):
            calls["n"] += 1
            if kw.get("force") is False:  # 第一轮
                raise ValueError("全文提取字段 methods 缺少有效 chunkId 引用")
            return {"extraction": _six_fields()}  # 第二轮成功

        async def fake_verify(paper, **kw):
            return {"status": "completed", "passed": True, "items": []}

        record, verification = await self._run(fake_extract, fake_verify)
        self.assertEqual(calls["n"], 2, "应重试一次")
        self.assertNotEqual(record.get("status"), "extraction_failed")
        self.assertEqual(verification.get("status"), "completed")

    async def test_format_error_exhausts_to_structured_failure(self):
        """始终抛 ValueError → 返回 extraction_failed + skipped_extraction_failed（不再静默吞）。"""
        calls = {"n": 0}

        async def fake_extract(paper, **kw):
            calls["n"] += 1
            raise ValueError("全文提取字段 research_topic 缺少有效 chunkId 引用")

        async def fake_verify(paper, **kw):  # 不应被调用
            raise AssertionError("提取失败时不应进入核查")

        record, verification = await self._run(fake_extract, fake_verify)
        self.assertEqual(calls["n"], 2, "max_retries=1 应共尝试 2 次")
        self.assertEqual(record.get("status"), "extraction_failed")
        self.assertEqual(verification.get("status"), "skipped_extraction_failed")
        self.assertIn("research_topic", verification.get("reason", ""))

    async def test_runtime_error_still_bubbles(self):
        """RuntimeError（模型不可用）不被捕获，继续冒泡走中断恢复（关键不变量）。"""

        async def fake_extract(paper, **kw):
            raise RuntimeError("全文提取模型调用失败：503")

        async def fake_verify(paper, **kw):
            raise AssertionError("不应到达")

        with self.assertRaises(RuntimeError):
            await self._run(fake_extract, fake_verify)


if __name__ == "__main__":
    unittest.main()
