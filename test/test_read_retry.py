"""5d 失败扫尾测试：失败分类 + 单次补考 + 跨崩溃累计预算。

覆盖 §18.7 清单：
- 失败分类器（可重试 / 不可重试）表驱动；
- 失败一次后成功、只重跑失败论文、永久错误不重试、补考预算用完即止；
- 全文下载失败的部分失败补考（真实 download 分支）与 note upsert 不重复；
- kill 在第二轮可续跑（恢复回来的部分失败/硬失败论文按累计预算继续）。

全部使用临时目录与 hermetic 桩，不触碰真实 data/。
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from langgraph.runtime import Runtime

from src.graph.read_node import (
    _read_one_paper as actual_read_one_paper,
    _run_read_node_async,
    _should_retry_failure,
    _should_retry_fulltext,
)
from src.graph.runtime import WorkflowRuntimeContext
from src.models.read_models import FullTextStatus, PaperReadResult, ReadNote, ReadRelevance
from src.paper_retrieval.download import DownloadedPaper
from src.paper_retrieval.models import PaperDocument
from src.repositories.node_persistence.read_persistence import ReadPersistenceSink, read_request_fingerprint
from src.repositories.sessions.sqlite import SQLiteSessionRepository


class FailureClassifierTest(unittest.TestCase):
    """失败分类器：只认明确的暂时性签名，未知错误保守不重试。"""

    def test_retryable_hard_failures(self):
        retryable = [
            {"error_kind": "TimeoutError", "message": "read timed out"},
            {"error_kind": "ProviderConnectionError", "message": "connection aborted"},
            {"error_kind": "ConnectionResetError", "message": "peer reset"},
            {"error_kind": "ProviderHttpError", "message": "too many requests", "status_code": 429},
            {"error_kind": "ProviderHttpError", "message": "bad gateway", "status_code": 503},
            {"error_kind": "RuntimeError", "message": "下载全文超时"},
            {"error_kind": "RuntimeError", "message": "getaddrinfo failed"},
        ]
        for failure in retryable:
            self.assertTrue(_should_retry_failure(failure), failure)

    def test_non_retryable_hard_failures(self):
        non_retryable = [
            None,
            {},
            {"error_kind": "RuntimeError", "message": "abstract boom"},
            # 笼统的 "temporary" 不构成暂时性签名，避免误伤永久错误。
            {"error_kind": "RuntimeError", "message": "temporary abstract failure"},
            {"error_kind": "ValueError", "message": "论文缺少标题，未交给模型判断"},
            {"error_kind": "ProviderHttpError", "message": "unauthorized", "status_code": 401},
            {"error_kind": "ProviderHttpError", "message": "not found", "status_code": 404},
            {"error_kind": "ProviderHttpError", "message": "bad request", "status_code": 400},
        ]
        for failure in non_retryable:
            self.assertFalse(_should_retry_failure(failure), failure)

    def test_retryable_fulltext_download_failures(self):
        for reason in ("下载全文超时", "下载地址返回 HTTP 429", "下载地址返回 HTTP 503", "下载全文失败：connection reset"):
            self.assertTrue(
                _should_retry_fulltext(FullTextStatus(status="download_failed", reason=reason)),
                reason,
            )

    def test_non_retryable_fulltext_failures(self):
        # parse_failed / no_url 是永久结果，永不重试。
        self.assertFalse(_should_retry_fulltext(FullTextStatus(status="parse_failed", reason="无法转换")))
        self.assertFalse(_should_retry_fulltext(FullTextStatus(status="no_url", reason="论文没有可尝试的全文地址")))
        for reason in (
            "下载地址返回 HTTP 404",
            "下载地址返回 HTTP 403",
            "文件超过允许大小",
            "下载内容不是可读取的 PDF 或 HTML",
            "全文地址只允许使用 http 或 https",
        ):
            self.assertFalse(
                _should_retry_fulltext(FullTextStatus(status="download_failed", reason=reason)),
                reason,
            )


class ReadRetryTest(unittest.TestCase):
    """用真实临时仓储验证阅读节点的 scene 内补考与跨崩溃累计预算。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.repo = SQLiteSessionRepository(storage_root=self.temp_dir.name)
        self.session_key = self.repo.create(title="Read retry").key
        self.thread_id = f"{self.session_key}:thread-1"
        self.request = {
            "topic": "agent review",
            "constraints": {"deep_read_limit": 1},
            "language": "zh",
        }

    def _state(self, papers):
        return {
            "request": self.request,
            "search_results": [paper.to_dict() for paper in papers],
            "read_results": [],
            "read_artifact_refs": [],
            "diagnostics": {},
        }

    def _runtime(self, turn_id):
        return Runtime(context=WorkflowRuntimeContext(
            session_key=self.session_key,
            turn_id=turn_id,
            thread_id=self.thread_id,
            session_repo=self.repo,
        ))

    def _run(self, papers, turn_id="turn-1"):
        return asyncio.run(_run_read_node_async(self._state(papers), self._runtime(turn_id)))

    def _sink(self, turn_id="turn-0"):
        return ReadPersistenceSink(
            self.repo,
            session_key=self.session_key,
            turn_id=turn_id,
            thread_id=self.thread_id,
            request_fingerprint=read_request_fingerprint(self.request),
        )

    def _note_payloads(self):
        return [
            json.loads(Path(item["path"]).read_text(encoding="utf-8"))
            for item in self.repo.get(self.session_key).artifacts
            if item["artifact_type"] == "paper_read_note"
        ]

    # ---- 摘要补考 ----

    def test_abstract_transient_failure_succeeds_on_retry_within_one_run(self):
        paper = PaperDocument(id="paper-1", title="Agent paper", abstract="abstract")
        calls = []

        async def flaky_abstract(paper, **kwargs):
            calls.append(paper.id)
            if len(calls) == 1:
                raise TimeoutError("read timed out")
            return ReadNote(short_summary="重试后完成"), ReadRelevance(relevant=False, reason="无关"), []

        with (
            mock.patch("src.graph.read_node.MIN_USABLE_PAPERS", 0),
            mock.patch("src.graph.read_node.load_read_agent_llm", return_value=None),
            mock.patch("src.graph.read_node._build_abstract_note", new=flaky_abstract),
        ):
            result = self._run([paper])

        self.assertEqual(calls, ["paper-1", "paper-1"])
        self.assertEqual(result["read_summary"]["failed_paper_count"], 0)
        self.assertEqual(result["read_summary"]["discarded_paper_count"], 1)

    def test_only_failed_paper_is_retried(self):
        papers = [
            PaperDocument(id="good", title="Good paper"),
            PaperDocument(id="bad", title="Bad paper"),
        ]
        calls = []

        async def fake_abstract(paper, **kwargs):
            calls.append(paper.id)
            if paper.id == "bad" and calls.count("bad") == 1:
                raise TimeoutError("read timed out")
            return ReadNote(short_summary="摘要"), ReadRelevance(relevant=False, reason="无关"), []

        with (
            mock.patch("src.graph.read_node.MIN_USABLE_PAPERS", 0),
            mock.patch("src.graph.read_node.load_read_agent_llm", return_value=None),
            mock.patch("src.graph.read_node._build_abstract_note", new=fake_abstract),
        ):
            result = self._run(papers)

        self.assertEqual(calls.count("good"), 1)
        self.assertEqual(calls.count("bad"), 2)
        self.assertEqual(result["read_summary"]["failed_paper_count"], 0)

    def test_permanent_abstract_failure_is_not_retried(self):
        paper = PaperDocument(id="paper-1", title="Agent paper")
        calls = []

        async def fake_abstract(paper, **kwargs):
            calls.append(paper.id)
            raise RuntimeError("abstract boom")

        with (
            mock.patch("src.graph.read_node.MIN_USABLE_PAPERS", 0),
            mock.patch("src.graph.read_node.load_read_agent_llm", return_value=None),
            mock.patch("src.graph.read_node._build_abstract_note", new=fake_abstract),
        ):
            with self.assertRaisesRegex(RuntimeError, "所有论文"):
                self._run([paper])

        self.assertEqual(calls, ["paper-1"])

    def test_abstract_retry_budget_exhausts_at_two_attempts(self):
        papers = [
            PaperDocument(id="good", title="Good paper"),
            PaperDocument(id="bad", title="Bad paper"),
        ]
        calls = []

        async def fake_abstract(paper, **kwargs):
            calls.append(paper.id)
            if paper.id == "bad":
                raise TimeoutError("read timed out")
            return ReadNote(short_summary="摘要"), ReadRelevance(relevant=False, reason="无关"), []

        with (
            mock.patch("src.graph.read_node.MIN_USABLE_PAPERS", 0),
            mock.patch("src.graph.read_node.load_read_agent_llm", return_value=None),
            mock.patch("src.graph.read_node._build_abstract_note", new=fake_abstract),
        ):
            result = self._run(papers)

        self.assertEqual(calls.count("bad"), 2)
        self.assertEqual(result["read_summary"]["failed_paper_count"], 1)
        self.assertEqual(result["read_summary"]["failed_papers"][0]["attempts"], 2)

    # ---- 全文下载补考（真实 download 分支，失败提前返回无需下游）----

    def _run_with_download(self, paper, download_side_effect):
        async def fake_abstract(paper, **kwargs):
            return ReadNote(short_summary="摘要"), ReadRelevance(relevant=True), []

        with (
            mock.patch("src.graph.read_node.load_read_agent_llm", return_value=None),
            mock.patch("src.graph.read_node._build_abstract_note", new=fake_abstract),
            mock.patch("src.graph.read_node.async_download_paper_fulltext", new=download_side_effect),
        ):
            return self._run([paper])

    def test_fulltext_download_retry_exhausts_budget(self):
        paper = PaperDocument(id="paper-1", title="Agent paper", abstract="abstract")
        calls = []

        async def fake_download(paper, **kwargs):
            calls.append(paper.id)
            return DownloadedPaper(status="download_failed", reason="下载全文超时")

        result = self._run_with_download(paper, fake_download)

        self.assertEqual(calls, ["paper-1", "paper-1"])
        self.assertEqual(result["read_results"][0]["full_text"]["status"], "download_failed")
        self.assertEqual(result["read_results"][0]["full_text"]["attempts"], 2)
        self.assertIsNone(result["read_results"][0]["failure"])
        self.assertEqual(result["read_summary"]["failed_paper_count"], 0)

    def test_fulltext_permanent_download_failure_not_retried(self):
        paper = PaperDocument(id="paper-1", title="Agent paper", abstract="abstract")
        calls = []

        async def fake_download(paper, **kwargs):
            calls.append(paper.id)
            return DownloadedPaper(status="download_failed", reason="下载地址返回 HTTP 404")

        result = self._run_with_download(paper, fake_download)

        self.assertEqual(calls, ["paper-1"])
        self.assertEqual(result["read_results"][0]["full_text"]["status"], "download_failed")
        self.assertEqual(result["read_results"][0]["full_text"]["attempts"], 1)
        self.assertIsNone(result["read_results"][0]["failure"])

    def test_fulltext_retry_success_upserts_single_note(self):
        paper = PaperDocument(id="paper-1", title="Agent paper", abstract="abstract")
        fulltext_calls = []

        async def fake_abstract(paper, **kwargs):
            return ReadNote(short_summary="摘要"), ReadRelevance(relevant=True), []

        async def fake_fulltext(item, **kwargs):
            if not kwargs.get("process_full_text"):
                return await actual_read_one_paper(item, **kwargs)
            fulltext_calls.append(item.paper.id)
            result = item.restored_result
            if len(fulltext_calls) == 1:
                result.full_text = FullTextStatus(status="download_failed", reason="下载全文超时", attempts=1)
                return result
            result.note.evidence_level = "full_text"
            result.full_text = FullTextStatus(status="chunks_saved")
            return result

        with (
            mock.patch("src.graph.read_node.load_read_agent_llm", return_value=None),
            mock.patch("src.graph.read_node._build_abstract_note", new=fake_abstract),
            mock.patch("src.graph.read_node._read_one_paper", new=fake_fulltext),
        ):
            result = self._run([paper])

        self.assertEqual(fulltext_calls, ["paper-1", "paper-1"])
        self.assertEqual(result["read_results"][0]["full_text"]["status"], "chunks_saved")
        self.assertEqual(len(self._note_payloads()), 1)

    # ---- 跨崩溃续跑：累计预算 ----

    def test_restored_partial_failure_is_retried_on_resume(self):
        """模拟"补考到一半崩溃"：note 停在 download_failed attempts=1，续跑应继续补考。"""
        paper = PaperDocument(id="paper-1", title="Agent paper", abstract="abstract")
        restored = PaperReadResult(
            paper=paper,
            note=ReadNote(short_summary="已保存摘要"),
            relevance=ReadRelevance(relevant=True, status="selected_for_deep_read"),
            full_text=FullTextStatus(status="download_failed", reason="下载全文超时", attempts=1),
        )
        self._sink().persist_paper(restored, stage="abstract_complete")
        calls = []

        async def fake_download(paper, **kwargs):
            calls.append(paper.id)
            return DownloadedPaper(status="download_failed", reason="下载全文超时")

        result = self._run_with_download(paper, fake_download)

        # 首轮全文波次跳过已失败论文，只有补考波次重下这一次。
        self.assertEqual(calls, ["paper-1"])
        self.assertEqual(result["read_results"][0]["full_text"]["attempts"], 2)
        self.assertEqual(len(self._note_payloads()), 1)

    def test_restored_hard_failure_budget_carries_over_resume(self):
        """失败 note 带回 attempts=1，续跑再失败累计到 2，不再触发补考。"""
        paper = PaperDocument(id="paper-1", title="Agent paper")
        failure = {"stage": "abstract", "error_kind": "TimeoutError", "message": "read timed out", "attempts": 1}
        self._sink().persist_paper(PaperReadResult(paper=paper, failure=failure))
        calls = []

        async def fake_abstract(paper, **kwargs):
            calls.append(paper.id)
            raise TimeoutError("read timed out")

        with (
            mock.patch("src.graph.read_node.MIN_USABLE_PAPERS", 0),
            mock.patch("src.graph.read_node.load_read_agent_llm", return_value=None),
            mock.patch("src.graph.read_node._build_abstract_note", new=fake_abstract),
        ):
            with self.assertRaisesRegex(RuntimeError, "所有论文"):
                self._run([paper])

        # 本轮只跑正常波次一次（prior=1 → attempts=2），不再补考。
        self.assertEqual(calls, ["paper-1"])
        payloads = self._note_payloads()
        self.assertEqual(payloads[0]["last_error"]["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
