import io
import json
import tempfile
import unittest
import zipfile
from unittest import mock

from fastapi.testclient import TestClient

from src.api import GatewayConfig, create_app
from src.api.routers.sessions import _trace_filename
from src.graph.checkpointing import build_checkpoint_bundle
from src.llm.config import CheckpointSettings
from src.repositories.sessions.sqlite import SQLiteSessionRepository
from src.services.sessions import SessionError
from src.repositories.settings.json import SettingsRepository
from src.services.trace_bundle import TraceBundleService


class TraceBundleTest(unittest.TestCase):
    """验证轨迹按回合隔离、zip 结构、全量导出和 HTTP 下载契约。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.repo = SQLiteSessionRepository(storage_root=self.temp_dir.name)
        self.session_key = self.repo.create(title="Trace").key
        self.service = TraceBundleService(self.repo)
        self._add_turn("turn-1", "2026-11-18T10:00:00+00:00")
        self._add_turn("turn-2", "2026-11-18T11:00:00+00:00")

    def _add_turn(self, turn_id: str, created_at: str):
        self.repo.append_message(self.session_key, "user", f"研究主题 {turn_id}", turn_id=turn_id)
        self.repo.append_event(
            self.session_key,
            "status",
            metadata={"turn_id": turn_id, "run_id": f"run-{turn_id}", "thread_id": f"{self.session_key}:{turn_id}", "status": "running"},
            created_at=created_at,
        )
        self.repo.append_event(
            self.session_key,
            "message",
            content="已完成",
            metadata={
                "turn_id": turn_id,
                "role": "assistant",
                # 与生产 reporter → session_runs 的持久化结构一致：事件 metadata 内还有消息 metadata。
                "metadata": {"diagnostics": {"compose_reply": {"status": "ok", "turn_id": turn_id}}},
            },
            created_at=created_at,
        )
        self.repo.append_event(
            self.session_key,
            "turn_end",
            metadata={"turn_id": turn_id, "status": "completed"},
            created_at=created_at,
        )
        self.repo.write_artifact(
            self.session_key,
            artifact_type="final_review",
            name=f"{turn_id}.md",
            content=f"# {turn_id}",
            relative_path=f"artifacts/final_review/{turn_id}/review.md",
            metadata={"turn_id": turn_id},
        )
        self.repo.append_message(self.session_key, "assistant", f"{turn_id} 的最终回复", turn_id=turn_id)

    def _zip(self, turn_id: str = "turn-1") -> zipfile.ZipFile:
        return zipfile.ZipFile(io.BytesIO(self.service.build_zip(self.session_key, turn_id)))

    def test_lists_turns_without_mixing_status_or_identity(self):
        traces = self.service.list_traces(self.session_key)["traces"]

        self.assertEqual([item["turn_id"] for item in traces], ["turn-2", "turn-1"])
        self.assertEqual(traces[0]["status"], "completed")
        self.assertEqual(traces[0]["run_id"], "run-turn-2")
        self.assertEqual(traces[0]["event_count"], 3)
        self.assertEqual(traces[0]["message_count"], 2)
        self.assertEqual(traces[0]["artifact_count"], 1)
        self.assertIn("/traces/turn-2", traces[0]["download_url"])

    def test_bundle_contains_only_requested_turn_and_expected_files(self):
        with self._zip() as archive:
            self.assertEqual(
                set(archive.namelist()),
                {
                    "manifest.json", "events.jsonl", "messages.jsonl", "artifacts.json",
                    "diagnostics.json", "llm_calls.jsonl", "tool_calls.jsonl",
                    "state_history.jsonl", "report.md",
                },
            )
            manifest = json.loads(archive.read("manifest.json"))
            events = [json.loads(line) for line in archive.read("events.jsonl").decode("utf-8").splitlines()]
            messages = [json.loads(line) for line in archive.read("messages.jsonl").decode("utf-8").splitlines()]
            artifacts = json.loads(archive.read("artifacts.json"))
            diagnostics = json.loads(archive.read("diagnostics.json"))
            report = archive.read("report.md").decode("utf-8")

        self.assertNotIn("trace_version", manifest)
        self.assertEqual(manifest["turn_id"], "turn-1")
        self.assertEqual(manifest["event_count"], 3)
        self.assertEqual(manifest["llm_call_count"], 0)
        self.assertEqual(manifest["tool_call_count"], 0)
        self.assertEqual([event["seq_no"] for event in events], [1, 2, 3])
        self.assertTrue(all(event["metadata"]["turn_id"] == "turn-1" for event in events))
        self.assertTrue(all(message["turn_id"] == "turn-1" for message in messages))
        self.assertEqual(artifacts[0]["artifact_id"], self.repo.get(self.session_key).artifacts[0]["id"])
        self.assertNotIn("path", artifacts[0])
        self.assertEqual(diagnostics["compose_reply"]["turn_id"], "turn-1")
        self.assertIn("研究主题 turn-1", report)
        self.assertNotIn("turn-2", report)

    def test_bundle_exports_full_fidelity_without_redaction(self):
        # 中文注释：V2 契约反转——全量导出，密钥/路径/超长内容原样保留，
        # 出口只在 report 里提示分享者自行检查、必要时更换 API key。
        secret = "sk-live-secret-value"
        windows_path = r"C:\Users\someone\data\sessions\private\note.json"
        posix_path = "/data/sessions/private/note.json"
        long_content = "x" * 25_000
        self.repo.append_message(self.session_key, "user", long_content, turn_id="turn-1")
        self.repo.append_event(
            self.session_key,
            "error",
            content=f"无法读取 {windows_path}、{posix_path}",
            metadata={
                "turn_id": "turn-1",
                "api_key": secret,
                "nested": {"authorization": "Bearer private-token", "source_path": windows_path},
            },
        )

        with self._zip() as archive:
            raw = b"\n".join(archive.read(name) for name in archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
            events = [json.loads(line) for line in archive.read("events.jsonl").decode("utf-8").splitlines()]
            messages = [json.loads(line) for line in archive.read("messages.jsonl").decode("utf-8").splitlines()]
            report = archive.read("report.md").decode("utf-8")

        self.assertIn(secret.encode(), raw)
        self.assertIn(b"private-token", raw)
        self.assertIn(windows_path.encode(), raw)
        self.assertIn(posix_path.encode(), raw)
        self.assertNotIn("truncated", manifest)
        self.assertEqual(events[-1]["metadata"]["api_key"], secret)
        self.assertEqual(events[-1]["metadata"]["nested"]["authorization"], "Bearer private-token")
        self.assertEqual(events[-1]["metadata"]["nested"]["source_path"], windows_path)
        self.assertIn(windows_path, events[-1]["content"])
        self.assertEqual(messages[-1]["content"], long_content)
        self.assertIn("未经脱敏", report)
        self.assertIn("建议更换相关 API key", report)

    def _append_trace_event(self, turn_id: str, event_type: str, metadata: dict):
        self.repo.append_event(
            self.session_key,
            event_type,
            content=str(metadata.get("content") or ""),
            metadata={"turn_id": turn_id, **metadata},
        )

    def test_llm_calls_ledger_exports_full_envelope(self):
        long_prompt = "论文全文" * 6000
        self._append_trace_event(
            "turn-1",
            "llm_call",
            {
                "event": "llm_call",
                "id": "llm-1",
                "parent_id": "turn-1:read",
                "type": "llm_call",
                "node_key": "read",
                "status": "completed",
                "started_at": "2026-11-18T10:00:01+00:00",
                "duration_ms": 8340,
                "tokens": {"input_tokens": 5200, "output_tokens": 610},
                "error": None,
                "payload": {
                    "operation": "chat",
                    "model": "deepseek-chat",
                    "attempts": 2,
                    "prompt": [{"role": "user", "content": long_prompt}],
                    "response": "阅读完成",
                },
            },
        )

        with self._zip() as archive:
            manifest = json.loads(archive.read("manifest.json"))
            rows = [json.loads(line) for line in archive.read("llm_calls.jsonl").decode("utf-8").splitlines()]
            report = archive.read("report.md").decode("utf-8")

        self.assertEqual(manifest["llm_call_count"], 1)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["id"], "llm-1")
        self.assertEqual(row["parent_id"], "turn-1:read")
        self.assertEqual(row["node_key"], "read")
        self.assertEqual(row["tokens"], {"input_tokens": 5200, "output_tokens": 610})
        self.assertEqual(row["payload"]["attempts"], 2)
        self.assertEqual(row["payload"]["prompt"][0]["content"], long_prompt)
        self.assertEqual(row["payload"]["response"], "阅读完成")
        self.assertIn("共 1 次", report)
        self.assertIn("5200 / 610", report)

    def test_tool_calls_ledger_merges_pairs_and_keeps_unfinished(self):
        self._append_trace_event(
            "turn-1",
            "tool_call",
            {
                "event": "tool_call",
                "id": "call-1",
                "parent_id": "turn-1:read",
                "node_key": "read",
                "stage": "downloading_full_text",
                "status": "running",
                "started_at": "2026-11-18T10:00:01+00:00",
                "payload": {"name": "download_fulltext", "args": {"paper_id": "P3"}},
            },
        )
        self._append_trace_event(
            "turn-1",
            "tool_result",
            {
                "event": "tool_result",
                "id": "result-1",
                "parent_id": "call-1",
                "node_key": "read",
                "status": "failed",
                "duration_ms": 60013,
                "error": {"message": "连接超时"},
                "payload": {"name": "download_fulltext", "result": {"status": "download_failed"}},
            },
        )
        # 只有开工单没有完工单（运行中崩溃）：保留 running，不丢信息。
        self._append_trace_event(
            "turn-1",
            "tool_call",
            {
                "event": "tool_call",
                "id": "call-2",
                "node_key": "read",
                "status": "running",
                "payload": {"name": "extract_fulltext", "args": {"paper_id": "P5"}},
            },
        )

        with self._zip() as archive:
            manifest = json.loads(archive.read("manifest.json"))
            rows = [json.loads(line) for line in archive.read("tool_calls.jsonl").decode("utf-8").splitlines()]
            report = archive.read("report.md").decode("utf-8")

        self.assertEqual(manifest["tool_call_count"], 2)
        self.assertEqual(len(rows), 2)
        merged = rows[0]
        self.assertEqual(merged["call_id"], "call-1")
        self.assertEqual(merged["name"], "download_fulltext")
        self.assertEqual(merged["args"], {"paper_id": "P3"})
        self.assertEqual(merged["status"], "failed")
        self.assertEqual(merged["duration_ms"], 60013)
        self.assertEqual(merged["error"], {"message": "连接超时"})
        self.assertEqual(merged["result"], {"status": "download_failed"})
        self.assertEqual(rows[1]["call_id"], "call-2")
        self.assertEqual(rows[1]["status"], "running")
        self.assertIsNone(rows[1]["result"])
        self.assertIn("失败 1 次", report)

    def test_ledgers_are_isolated_per_turn(self):
        self._append_trace_event(
            "turn-2",
            "llm_call",
            {"event": "llm_call", "id": "llm-other", "status": "completed", "payload": {"model": "other"}},
        )

        with self._zip("turn-1") as archive:
            self.assertEqual(archive.read("llm_calls.jsonl"), b"")
            self.assertEqual(archive.read("tool_calls.jsonl"), b"")
            manifest = json.loads(archive.read("manifest.json"))
        self.assertEqual(manifest["llm_call_count"], 0)
        self.assertEqual(manifest["tool_call_count"], 0)

    def test_turn_detail_returns_only_requested_turn(self):
        detail = self.service.turn_detail(self.session_key, "turn-1")

        self.assertEqual(detail["turn_id"], "turn-1")
        self.assertEqual([event["seq_no"] for event in detail["events"]], [1, 2, 3])
        self.assertTrue(all(event["metadata"]["turn_id"] == "turn-1" for event in detail["events"]))
        self.assertEqual(len(detail["messages"]), 2)
        self.assertEqual(len(detail["artifacts"]), 1)
        self.assertIn("id", detail["artifacts"][0])

    def test_turn_detail_missing_turn_is_a_404(self):
        with self.assertRaises(SessionError) as missing:
            self.service.turn_detail(self.session_key, "missing")
        self.assertEqual(missing.exception.status, 404)

    def test_http_turn_detail_endpoint(self):
        settings = SettingsRepository(
            initial={
                "providers": {"openai": {"api_key": "sk-test", "api_base": "https://api.openai.com/v1"}},
                "agents": {"default_agent": {"model_name": "gpt-5-mini", "provider": "openai"}},
                "embedding_profiles": {},
            }
        )
        app = create_app(
            settings_repo=settings,
            sessions_repo=self.repo,
            config=GatewayConfig(),
            checkpoint_bundle=build_checkpoint_bundle(CheckpointSettings(backend="memory")),
        )
        with TestClient(app) as client:
            detail = client.get(f"/api/sessions/{self.session_key}/traces/turn-1/events")
            missing = client.get(f"/api/sessions/{self.session_key}/traces/missing/events")

        self.assertEqual(detail.status_code, 200)
        payload = detail.json()
        self.assertEqual(payload["turn_id"], "turn-1")
        self.assertEqual(len(payload["events"]), 3)
        self.assertIn("state_history", payload)
        self.assertEqual(missing.status_code, 404)

    def test_state_history_in_zip_and_turn_detail(self):
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        from src.graph.checkpointing import build_checkpoint_bundle
        from src.llm.config import CheckpointSettings

        class _S(TypedDict, total=False):
            counter: int

        bundle = build_checkpoint_bundle(CheckpointSettings(backend="memory"))
        self.addCleanup(bundle.close)
        graph = StateGraph(_S)
        graph.add_node("bump", lambda state: {"counter": 1})
        graph.add_edge(START, "bump")
        graph.add_edge("bump", END)
        app = graph.compile(checkpointer=bundle.saver)
        thread_id = f"{self.session_key}:turn-1"
        app.invoke({"counter": 0}, config={"configurable": {"thread_id": thread_id}})

        service = TraceBundleService(self.repo, checkpoint_bundle=bundle)
        with zipfile.ZipFile(io.BytesIO(service.build_zip(self.session_key, "turn-1"))) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            rows = [json.loads(line) for line in archive.read("state_history.jsonl").decode("utf-8").splitlines()]
            report = archive.read("report.md").decode("utf-8")
        detail = service.turn_detail(self.session_key, "turn-1")

        self.assertEqual(manifest["state_history_count"], len(rows))
        self.assertGreaterEqual(len(rows), 3)
        self.assertEqual(rows[-1]["changed"], {"counter": 1})
        self.assertIn("状态快照", report)
        self.assertEqual(detail["state_history"], rows)

    def test_state_history_empty_without_bundle(self):
        with self._zip() as archive:
            self.assertEqual(archive.read("state_history.jsonl"), b"")
            manifest = json.loads(archive.read("manifest.json"))
        self.assertEqual(manifest["state_history_count"], 0)
        self.assertEqual(self.service.turn_detail(self.session_key, "turn-1")["state_history"], [])

    def test_turn_ids_with_whitespace_remain_exact_and_isolated(self):
        padded_turn_id = " turn-1 "
        self.repo.append_message(self.session_key, "user", "带空格的独立回合", turn_id=padded_turn_id)

        traces = self.service.list_traces(self.session_key)["traces"]
        self.assertIn(padded_turn_id, [item["turn_id"] for item in traces])
        padded_summary = next(item for item in traces if item["turn_id"] == padded_turn_id)
        self.assertEqual(padded_summary["message_count"], 1)
        self.assertEqual(padded_summary["event_count"], 0)

        with self._zip(padded_turn_id) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            messages = [json.loads(line) for line in archive.read("messages.jsonl").decode("utf-8").splitlines()]
        self.assertEqual(manifest["turn_id"], padded_turn_id)
        self.assertEqual([message["content"] for message in messages], ["带空格的独立回合"])

    def test_process_recovery_marks_matching_running_turn_failed(self):
        turn_id = "recover-me"
        thread_id = f"{self.session_key}:{turn_id}"
        self.repo.append_event(
            self.session_key,
            "status",
            metadata={"turn_id": turn_id, "thread_id": thread_id, "status": "running"},
            created_at="2026-11-18T12:00:00+00:00",
        )
        # 中文注释：当前契约要求中断事件由生产者自带 turn_id（见 recover_interrupted_runs），
        # Trace 只按字段分组，不做归属推断。
        self.repo.append_event(
            self.session_key,
            "run_interrupted",
            metadata={
                "turn_id": turn_id,
                "status": "failed",
                "thread_id": thread_id,
                "recovery_status": "process_restarted",
            },
            created_at="2026-11-18T12:01:00+00:00",
        )

        # 中文注释：另一个没有收到中断事件的 running 回合不能被波及——
        # 终态只落在事件自带的 turn_id 上，不存在跨回合猜测。
        self.repo.append_event(
            self.session_key,
            "status",
            metadata={"turn_id": "keep-running", "thread_id": f"{self.session_key}:keep-running", "status": "running"},
            created_at="2026-11-18T12:00:30+00:00",
        )

        summary = next(item for item in self.service.list_traces(self.session_key)["traces"] if item["turn_id"] == turn_id)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["finished_at"], "2026-11-18T12:01:00+00:00")
        untouched = next(item for item in self.service.list_traces(self.session_key)["traces"] if item["turn_id"] == "keep-running")
        self.assertEqual(untouched["status"], "running")
        with self._zip(turn_id) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["finished_at"], "2026-11-18T12:01:00+00:00")

    def test_manifest_uses_actual_events_when_summary_is_stale(self):
        stale_summary = dict(
            next(item for item in self.repo.list_trace_turns(self.session_key) if item["turn_id"] == "turn-1")
        )
        stale_summary["status"] = "running"
        stale_summary["finished_at"] = None

        with mock.patch.object(self.repo, "list_trace_turns", return_value=[stale_summary]):
            payload = self.service.build_zip(self.session_key, "turn-1")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["finished_at"], "2026-11-18T10:00:00+00:00")

    def test_message_only_turn_has_an_empty_but_valid_trace(self):
        self.repo.append_message(self.session_key, "user", "只有消息", turn_id="empty-events")

        with self._zip("empty-events") as archive:
            self.assertEqual(archive.read("events.jsonl"), b"")
            manifest = json.loads(archive.read("manifest.json"))
            messages = archive.read("messages.jsonl").decode("utf-8").splitlines()

        self.assertEqual(manifest["event_count"], 0)
        self.assertEqual(manifest["message_count"], 1)
        self.assertEqual(len(messages), 1)

    def test_unknown_session_or_turn_is_a_404(self):
        with self.assertRaises(SessionError) as missing_session:
            self.service.build_zip("missing", "turn-1")
        with self.assertRaises(SessionError) as missing_turn:
            self.service.build_zip(self.session_key, "missing")

        self.assertEqual(missing_session.exception.status, 404)
        self.assertEqual(missing_turn.exception.status, 404)

    def test_download_filename_is_ascii_and_cannot_inject_headers(self):
        special_turn_id = "回合/1"
        filename = _trace_filename("回合/1\r\nX-Test: injected")
        self.repo.append_message(self.session_key, "user", "特殊回合", turn_id=special_turn_id)

        trace = next(
            item for item in self.service.list_traces(self.session_key)["traces"] if item["turn_id"] == special_turn_id
        )
        self.assertTrue(trace["download_url"].endswith("/%E5%9B%9E%E5%90%88%2F1"))
        self.assertTrue(filename.isascii())
        self.assertNotIn("\r", filename)
        self.assertNotIn("\n", filename)
        self.assertEqual(filename, "paper-agent-trace-___1__X-Test__injected.zip")

    def test_http_list_download_and_missing_turn(self):
        settings = SettingsRepository(
            initial={
                "providers": {"openai": {"api_key": "sk-test", "api_base": "https://api.openai.com/v1"}},
                "agents": {"default_agent": {"model_name": "gpt-5-mini", "provider": "openai"}},
                "embedding_profiles": {},
            }
        )
        app = create_app(
            settings_repo=settings,
            sessions_repo=self.repo,
            config=GatewayConfig(),
            checkpoint_bundle=build_checkpoint_bundle(CheckpointSettings(backend="memory")),
        )
        with TestClient(app) as client:
            listed = client.get(f"/api/sessions/{self.session_key}/traces")
            downloaded = client.get(f"/api/sessions/{self.session_key}/traces/turn-1")
            missing = client.get(f"/api/sessions/{self.session_key}/traces/missing")

        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()["traces"]), 2)
        self.assertEqual(downloaded.status_code, 200)
        self.assertEqual(downloaded.headers["content-type"], "application/zip")
        self.assertIn("paper-agent-trace-turn-1.zip", downloaded.headers["content-disposition"])
        with zipfile.ZipFile(io.BytesIO(downloaded.content)) as archive:
            self.assertEqual(json.loads(archive.read("manifest.json"))["turn_id"], "turn-1")
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
