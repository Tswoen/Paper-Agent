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
from src.services.trace_bundle import MAX_STRING_LENGTH, TraceBundleService


class TraceBundleTest(unittest.TestCase):
    """验证轨迹按回合隔离、zip 结构、脱敏和 HTTP 下载契约。"""

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
                {"manifest.json", "events.jsonl", "messages.jsonl", "artifacts.json", "diagnostics.json", "report.md"},
            )
            manifest = json.loads(archive.read("manifest.json"))
            events = [json.loads(line) for line in archive.read("events.jsonl").decode("utf-8").splitlines()]
            messages = [json.loads(line) for line in archive.read("messages.jsonl").decode("utf-8").splitlines()]
            artifacts = json.loads(archive.read("artifacts.json"))
            diagnostics = json.loads(archive.read("diagnostics.json"))
            report = archive.read("report.md").decode("utf-8")

        self.assertEqual(manifest["trace_version"], 1)
        self.assertEqual(manifest["turn_id"], "turn-1")
        self.assertEqual(manifest["event_count"], 3)
        self.assertEqual([event["seq_no"] for event in events], [1, 2, 3])
        self.assertTrue(all(event["metadata"]["turn_id"] == "turn-1" for event in events))
        self.assertTrue(all(message["turn_id"] == "turn-1" for message in messages))
        self.assertEqual(artifacts[0]["artifact_id"], self.repo.get(self.session_key).artifacts[0]["id"])
        self.assertNotIn("path", artifacts[0])
        self.assertEqual(diagnostics["compose_reply"]["turn_id"], "turn-1")
        self.assertIn("研究主题 turn-1", report)
        self.assertNotIn("turn-2", report)

    def test_bundle_redacts_secrets_paths_and_truncates_long_payloads(self):
        secret = "sk-live-secret-value"
        windows_path = r"C:\Users\someone\data\sessions\private\note.json"
        posix_path = "/data/sessions/private/note.json"
        usr_path = "/usr/local/private/note.json"
        unc_path = r"\\server\share\note.json"
        file_url = "file:///data/sessions/private/note.json"
        web_url = "https://example.com/data/sessions/private"
        self.repo.append_message(self.session_key, "user", "x" * (MAX_STRING_LENGTH + 50), turn_id="turn-1")
        self.repo.append_event(
            self.session_key,
            "error",
            content=f"无法读取 {windows_path}、{posix_path}、{usr_path}、{unc_path}、{file_url}；文档见 {web_url}",
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

        self.assertNotIn(secret.encode(), raw)
        self.assertNotIn(b"private-token", raw)
        self.assertNotIn(windows_path.encode(), raw)
        self.assertNotIn(posix_path.encode(), raw)
        self.assertNotIn(usr_path.encode(), raw)
        self.assertNotIn(unc_path.encode(), raw)
        self.assertNotIn(file_url.encode(), raw)
        self.assertNotIn(b"C:\\Users", raw)
        self.assertIn(web_url.encode(), raw)
        self.assertTrue(manifest["truncated"])
        self.assertEqual(events[-1]["metadata"]["api_key"], "[REDACTED]")
        self.assertEqual(events[-1]["metadata"]["nested"]["authorization"], "[REDACTED]")
        self.assertNotIn("source_path", events[-1]["metadata"]["nested"])
        sanitized_content = events[-1]["content"]
        for path in (windows_path, posix_path, usr_path, unc_path, file_url):
            self.assertNotIn(path, sanitized_content)
        self.assertIn("[REMOVED_PATH]", sanitized_content)
        self.assertIn(web_url, sanitized_content)
        self.assertTrue(messages[-1]["content"].endswith("…[truncated]"))

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
