from __future__ import annotations

import io
import json
import re
import zipfile
from typing import Any
from urllib.parse import quote

from src.models.artifacts import to_artifact_ref
from src.models.sessions import utc_now
from src.repositories.sessions.base import SessionRepository
from src.services.sessions import SessionError


JsonObject = dict[str, Any]
TRACE_VERSION = 1
MAX_STRING_LENGTH = 20_000
_BUNDLE_FILES = (
    "manifest.json",
    "events.jsonl",
    "messages.jsonl",
    "artifacts.json",
    "diagnostics.json",
    "report.md",
)
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "passwd",
    "secret",
    "client_secret",
    "access_token",
    "refresh_token",
}
_PATH_KEYS = {
    "path",
    "local_path",
    "absolute_path",
    "source_path",
    "markdown_path",
    "storage_root",
    "vector_store_path",
    "checkpoint_path",
}
_SECRET_PATTERNS = (
    (re.compile(r"\bsk-[A-Za-z0-9_-]{4,}\b"), "[REDACTED]"),
    (re.compile(r"\blsv2_[A-Za-z0-9_-]{4,}\b"), "[REDACTED]"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [REDACTED]"),
)
_WINDOWS_PATH_PATTERN = re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Z]:[\\/](?:[^\s\"'<>|]+)")
_UNC_PATH_PATTERN = re.compile(r"(?<![:/\w])(?:\\\\[^\\\s]+\\|//[^/\s]+/)[^\s\"'<>|]+")
# 中文说明：POSIX 根目录不应靠白名单枚举；排除 URL 的 :// 前缀后，覆盖所有绝对路径。
_POSIX_PATH_PATTERN = re.compile(r"(?<![:/\w])/(?:[^\s/\"'<>|]+/)*[^\s/\"'<>|]+")
_FILE_URL_PATTERN = re.compile(r"(?i)\bfile://[^\s\"'<>|]+")


class TraceBundleService:
    """把某次回合已经持久化的数据整理成可查看、可下载的轨迹包。"""

    def __init__(self, repo: SessionRepository):
        self._repo = repo

    def list_traces(self, session_key: str) -> JsonObject:
        """列出会话内可下载的回合，最新的排在前面。"""

        traces = []
        for item in self._repo.list_trace_turns(session_key):
            trace = dict(item)
            trace["download_url"] = (
                f"/api/sessions/{quote(session_key, safe='')}/traces/{quote(str(trace['turn_id']), safe='')}"
            )
            traces.append(trace)
        return {"traces": traces}

    def build_zip(self, session_key: str, turn_id: str) -> bytes:
        """在内存中生成一个 turn 的 zip，不写入会话目录或长期临时文件。"""

        summaries = self._repo.list_trace_turns(session_key)
        summary = next((item for item in summaries if item.get("turn_id") == turn_id), None)
        if summary is None:
            raise SessionError(f"trace not found: {turn_id}", 404)

        events = self._repo.list_events(session_key, turn_id)
        messages = self._repo.list_messages(session_key, turn_id)
        artifacts = self._repo.list_artifacts(session_key, turn_id)
        # 运行中下载时列表和正文可能读到不同时间点；manifest 以实际打包内容为基准重算。
        snapshot = _snapshot_summary(summary, events, messages, artifacts)
        sanitizer = _TraceSanitizer()
        safe_events = [sanitizer.sanitize(event) for event in events]
        safe_messages = [sanitizer.sanitize(message) for message in messages]
        safe_artifacts = [sanitizer.sanitize(to_artifact_ref(artifact)) for artifact in artifacts]
        diagnostics = _extract_diagnostics(safe_events)

        manifest = {
            "trace_version": TRACE_VERSION,
            "session_key": session_key,
            "turn_id": turn_id,
            "run_id": snapshot.get("run_id"),
            "thread_id": snapshot.get("thread_id"),
            "status": snapshot.get("status") or "unknown",
            "started_at": snapshot.get("started_at"),
            "finished_at": snapshot.get("finished_at"),
            "generated_at": utc_now(),
            "event_count": len(safe_events),
            "message_count": len(safe_messages),
            "artifact_count": len(safe_artifacts),
            "truncated": sanitizer.truncated,
            "files": list(_BUNDLE_FILES),
        }
        report = _build_report(manifest, safe_messages, safe_events, safe_artifacts)
        files = {
            "manifest.json": _json_bytes(manifest),
            "events.jsonl": _jsonl_bytes(safe_events),
            "messages.jsonl": _jsonl_bytes(safe_messages),
            "artifacts.json": _json_bytes(safe_artifacts),
            "diagnostics.json": _json_bytes(diagnostics),
            "report.md": report.encode("utf-8"),
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in _BUNDLE_FILES:
                archive.writestr(name, files[name])
        return buffer.getvalue()


class _TraceSanitizer:
    """递归清理导出数据中的密钥、本地路径和超长文本。"""

    def __init__(self):
        self.truncated = False

    def sanitize(self, value: Any, *, key: str = "") -> Any:
        normalized_key = key.lower().replace("-", "_")
        if normalized_key in _SENSITIVE_KEYS:
            return "[REDACTED]"
        if isinstance(value, dict):
            sanitized: JsonObject = {}
            for item_key, item_value in value.items():
                normalized_item_key = str(item_key)
                if normalized_item_key.lower().replace("-", "_") in _PATH_KEYS:
                    continue
                sanitized[normalized_item_key] = self.sanitize(item_value, key=normalized_item_key)
            return sanitized
        if isinstance(value, (list, tuple)):
            return [self.sanitize(item) for item in value]
        if isinstance(value, str):
            return self._sanitize_text(value)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return self._sanitize_text(str(value))

    def _sanitize_text(self, value: str) -> str:
        sanitized = value
        for pattern, replacement in _SECRET_PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)
        sanitized = _FILE_URL_PATTERN.sub("[REMOVED_PATH]", sanitized)
        sanitized = _WINDOWS_PATH_PATTERN.sub("[REMOVED_PATH]", sanitized)
        sanitized = _UNC_PATH_PATTERN.sub("[REMOVED_PATH]", sanitized)
        sanitized = _POSIX_PATH_PATTERN.sub("[REMOVED_PATH]", sanitized)
        if len(sanitized) > MAX_STRING_LENGTH:
            self.truncated = True
            sanitized = sanitized[:MAX_STRING_LENGTH] + "\n…[truncated]"
        return sanitized


def _snapshot_summary(
    summary: JsonObject,
    events: list[JsonObject],
    messages: list[JsonObject],
    artifacts: list[JsonObject],
) -> JsonObject:
    """从实际导出的记录重算状态和计数，避免活动 turn 的清单与正文矛盾。"""

    snapshot = dict(summary)
    observed_status = ""
    observed_finished_at: str | None = None
    timestamps = [
        str(item["created_at"])
        for item in [*events, *messages, *artifacts]
        if item.get("created_at")
    ]
    if timestamps:
        snapshot["started_at"] = min(timestamps)
        snapshot["updated_at"] = max(timestamps)

    for event in events:
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        snapshot["run_id"] = str(metadata.get("run_id") or "") or snapshot.get("run_id")
        snapshot["thread_id"] = str(metadata.get("thread_id") or "") or snapshot.get("thread_id")
        event_type = str(event.get("event_type") or "")
        event_status = str(metadata.get("status") or "").strip()
        if event_type == "status" and event_status:
            observed_status = event_status
        elif event_type in {"error", "run_interrupted"}:
            observed_status = event_status or "failed"
            observed_finished_at = str(event.get("created_at") or "") or None
        elif event_type == "turn_end":
            observed_status = event_status or "completed"
            observed_finished_at = str(event.get("created_at") or "") or None

    # 中文注释：终态一律以实际打包的事件为准。进程中断的 run_interrupted 事件
    # 由生产者带上 turn_id，会正常出现在本回合事件列表里，无需另读会话摘要兜底。
    if observed_status:
        snapshot["status"] = observed_status
        snapshot["finished_at"] = observed_finished_at
    return snapshot


def _extract_diagnostics(events: list[JsonObject]) -> JsonObject:
    """复用最终消息事件中的诊断信息，没有时显式返回空对象。"""

    for event in reversed(events):
        metadata = event.get("metadata")
        if not isinstance(metadata, dict):
            continue
        # 中文注释：当前消息事件经 reporter → 持久化后是两层结构——外层是持久化
        # 信封（整个事件去掉 content），内层是事件自带的 metadata，diagnostics 在内层。
        nested_metadata = metadata.get("metadata")
        if not isinstance(nested_metadata, dict):
            continue
        diagnostics = nested_metadata.get("diagnostics")
        if isinstance(diagnostics, dict):
            return diagnostics
    return {}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _jsonl_bytes(items: list[JsonObject]) -> bytes:
    return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items).encode("utf-8")


def _build_report(
    manifest: JsonObject,
    messages: list[JsonObject],
    events: list[JsonObject],
    artifacts: list[JsonObject],
) -> str:
    """生成不包含大段模型输出的简短人工可读摘要。"""

    lines = [
        "# Paper Agent 运行轨迹",
        "",
        f"- Trace 版本：{manifest['trace_version']}",
        f"- 会话：`{manifest['session_key']}`",
        f"- 回合：`{manifest['turn_id']}`",
        f"- 状态：{manifest['status']}",
        f"- 开始时间：{manifest.get('started_at') or '未知'}",
        f"- 结束时间：{manifest.get('finished_at') or '未知'}",
        f"- 事件 / 消息 / 产物：{manifest['event_count']} / {manifest['message_count']} / {manifest['artifact_count']}",
        "",
    ]
    user_message = next((item for item in messages if item.get("role") == "user"), None)
    if user_message:
        lines.extend(["## 用户输入", "", _preview(str(user_message.get("content") or "")), ""])

    errors = []
    for event in events:
        if event.get("event_type") != "error":
            continue
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        errors.append(str(metadata.get("message") or event.get("content") or "未知错误"))
    if errors:
        lines.extend(["## 错误", ""])
        lines.extend(f"- {_preview(message)}" for message in errors)
        lines.append("")

    lines.extend(["## 产物", ""])
    if artifacts:
        for artifact in artifacts:
            name = str(artifact.get("name") or artifact.get("artifact_id") or "未命名产物")
            artifact_type = str(artifact.get("artifact_type") or "unknown")
            size = artifact.get("size") if isinstance(artifact.get("size"), int) else 0
            lines.append(f"- `{name}`（{artifact_type}，{size} bytes）")
    else:
        lines.append("本次回合没有登记产物。")
    lines.extend(["", "## 数据说明", "", "详细事件、消息和诊断信息请查看同目录下的 JSON/JSONL 文件。", ""])
    return "\n".join(lines)


def _preview(value: str, limit: int = 300) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[:limit] + "…"
