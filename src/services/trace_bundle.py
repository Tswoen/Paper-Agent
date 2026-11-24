from __future__ import annotations

import io
import json
import zipfile
from typing import Any
from urllib.parse import quote

from src.models.artifacts import to_artifact_ref
from src.models.sessions import utc_now
from src.repositories.sessions.base import SessionRepository
from src.services.sessions import SessionError
from src.services.trace_state_history import build_state_history


JsonObject = dict[str, Any]
_BUNDLE_FILES = (
    "manifest.json",
    "events.jsonl",
    "messages.jsonl",
    "artifacts.json",
    "diagnostics.json",
    "llm_calls.jsonl",
    "tool_calls.jsonl",
    "state_history.jsonl",
    "report.md",
)
# 中文说明：V2 起全量导出、不再脱敏（用户拍板：防御不完备，反而可能洗掉排错信息）。
# 出口只留醒目提示，敏感信息的处置责任交给分享者；漏 key 的 remedy 是更换 key。
_EXPORT_WARNING = (
    "> **安全提示**：本包为全量导出，未经脱敏，可能包含 API key 与本机路径等敏感信息。\n"
    "> 分享前请自行检查内容；若已外发，建议更换相关 API key。\n"
)


class TraceBundleService:
    """把某次回合已经持久化的数据整理成可查看、可下载的轨迹包。"""

    def __init__(self, repo: SessionRepository, checkpoint_bundle: Any = None):
        self._repo = repo
        # 中文注释：checkpoint bundle 可选——没有它（旧测试/裸脚本）时
        # state_history 优雅为空，其余导出内容不受影响。
        self._checkpoint_bundle = checkpoint_bundle

    def _state_history(self, thread_id: str | None) -> list[JsonObject]:
        saver = getattr(self._checkpoint_bundle, "saver", None)
        return build_state_history(saver, thread_id)

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

    def turn_detail(self, session_key: str, turn_id: str) -> JsonObject:
        """读取单个回合的事件/消息/产物正文，供轨迹抽屉按需取数。

        中文注释：与 webui-thread 的记录形状保持一致（产物带 id），
        前端抽屉无需为按回合取数做结构适配。
        """

        summaries = self._repo.list_trace_turns(session_key)
        summary = next((item for item in summaries if item.get("turn_id") == turn_id), None)
        if summary is None:
            raise SessionError(f"trace not found: {turn_id}", 404)
        return {
            "turn_id": turn_id,
            "events": self._repo.list_events(session_key, turn_id),
            "messages": self._repo.list_messages(session_key, turn_id),
            "artifacts": self._repo.list_artifacts(session_key, turn_id),
            "state_history": self._state_history(summary.get("thread_id")),
        }

    def build_zip(self, session_key: str, turn_id: str) -> bytes:
        """在内存中生成一个 turn 的 zip，不写入会话目录或长期临时文件。

        中文注释：全量保真导出——事件、消息、产物原文打包，不脱敏、不截断。
        llm_calls/tool_calls 是从事件整理出的账本视图，原始事件仍全量保留在
        events.jsonl 里。
        """

        summaries = self._repo.list_trace_turns(session_key)
        summary = next((item for item in summaries if item.get("turn_id") == turn_id), None)
        if summary is None:
            raise SessionError(f"trace not found: {turn_id}", 404)

        events = self._repo.list_events(session_key, turn_id)
        messages = self._repo.list_messages(session_key, turn_id)
        artifacts = self._repo.list_artifacts(session_key, turn_id)
        # 运行中下载时列表和正文可能读到不同时间点；manifest 以实际打包内容为基准重算。
        snapshot = _snapshot_summary(summary, events, messages, artifacts)
        artifact_refs = [to_artifact_ref(artifact) for artifact in artifacts]
        diagnostics = _extract_diagnostics(events)
        llm_calls = _llm_call_rows(events)
        tool_calls = _tool_call_rows(events)
        state_history = self._state_history(snapshot.get("thread_id"))

        manifest = {
            "session_key": session_key,
            "turn_id": turn_id,
            "run_id": snapshot.get("run_id"),
            "thread_id": snapshot.get("thread_id"),
            "status": snapshot.get("status") or "unknown",
            "started_at": snapshot.get("started_at"),
            "finished_at": snapshot.get("finished_at"),
            "generated_at": utc_now(),
            "event_count": len(events),
            "message_count": len(messages),
            "artifact_count": len(artifact_refs),
            "llm_call_count": len(llm_calls),
            "tool_call_count": len(tool_calls),
            "state_history_count": len(state_history),
            "files": list(_BUNDLE_FILES),
        }
        report = _build_report(manifest, messages, events, artifact_refs, llm_calls, tool_calls, state_history)
        files = {
            "manifest.json": _json_bytes(manifest),
            "events.jsonl": _jsonl_bytes(events),
            "messages.jsonl": _jsonl_bytes(messages),
            "artifacts.json": _json_bytes(artifact_refs),
            "diagnostics.json": _json_bytes(diagnostics),
            "llm_calls.jsonl": _jsonl_bytes(llm_calls),
            "tool_calls.jsonl": _jsonl_bytes(tool_calls),
            "state_history.jsonl": _jsonl_bytes(state_history),
            "report.md": report.encode("utf-8"),
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in _BUNDLE_FILES:
                archive.writestr(name, files[name])
        return buffer.getvalue()


def _llm_call_rows(events: list[JsonObject]) -> list[JsonObject]:
    """把 llm_call 事件整理成账本行：信封原文 + 持久化序号/时间。"""

    rows: list[JsonObject] = []
    for event in events:
        if str(event.get("event_type") or "") != "llm_call":
            continue
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        rows.append(
            {
                "seq_no": event.get("seq_no"),
                "created_at": event.get("created_at"),
                **metadata,
            }
        )
    return rows


def _tool_call_rows(events: list[JsonObject]) -> list[JsonObject]:
    """把 tool_call/tool_result 按 call_id 合并成一行，方便直接回看。

    中文注释：只有开工单没有完工单（如运行中崩溃）时保留 running 状态；
    孤儿完工单也原样导出，全量保真不丢信息。
    """

    calls: dict[str, JsonObject] = {}
    results: dict[str, JsonObject] = {}
    order: list[str] = []
    for event in events:
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        event_type = str(event.get("event_type") or "")
        if event_type == "tool_call":
            call_id = str(metadata.get("id") or "")
            if not call_id:
                continue
            payload = metadata.get("payload") if isinstance(metadata.get("payload"), dict) else {}
            calls[call_id] = {
                "call_id": call_id,
                "name": payload.get("name"),
                "node_key": metadata.get("node_key"),
                "stage": metadata.get("stage"),
                "status": "running",
                "started_at": metadata.get("started_at"),
                "created_at": event.get("created_at"),
                "seq_no": event.get("seq_no"),
                "args": payload.get("args") if payload.get("args") is not None else {},
                "result": None,
                "duration_ms": None,
                "error": None,
            }
            order.append(call_id)
        elif event_type == "tool_result":
            parent_id = str(metadata.get("parent_id") or "")
            if parent_id:
                results[parent_id] = metadata

    rows: list[JsonObject] = []
    for call_id in order:
        row = calls[call_id]
        result = results.pop(call_id, None)
        if result is not None:
            payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
            row["status"] = result.get("status") or row["status"]
            row["duration_ms"] = result.get("duration_ms")
            row["error"] = result.get("error")
            row["result"] = payload.get("result")
        rows.append(row)
    for call_id, result in results.items():
        payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        rows.append(
            {
                "call_id": call_id,
                "name": payload.get("name"),
                "node_key": result.get("node_key"),
                "stage": result.get("stage"),
                "status": result.get("status"),
                "started_at": result.get("started_at"),
                "created_at": None,
                "seq_no": None,
                "args": None,
                "result": payload.get("result"),
                "duration_ms": result.get("duration_ms"),
                "error": result.get("error"),
            }
        )
    return rows


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
    llm_calls: list[JsonObject],
    tool_calls: list[JsonObject],
    state_history: list[JsonObject],
) -> str:
    """生成不包含大段模型输出的简短人工可读摘要。"""

    lines = [
        "# Paper Agent 运行轨迹",
        "",
        _EXPORT_WARNING,
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

    llm_failed = sum(1 for row in llm_calls if row.get("status") == "failed")
    input_tokens = sum(int((row.get("tokens") or {}).get("input_tokens") or 0) for row in llm_calls)
    output_tokens = sum(int((row.get("tokens") or {}).get("output_tokens") or 0) for row in llm_calls)
    tool_failed = sum(1 for row in tool_calls if row.get("status") == "failed")
    lines.extend(
        [
            "## 模型调用",
            "",
            f"- 共 {len(llm_calls)} 次，失败 {llm_failed} 次；token 输入 / 输出：{input_tokens} / {output_tokens}",
            "- 逐条明细（含完整 prompt/response）见 `llm_calls.jsonl`。",
            "",
            "## 工具调用",
            "",
            f"- 共 {len(tool_calls)} 次，失败 {tool_failed} 次",
            "- 逐条明细（参数/结果/耗时）见 `tool_calls.jsonl`。",
            "",
            "## 状态快照",
            "",
            f"- 共 {len(state_history)} 张快照（按 thread 导出；续跑时一个 thread 可能跨多个回合）",
            "- 每张快照记录与上一张的字段差分（完整新值），见 `state_history.jsonl`。",
            "",
        ]
    )

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
    lines.extend(
        [
            "",
            "## 数据说明",
            "",
            "详细事件、消息和诊断信息请查看同目录下的 JSON/JSONL 文件；",
            "模型与工具调用账本分别在 llm_calls.jsonl 与 tool_calls.jsonl，状态快照历史在 state_history.jsonl。",
            "",
        ]
    )
    return "\n".join(lines)


def _preview(value: str, limit: int = 300) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[:limit] + "…"
