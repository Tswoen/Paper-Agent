from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.models.artifacts import to_artifact_ref
from src.models.sessions import utc_now
from src.repositories.node_persistence.read_persistence import _safe_path_component
from src.repositories.sessions.base import SessionRepository


JsonObject = dict[str, Any]


@dataclass(slots=True)
class WritingPersistenceResult:
    """保存写作产物写盘后的轻量引用，供状态和前端使用。"""

    artifacts: list[JsonObject]


@dataclass(slots=True)
class LoadedWritingSection:
    """通过归属与请求指纹校验的单节稳定点及其产物引用。"""

    section_id: str
    section: JsonObject
    artifact: JsonObject


class WritingPersistenceSink:
    """把每节写作结论和最终报告写入当前会话的产物目录。

    中文注释：与 read 的 note 稳定点模式同构——小节级幂等由 section.json 负责，
    节点级恢复由官方 Checkpointer 负责。崩溃续跑时已完成小节直接复用，
    只重写缺失小节。
    """

    def __init__(
        self,
        repo: SessionRepository,
        *,
        session_key: str,
        turn_id: str,
        thread_id: str,
        request_fingerprint: str,
    ):
        """初始化写盘上下文；thread 路径在续跑换 turn 后仍保持稳定。"""

        self._repo = repo
        self._session_key = session_key
        self._turn_id = turn_id
        self._thread_id = thread_id
        self._request_fingerprint = request_fingerprint

    def persist_section(self, section: JsonObject) -> WritingPersistenceResult:
        """一节正文完成后立即原子写入稳定点，重复写只更新原记录。"""

        section_id = str(section.get("section_id") or "")
        payload = {
            "section_id": section_id,
            "thread_id": self._thread_id,
            "request_fingerprint": self._request_fingerprint,
            "section": dict(section),
            "updated_at": utc_now(),
        }
        record = self._repo.write_artifact(
            self._session_key,
            artifact_type="writing_section",
            name=f"{_safe_path_component(section_id)}_section.json",
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            relative_path=self._section_path(section_id),
            metadata={
                "turn_id": self._turn_id,
                "thread_id": self._thread_id,
                "section_id": section_id,
                "request_fingerprint": self._request_fingerprint,
                "format": "json",
            },
            replace_existing=True,
        )
        return WritingPersistenceResult(artifacts=[to_artifact_ref(record)])

    def load_section(self, section_id: str) -> LoadedWritingSection | None:
        """读取一节有效稳定点；损坏、归属或请求不匹配、正文为空都按未恢复处理。"""

        record = self._find_artifact(self._section_path(section_id))
        if record is None:
            return None
        artifact_id = str(record.get("id") or "")
        if not artifact_id:
            return None
        path = self._repo.read_artifact_path(self._session_key, artifact_id)
        if path is None:
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        if str(payload.get("section_id") or "") != section_id:
            return None
        if str(payload.get("thread_id") or "") != self._thread_id:
            return None
        if str(payload.get("request_fingerprint") or "") != self._request_fingerprint:
            return None
        section = payload.get("section")
        if not isinstance(section, dict):
            return None
        if str(section.get("section_id") or "") != section_id:
            return None
        if not str(section.get("content") or "").strip():
            return None
        return LoadedWritingSection(
            section_id=section_id,
            section=section,
            artifact=to_artifact_ref(record),
        )

    def persist_report(self, report: JsonObject) -> WritingPersistenceResult:
        """写入最终写作报告；按 thread 稳定路径覆盖更新，续跑不产生重复成品。"""

        record = self._repo.write_artifact(
            self._session_key,
            artifact_type="writing",
            name="writing.json",
            content=json.dumps(report, ensure_ascii=False, indent=2),
            relative_path=self._report_path(),
            metadata={
                "turn_id": self._turn_id,
                "thread_id": self._thread_id,
                "request_fingerprint": self._request_fingerprint,
                "format": "json",
            },
            replace_existing=True,
        )
        return WritingPersistenceResult(artifacts=[to_artifact_ref(record)])

    def _section_path(self, section_id: str) -> str:
        return (
            f"artifacts/writing/threads/{_safe_path_component(self._thread_id)}"
            f"/sections/{_safe_path_component(section_id)}/section.json"
        )

    def _report_path(self) -> str:
        return f"artifacts/writing/threads/{_safe_path_component(self._thread_id)}/writing.json"

    def _find_artifact(self, relative_path: str) -> JsonObject | None:
        """按相对路径定位产物记录，再通过现有安全接口按 ID 读取。"""

        normalized = relative_path.replace("\\", "/").lstrip("/")
        artifacts = self._repo.get(self._session_key).artifacts
        for record in reversed(artifacts):
            candidate = str(record.get("path") or "").replace("\\", "/")
            if candidate == normalized or candidate.endswith(f"/{normalized}"):
                return record
        return None
