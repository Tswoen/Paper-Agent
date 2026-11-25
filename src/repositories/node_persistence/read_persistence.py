from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from src.models.artifacts import to_artifact_ref
from src.models.read_models import PaperReadResult
from src.models.sessions import utc_now
from src.repositories.sessions.base import SessionRepository


JsonObject = dict[str, Any]
_READ_NOTE_STAGES = frozenset({"abstract_complete", "fulltext_complete", "discarded"})


@dataclass(slots=True)
class ReadPersistenceResult:
    """保存阅读产物写盘后的轻量引用，供状态和前端使用。"""

    artifacts: list[JsonObject] = field(default_factory=list)


@dataclass(slots=True)
class LoadedPaperNote:
    """通过归属与请求指纹校验的单篇笔记及其稳定产物引用。"""

    stage: str | None
    result: JsonObject
    artifact: JsonObject
    last_error: JsonObject | None = None


def read_request_fingerprint(request: JsonObject) -> str:
    """对影响阅读结果的请求做规范化摘要，字典键顺序不影响结果。"""

    constraints = request.get("constraints")
    payload = {
        "topic": str(request.get("topic") or ""),
        "constraints": constraints if isinstance(constraints, dict) else {},
        "language": str(request.get("language") or "zh"),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class ReadPersistenceSink:
    """把每篇论文的阅读结论和最终汇总写入当前会话的产物目录。"""

    def __init__(
        self,
        repo: SessionRepository,
        *,
        session_key: str,
        turn_id: str,
        thread_id: str,
        request_fingerprint: str,
    ):
        """初始化写盘上下文；thread 路径在续跑换 turn 后仍保持稳定。

        中文注释：thread_id 与 request_fingerprint 必填——它们决定 note 的归属
        与可恢复性，漏传会立刻报错，而不是静默写出无法恢复的笔记。
        """

        self._repo = repo
        self._session_key = session_key
        self._turn_id = turn_id
        self._thread_id = thread_id
        self._request_fingerprint = request_fingerprint

    def persist_paper(self, result: PaperReadResult, *, stage: str | None = None) -> ReadPersistenceResult:
        """在一篇论文完成一个稳定阶段后原子写入笔记，重复写只更新原记录。"""

        paper_id = _safe_path_component(result.paper.id)
        resolved_stage = stage if stage is not None else _note_stage(result)
        if resolved_stage is not None and resolved_stage not in _READ_NOTE_STAGES:
            raise ValueError(f"unsupported read note stage: {resolved_stage}")
        failure = result.failure if isinstance(result.failure, dict) else None
        payload = {
            "paper_id": result.paper.id,
            "thread_id": self._thread_id,
            "stage": resolved_stage,
            "request_fingerprint": self._request_fingerprint,
            "result": result.to_dict(),
            "last_error": failure,
            "updated_at": utc_now(),
        }
        record = self._repo.write_artifact(
            self._session_key,
            artifact_type="paper_read_note",
            name=f"{paper_id}_note.json",
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            relative_path=self._paper_note_path(result.paper.id),
            metadata={
                "turn_id": self._turn_id,
                "thread_id": self._thread_id,
                "paper_id": result.paper.id,
                "note_stage": resolved_stage or "",
                "request_fingerprint": self._request_fingerprint,
                "format": "json",
            },
            replace_existing=True,
        )
        return ReadPersistenceResult(artifacts=[_artifact_ref(record)])

    def load_paper_note(self, paper_id: str) -> LoadedPaperNote | None:
        """读取一篇有效笔记；损坏、归属或请求不匹配时都按未恢复处理。"""

        relative_path = self._paper_note_path(paper_id)
        record = self._find_artifact(relative_path)
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
        if str(payload.get("paper_id") or "") != paper_id:
            return None
        if str(payload.get("thread_id") or "") != self._thread_id:
            return None
        if str(payload.get("request_fingerprint") or "") != self._request_fingerprint:
            return None
        result = payload.get("result")
        if not isinstance(result, dict):
            return None
        raw_stage = payload.get("stage")
        stage = str(raw_stage) if raw_stage else None
        if stage is not None and stage not in _READ_NOTE_STAGES:
            return None
        last_error = payload.get("last_error")
        return LoadedPaperNote(
            stage=stage,
            result=result,
            artifact=_artifact_ref(record),
            last_error=last_error if isinstance(last_error, dict) else None,
        )

    def persist_summary(self, summary: JsonObject, results: list[PaperReadResult]) -> ReadPersistenceResult:
        """写入一次阅读任务的汇总清单，便于后续节点快速了解整体处理情况。"""

        payload = {
            "turn_id": self._turn_id,
            "thread_id": self._thread_id,
            "created_at": utc_now(),
            "summary": dict(summary),
            "papers": [result.to_dict() for result in results],
        }
        record = self._repo.write_artifact(
            self._session_key,
            artifact_type="paper_read_manifest",
            name="read_manifest.json",
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            relative_path=f"artifacts/read/threads/{_safe_path_component(self._thread_id)}/read_manifest.json",
            metadata={
                "turn_id": self._turn_id,
                "thread_id": self._thread_id,
                "request_fingerprint": self._request_fingerprint,
                "format": "json",
            },
            replace_existing=True,
        )
        return ReadPersistenceResult(artifacts=[_artifact_ref(record)])

    def _paper_note_path(self, paper_id: str) -> str:
        return (
            f"artifacts/read/threads/{_safe_path_component(self._thread_id)}"
            f"/papers/{_safe_path_component(paper_id)}/note.json"
        )

    def _find_artifact(self, relative_path: str) -> JsonObject | None:
        """兼容仓储里的相对/绝对位置，再通过现有安全接口按 ID 读取。"""

        normalized = relative_path.replace("\\", "/").lstrip("/")
        artifacts = self._repo.get(self._session_key).artifacts
        for record in reversed(artifacts):
            candidate = str(record.get("path") or "").replace("\\", "/")
            if candidate == normalized or candidate.endswith(f"/{normalized}"):
                return record
        return None


def _note_stage(result: PaperReadResult) -> str | None:
    """失败不伪造完成阶段；部分全文失败仍保留已有摘要进度。"""

    if result.failure is not None:
        return None
    if result.relevance.discarded:
        return "discarded"
    if result.note.evidence_level == "full_text":
        return "fulltext_complete"
    return "abstract_complete"


def _safe_path_component(value: str) -> str:
    """把论文编号转换成适合放进文件路径的短名称。"""

    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
    return cleaned[:100] or "paper"


def _artifact_ref(record: JsonObject) -> JsonObject:
    """把仓储返回的记录收敛为状态中需要的产物引用字段（不携带本地路径）。"""

    return to_artifact_ref(record)
