from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.models.read_models import PaperReadResult
from src.models.sessions import utc_now
from src.repositories.sessions.base import SessionRepository


JsonObject = dict[str, Any]


@dataclass(slots=True)
class ReadPersistenceResult:
    """保存阅读产物写盘后的轻量引用，供状态和前端使用。"""

    artifacts: list[JsonObject] = field(default_factory=list)


class ReadPersistenceSink:
    """把每篇论文的阅读结论和最终汇总写入当前会话的产物目录。"""

    def __init__(self, repo: SessionRepository, *, session_key: str, turn_id: str):
        """初始化会话写盘所需的仓储对象和本次回合编号。"""

        self._repo = repo
        self._session_key = session_key
        self._turn_id = turn_id

    def persist_paper(self, result: PaperReadResult) -> ReadPersistenceResult:
        """在一篇论文处理结束后立刻写入笔记，防止后续失败丢失已完成结果。"""

        paper_id = _safe_path_component(result.paper.id)
        record = self._repo.write_artifact(
            self._session_key,
            artifact_type="paper_read_note",
            name=f"{paper_id}_note.json",
            content=json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            relative_path=f"artifacts/read/{self._turn_id}/papers/{paper_id}/note.json",
            metadata={"turn_id": self._turn_id, "paper_id": result.paper.id, "format": "json"},
        )
        return ReadPersistenceResult(artifacts=[_artifact_ref(record)])

    def persist_summary(self, summary: JsonObject, results: list[PaperReadResult]) -> ReadPersistenceResult:
        """写入一次阅读任务的汇总清单，便于后续节点快速了解整体处理情况。"""

        payload = {
            "turn_id": self._turn_id,
            "created_at": utc_now(),
            "summary": dict(summary),
            "papers": [result.to_dict() for result in results],
        }
        record = self._repo.write_artifact(
            self._session_key,
            artifact_type="paper_read_manifest",
            name="read_manifest.json",
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            relative_path=f"artifacts/read/{self._turn_id}/read_manifest.json",
            metadata={"turn_id": self._turn_id, "format": "json"},
        )
        return ReadPersistenceResult(artifacts=[_artifact_ref(record)])

    def persist_checkpoint(self, checkpoint: JsonObject) -> ReadPersistenceResult:
        """保存阅读节点的可恢复现场，供模型或 embedding 修好后继续执行。"""

        payload = dict(checkpoint)
        payload.setdefault("turn_id", self._turn_id)
        payload.setdefault("created_at", utc_now())
        record = self._repo.write_artifact(
            self._session_key,
            artifact_type="paper_read_checkpoint",
            name="read_checkpoint.json",
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            relative_path=f"artifacts/read/{self._turn_id}/read_checkpoint.json",
            metadata={
                "turn_id": self._turn_id,
                "format": "json",
                "recovery_status": str(payload.get("recovery_status") or "waiting_model"),
                "next_position": payload.get("next_position", 0),
            },
        )
        return ReadPersistenceResult(artifacts=[_artifact_ref(record)])


def _safe_path_component(value: str) -> str:
    """把论文编号转换成适合放进文件路径的短名称。"""

    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
    return cleaned[:100] or "paper"


def _artifact_ref(record: JsonObject) -> JsonObject:
    """把仓储返回的记录收敛为状态中需要的产物引用字段。"""

    return {
        "artifact_id": str(record["id"]),
        "artifact_type": str(record["artifact_type"]),
        "name": str(record["name"]),
        "path": str(record["path"]),
        "size": int(record["size"]),
        "created_at": str(record["created_at"]),
        "metadata": dict(record.get("metadata") or {}),
    }
