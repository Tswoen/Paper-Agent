from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from src.agents.searchAgent import SearchIntent
from src.models.sessions import utc_now
from src.paper_retrieval.models import PaperDocument
from src.repositories.sessions.base import SessionRepository


JsonObject = dict[str, Any]


@dataclass(slots=True)
class SearchArtifactRef:
    """描述一次检索落盘后生成的单个产物引用。"""

    artifact_id: str
    artifact_type: str
    name: str
    path: str
    size: int
    created_at: str
    metadata: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        """把产物引用转成普通字典，方便写回图状态。"""

        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "name": self.name,
            "path": self.path,
            "size": self.size,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class SearchPersistenceResult:
    """承载一次检索写盘后需要给上层继续使用的结果。"""

    artifacts: list[SearchArtifactRef] = field(default_factory=list)
    manifest: JsonObject = field(default_factory=dict)

    def to_state_refs(self) -> list[JsonObject]:
        """把产物引用转换成适合放进状态里的轻量结构。"""

        return [artifact.to_dict() for artifact in self.artifacts]


class SearchPersistenceSink:
    """负责把检索阶段产生的结构化数据写入会话产物目录。"""

    def __init__(self, repo: SessionRepository, session_key: str, turn_id: str):
        """初始化检索写盘组件。"""

        self.repo = repo
        self.session_key = session_key
        self.turn_id = turn_id

    def persist(
        self,
        *,
        topic: str,
        intent: SearchIntent,
        raw_papers: list[PaperDocument],
        scored_papers: list[JsonObject],
        selected_papers: list[PaperDocument],
        search_summary: JsonObject,
        search_output: JsonObject,
        agent_diagnostics: JsonObject,
        search_halted: bool,
    ) -> SearchPersistenceResult:
        """把一次检索产物写成多个 JSON 文件，并返回它们的引用。"""

        now = utc_now()
        base_dir = f"artifacts/search/{self.turn_id}"
        artifacts: list[SearchArtifactRef] = []

        intent_artifact = self._write_json_artifact(
            artifact_type="paper_search_intent",
            name="search_intent.json",
            relative_path=f"{base_dir}/search_intent.json",
            payload={
                "turn_id": self.turn_id,
                "topic": topic,
                "intent": _intent_to_dict(intent),
                "agent_diagnostics": dict(agent_diagnostics),
                "search_halted": search_halted,
                "created_at": now,
            },
        )
        artifacts.append(intent_artifact)

        raw_artifact = self._write_json_artifact(
            artifact_type="paper_search_raw_results",
            name="search_raw_results.json",
            relative_path=f"{base_dir}/search_raw_results.json",
            payload={
                "turn_id": self.turn_id,
                "topic": topic,
                "count": len(raw_papers),
                "papers": [_paper_to_dict(paper) for paper in raw_papers],
                "created_at": now,
            },
        )
        artifacts.append(raw_artifact)

        ranked_artifact = self._write_json_artifact(
            artifact_type="paper_search_ranked_results",
            name="search_ranked_results.json",
            relative_path=f"{base_dir}/search_ranked_results.json",
            payload={
                "turn_id": self.turn_id,
                "topic": topic,
                "scored_papers": list(scored_papers),
                "selected_papers": [_paper_to_dict(paper) for paper in selected_papers],
                "created_at": now,
            },
        )
        artifacts.append(ranked_artifact)

        output_artifact = self._write_json_artifact(
            artifact_type="paper_search_output",
            name="search_output.json",
            relative_path=f"{base_dir}/search_output.json",
            payload={
                "turn_id": self.turn_id,
                "topic": topic,
                "summary": dict(search_summary),
                "output": dict(search_output),
                "created_at": now,
            },
        )
        artifacts.append(output_artifact)

        manifest = {
            "turn_id": self.turn_id,
            "topic": topic,
            "search_halted": search_halted,
            "raw_paper_count": len(raw_papers),
            "selected_paper_count": len(selected_papers),
            "summary": dict(search_summary),
            "artifacts": [artifact.to_dict() for artifact in artifacts],
            "created_at": now,
        }
        manifest_artifact = self._write_json_artifact(
            artifact_type="paper_search_manifest",
            name="search_manifest.json",
            relative_path=f"{base_dir}/search_manifest.json",
            payload=manifest,
        )
        artifacts.append(manifest_artifact)
        manifest["artifacts"] = [artifact.to_dict() for artifact in artifacts]
        return SearchPersistenceResult(artifacts=artifacts, manifest=manifest)

    def _write_json_artifact(
        self,
        *,
        artifact_type: str,
        name: str,
        relative_path: str,
        payload: JsonObject,
    ) -> SearchArtifactRef:
        """把一个 JSON 产物写盘，并返回统一的产物引用。"""

        record = self.repo.write_artifact(
            self.session_key,
            artifact_type=artifact_type,
            name=name,
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            relative_path=relative_path,
            metadata={"turn_id": self.turn_id, "format": "json"},
        )
        return SearchArtifactRef(
            artifact_id=str(record["id"]),
            artifact_type=str(record["artifact_type"]),
            name=str(record["name"]),
            path=str(record["path"]),
            size=int(record["size"]),
            created_at=str(record["created_at"]),
            metadata=dict(record.get("metadata") or {}),
        )


def _intent_to_dict(intent: SearchIntent) -> JsonObject:
    """把检索意图转成普通字典，方便写到 JSON 产物里。"""

    return asdict(intent)


def _paper_to_dict(paper: PaperDocument) -> JsonObject:
    """把论文对象转成普通字典，方便写盘和调试。"""

    return paper.to_dict()

