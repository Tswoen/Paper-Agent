from __future__ import annotations

from typing import Any, Mapping


JsonObject = dict[str, Any]


def to_artifact_ref(record: Mapping[str, Any]) -> JsonObject:
    """把仓储的产物记录收敛成 State / 事件用的引用。

    中文说明：引用只保留"是哪个产物"的身份字段，不携带"它在哪台机器的哪个目录"。
    仓储记录里的 `path` 是本地位置信息，进入快照后会随机器、工作目录一起失效，
    所以这里有意不复制；需要真实文件时，调用方用
    `SessionRepository.read_artifact_path(session_key, artifact_id)` 按 id 解析。
    """

    return {
        "artifact_id": str(record.get("id") or ""),
        "artifact_type": str(record.get("artifact_type") or ""),
        "name": str(record.get("name") or ""),
        "size": int(record.get("size") or 0),
        "created_at": str(record.get("created_at") or ""),
        "metadata": dict(record.get("metadata") or {}),
    }