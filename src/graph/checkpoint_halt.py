from __future__ import annotations

from collections.abc import Callable
from typing import Any, NoReturn

from src.graph.state_models import JsonObject, State


PersistCheckpoint = Callable[[JsonObject], Any]


def halt_with_checkpoint(
    state: State,
    *,
    checkpoint: JsonObject,
    error: BaseException,
    persist_checkpoint: PersistCheckpoint | None,
    reporter: Any,
    results_payload: list[JsonObject],
    artifact_refs: list[JsonObject],
    checkpoint_key: str,
    results_key: str,
    artifact_refs_key: str,
    diagnostics_key: str,
    current_step: str,
    failure_stage: str,
    recovery_status: str,
    total: int,
    completed: int,
    next_position: int,
    artifact_stage: str = "read_checkpoint_saved",
) -> NoReturn:
    """保存可恢复现场、通知前端，然后抛出原始错误中断当前节点。"""

    checkpoint_artifacts: list[JsonObject] = []
    if persist_checkpoint is not None:
        try:
            # 中文注释：优先把现场保存成会话产物文件。这样前端或调用方不仅能从
            # 失败事件里拿到 checkpoint，也能从产物列表里找到同一份恢复文件。
            persisted = persist_checkpoint(checkpoint)
            checkpoint_artifacts.extend(list(getattr(persisted, "artifacts", []) or []))
            checkpoint["checkpoint_artifact_refs"] = checkpoint_artifacts
            checkpoint[artifact_refs_key] = artifact_refs + checkpoint_artifacts
            if reporter is not None:
                for artifact in checkpoint_artifacts:
                    # 中文注释：这里复用节点已有的产物上报能力，让界面能立刻显示
                    # “恢复现场文件已保存”，不需要等整个 run 结束后再刷新。
                    reporter.artifact(artifact, stage=artifact_stage)
        except Exception as exc:
            # 中文注释：checkpoint 写盘失败时不能吞掉整个恢复信息。这里把错误也
            # 写进 checkpoint，后面仍会放进事件 metadata 和 state，保证至少能恢复。
            checkpoint["checkpoint_persistence_error"] = str(exc)

    if reporter is not None:
        # 中文注释：failed 状态的 runtime_event 是前端实时感知“需要用户处理后继续”的主要信号。
        # checkpoint 会放进事件详情里，前端以后做“继续执行”按钮时不用额外读文件。
        reporter.failed(
            str(error),
            stage=failure_stage,
            total=total,
            completed=completed,
            next_position=next_position,
            recovery_status=recovery_status,
            checkpoint=checkpoint,
            checkpoint_artifact_refs=checkpoint_artifacts,
        )

    # 中文注释：虽然下面马上会抛出异常，仍然把现场写回 state。这样单元测试、
    # 图调试器或外层异常处理逻辑都能拿到最新的恢复信息。
    diagnostics = dict(state.get("diagnostics") or {})
    diagnostics[diagnostics_key] = checkpoint
    state["diagnostics"] = diagnostics
    state[checkpoint_key] = checkpoint
    state[results_key] = results_payload
    state[artifact_refs_key] = artifact_refs + checkpoint_artifacts
    state["current_step"] = current_step
    raise error
