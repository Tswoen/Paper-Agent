from __future__ import annotations

from typing import Any

from src.utils import get_logger


JsonObject = dict[str, Any]
logger = get_logger(__name__)

# 中文注释：官方 channel_values 里混着框架内部通道（__start__、branch:to:* 等），
# 它们不是业务状态，导出时一律过滤，只保留 State 的业务字段。
_INTERNAL_PREFIXES = ("__", "branch:")


def _state_channels(checkpoint: JsonObject) -> JsonObject:
    values = checkpoint.get("channel_values")
    if not isinstance(values, dict):
        return {}
    return {
        key: value
        for key, value in values.items()
        if not str(key).startswith(_INTERNAL_PREFIXES)
    }


def build_state_history(saver: Any, thread_id: str | None) -> list[JsonObject]:
    """把一个线程的 checkpoint 序列整理成"状态快照历史"账本行。

    中文注释：
    - 官方 saver.list 返回新→旧，这里反转为时间正序；
    - 每行只记"与上一张快照的差分"（变化的键记完整新值，全量保真不截断），
      第一张业务快照相对空状态即全量；
    - 纯读取：不写不删，不碰恢复机制；saver 缺失/线程已清理/读取失败时
      优雅返回空列表，绝不影响导出主流程。
    """

    if saver is None or not thread_id:
        return []
    config = {"configurable": {"thread_id": thread_id}}
    try:
        tuples = list(saver.list(config))
    except Exception:
        logger.warning("读取 checkpoint 历史失败（state_history 置空）", exc_info=True)
        return []
    tuples.reverse()

    rows: list[JsonObject] = []
    previous: JsonObject = {}
    for seq, item in enumerate(tuples, start=1):
        checkpoint = item.checkpoint if isinstance(item.checkpoint, dict) else {}
        current = _state_channels(checkpoint)
        changed = {
            key: value
            for key, value in current.items()
            if key not in previous or previous[key] != value
        }
        removed_keys = sorted(set(previous) - set(current))
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        parent_config = item.parent_config if isinstance(item.parent_config, dict) else {}
        rows.append(
            {
                "seq": seq,
                "checkpoint_id": checkpoint.get("id"),
                "parent_checkpoint_id": (parent_config.get("configurable") or {}).get("checkpoint_id"),
                "created_at": checkpoint.get("ts"),
                "source": metadata.get("source"),
                "step": metadata.get("step"),
                "changed": changed,
                "removed_keys": removed_keys,
            }
        )
        previous = current
    return rows
