from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from src.llm.config import CheckpointSettings
from src.models.sessions import utc_now


JsonObject = dict[str, Any]

# 中文注释：只有这些终态才允许“继续执行”。running 不算——进程还活着时，
# 同一会话会被 start_run 的并发保护挡住；进程重启后由 recover_interrupted_runs
# 把它改写成 interrupted，才进入可续跑集合。
RESUMABLE_STATUSES = ("failed", "cancelled", "interrupted")
_FINISHED_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})

# 中文注释：checkpoint 由官方 saver 自己建表；这里只建“哪个 thread 属于哪个
# 会话、版本是多少、现在什么状态”这份小账本，用于续跑选择、版本校验和保留期清理。
_REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoint_runs (
    thread_id TEXT PRIMARY KEY,
    session_key TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    run_id TEXT,
    graph_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checkpoint_runs_session
    ON checkpoint_runs (session_key, updated_at);
CREATE INDEX IF NOT EXISTS idx_checkpoint_runs_status
    ON checkpoint_runs (status, finished_at);
"""


class CheckpointRegistry(ABC):
    """记录 run 与 checkpoint thread 对应关系的最小账本。

    中文注释：LangGraph 的 checkpointer 只回答“某个 thread 的最后快照是什么”，
    它不知道“哪个 thread 属于哪个会话、是否已经结束、版本是多少”。这些信息只
    用于启动续跑、版本校验和保留期清理，所以单独成表，不混进 State。

    TODO(4-postgres): 上 Postgres 时新增 PostgresCheckpointRegistry 实现，
    SessionRunService 只依赖本接口，不再关心底层存储。
    """

    @abstractmethod
    def mark_started(
        self,
        *,
        thread_id: str,
        session_key: str,
        turn_id: str,
        run_id: str | None,
        graph_version: int,
        started_at: str | None = None,
    ) -> None:
        """登记一次 run 开始；续跑时复用同一 thread_id 并保留首次记录的版本号。"""

    @abstractmethod
    def mark_finished(self, *, thread_id: str, status: str, finished_at: str | None = None) -> None:
        """登记一次 run 的终态；status 必须是 completed/failed/cancelled/interrupted。"""

    @abstractmethod
    def get(self, thread_id: str) -> JsonObject | None:
        """按 thread_id 读取一条登记记录。"""

    @abstractmethod
    def latest_resumable(self, session_key: str) -> JsonObject | None:
        """返回当前会话最近一次可续跑的 run；没有时返回 None。"""

    @abstractmethod
    def list_threads(self, session_key: str | None = None) -> list[JsonObject]:
        """按更新时间倒序列出登记记录，可按会话过滤。"""

    @abstractmethod
    def delete_thread(self, thread_id: str) -> bool:
        """删除一条登记记录，返回是否真的删到了行。"""

    @abstractmethod
    def close(self) -> None:
        """释放底层连接；内存实现是空操作。"""


class SqliteCheckpointRegistry(CheckpointRegistry):
    """把 run 账本保存在独立 SQLite 文件里。

    中文注释：注册表和官方 saver 共用同一个数据库文件，但各自持有一条连接，
    避免两边的隐式事务互相提交。注册表只做短事务，不会长时间占住写锁。
    """

    def __init__(self, database_path: Path | str):
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.executescript(_REGISTRY_SCHEMA)
        self._connection.commit()
        self._closed = False

    def mark_started(
        self,
        *,
        thread_id: str,
        session_key: str,
        turn_id: str,
        run_id: str | None,
        graph_version: int,
        started_at: str | None = None,
    ) -> None:
        now = started_at or utc_now()
        with self._connection:
            # 中文注释：续跑复用同一 thread_id 时，只把状态改回 running，
            # 不覆盖首次登记的 graph_version——版本校验要以最初那份存档为准。
            self._connection.execute(
                """
                INSERT INTO checkpoint_runs (
                    thread_id, session_key, turn_id, run_id, graph_version,
                    status, started_at, finished_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'running', ?, NULL, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    status = 'running',
                    started_at = excluded.started_at,
                    finished_at = NULL,
                    updated_at = excluded.updated_at
                """,
                (thread_id, session_key, turn_id, run_id, graph_version, now, now),
            )

    def mark_finished(self, *, thread_id: str, status: str, finished_at: str | None = None) -> None:
        if status not in _FINISHED_STATUSES:
            raise ValueError(f"不支持的 checkpoint run 终态：{status!r}")
        now = finished_at or utc_now()
        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE checkpoint_runs
                SET status = ?, finished_at = ?, updated_at = ?
                WHERE thread_id = ?
                """,
                (status, now, now, thread_id),
            )
        if cursor.rowcount == 0:
            raise KeyError(f"checkpoint run 不存在：{thread_id}")

    def get(self, thread_id: str) -> JsonObject | None:
        row = self._connection.execute(
            "SELECT * FROM checkpoint_runs WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def latest_resumable(self, session_key: str) -> JsonObject | None:
        placeholders = ", ".join("?" for _ in RESUMABLE_STATUSES)
        row = self._connection.execute(
            f"""
            SELECT * FROM checkpoint_runs
            WHERE session_key = ? AND status IN ({placeholders})
            ORDER BY COALESCE(finished_at, updated_at) DESC
            LIMIT 1
            """,
            (session_key, *RESUMABLE_STATUSES),
        ).fetchone()
        return dict(row) if row is not None else None

    def list_threads(self, session_key: str | None = None) -> list[JsonObject]:
        if session_key is None:
            rows = self._connection.execute(
                "SELECT * FROM checkpoint_runs ORDER BY updated_at DESC"
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM checkpoint_runs WHERE session_key = ? ORDER BY updated_at DESC",
                (session_key,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_thread(self, thread_id: str) -> bool:
        with self._connection:
            cursor = self._connection.execute(
                "DELETE FROM checkpoint_runs WHERE thread_id = ?",
                (thread_id,),
            )
        return cursor.rowcount > 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connection.close()


class MemoryCheckpointRegistry(CheckpointRegistry):
    """内存版 run 账本，只用于测试和临时运行。"""

    def __init__(self) -> None:
        self._rows: dict[str, JsonObject] = {}

    def mark_started(
        self,
        *,
        thread_id: str,
        session_key: str,
        turn_id: str,
        run_id: str | None,
        graph_version: int,
        started_at: str | None = None,
    ) -> None:
        now = started_at or utc_now()
        previous = self._rows.get(thread_id)
        self._rows[thread_id] = {
            "thread_id": thread_id,
            "session_key": session_key,
            "turn_id": turn_id,
            "run_id": run_id,
            # 中文注释：与 SQLite 实现保持一致，续跑不覆盖首次登记的版本号。
            "graph_version": previous["graph_version"] if previous else graph_version,
            "status": "running",
            "started_at": now,
            "finished_at": None,
            "updated_at": now,
        }

    def mark_finished(self, *, thread_id: str, status: str, finished_at: str | None = None) -> None:
        if status not in _FINISHED_STATUSES:
            raise ValueError(f"不支持的 checkpoint run 终态：{status!r}")
        row = self._rows.get(thread_id)
        if row is None:
            raise KeyError(f"checkpoint run 不存在：{thread_id}")
        now = finished_at or utc_now()
        row["status"] = status
        row["finished_at"] = now
        row["updated_at"] = now

    def get(self, thread_id: str) -> JsonObject | None:
        row = self._rows.get(thread_id)
        return dict(row) if row is not None else None

    def latest_resumable(self, session_key: str) -> JsonObject | None:
        candidates = [
            row
            for row in self._rows.values()
            if row["session_key"] == session_key and row["status"] in RESUMABLE_STATUSES
        ]
        if not candidates:
            return None
        latest = max(candidates, key=lambda row: row["finished_at"] or row["updated_at"])
        return dict(latest)

    def list_threads(self, session_key: str | None = None) -> list[JsonObject]:
        rows = [
            row
            for row in self._rows.values()
            if session_key is None or row["session_key"] == session_key
        ]
        rows.sort(key=lambda row: row["updated_at"], reverse=True)
        return [dict(row) for row in rows]

    def delete_thread(self, thread_id: str) -> bool:
        return self._rows.pop(thread_id, None) is not None

    def close(self) -> None:
        self._rows.clear()


class CheckpointStore:
    """把官方 saver 与 run 账本打包在一起，供服务层一次性注入。"""

    def __init__(
        self,
        *,
        settings: CheckpointSettings,
        saver: BaseCheckpointSaver,
        registry: CheckpointRegistry,
        connection: sqlite3.Connection | None = None,
    ):
        self.settings = settings
        self.saver = saver
        self.registry = registry
        self._connection = connection
        self._closed = False

    @property
    def backend(self) -> str:
        return self.settings.backend

    @property
    def is_persistent(self) -> bool:
        return self.backend == "sqlite"

    def close(self) -> None:
        """先关注册表，再关官方 saver 持有的连接；可重复调用。"""

        if self._closed:
            return
        self._closed = True
        self.registry.close()
        if self._connection is not None:
            self._connection.close()

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def build_checkpoint_store(settings: CheckpointSettings) -> CheckpointStore:
    """按配置创建 checkpoint 存储；节点看不到这个对象，只在 compile 阶段使用 saver。

    中文说明：sqlite 会在构造时就把文件和表建好（官方 saver 的 setup 还会切到
    WAL），这样启动阶段就能发现磁盘/路径问题，而不是等第一次 run 才炸。
    """

    backend = str(settings.backend or "").strip().lower()
    if backend == "memory":
        return CheckpointStore(
            settings=settings,
            saver=InMemorySaver(),
            registry=MemoryCheckpointRegistry(),
        )
    if backend == "sqlite":
        database_path = Path(settings.path or "data/checkpoints.db")
        database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database_path, check_same_thread=False)
        try:
            saver = SqliteSaver(connection)
            # 中文注释：先让官方 saver 建表并切到 WAL，再创建注册表连接；
            # 顺序反了会让注册表先在 rollback journal 模式下写一次。
            saver.setup()
            registry = SqliteCheckpointRegistry(database_path)
        except Exception:
            connection.close()
            raise
        return CheckpointStore(
            settings=settings,
            saver=saver,
            registry=registry,
            connection=connection,
        )
    raise ValueError(f"不支持的 checkpoint backend：{settings.backend!r}")