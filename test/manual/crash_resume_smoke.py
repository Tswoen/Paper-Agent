"""手工验收：真实进程 kill 后，另一个进程能否从 checkpoint 续跑。

用法（仓库根目录）：

    .\\.venv\\Scripts\\python.exe test\\manual\\crash_resume_smoke.py

脚本会自己启动两个子进程：

1. crash：跑到分析节点时 os._exit(1)，模拟进程被 kill；
2. resume：用同一个临时目录重新打开会话库/存档库，对账后续跑。

验收点：检索只调用一次、search_manifest.json 只有一份、账本 completed、
最终 current_step == reply。

这个文件不以 test_ 开头，也不在 test/manual 下被 unittest discover 收集。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = REPO_ROOT / "test"

# 子进程直接运行本文件时，保证能导入 src/ 和 test/ 下的辅助模块。
for path in (REPO_ROOT, TEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _make_repo(workdir: Path):
    from src.repositories.sessions.sqlite import SQLiteSessionRepository

    return SQLiteSessionRepository(storage_root=workdir / "data" / "sessions")


def _make_bundle(workdir: Path):
    from src.graph.checkpointing import build_checkpoint_bundle
    from src.llm.config import CheckpointSettings

    return build_checkpoint_bundle(
        CheckpointSettings(
            backend="sqlite",
            path=str(workdir / "data" / "checkpoints.db"),
        )
    )


class _FlakyCrashAgent:
    """第一次综合分析的节点直接退出整个进程。"""

    async def async_analyse_subtopic(self, *, topic, group, usage_callback=None):
        from src.agents.analyseAgent import AnalyseModelResult

        return AnalyseModelResult(parsed=None, reason="stub: 子主题走兜底")

    async def async_analyse_overall(self, *, topic, subtopic_analyses, usage_callback=None):
        # 中文注释：先留下到达标记，再硬杀进程；父进程用它区分“真的崩了”和“脚本提前失败”。
        marker = Path(os.environ["CRASH_MARKER"])
        marker.write_text("reached", encoding="utf-8")
        os._exit(1)


class _SuccessAgent:
    """续跑时让分析节点正常返回。"""

    async def async_analyse_subtopic(self, *, topic, group, usage_callback=None):
        from src.agents.analyseAgent import AnalyseModelResult

        return AnalyseModelResult(parsed=None, reason="stub: 子主题走兜底")

    async def async_analyse_overall(self, *, topic, subtopic_analyses, usage_callback=None):
        from src.agents.analyseAgent import AnalyseModelResult

        return AnalyseModelResult(parsed={}, raw_model_output="{}")


class _CountingSearchService:
    """每次检索往 marker 文件追加一行；_StubService 负责返回稳定结果。"""

    def __init__(self, marker: Path):
        from test_search_graph import _StubService

        self._stub = _StubService()
        self._marker = marker

    async def async_search(self, **kwargs):
        with self._marker.open("a", encoding="utf-8") as handle:
            handle.write("search\n")
        return await self._stub.async_search(**kwargs)


def _load_test_helpers():
    from graph_test_support import hermetic_graph_llms
    from test_search_graph import _fake_backends, _fake_snapshot, _read_snapshot

    return hermetic_graph_llms, _fake_backends, _fake_snapshot, _read_snapshot


def _run_crash(workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    repo = _make_repo(workdir)
    # 中文注释：先加载会话仓储，再导入 services，避免包初始化顺序造成的循环导入。
    from src.services.paper_workflow_runtime import build_paper_workflow_message_handler
    from src.services.session_runs import SessionRunService

    session = repo.create("Crash resume smoke")
    turn_id = "t1"
    thread_id = f"{session.key}:{turn_id}"
    state = {"session_key": session.key, "turn_id": turn_id, "thread_id": thread_id}
    (workdir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    search_marker = workdir / "search_calls.log"
    search_marker.write_text("", encoding="utf-8")
    crash_marker = workdir / "crash_reached.marker"
    os.environ["CRASH_MARKER"] = str(crash_marker)

    bundle = _make_bundle(workdir)
    service = SessionRunService(
        repo,
        message_handler=build_paper_workflow_message_handler(repo),
        checkpoint_bundle=bundle,
    )
    hermetic_graph_llms, _fake_backends, _fake_snapshot, _read_snapshot = _load_test_helpers()

    async def exercise() -> None:
        with hermetic_graph_llms(), _fake_backends(
            _CountingSearchService(search_marker),
            search_snapshot=_fake_snapshot(),
            read_snapshot=_read_snapshot(),
        ), mock.patch(
            "src.graph.analyse_node.build_analyse_agent",
            lambda *args, **kwargs: _FlakyCrashAgent(),
        ):
            accepted = await service.start_run(
                session.key,
                {
                    "content": "multi-agent literature review",
                    "turn_id": turn_id,
                    "constraints": {"sources": ["openalex"], "max_results": 3},
                },
            )
            task = service.broker.get_run(accepted["run_id"]).task
            await task

    asyncio.run(exercise())
    # 走到这里说明图没有在分析节点硬杀，属于脚本/场景失败；用非 1 的退出码让父进程识别。
    raise SystemExit(2)


def _run_resume(workdir: Path) -> None:
    state = json.loads((workdir / "state.json").read_text(encoding="utf-8"))
    session_key = str(state["session_key"])
    thread_id = str(state["thread_id"])
    search_marker = workdir / "search_calls.log"

    repo = _make_repo(workdir)
    bundle = _make_bundle(workdir)
    # 中文注释：先加载会话仓储，再导入 services，避免包初始化顺序造成的循环导入。
    from graph import build_graph
    from src.services.paper_workflow_runtime import build_paper_workflow_message_handler
    from src.services.session_runs import SessionRunService

    service = SessionRunService(
        repo,
        message_handler=build_paper_workflow_message_handler(repo),
        checkpoint_bundle=bundle,
    )
    hermetic_graph_llms, _fake_backends, _fake_snapshot, _read_snapshot = _load_test_helpers()

    recovered = service.recover_interrupted_runs()

    async def exercise():
        with hermetic_graph_llms(), _fake_backends(
            _CountingSearchService(search_marker),
            search_snapshot=_fake_snapshot(),
            read_snapshot=_read_snapshot(),
        ), mock.patch(
            "src.graph.analyse_node.build_analyse_agent",
            lambda *args, **kwargs: _SuccessAgent(),
        ):
            accepted = await service.start_run(
                session_key,
                {
                    "content": "继续上次失败的位置",
                    "turn_id": "t2",
                    "resume_from_last_checkpoint": True,
                },
            )
            task = service.broker.get_run(accepted["run_id"]).task
            await asyncio.wait_for(task, timeout=120)

        graph = build_graph(checkpointer=bundle.saver)
        return await graph.aget_state({"configurable": {"thread_id": thread_id}})

    try:
        snapshot = asyncio.run(exercise())
        row = bundle.registry.get(thread_id)
        artifact_names = [
            artifact["name"]
            for artifact in repo.get(session_key).thread()["artifacts"]
        ]
        search_count = len(
            [line for line in search_marker.read_text(encoding="utf-8").splitlines() if line.strip()]
        )
        with sqlite3.connect(workdir / "data" / "checkpoints.db") as connection:
            checkpoint_count = connection.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()[0]

        assert recovered == [session_key], f"recover 结果异常: {recovered}"
        assert row is not None, "账本行不存在"
        assert row["status"] == "completed", f"账本状态异常: {row['status']}"
        assert snapshot.values.get("current_step") == "reply", (
            f"最终步骤异常: {snapshot.values.get('current_step')}"
        )
        assert search_count == 1, f"检索次数异常: {search_count}"
        assert artifact_names.count("search_manifest.json") == 1, (
            f"manifest 数量异常: {artifact_names}"
        )
        assert checkpoint_count > 1, f"checkpoint 数量异常: {checkpoint_count}"

        print(f"[resume] recovered={recovered}")
        print(f"[resume] registry={row['status']}")
        print(f"[resume] current_step={snapshot.values.get('current_step')}")
        print(f"[resume] search_calls={search_count}")
        print(f"[resume] manifest={artifact_names.count('search_manifest.json')}")
        print(f"[resume] checkpoints={checkpoint_count}")
    finally:
        bundle.close()


def _spawn(role: str, workdir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--role",
            role,
            "--workdir",
            str(workdir),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )


def _print_tail(label: str, output: str) -> None:
    lines = output.strip().splitlines()
    print(f"--- {label} (last {min(40, len(lines))} lines) ---")
    for line in lines[-40:]:
        print(line)


def _run_parent() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="paper-agent-crash-"))
    succeeded = False
    try:
        crash = _spawn("crash", workdir)
        print(f"[crash] exit={crash.returncode} (expected 1)")
        crash_marker = workdir / "crash_reached.marker"
        if crash.returncode != 1 or not crash_marker.exists():
            _print_tail("crash stdout", crash.stdout)
            _print_tail("crash stderr", crash.stderr)
            raise AssertionError("crash 子进程没有在分析节点被 kill")

        resume = _spawn("resume", workdir)
        print(f"[resume] exit={resume.returncode} (expected 0)")
        if resume.returncode != 0:
            _print_tail("resume stdout", resume.stdout)
            _print_tail("resume stderr", resume.stderr)
            raise AssertionError("resume 子进程失败")
        _print_tail("resume stdout", resume.stdout)

        search_marker = workdir / "search_calls.log"
        search_count = len(
            [line for line in search_marker.read_text(encoding="utf-8").splitlines() if line.strip()]
        )
        with sqlite3.connect(workdir / "data" / "checkpoints.db") as connection:
            checkpoint_count = connection.execute(
                "SELECT COUNT(*) FROM checkpoints"
            ).fetchone()[0]
            registry = connection.execute(
                "SELECT status FROM checkpoint_runs"
            ).fetchone()

        print(f"[parent] search_calls={search_count}")
        print(f"[parent] checkpoints={checkpoint_count}")
        print(f"[parent] registry={registry[0] if registry else None}")

        assert search_count == 1, f"检索次数异常: {search_count}"
        assert checkpoint_count > 1, f"checkpoint 数量异常: {checkpoint_count}"
        assert registry is not None and registry[0] == "completed", (
            f"账本状态异常: {registry}"
        )
        succeeded = True
        print("PASS: real kill -> restart -> resume")
    finally:
        if succeeded:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"[parent] 失败现场保留在: {workdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="真实进程 kill 后续跑冒烟测试")
    parser.add_argument("--role", choices=["parent", "crash", "resume"], default="parent")
    parser.add_argument("--workdir", default="")
    args = parser.parse_args()

    if args.role == "crash":
        _run_crash(Path(args.workdir))
    elif args.role == "resume":
        _run_resume(Path(args.workdir))
    else:
        _run_parent()


if __name__ == "__main__":
    main()