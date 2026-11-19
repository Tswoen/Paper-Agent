"""手工验收：真实进程 kill 后，另一个进程能否从 checkpoint 续跑。

用法（仓库根目录）：

    .\\.venv\\Scripts\\python.exe test\\manual\\crash_resume_smoke.py

脚本默认依次跑两个场景，每个场景各启动两个子进程：

1. analyse：crash 在分析节点 os._exit(1)；resume 对账后从分析节点续跑。
   验收点：检索只调用一次、search_manifest.json 只有一份、账本 completed、
   最终 current_step == reply。
2. read：crash 在阅读节点内部——第一篇论文的摘要 note 落盘后、第二篇摘要
   进行中 os._exit(1)；resume 重跑 read 节点。
   验收点：检索不重跑、已完成摘要的论文不重跑摘要（note 幂等）、
   未完成论文重做摘要、账本 completed、最终 current_step == reply。

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
import time
from contextlib import nullcontext
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


def _make_read_crash_abstract(workdir: Path):
    """read 场景的崩溃桩：第一篇摘要正常完成，第二篇等首篇 note 落盘后硬杀进程。"""

    from src.models.read_models import ReadNote, ReadRelevance

    async def fake_abstract(paper, **kwargs):
        marker = Path(os.environ["ABSTRACT_MARKER"])
        with marker.open("a", encoding="utf-8") as handle:
            handle.write(f"{paper.id}\n")
        if paper.id == "graph-paper-arxiv":
            return ReadNote(short_summary="崩溃前已完成的摘要"), ReadRelevance(relevant=True), []
        # 中文注释：等第一篇的 note 真正落盘后再杀进程，保证崩溃现场里
        # 一定存在一个可复用的摘要稳定点。
        deadline = time.monotonic() + 30
        sessions_dir = workdir / "data" / "sessions"
        while time.monotonic() < deadline:
            if list(sessions_dir.glob("**/artifacts/read/threads/*/papers/*/note.json")):
                Path(os.environ["CRASH_MARKER"]).write_text("reached", encoding="utf-8")
                os._exit(1)
            await asyncio.sleep(0.05)
        raise SystemExit(3)

    return fake_abstract


def _make_read_resume_abstract():
    """read 场景的续跑桩：记录摘要调用，再委托真实实现（用 hermetic 模型）。"""

    from src.graph import read_node

    real_abstract = read_node._build_abstract_note

    async def logging_abstract(paper, **kwargs):
        marker = Path(os.environ["ABSTRACT_MARKER"])
        with marker.open("a", encoding="utf-8") as handle:
            handle.write(f"{paper.id}\n")
        return await real_abstract(paper, **kwargs)

    return logging_abstract


def _run_crash(workdir: Path, scenario: str) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    repo = _make_repo(workdir)
    # 中文注释：先加载会话仓储，再导入 services，避免包初始化顺序造成的循环导入。
    from src.services.paper_workflow_runtime import build_paper_workflow_message_handler
    from src.services.session_runs import SessionRunService

    session = repo.create("Crash resume smoke")
    turn_id = "t1"
    thread_id = f"{session.key}:{turn_id}"
    state = {"session_key": session.key, "turn_id": turn_id, "thread_id": thread_id, "scenario": scenario}
    (workdir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    search_marker = workdir / "search_calls.log"
    search_marker.write_text("", encoding="utf-8")
    abstract_marker = workdir / "abstract_calls.log"
    abstract_marker.write_text("", encoding="utf-8")
    crash_marker = workdir / "crash_reached.marker"
    os.environ["CRASH_MARKER"] = str(crash_marker)
    os.environ["ABSTRACT_MARKER"] = str(abstract_marker)

    bundle = _make_bundle(workdir)
    service = SessionRunService(
        repo,
        message_handler=build_paper_workflow_message_handler(repo),
        checkpoint_bundle=bundle,
    )
    hermetic_graph_llms, _fake_backends, _fake_snapshot, _read_snapshot = _load_test_helpers()

    if scenario == "read":
        # 中文注释：两篇论文进阅读节点；第一篇摘要完成并落盘 note 后，
        # 第二篇摘要进行中硬杀进程。
        constraints = {"sources": ["openalex", "arxiv"], "max_results": 3}
        crash_patch = mock.patch(
            "src.graph.read_node._build_abstract_note",
            new=_make_read_crash_abstract(workdir),
        )
        agent_patch = mock.patch(
            "src.graph.analyse_node.build_analyse_agent",
            lambda *args, **kwargs: _SuccessAgent(),
        )
    else:
        constraints = {"sources": ["openalex"], "max_results": 3}
        crash_patch = nullcontext()
        agent_patch = mock.patch(
            "src.graph.analyse_node.build_analyse_agent",
            lambda *args, **kwargs: _FlakyCrashAgent(),
        )

    async def exercise() -> None:
        with hermetic_graph_llms(), _fake_backends(
            _CountingSearchService(search_marker),
            search_snapshot=_fake_snapshot(),
            read_snapshot=_read_snapshot(),
        ), agent_patch, crash_patch:
            accepted = await service.start_run(
                session.key,
                {
                    "content": "multi-agent literature review",
                    "turn_id": turn_id,
                    "constraints": constraints,
                },
            )
            task = service.broker.get_run(accepted["run_id"]).task
            await task

    asyncio.run(exercise())
    # 走到这里说明图没有在预期位置硬杀，属于脚本/场景失败；用非 1 的退出码让父进程识别。
    raise SystemExit(2)


def _run_resume(workdir: Path, scenario: str) -> None:
    state = json.loads((workdir / "state.json").read_text(encoding="utf-8"))
    session_key = str(state["session_key"])
    thread_id = str(state["thread_id"])
    search_marker = workdir / "search_calls.log"
    abstract_marker = workdir / "abstract_calls.log"
    os.environ["ABSTRACT_MARKER"] = str(abstract_marker)

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

    if scenario == "read":
        abstract_patch = mock.patch(
            "src.graph.read_node._build_abstract_note",
            new=_make_read_resume_abstract(),
        )
    else:
        abstract_patch = nullcontext()

    async def exercise():
        with hermetic_graph_llms(), _fake_backends(
            _CountingSearchService(search_marker),
            search_snapshot=_fake_snapshot(),
            read_snapshot=_read_snapshot(),
        ), mock.patch(
            "src.graph.analyse_node.build_analyse_agent",
            lambda *args, **kwargs: _SuccessAgent(),
        ), abstract_patch:
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
        artifacts = repo.get(session_key).thread()["artifacts"]
        artifact_names = [artifact["name"] for artifact in artifacts]
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

        if scenario == "read":
            abstract_calls = [
                line.strip()
                for line in abstract_marker.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            # 中文注释：arxiv 在崩溃前已完成摘要——续跑绝不能重跑它（全程恰好 1 次）；
            # openalex 崩溃时未完成——崩溃尝试 1 次 + 续跑重做 1 次 = 2 次。
            assert abstract_calls.count("graph-paper-arxiv") == 1, (
                f"已完成摘要被重跑: {abstract_calls}"
            )
            assert abstract_calls.count("graph-paper-openalex") == 2, (
                f"未完成论文重做次数异常: {abstract_calls}"
            )
            arxiv_notes = [
                artifact
                for artifact in artifacts
                if artifact["artifact_type"] == "paper_read_note"
                and artifact["metadata"].get("paper_id") == "graph-paper-arxiv"
            ]
            assert len(arxiv_notes) == 1, f"arxiv note 数量异常: {len(arxiv_notes)}"
            note_path = repo.read_artifact_path(session_key, arxiv_notes[0]["id"])
            note_payload = json.loads(note_path.read_text(encoding="utf-8"))
            assert note_payload["result"]["note"]["short_summary"] == "崩溃前已完成的摘要", (
                "崩溃前的摘要 note 没有被复用"
            )
            print(f"[resume] abstract_calls={abstract_calls}")
            print("[resume] note_reuse=ok")
    finally:
        bundle.close()


def _spawn(role: str, workdir: Path, scenario: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--role",
            role,
            "--scenario",
            scenario,
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


def _run_scenario(scenario: str) -> None:
    workdir = Path(tempfile.mkdtemp(prefix=f"paper-agent-crash-{scenario}-"))
    succeeded = False
    try:
        crash = _spawn("crash", workdir, scenario)
        print(f"[{scenario}][crash] exit={crash.returncode} (expected 1)")
        crash_marker = workdir / "crash_reached.marker"
        if crash.returncode != 1 or not crash_marker.exists():
            _print_tail("crash stdout", crash.stdout)
            _print_tail("crash stderr", crash.stderr)
            raise AssertionError(f"crash 子进程没有在预期位置被 kill（场景 {scenario}）")

        resume = _spawn("resume", workdir, scenario)
        print(f"[{scenario}][resume] exit={resume.returncode} (expected 0)")
        if resume.returncode != 0:
            _print_tail("resume stdout", resume.stdout)
            _print_tail("resume stderr", resume.stderr)
            raise AssertionError(f"resume 子进程失败（场景 {scenario}）")
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

        print(f"[{scenario}][parent] search_calls={search_count}")
        print(f"[{scenario}][parent] checkpoints={checkpoint_count}")
        print(f"[{scenario}][parent] registry={registry[0] if registry else None}")

        assert search_count == 1, f"检索次数异常: {search_count}"
        assert checkpoint_count > 1, f"checkpoint 数量异常: {checkpoint_count}"
        assert registry is not None and registry[0] == "completed", (
            f"账本状态异常: {registry}"
        )
        succeeded = True
        print(f"PASS: real kill -> restart -> resume [{scenario}]")
    finally:
        if succeeded:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"[parent] 失败现场保留在: {workdir}")


def _run_parent() -> None:
    for scenario in ("analyse", "read"):
        _run_scenario(scenario)
    print("PASS: real kill -> restart -> resume (analyse + read)")


def main() -> None:
    parser = argparse.ArgumentParser(description="真实进程 kill 后续跑冒烟测试")
    parser.add_argument("--role", choices=["parent", "crash", "resume"], default="parent")
    parser.add_argument("--scenario", choices=["analyse", "read"], default="analyse")
    parser.add_argument("--workdir", default="")
    args = parser.parse_args()

    if args.role == "crash":
        _run_crash(Path(args.workdir), args.scenario)
    elif args.role == "resume":
        _run_resume(Path(args.workdir), args.scenario)
    else:
        _run_parent()


if __name__ == "__main__":
    main()