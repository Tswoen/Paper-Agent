from __future__ import annotations

import asyncio
import json
from typing import Any, cast

from src.agents.writingAgent import WritingAgent, build_writing_agent, load_writing_agent_llm
from src.graph.runtime import WorkflowRuntimeContext
from src.graph.state_models import JsonObject, State
from src.llm import ProviderSnapshot, SystemConfig
from src.models.sessions import utc_now
from src.repositories.sessions.base import SessionRepository


WRITING_VERSION = "1.0"


def run_writing_node():
    """生成论文正文写作节点。

    中文说明：
    大纲节点只负责“怎么写”，这个节点负责“真正写出来”。
    它会把大纲里的每个小节当成一个独立写作任务，按顺序交给 WritingAgent。
    """

    async def _node(state: State) -> State:
        """按大纲顺序逐节写作，并把结果保存进共享状态。"""

        request = state.get("request")
        if request is None:
            raise ValueError("写作节点缺少用户综述主题，无法继续生成正文")
        outline = dict(state.get("writing_outline") or {})
        if not outline:
            raise ValueError("写作节点缺少写作大纲，无法知道要写哪些小节")

        reporter = _resolve_reporter(state)
        llm = _resolve_llm(state)
        agent = build_writing_agent(llm)
        read_results = list(state.get("read_results") or [])
        cache_dir = SystemConfig.load().read.paper_cache_dir
        section_tasks = _flatten_outline(outline)
        written_sections: list[JsonObject] = []

        if reporter is not None:
            reporter.started(
                f"准备撰写 {len(section_tasks)} 个小节",
                stage="writing_start",
                total=len(section_tasks),
            )

        for index, section_task in enumerate(section_tasks, start=1):
            if reporter is not None:
                reporter.progress(
                    "正在撰写小节正文",
                    stage="writing_section",
                    completed=index - 1,
                    total=len(section_tasks),
                    section_id=section_task["section_id"],
                )
            previous_sections = _resolve_previous_sections(
                requested_refs=list(section_task.get("ref_sections") or []),
                written_sections=written_sections,
            )
            section_result = await agent.async_write_section(
                section_id=str(section_task["section_id"]),
                task=str(section_task.get("task") or ""),
                evidence_map=list(section_task.get("evidence_map") or []),
                previous_sections=previous_sections,
                word_count=int(section_task.get("word_count") or 800),
                read_results=read_results,
                cache_dir=cache_dir,
            )
            section_result.update(
                chapter_key=section_task["chapter_key"],
                section_key=section_task["section_key"],
                chapter_description=section_task.get("chapter_description") or "",
                ref_sections=list(section_task.get("ref_sections") or []),
            )
            written_sections.append(section_result)
            if reporter is not None:
                reporter.progress(
                    "小节正文已完成",
                    stage="writing_section_done",
                    completed=index,
                    total=len(section_tasks),
                    section_id=section_task["section_id"],
                    cited_paper_count=len(section_result.get("cited_paper_ids") or []),
                )

        writing_report = _build_writing_report(
            topic=request.topic,
            outline=outline,
            sections=written_sections,
            model_used=llm.model if isinstance(llm, ProviderSnapshot) else "unavailable",
        )

        artifact_refs = list(state.get("writing_artifact_refs") or [])
        persisted = await _persist_writing_if_possible(state, writing_report)
        if persisted:
            artifact_refs.append(persisted)
            if reporter is not None:
                reporter.artifact(persisted, stage="writing_artifact_ready")

        diagnostics = dict(state.get("diagnostics") or {})
        diagnostics["writing"] = {
            "status": "ok",
            "section_count": len(written_sections),
            "used_llm": isinstance(llm, ProviderSnapshot),
            "message": "正文写作节点已完成",
        }

        if reporter is not None:
            reporter.completed(
                "正文写作节点已完成",
                stage="writing_done",
                section_count=len(written_sections),
            )

        updated = dict(state)
        updated.update(
            writing_sections=written_sections,
            writing_report=writing_report,
            writing_artifact_refs=artifact_refs,
            diagnostics=diagnostics,
            current_step="write",
        )
        return cast(State, updated)

    return _node


def _flatten_outline(outline: JsonObject) -> list[JsonObject]:
    """把章节大纲拍平成按顺序执行的小节任务列表。"""

    tasks: list[JsonObject] = []
    for chapter_key, chapter in outline.items():
        if not isinstance(chapter, dict):
            continue
        sections = chapter.get("Sections")
        if not isinstance(sections, dict):
            continue
        for section_key, section in sections.items():
            if not isinstance(section, dict):
                continue
            section_id = f"{chapter_key}.{section_key}"
            tasks.append(
                {
                    "section_id": section_id,
                    "chapter_key": chapter_key,
                    "section_key": section_key,
                    "chapter_description": str(chapter.get("description") or ""),
                    "task": str(section.get("task") or ""),
                    "evidence_map": list(section.get("evidence-map") or []),
                    "ref_sections": list(section.get("ref-sections") or []),
                    "word_count": int(section.get("word-count") or 800),
                }
            )
    return tasks


def _resolve_previous_sections(*, requested_refs: list[Any], written_sections: list[JsonObject]) -> list[JsonObject]:
    """根据大纲里的 ref-sections 找出当前小节需要参考的前文。

    中文注释：
    ref-sections 可能写成 Chapter1.section2，也可能只写 Chapter1。
    这里做最简单的匹配：完整小节编号精确匹配，章节编号匹配该章节下所有已写小节。
    """

    if not requested_refs:
        return []
    resolved: list[JsonObject] = []
    seen: set[str] = set()
    ref_texts = [str(ref or "").strip() for ref in requested_refs if str(ref or "").strip()]
    for section in written_sections:
        section_id = str(section.get("section_id") or "").strip()
        chapter_key = str(section.get("chapter_key") or "").strip()
        if not section_id:
            continue
        matched = section_id in ref_texts or chapter_key in ref_texts
        if matched and section_id not in seen:
            resolved.append(
                {
                    "section_id": section_id,
                    "content": str(section.get("content") or ""),
                    "cited_paper_ids": list(section.get("cited_paper_ids") or []),
                }
            )
            seen.add(section_id)
    return resolved


def _build_writing_report(
    *,
    topic: str,
    outline: JsonObject,
    sections: list[JsonObject],
    model_used: str,
) -> JsonObject:
    """整理正文写作节点的完整产物。"""

    return {
        "writing_version": WRITING_VERSION,
        "topic": topic,
        "writing_outline": outline,
        "sections": sections,
        "content_markdown": _sections_to_markdown(sections),
        "cited_paper_ids": _collect_cited_paper_ids(sections),
        "execution_metadata": {
            "model_used": model_used,
            "section_count": len(sections),
            "created_at": utc_now(),
        },
    }


def _sections_to_markdown(sections: list[JsonObject]) -> str:
    """把所有小节正文拼成一份 Markdown，方便用户直接预览。"""

    blocks: list[str] = []
    current_chapter = ""
    for section in sections:
        chapter_key = str(section.get("chapter_key") or "")
        if chapter_key and chapter_key != current_chapter:
            blocks.append(f"# {chapter_key}")
            current_chapter = chapter_key
        section_id = str(section.get("section_id") or "")
        content = str(section.get("content") or "").strip()
        blocks.append(f"## {section_id}\n\n{content}")
    return "\n\n".join(blocks).strip()


def _collect_cited_paper_ids(sections: list[JsonObject]) -> list[str]:
    """汇总所有小节实际引用到的 paperId。"""

    seen: set[str] = set()
    result: list[str] = []
    for section in sections:
        for paper_id in list(section.get("cited_paper_ids") or []):
            text = str(paper_id or "").strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            result.append(text)
    return result


def _resolve_llm(state: State) -> ProviderSnapshot | None:
    """优先使用外部注入的正文写作模型，没有注入时读取默认配置。"""

    injected = state.get("writing_node_llm")
    if isinstance(injected, ProviderSnapshot):
        return injected
    if injected is None:
        return load_writing_agent_llm()
    return None


def _resolve_reporter(state: State):
    """从运行上下文里取出正文写作节点的进度上报器。"""

    runtime = cast(WorkflowRuntimeContext | None, state.get("runtime_context"))
    if runtime is None or runtime.sync_port is None:
        return None
    return runtime.sync_port.for_node("write", "正文写作")


async def _persist_writing_if_possible(state: State, report: JsonObject) -> JsonObject | None:
    """如果当前有会话仓库，就把正文写作结果保存成 JSON 产物。"""

    repo = cast(SessionRepository | None, state.get("session_repo"))
    session_key = _optional_text(state.get("session_key"))
    turn_id = _optional_text(state.get("turn_id"))
    if repo is None or not session_key or not turn_id:
        return None
    try:
        record = await asyncio.to_thread(
            repo.write_artifact,
            session_key,
            "writing",
            "writing.json",
            json.dumps(report, ensure_ascii=False, indent=2),
            relative_path=f"artifacts/writing/{turn_id}/writing.json",
            metadata={"turn_id": turn_id, "format": "json", "writing_version": WRITING_VERSION},
        )
    except Exception:
        return None
    return {
        "artifact_id": str(record["id"]),
        "artifact_type": str(record["artifact_type"]),
        "name": str(record["name"]),
        "path": str(record["path"]),
        "size": int(record["size"]),
        "created_at": str(record["created_at"]),
        "metadata": dict(record.get("metadata") or {}),
    }


def _optional_text(value: Any) -> str | None:
    """把可选值整理成非空字符串。"""

    text = str(value or "").strip()
    return text or None
