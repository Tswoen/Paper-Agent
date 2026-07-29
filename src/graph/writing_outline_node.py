from __future__ import annotations

import asyncio
import json
from typing import Any, cast

from src.agents.writingOutlineAgent import WritingOutlineAgent, build_writing_outline_agent, load_writing_outline_agent_llm
from src.graph.runtime import WorkflowRuntimeContext
from src.graph.state_models import JsonObject, State
from src.llm import ProviderSnapshot
from src.models.sessions import utc_now
from src.repositories.sessions.base import SessionRepository


WRITING_OUTLINE_VERSION = "1.0"


def run_writing_outline_node():
    """生成论文写作大纲节点。

    中文说明：
    这个节点只负责“写作前的规划”，也就是章节和小节怎么安排。
    小节正文写作还没有设计好，所以这里不会生成正文，避免后面不好改。
    """

    async def _node(state: State) -> State:
        """从分析报告里读取 overall_framework，并生成结构化写作大纲。"""

        request = state.get("request")
        if request is None:
            raise ValueError("写作大纲节点缺少用户综述主题，无法继续生成大纲")

        reporter = _resolve_reporter(state)
        analysis_report = dict(state.get("analysis_report") or {})
        llm = _resolve_llm(state)
        agent = build_writing_outline_agent(llm)

        if reporter is not None:
            reporter.started("正在根据分析结果生成写作大纲", stage="writing_outline_start")

        outline, raw_model_output, reason = await _generate_outline(state, agent=agent)
        used_llm = outline is not None and reason == "ok"
        if outline is None:
            outline = _fallback_outline(topic=request.topic, analysis_report=analysis_report)

        report = {
            "outline_version": WRITING_OUTLINE_VERSION,
            "topic": request.topic,
            # 中文说明：writing_outline 是后续正文写作最应该直接读取的核心对象。
            "writing_outline": outline,
            "execution_metadata": {
                "used_llm": used_llm,
                "model_used": llm.model if isinstance(llm, ProviderSnapshot) else "unavailable",
                "created_at": utc_now(),
                "message": "已使用模型生成写作大纲" if used_llm else reason,
            },
        }

        artifact_refs = list(state.get("writing_outline_artifact_refs") or [])
        persisted = await _persist_outline_if_possible(state, report)
        if persisted:
            artifact_refs.append(persisted)
            if reporter is not None:
                reporter.artifact(persisted, stage="writing_outline_artifact_ready")

        diagnostics = dict(state.get("diagnostics") or {})
        diagnostics["writing_outline"] = {
            "used_llm": used_llm,
            "status": "ok" if used_llm else "fallback",
            "message": report["execution_metadata"]["message"],
            "raw_model_output": raw_model_output,
        }

        if reporter is not None:
            reporter.completed(
                "写作大纲节点已完成",
                stage="writing_outline_done",
                chapter_count=len(outline),
                used_llm=used_llm,
            )

        updated = dict(state)
        updated.update(
            writing_outline=outline,
            writing_outline_report=report,
            writing_outline_artifact_refs=artifact_refs,
            diagnostics=diagnostics,
            current_step="write_outline",
        )
        return cast(State, updated)

    return _node


async def _generate_outline(state: State, *, agent: WritingOutlineAgent) -> tuple[JsonObject | None, str, str]:
    """调用 Agent 生成大纲，并把空结果当作失败处理。"""

    outline, raw_model_output, reason = await agent.async_generate_outline(dict(state))
    if not _outline_is_complete(outline):
        return None, raw_model_output, reason if reason != "ok" else "模型返回的大纲为空"
    return outline, raw_model_output, reason


def _outline_is_complete(outline: JsonObject | None) -> bool:
    """检查大纲是否真的包含章节和小节。

    中文说明：
    模型有时会返回一个能解析的 JSON，但里面缺字段。
    这种结果对后续写正文没有帮助，所以这里直接判定为不可用，让节点走兜底大纲。
    """

    if not outline:
        return False
    for chapter in outline.values():
        if not isinstance(chapter, dict):
            return False
        if not str(chapter.get("description") or "").strip():
            return False
        sections = chapter.get("Sections")
        if not isinstance(sections, dict) or not sections:
            return False
        for section in sections.values():
            if not isinstance(section, dict):
                return False
            for key in ("task", "evidence-map", "ref-sections", "word-count"):
                if key not in section:
                    return False
    return True


def _fallback_outline(*, topic: str, analysis_report: JsonObject) -> JsonObject:
    """模型不可用时生成一份保守大纲。

    中文说明：
    这份兜底大纲不假装自己做了复杂判断，只把分析节点已有的信息安排进常见综述结构。
    后续用户可以拿到一个字段完整的对象，前端和下一步节点也不会因为空值出错。
    """

    overall_framework = str(analysis_report.get("overall_framework") or "暂无可用的总体分析。").strip()
    subtopic_analyses = [item for item in list(analysis_report.get("subtopic_analyses") or []) if isinstance(item, dict)]
    topic_text = topic or str(analysis_report.get("topic") or "当前主题")

    outline: JsonObject = {
        "Chapter1": {
            "description": f"本章只说明《{topic_text}》的研究背景、问题来源和综述范围，不提前展开具体论文比较。",
            "Sections": {
                "section1": {
                    "task": "交代研究背景，并说明为什么这个主题值得做综述。",
                    "evidence-map": [overall_framework],
                    "ref-sections": [],
                    "word-count": 600,
                },
                "section2": {
                    "task": "界定本文综述边界，说明本文会围绕哪些问题展开，哪些内容暂不展开。",
                    "evidence-map": _subtopic_names(subtopic_analyses),
                    "ref-sections": ["Chapter1.section1"],
                    "word-count": 500,
                },
            },
        },
        "Chapter2": {
            "description": "本章只梳理已有研究的主要方向和代表性证据，重点写清楚各方向已经解决了什么。",
            "Sections": _research_status_sections(subtopic_analyses),
        },
        "Chapter3": {
            "description": "本章只讨论已有研究之间的分歧、不足和仍未解决的问题，不重复铺陈第二章的研究现状。",
            "Sections": {
                "section1": {
                    "task": "归纳不同研究之间的一致点和分歧点，写清楚争议来自方法、数据还是研究对象差异。",
                    "evidence-map": _evidence_from_field(subtopic_analyses, "矛盾点"),
                    "ref-sections": ["Chapter2"],
                    "word-count": 900,
                },
                "section2": {
                    "task": "总结仍然缺少研究的问题，并说明这些空白为什么会影响后续研究。",
                    "evidence-map": _evidence_from_field(subtopic_analyses, "研究空白"),
                    "ref-sections": ["Chapter3.section1"],
                    "word-count": 800,
                },
            },
        },
        "Chapter4": {
            "description": "本章只做收束和展望，把前文的发现整理成后续研究方向，不再引入新的大块材料。",
            "Sections": {
                "section1": {
                    "task": "概括全文主线，说明各章节共同支持了什么判断。",
                    "evidence-map": [overall_framework],
                    "ref-sections": ["Chapter1", "Chapter2", "Chapter3"],
                    "word-count": 600,
                },
                "section2": {
                    "task": "提出后续研究可以继续推进的方向，保持和第三章的研究空白对应。",
                    "evidence-map": _evidence_from_field(subtopic_analyses, "研究空白"),
                    "ref-sections": ["Chapter3.section2", "Chapter4.section1"],
                    "word-count": 500,
                },
            },
        },
    }
    return outline


def _research_status_sections(subtopic_analyses: list[JsonObject]) -> JsonObject:
    """把每个子主题变成第二章的一个小节。"""

    if not subtopic_analyses:
        return {
            "section1": {
                "task": "根据总体分析梳理已有研究现状，并保留后续补充论文证据的位置。",
                "evidence-map": [],
                "ref-sections": ["Chapter1.section2"],
                "word-count": 1000,
            }
        }

    sections: JsonObject = {}
    for index, item in enumerate(subtopic_analyses, start=1):
        subtopic = str(item.get("subtopic") or f"子主题{index}").strip()
        sections[f"section{index}"] = {
            "task": f"围绕“{subtopic}”梳理已有研究现状，先写主要结论，再写支撑这些结论的论文证据。",
            "evidence-map": _section_evidence(item),
            "ref-sections": ["Chapter1.section2"],
            "word-count": 800,
        }
    return sections


def _section_evidence(item: JsonObject) -> list[Any]:
    """整理单个子主题能用的证据。"""

    evidence: list[Any] = []
    summary = str(item.get("subtopic_summary") or item.get("研究现状") or "").strip()
    if summary:
        evidence.append(summary)
    paper_ids = item.get("paperIds")
    if isinstance(paper_ids, list) and paper_ids:
        evidence.append({"paperIds": paper_ids})
    return evidence


def _subtopic_names(subtopic_analyses: list[JsonObject]) -> list[str]:
    """提取子主题名称，作为综述范围的依据。"""

    names: list[str] = []
    for item in subtopic_analyses:
        name = str(item.get("subtopic") or "").strip()
        if name:
            names.append(name)
    return names


def _evidence_from_field(subtopic_analyses: list[JsonObject], field: str) -> list[Any]:
    """从子主题分析中提取某一类证据，比如矛盾点或研究空白。"""

    evidence: list[Any] = []
    for item in subtopic_analyses:
        value = item.get(field)
        if isinstance(value, list):
            evidence.extend(value)
        elif value:
            evidence.append(value)
    return evidence


def _resolve_llm(state: State) -> ProviderSnapshot | None:
    """优先使用外部注入的写作大纲模型，没有注入时读取默认配置。"""

    injected = state.get("writing_outline_node_llm")
    if isinstance(injected, ProviderSnapshot):
        return injected
    if injected is None:
        return load_writing_outline_agent_llm()
    return None


def _resolve_reporter(state: State):
    """从运行上下文里取出写作大纲节点的进度上报器。"""

    runtime = cast(WorkflowRuntimeContext | None, state.get("runtime_context"))
    if runtime is None or runtime.sync_port is None:
        return None
    return runtime.sync_port.for_node("write_outline", "写作大纲")


async def _persist_outline_if_possible(state: State, report: JsonObject) -> JsonObject | None:
    """如果当前有会话仓库，就把写作大纲保存成 JSON 产物。"""

    repo = cast(SessionRepository | None, state.get("session_repo"))
    session_key = _optional_text(state.get("session_key"))
    turn_id = _optional_text(state.get("turn_id"))
    if repo is None or not session_key or not turn_id:
        return None
    try:
        record = await asyncio.to_thread(
            repo.write_artifact,
            session_key,
            "writing_outline",
            "writing_outline.json",
            json.dumps(report, ensure_ascii=False, indent=2),
            relative_path=f"artifacts/writing_outline/{turn_id}/writing_outline.json",
            metadata={"turn_id": turn_id, "format": "json", "outline_version": WRITING_OUTLINE_VERSION},
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
