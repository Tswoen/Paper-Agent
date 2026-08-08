from __future__ import annotations

import asyncio
import json
from typing import Any, cast

from src.agents.writingOutlineAgent import (
    OVERALL_ANALYSIS_FIELDS,
    WritingOutlineAgent,
    build_writing_outline_agent,
    load_writing_outline_agent_llm,
)
from src.graph.runtime import WorkflowRuntimeContext
from src.graph.state_models import JsonObject, State
from src.llm import ProviderSnapshot
from src.models.sessions import utc_now
from src.repositories.sessions.base import SessionRepository


# 中文说明：标题字段和正文范围发生了变化，使用新版本号便于识别旧产物。
WRITING_OUTLINE_VERSION = "1.1"


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

        def report_outline_usage(usage: JsonObject) -> None:
            """把写作大纲模型返回的真实 token 用量更新到大纲卡片。"""

            if reporter is not None:
                reporter.progress("写作大纲模型调用完成", stage="writing_outline", **usage)

        outline, raw_model_output, reason = await _generate_outline(
            state,
            agent=agent,
            usage_callback=report_outline_usage,
        )
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


async def _generate_outline(state: State, *, agent: WritingOutlineAgent, usage_callback: Any | None = None) -> tuple[JsonObject | None, str, str]:
    """调用 Agent 生成大纲，并把空结果当作失败处理。"""

    outline, raw_model_output, reason = await agent.async_generate_outline(dict(state), usage_callback=usage_callback)
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
        if not str(chapter.get("title") or "").strip():
            return False
        if not str(chapter.get("description") or "").strip():
            return False
        sections = chapter.get("Sections")
        if not isinstance(sections, dict) or not sections:
            return False
        for section in sections.values():
            if not isinstance(section, dict):
                return False
            if not str(section.get("title") or "").strip():
                return False
            for key in ("title", "task", "evidence-map", "ref-sections", "word-count"):
                if key not in section:
                    return False
    return True


def _fallback_outline(*, topic: str, analysis_report: JsonObject) -> JsonObject:
    """模型不可用时生成一份保守大纲。

    中文说明：
    这份兜底大纲不假装自己做了复杂判断，只把分析节点已有的信息安排进常见综述结构。
    后续用户可以拿到一个字段完整的对象，前端和下一步节点也不会因为空值出错。
    """

    topic_text = topic or str(analysis_report.get("topic") or "当前主题")
    overall_analysis = dict(analysis_report.get("overall_analysis") or {})

    # 中文说明：兜底结构从正文第一章开始，不再放摘要、引言和参考文献。
    # 每个章节和小节都写出独立标题，前端展示和后续正文写作都可以直接使用。
    outline: JsonObject = {
        "Chapter1": {
            "title": "相关研究现状",
            "description": f"围绕《{topic_text}》梳理已有研究的主要方向、代表性发现及其适用范围。",
            "Sections": {
                "section1": {
                    "title": "总体研究现状",
                    "task": "梳理该领域的研究范围、核心问题、主要方向和总体发展状态。",
                    "evidence-map": _available_evidence_fields(
                        overall_analysis,
                        "领域整体研究概况",
                        "各子主题横向差异对比分析",
                    ),
                    "ref-sections": [],
                    "word-count": 1000,
                }
            },
        },
        "Chapter2": {
            "title": "研究方法与技术演进",
            "description": "比较不同研究采用的方法、技术路线和演进趋势，说明方法差异如何影响研究结果。",
            "Sections": {
                "section1": {
                    "title": "方法与技术路线",
                    "task": "归纳各研究使用的方法和技术路线，说明它们分别解决了哪些问题。",
                    "evidence-map": _available_evidence_fields(overall_analysis, "领域技术与研究方法迭代脉络"),
                    "ref-sections": ["Chapter1"],
                    "word-count": 900,
                },
                "section2": {
                    "title": "研究时序演化",
                    "task": "按照研究发展顺序梳理关键变化，说明研究重点如何从早期问题逐步转向当前问题。",
                    "evidence-map": _available_evidence_fields(overall_analysis, "领域研究时序演化脉络"),
                    "ref-sections": ["Chapter1"],
                    "word-count": 800,
                },
            },
        },
        "Chapter3": {
            "title": "研究争议、空白与发展方向",
            "description": "在前文研究现状和方法比较的基础上，归纳主要争议、研究不足及可继续推进的方向。",
            "Sections": {
                "section1": {
                    "title": "研究共识与核心争议",
                    "task": "归纳不同研究之间的一致点和分歧点，说明争议来自方法、数据还是研究对象差异。",
                    "evidence-map": _available_evidence_fields(
                        overall_analysis,
                        "领域全域共性研究共识",
                        "领域核心研究争议与矛盾体系",
                    ),
                    "ref-sections": ["Chapter2"],
                    "word-count": 900,
                },
                "section2": {
                    "title": "研究空白与后续方向",
                    "task": "总结仍然缺少研究的问题，并提出与这些空白对应的后续研究方向。",
                    "evidence-map": _available_evidence_fields(
                        overall_analysis,
                        "领域系统性研究空白与局限",
                        "领域整体总结与研究展望",
                    ),
                    "ref-sections": ["Chapter3.section1"],
                    "word-count": 800,
                },
            },
        },
    }
    return outline


def _available_evidence_fields(overall_analysis: JsonObject, *fields: str) -> list[str]:
    """只保留当前全局分析中确实有内容的字段名。"""

    return [
        field
        for field in fields
        if field in OVERALL_ANALYSIS_FIELDS and str(overall_analysis.get(field) or "").strip()
    ]


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
