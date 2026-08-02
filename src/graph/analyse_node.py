from __future__ import annotations

import asyncio
import json
import re
from typing import Any, cast

from src.agents.analyseAgent import AnalyseAgent, build_analyse_agent, load_analyse_agent_llm
from src.graph.runtime import WorkflowRuntimeContext
from src.graph.state_models import JsonObject, State
from src.llm import ProviderSnapshot
from src.models.sessions import utc_now
from src.repositories.sessions.base import SessionRepository


ANALYSIS_VERSION = "2.0"


def run_analyse_node():
    """生成论文分析节点。

    中文说明：
    阅读节点已经把每篇论文整理成结构化摘要，分析节点只做三件事：
    1. 按子主题把这些结构化摘要分组；
    2. 让模型分别分析每个子主题，再做一次全局综合分析；
    3. 把最终报告放回 state，后面的回复节点和前端就能直接使用。
    """

    async def _node(state: State) -> State:
        """执行分析节点，并尽量保证模型不可用时也返回稳定结构。"""

        request = state.get("request")
        if request is None:
            raise ValueError("分析节点缺少用户综述主题，无法继续分析")

        reporter = _resolve_reporter(state)
        read_results = list(state.get("read_results") or [])
        read_summary = dict(state.get("read_summary") or {})
        search_summary = dict(state.get("search_summary") or {})
        llm = _resolve_llm(state)
        agent = build_analyse_agent(llm)

        if reporter is not None:
            reporter.started("正在按子主题分析论文", stage="analyse_start")

        groups = _build_subtopic_groups(read_results, read_summary, search_summary)
        model_used = llm.model if isinstance(llm, ProviderSnapshot) else "unavailable"

        if not groups:
            report = _empty_report(topic=request.topic, model_used=model_used)
        else:
            subtopic_analyses: list[JsonObject] = []
            for index, group in enumerate(groups, start=1):
                if reporter is not None:
                    reporter.progress(
                        f"正在分析子主题：{group['subtopic']}",
                        stage="analyse_subtopic",
                        completed=index - 1,
                        total=len(groups),
                        subtopic=group["subtopic"],
                    )
                analysis = await _analyse_one_subtopic(
                    group,
                    topic=request.topic,
                    agent=agent,
                )
                subtopic_analyses.append(analysis)

                # 中文说明：上面的 progress 事件表示“正在处理当前子主题”，所以第三个子主题开始时显示的是 2/3。
                # 子主题真正处理完成后，需要再发一个完成状态，前端才能显示 3/3，并明确知道综合分析可以开始了。
                # 这里使用 progress 配合 completed 状态，只更新“分析子主题”这一步，不会提前结束整个分析节点。
                if reporter is not None:
                    reporter.progress(
                        f"子主题分析完成：{group['subtopic']}",
                        stage="analyse_subtopic",
                        runtime_status="completed",
                        completed=index,
                        total=len(groups),
                        subtopic=group["subtopic"],
                    )

            if reporter is not None:
                reporter.progress("正在综合所有子主题的分析结果", stage="analyse_overall")

            overall_analysis = await _analyse_overall(
                topic=request.topic,
                subtopic_analyses=subtopic_analyses,
                agent=agent,
            )
            report = _build_final_report(
                topic=request.topic,
                subtopic_analyses=subtopic_analyses,
                overall_analysis=overall_analysis,
                model_used=model_used,
            )

        artifact_refs = list(state.get("analysis_artifact_refs") or [])
        persisted = await _persist_report_if_possible(state, report)
        if persisted:
            artifact_refs.append(persisted)
            if reporter is not None:
                reporter.artifact(persisted, stage="analysis_artifact_ready")

        if reporter is not None:
            reporter.completed(
                "论文分析节点已完成",
                stage="analyse_done",
                subtopic_count=report["execution_metadata"]["subtopic_count"],
                total_papers_analyzed=report["execution_metadata"]["total_papers_analyzed"],
            )

        updated = dict(state)
        updated.update(
            analysis_report=report,
            analysis_artifact_refs=artifact_refs,
            current_step="analyse",
        )
        return cast(State, updated)

    return _node


async def _analyse_one_subtopic(group: JsonObject, *, topic: str, agent: AnalyseAgent) -> JsonObject:
    """分析一个子主题。

    中文说明：
    这里的输入只使用阅读节点产出的结构化摘要，不重新读取 PDF，也不重新检索论文。
    这样逻辑简单，用户想排查结果时也能直接回到 read_results 找来源。
    """

    result = await agent.async_analyse_subtopic(topic=topic, group=group)
    if result.parsed is None:
        return _fallback_subtopic_analysis(group, reason=result.reason)
    return _normalize_subtopic_analysis(result.parsed, group)


async def _analyse_overall(topic: str, subtopic_analyses: list[JsonObject], agent: AnalyseAgent) -> JsonObject:
    """综合所有子主题的分析摘要，得到全局分析。"""

    result = await agent.async_analyse_overall(topic=topic, subtopic_analyses=subtopic_analyses)
    if result.parsed is None:
        return _fallback_overall_analysis(topic, subtopic_analyses, reason=result.reason)
    group = {
        "subtopic": "全局综合分析",
        "search_keyword": "",
        "paper_count": sum(int(item.get("paper_count") or 0) for item in subtopic_analyses),
        "paperIds": _paper_ids_from_analyses(subtopic_analyses),
        "papers": [],
    }
    return _normalize_subtopic_analysis(result.parsed, group)


def _build_subtopic_groups(read_results: list[JsonObject], read_summary: JsonObject, search_summary: JsonObject) -> list[JsonObject]:
    """把阅读结果按子主题分组。

    中文说明：
    优先用 read_results 里的论文 metadata.search_subtopics，因为这里能拿到完整阅读笔记。
    read_summary.subtopics 主要作为关键词补充，避免重复遍历时丢掉检索词。
    """

    keywords = _subtopic_keywords(read_summary, search_summary)
    grouped: dict[str, JsonObject] = {}
    for item in read_results:
        paper = dict(item.get("paper") or {})
        if not paper:
            continue
        origins = _paper_origins(paper)
        if not origins:
            origins = [{"subtopic": "综合阅读", "keyword": ""}]
        for origin in origins:
            subtopic = str(origin.get("subtopic") or "综合阅读").strip() or "综合阅读"
            keyword = str(origin.get("keyword") or keywords.get(subtopic) or "").strip()
            group = grouped.setdefault(
                subtopic,
                {
                    "subtopic": subtopic,
                    "search_keyword": keyword,
                    "papers": [],
                },
            )
            if keyword and not group.get("search_keyword"):
                group["search_keyword"] = keyword
            group["papers"].append(_paper_analysis_input(item))

    groups = list(grouped.values())
    for group in groups:
        # 中文说明：同一篇论文可能被多个来源命中；在同一个子主题里只保留一次，避免模型重复统计。
        group["papers"] = _deduplicate_paper_inputs(list(group["papers"]))
        group["paper_count"] = len(group["papers"])
    return groups


def _paper_analysis_input(item: JsonObject) -> JsonObject:
    """把单篇阅读结果压成分析模型需要的结构化摘要。"""

    paper = dict(item.get("paper") or {})
    note = dict(item.get("note") or {})
    relevance = dict(item.get("relevance") or {})
    full_text = dict(item.get("full_text") or {})
    extraction = dict(item.get("extraction") or {})
    paper_id = str(paper.get("paperId") or paper.get("id") or "").strip()
    return {
        "paperId": paper_id,
        "title": _shorten(paper.get("title"), 220),
        "year": paper.get("year") or "",
        "authors": list(paper.get("authors") or [])[:8],
        "venue": paper.get("journal_conference") or paper.get("journal/conference") or paper.get("venue") or "",
        "abstract": _shorten(paper.get("abstract"), 900),
        "structured_summary": {
            "main_question": _shorten(note.get("main_question"), 400),
            "methods": _shorten_list(note.get("methods"), 8, 180),
            "datasets": _shorten_list(note.get("datasets"), 8, 180),
            "contributions": _shorten_list(note.get("contributions"), 8, 220),
            "limitations": _shorten_list(note.get("limitations"), 8, 220),
            "main_results": _shorten_list(note.get("main_results"), 8, 220),
            "short_summary": _shorten(note.get("short_summary"), 500),
            "evidence_level": note.get("evidence_level") or "",
        },
        "relevance": {
            "score": relevance.get("score", 0),
            "decision": relevance.get("decision") or "",
            "reason": _shorten(relevance.get("reason"), 300),
        },
        "full_text_status": full_text.get("status") or "",
        "extraction": _shorten_json(extraction, 1200),
        "warnings": _shorten_list(item.get("warnings"), 5, 160),
    }


def _normalize_subtopic_analysis(parsed: JsonObject, group: JsonObject) -> JsonObject:
    """整理模型返回的 JSON，并由服务端补齐流程需要的论文信息。"""

    paper_ids = [str(paper.get("paperId") or "").strip() for paper in group.get("papers", []) if str(paper.get("paperId") or "").strip()]
    if not paper_ids:
        paper_ids = _string_list(group.get("paperIds"))
    fallback = _fallback_subtopic_analysis(group, reason="")
    return {
        # 这些字段由程序根据分析输入生成，模型不需要重复输出，避免格式出错。
        "subtopic": str(group.get("subtopic") or "综合分析"),
        "search_keyword": str(group.get("search_keyword") or ""),
        "paper_count": int(group.get("paper_count") or len(paper_ids)),
        "paperIds": paper_ids,
        "研究现状": _text_with_citation(parsed.get("研究现状") or fallback["研究现状"], paper_ids),
        "一致点": _text_list_with_citation(parsed.get("一致点"), paper_ids),
        "矛盾点": _text_with_citation(parsed.get("矛盾点") or fallback["矛盾点"], paper_ids),
        "研究空白": _text_with_citation(parsed.get("研究空白") or fallback["研究空白"], paper_ids),
        "时间线演化": _text_with_citation(parsed.get("时间线演化") or fallback["时间线演化"], paper_ids),
        "技术方法栈演变": _text_with_citation(parsed.get("技术方法栈演变") or fallback["技术方法栈演变"], paper_ids),
    }


def _fallback_subtopic_analysis(group: JsonObject, *, reason: str) -> JsonObject:
    """生成兜底子主题分析。

    中文说明：
    兜底结果不冒充真正的深度分析，只保证字段齐全，并把可引用的论文编号列出来。
    这样前端或后续节点不会因为模型暂时不可用而拿到空结构。
    """

    paper_ids = [str(paper.get("paperId") or "").strip() for paper in group.get("papers", []) if str(paper.get("paperId") or "").strip()]
    if not paper_ids:
        paper_ids = _string_list(group.get("paperIds"))
    citation = _citation_tail(paper_ids)
    summary = f"当前子主题共有 {len(paper_ids)} 篇论文可用于分析，建议结合这些论文的结构化摘要继续判断。{citation}"
    if reason:
        summary = f"{summary} 说明：{reason}。"
    return {
        "subtopic": str(group.get("subtopic") or "综合分析"),
        "search_keyword": str(group.get("search_keyword") or ""),
        "paper_count": int(group.get("paper_count") or len(paper_ids)),
        "paperIds": paper_ids,
        "研究现状": summary,
        "一致点": [],
        "矛盾点": f"暂未能稳定归纳论文之间的矛盾点，需要模型进一步分析。{citation}",
        "研究空白": f"暂未能稳定归纳研究空白，需要模型进一步分析。{citation}",
        "时间线演化": _fallback_timeline(group),
        "技术方法栈演变": f"暂未能稳定归纳方法栈演变，需要模型进一步分析。{citation}",
    }


def _fallback_overall_analysis(topic: str, subtopic_analyses: list[JsonObject], *, reason: str) -> JsonObject:
    """生成兜底全局分析。"""

    all_ids = _paper_ids_from_analyses(subtopic_analyses)
    citation = _citation_tail(all_ids)
    summary = f"《{topic}》目前已完成 {len(subtopic_analyses)} 个子主题的初步分析，后续可继续比较子主题之间的共识、矛盾和空白。{citation}"
    if reason:
        summary = f"{summary} 说明：{reason}。"
    return {
        "subtopic": "全局综合分析",
        "search_keyword": "",
        "paper_count": sum(int(item.get("paper_count") or 0) for item in subtopic_analyses),
        "paperIds": all_ids,
        "研究现状": summary,
        "一致点": [],
        "矛盾点": f"暂未能稳定归纳全局矛盾点，需要模型进一步分析。{citation}",
        "研究空白": f"暂未能稳定归纳全局研究空白，需要模型进一步分析。{citation}",
        "时间线演化": f"暂未能稳定归纳全局时间线演化，需要模型进一步分析。{citation}",
        "技术方法栈演变": f"暂未能稳定归纳全局方法栈演变。{citation}",
    }


def _build_final_report(
    *,
    topic: str,
    subtopic_analyses: list[JsonObject],
    overall_analysis: JsonObject,
    model_used: str,
) -> JsonObject:
    """组装最终报告。"""

    total_papers = len(set(_paper_ids_from_analyses(subtopic_analyses)))
    return {
        "analysis_version": ANALYSIS_VERSION,
        "topic": topic,
        "overall_framework": overall_analysis.get("研究现状") or "",
        "overall_analysis": overall_analysis,
        "subtopic_analyses": subtopic_analyses,
        "execution_metadata": {
            "total_papers_analyzed": total_papers,
            "subtopic_count": len(subtopic_analyses),
            "model_used": model_used,
            "created_at": utc_now(),
        },
    }


def _empty_report(*, topic: str, model_used: str) -> JsonObject:
    """没有阅读结果时返回空报告。"""

    return {
        "analysis_version": ANALYSIS_VERSION,
        "topic": topic,
        "overall_framework": "暂无可分析的阅读结果。",
        "overall_analysis": _fallback_overall_analysis(topic, [], reason="阅读节点没有产出论文摘要"),
        "subtopic_analyses": [],
        "execution_metadata": {
            "total_papers_analyzed": 0,
            "subtopic_count": 0,
            "model_used": model_used,
            "created_at": utc_now(),
        },
    }


def _resolve_llm(state: State) -> ProviderSnapshot | None:
    """优先使用外部注入的分析模型，没有注入时读取 AnalyseAgent 自己的模型配置。"""

    injected = state.get("analysis_node_llm")
    if isinstance(injected, ProviderSnapshot):
        return injected
    if injected is None:
        return load_analyse_agent_llm()
    return None


def _resolve_reporter(state: State):
    """从运行上下文里取出分析节点的上报器。"""

    runtime = cast(WorkflowRuntimeContext | None, state.get("runtime_context"))
    if runtime is None or runtime.sync_port is None:
        return None
    return runtime.sync_port.for_node("analyse", "论文分析")


async def _persist_report_if_possible(state: State, report: JsonObject) -> JsonObject | None:
    """有会话仓库时保存分析报告。"""

    repo = cast(SessionRepository | None, state.get("session_repo"))
    session_key = _optional_text(state.get("session_key"))
    turn_id = _optional_text(state.get("turn_id"))
    if repo is None or not session_key or not turn_id:
        return None
    try:
        record = await asyncio.to_thread(
            repo.write_artifact,
            session_key,
            "paper_analysis_report",
            "analysis_report.json",
            json.dumps(report, ensure_ascii=False, indent=2),
            relative_path=f"artifacts/analysis/{turn_id}/analysis_report.json",
            metadata={"turn_id": turn_id, "format": "json", "analysis_version": ANALYSIS_VERSION},
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


def _subtopic_keywords(read_summary: JsonObject, search_summary: JsonObject) -> dict[str, str]:
    """整理子主题对应的检索关键词。"""

    keywords: dict[str, str] = {}
    for source in (search_summary, read_summary):
        for item in list(source.get("subtopics") or []):
            if not isinstance(item, dict):
                continue
            subtopic = str(item.get("subtopic") or "").strip()
            keyword = str(item.get("keyword") or item.get("search_keyword") or "").strip()
            if subtopic and keyword and subtopic not in keywords:
                keywords[subtopic] = keyword
    return keywords


def _paper_origins(paper: JsonObject) -> list[JsonObject]:
    """从论文 metadata 中读取它属于哪些子主题。"""

    metadata = paper.get("metadata") if isinstance(paper.get("metadata"), dict) else {}
    origins = metadata.get("search_subtopics") if isinstance(metadata, dict) else []
    if not isinstance(origins, list):
        return []
    cleaned: list[JsonObject] = []
    for origin in origins:
        if not isinstance(origin, dict):
            continue
        subtopic = str(origin.get("subtopic") or "").strip()
        if not subtopic:
            continue
        cleaned.append({"subtopic": subtopic, "keyword": str(origin.get("keyword") or "").strip()})
    return cleaned


def _deduplicate_paper_inputs(papers: list[JsonObject]) -> list[JsonObject]:
    """同一组里按 paperId 去重。"""

    seen: set[str] = set()
    deduplicated: list[JsonObject] = []
    for paper in papers:
        paper_id = str(paper.get("paperId") or "").strip()
        if paper_id and paper_id in seen:
            continue
        if paper_id:
            seen.add(paper_id)
        deduplicated.append(paper)
    return deduplicated


def _fallback_timeline(group: JsonObject) -> str:
    """根据年份生成一个很保守的兜底时间线。"""

    papers = list(group.get("papers") or [])
    years = [int(paper["year"]) for paper in papers if str(paper.get("year") or "").isdigit()]
    paper_ids = [str(paper.get("paperId") or "").strip() for paper in papers if str(paper.get("paperId") or "").strip()]
    if not years:
        return f"暂无足够的年份信息来归纳时间线演化。{_citation_tail(paper_ids)}"
    return f"现有论文的发表年份覆盖 {min(years)} 至 {max(years)} 年，具体演化过程仍需要模型进一步归纳。{_citation_tail(paper_ids)}"


def _paper_ids_from_analyses(analyses: list[JsonObject]) -> list[str]:
    """从子主题分析里尽量收集 paperId。"""

    ids: list[str] = []
    pattern = re.compile(r"\[([^\[\]]+)\]")
    for analysis in analyses:
        for paper_id in _list_value(analysis.get("paperIds")):
            text_id = str(paper_id or "").strip()
            if text_id and text_id not in ids:
                ids.append(text_id)
        text = json.dumps(analysis, ensure_ascii=False)
        for match in pattern.findall(text):
            paper_id = match.strip()
            # 中文说明：JSON 数组会长得像 ["P1"]，它不是正文里的 [paperId] 引用。
            # 这里跳过带引号或逗号的内容，只保留真正的文本引用。
            if '"' in paper_id or "'" in paper_id or "," in paper_id:
                continue
            if paper_id and paper_id not in ids:
                ids.append(paper_id)
    return ids


def _text_with_citation(value: Any, paper_ids: list[str]) -> str:
    """保证关键文本字段至少带一个论文引用。"""

    text = str(value or "").strip()
    if not text:
        text = "暂无稳定结论。"
    if paper_ids and not re.search(r"\[[^\[\]]+\]", text):
        text = f"{text}{_citation_tail(paper_ids[:3])}"
    return text


def _citation_tail(paper_ids: list[str]) -> str:
    """把 paperId 列表变成 [paperId] 引用尾巴。"""

    cleaned = [paper_id for paper_id in paper_ids if paper_id]
    return "".join(f"[{paper_id}]" for paper_id in cleaned[:6])


def _list_value(value: Any) -> list[Any]:
    """把模型返回的列表字段整理成列表。"""

    return value if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    """只保留列表中的非空文本，避免把错误的复杂结构写入结果。"""

    return [str(item).strip() for item in _list_value(value) if str(item).strip()]


def _text_list_with_citation(value: Any, paper_ids: list[str]) -> list[str]:
    """整理一致点数组，并保证每一条都至少带一个论文引用。"""

    return [_text_with_citation(item, paper_ids) for item in _string_list(value)]


def _optional_text(value: Any) -> str | None:
    """把可选值整理成非空字符串。"""

    text = str(value or "").strip()
    return text or None


def _shorten(value: Any, limit: int) -> str:
    """截短很长的文本，避免一次分析塞进过多无关内容。"""

    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _shorten_list(value: Any, max_items: int, item_limit: int) -> list[str]:
    """截短列表字段，保留最重要的前几条。"""

    if not isinstance(value, list):
        return []
    return [_shorten(item, item_limit) for item in value[:max_items] if str(item or "").strip()]


def _shorten_json(value: JsonObject, limit: int) -> JsonObject:
    """把复杂提取结果转成短 JSON，避免提示词太长。"""

    if not value:
        return {}
    text = _shorten(json.dumps(value, ensure_ascii=False), limit)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"summary": text}
    return parsed if isinstance(parsed, dict) else {"summary": text}
