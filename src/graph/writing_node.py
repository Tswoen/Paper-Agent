from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, cast

from src.agents.writingAgent import WritingAgent, build_writing_agent, load_writing_agent_llm
from src.graph.runtime import WorkflowRuntimeContext
from src.graph.state_models import JsonObject, State
from src.llm import ProviderSnapshot, SystemConfig
from src.models.sessions import utc_now
from src.paper_retrieval.models import PaperDocument
from src.repositories.sessions.base import SessionRepository
from src.utils.read_utils.cache import safe_cache_name


# 中文说明：本版本开始，写作产物同时包含摘要和按正文引用整理的参考文献。
WRITING_VERSION = "1.1"


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
        # State 可能只带有本轮阅读结果，因此额外读取同一会话历史轮次的论文笔记。
        # 这样从已有会话恢复写作时，也能找到没有进入本轮 State 的论文摘要。
        session_read_results = await asyncio.to_thread(_load_session_read_results, state)
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
            def report_section_phase(message: str, *, task=section_task) -> None:
                """把当前小节的内部阶段更新到它自己的卡片上。"""

                if reporter is not None:
                    reporter.progress(
                        message,
                        stage="writing_section",
                        completed=index - 1,
                        total=len(section_tasks),
                        section_id=task["section_id"],
                        event_key=f"writing_section:{task['section_id']}",
                        stage_title=str(task.get("section_title") or task["section_id"]),
                    )

            if reporter is not None:
                reporter.progress(
                    "正在撰写小节正文",
                    stage="writing_section",
                    completed=index - 1,
                    total=len(section_tasks),
                    section_id=section_task["section_id"],
                    # 中文说明：每个小节都使用独立事件键，避免所有小节被前端合并成一张卡。
                    event_key=f"writing_section:{section_task['section_id']}",
                    stage_title=str(section_task.get("section_title") or section_task["section_id"]),
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
                session_read_results=session_read_results,
                progress_callback=report_section_phase,
            )
            section_result.update(
                chapter_key=section_task["chapter_key"],
                section_key=section_task["section_key"],
                chapter_title=section_task.get("chapter_title") or section_task["chapter_key"],
                section_title=section_task.get("section_title") or section_task["section_key"],
                chapter_description=section_task.get("chapter_description") or "",
                ref_sections=list(section_task.get("ref_sections") or []),
            )
            written_sections.append(section_result)
            if reporter is not None:
                reporter.progress(
                    "小节正文已完成",
                    stage="writing_section_done",
                    runtime_status="completed",
                    completed=index,
                    total=len(section_tasks),
                    section_id=section_task["section_id"],
                    cited_paper_count=len(section_result.get("cited_paper_ids") or []),
                    # 中文说明：完成事件沿用开始时的小节事件键，前端才能更新原卡片。
                    event_key=f"writing_section:{section_task['section_id']}",
                    stage_title=str(section_task.get("section_title") or section_task["section_id"]),
                )

        if reporter is not None:
            reporter.progress("正在根据正文生成摘要", stage="writing_abstract")
        abstract, abstract_status = await agent.async_write_abstract(
            topic=request.topic,
            sections=written_sections,
            word_count=300,
        )

        # 中文说明：引用顺序以正文里 paperId 第一次出现的位置为准，
        # 这样最后的参考文献编号和正文阅读顺序一致。
        cited_paper_ids = _extract_paper_ids_from_sections(written_sections)
        paper_metadata = _collect_paper_metadata(
            cited_paper_ids,
            search_results=list(state.get("search_results") or []),
            read_results=read_results,
            session_read_results=session_read_results,
            cache_dir=Path(str(cache_dir)),
        )
        references = _build_references(cited_paper_ids, paper_metadata)
        if reporter is not None:
            # 中文说明：摘要生成结束后发送完成状态，避免摘要卡一直停留在“处理中”。
            reporter.progress(
                "摘要写作已完成",
                stage="writing_abstract",
                runtime_status="completed",
                abstract_status=abstract_status,
            )
            reporter.progress(
                "正在整理参考文献",
                stage="writing_references",
                cited_paper_count=len(cited_paper_ids),
            )

        writing_report = _build_writing_report(
            topic=request.topic,
            outline=outline,
            sections=written_sections,
            abstract=abstract,
            abstract_status=abstract_status,
            references=references,
            model_used=llm.model if isinstance(llm, ProviderSnapshot) else "unavailable",
        )

        if reporter is not None:
            # 中文说明：参考文献已经在内存中整理完毕，文件保存是后续独立步骤。
            reporter.progress(
                "参考文献已生成",
                stage="writing_references",
                runtime_status="completed",
                cited_paper_count=len(cited_paper_ids),
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
            "abstract_status": abstract_status,
            "reference_count": len(references),
            "used_llm": isinstance(llm, ProviderSnapshot),
            "message": "正文、摘要和参考文献已完成",
        }

        if reporter is not None:
            reporter.completed(
                "正文、摘要和参考文献已完成",
                stage="writing_done",
                section_count=len(written_sections),
                reference_count=len(references),
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


def _load_session_read_results(state: State) -> list[JsonObject]:
    """读取当前会话所有轮次保存的论文阅读笔记。

    中文说明：会话产物记录里既有阅读汇总，也有每篇论文自己的 note.json。
    这里只读取每篇论文的 note.json，因为它同时保存了 paperId、论文信息和阅读笔记，
    内容完整且不会把同一轮汇总数据重复传给写作 Agent。
    """

    repository = state.get("session_repo")
    session_key = str(state.get("session_key") or "").strip()
    if repository is None or not session_key:
        return []

    try:
        session = repository.get(session_key)
    except Exception:
        # 会话刚被删除或产物暂时不可读时，继续使用 State 和 paper_cache。
        return []

    results: list[JsonObject] = []
    for artifact in list(getattr(session, "artifacts", []) or []):
        if not isinstance(artifact, dict) or artifact.get("artifact_type") != "paper_read_note":
            continue
        path = Path(str(artifact.get("path") or ""))
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            results.append(payload)
    return results


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
                    "chapter_title": str(chapter.get("title") or chapter_key),
                    "section_title": str(section.get("title") or section_key),
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
    abstract: str,
    abstract_status: str,
    references: list[JsonObject],
    model_used: str,
) -> JsonObject:
    """整理正文、摘要和参考文献组成的完整写作产物。"""

    return {
        "writing_version": WRITING_VERSION,
        "topic": topic,
        "writing_outline": outline,
        "sections": sections,
        "abstract": abstract,
        "references": references,
        "references_markdown": _references_markdown(references),
        "content_markdown": _compose_content_markdown(abstract, sections, references),
        "cited_paper_ids": [str(item.get("paperId") or "") for item in references if str(item.get("paperId") or "").strip()],
        "execution_metadata": {
            "model_used": model_used,
            "section_count": len(sections),
            "abstract_status": abstract_status,
            "reference_count": len(references),
            "created_at": utc_now(),
        },
    }


def _extract_paper_ids_from_sections(sections: list[JsonObject]) -> list[str]:
    """从小节正文的方括号引用中提取 paperId，并按首次出现顺序去重。"""

    # 中文说明：正文约定使用 [paperId] 标记引用。先收集 Agent 返回的 paperId，
    # 用它过滤普通的 Markdown 方括号文字；随后再保留带数字、冒号或斜杠的未知编号。
    declared_ids = _collect_cited_paper_ids(sections)
    declared_by_key = {paper_id.lower(): paper_id for paper_id in declared_ids}
    found: list[str] = []
    seen: set[str] = set()
    citation_pattern = re.compile(r"\[([^\[\]\r\n]+)\]")
    for section in sections:
        content = str(section.get("content") or "")
        for match in citation_pattern.finditer(content):
            candidate = match.group(1).strip().strip('"').strip("'")
            if not candidate or any(character.isspace() for character in candidate):
                continue
            paper_id = declared_by_key.get(candidate.lower(), candidate)
            if candidate.lower() not in declared_by_key and not re.search(r"\d|[:/.]", candidate):
                continue
            key = paper_id.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(paper_id)

        # 中文说明：如果模型把引用列在结构化字段里但正文没有重复写出，仍保留该引用，
        # 避免正文内容和写作 Agent 的引用记录不一致。
        for paper_id in list(section.get("cited_paper_ids") or []):
            text = str(paper_id or "").strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                found.append(text)
    return found


def _collect_paper_metadata(
    paper_ids: list[str],
    *,
    search_results: list[Any],
    read_results: list[JsonObject],
    session_read_results: list[JsonObject],
    cache_dir: Path,
) -> dict[str, JsonObject]:
    """从当前状态、会话阅读产物和本地缓存汇总论文元数据。"""

    metadata_by_id: dict[str, JsonObject] = {}

    def add_payload(payload: Any) -> None:
        if isinstance(payload, PaperDocument):
            paper = payload.to_dict()
        elif isinstance(payload, dict):
            paper = dict(payload.get("paper") or payload)
        else:
            return
        paper_id = str(paper.get("paperId") or paper.get("id") or "").strip()
        if not paper_id:
            return
        metadata_by_id.setdefault(paper_id.lower(), paper)

    for paper in search_results:
        add_payload(paper)
    for result in [*read_results, *session_read_results]:
        add_payload(result.get("paper") if isinstance(result, dict) else None)

    for paper_id in paper_ids:
        key = paper_id.lower()
        if key in metadata_by_id:
            continue
        for directory in _paper_metadata_dirs(cache_dir, paper_id):
            metadata_path = directory / "metadata.json"
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            paper = dict(payload.get("paper") or payload) if isinstance(payload, dict) else {}
            cached_ids = {
                str(payload.get("paperId") or "").strip().lower() if isinstance(payload, dict) else "",
                str(paper.get("paperId") or "").strip().lower(),
                str(paper.get("id") or "").strip().lower(),
            }
            if key not in cached_ids:
                continue
            add_payload(payload)
            if key in metadata_by_id:
                break
    return metadata_by_id


def _paper_metadata_dirs(cache_dir: Path, paper_id: str) -> list[Path]:
    """定位某个 paperId 可能对应的缓存目录。"""

    if not cache_dir.is_dir():
        return []
    direct = cache_dir / safe_cache_name(paper_id)
    directories = [direct] if direct.is_dir() else []
    for candidate in cache_dir.iterdir():
        if candidate.is_dir() and candidate not in directories:
            directories.append(candidate)
    return directories


def _build_references(paper_ids: list[str], metadata_by_id: dict[str, JsonObject]) -> list[JsonObject]:
    """按正文首次引用顺序生成 GB/T 7714 参考文献条目。"""

    references: list[JsonObject] = []
    for index, paper_id in enumerate(paper_ids, start=1):
        metadata = dict(metadata_by_id.get(paper_id.lower()) or {})
        references.append(
            {
                "index": index,
                "paperId": paper_id,
                "citation": _format_gbt7714_reference(paper_id, metadata),
                "metadata": metadata,
            }
        )
    return references


def _format_gbt7714_reference(paper_id: str, paper: JsonObject) -> str:
    """使用论文元数据生成常见的 GB/T 7714 顺序编码制格式。"""

    title = str(paper.get("title") or paper_id).strip()
    extra_metadata = paper.get("metadata") if isinstance(paper.get("metadata"), dict) else {}
    authors = _format_reference_authors(paper.get("authors") or paper.get("author"))
    resource_type = _reference_resource_type(paper)
    container = str(
        paper.get("journal_conference")
        or paper.get("journal/conference")
        or paper.get("journal")
        or paper.get("venue")
        or extra_metadata.get("journal")
        or ""
    ).strip()
    year = str(paper.get("year") or paper.get("publication_date") or "").strip()[:4]
    volume = str(paper.get("volume") or "").strip()
    issue = str(paper.get("issue") or "").strip()
    pages = str(
        paper.get("pages")
        or paper.get("page_range")
        or extra_metadata.get("pages")
        or (
            f"{paper.get('page_start')}-{paper.get('page_end')}"
            if paper.get("page_start") is not None and paper.get("page_end") is not None
            else ""
        )
    ).strip()
    doi = str(paper.get("doi") or "").strip()
    url = str(paper.get("url") or "").strip()

    citation = f"{authors + '. ' if authors else ''}{title}[{resource_type}]"
    if container:
        citation += f". {container}"
    if year:
        citation += f", {year}"
    if volume:
        citation += f", {volume}"
        if issue:
            citation += f"({issue})"
    elif issue:
        citation += f", ({issue})"
    if pages:
        citation += f": {pages}"
    citation += "."
    if doi:
        citation += f" DOI: {doi}."
    elif url:
        citation += f" {url}."
    return citation


def _format_reference_authors(value: Any) -> str:
    """整理作者字段，超过三位时按 GB/T 7714 习惯使用 et al.。"""

    if isinstance(value, str):
        authors = [item.strip() for item in re.split(r"[,;，；]", value) if item.strip()]
    elif isinstance(value, list):
        authors = []
        for item in value:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("author") or "").strip()
            else:
                name = str(item or "").strip()
            if name:
                authors.append(name)
    else:
        authors = []
    if len(authors) > 3:
        suffix = "等" if any("\u4e00" <= character <= "\u9fff" for character in authors[0]) else "et al"
        return ", ".join(authors[:3]) + ("，" if suffix == "等" else ", ") + suffix
    return ", ".join(authors)


def _reference_resource_type(paper: JsonObject) -> str:
    """根据元数据推断参考文献类型，缺少信息时按期刊论文处理。"""

    type_text = " ".join(
        str(paper.get(key) or "")
        for key in ("type", "document_type", "publication_type", "source")
    ).lower()
    if "conference" in type_text or "proceedings" in type_text:
        return "C"
    if "thesis" in type_text or "dissertation" in type_text:
        return "D"
    if "book" in type_text:
        return "M"
    if "arxiv" in type_text or "preprint" in type_text:
        return "EB/OL"
    return "J"


def _compose_content_markdown(
    abstract: str,
    sections: list[JsonObject],
    references: list[JsonObject],
) -> str:
    """按摘要、正文、参考文献顺序拼接最终 Markdown。"""

    blocks: list[str] = []
    if abstract.strip():
        blocks.append(f"# 摘要\n\n{abstract.strip()}")
    body = _sections_to_markdown(sections)
    if body:
        blocks.append(body)
    if references:
        reference_text = _references_markdown(references)
        if reference_text:
            blocks.append(f"# 参考文献\n\n{reference_text}")
    return "\n\n".join(blocks).strip()


def _references_markdown(references: list[JsonObject]) -> str:
    """把结构化参考文献条目拼成带编号的 Markdown 文本。"""

    return "\n".join(
        f"[{item.get('index')}] {item.get('citation')}"
        for item in references
        if str(item.get("citation") or "").strip()
    )


def _sections_to_markdown(sections: list[JsonObject]) -> str:
    """把所有小节正文拼成一份 Markdown，方便用户直接预览。"""

    blocks: list[str] = []
    current_chapter = ""
    for section in sections:
        chapter_key = str(section.get("chapter_key") or "")
        if chapter_key and chapter_key != current_chapter:
            blocks.append(f"# {section.get('chapter_title') or chapter_key}")
            current_chapter = chapter_key
        section_id = str(section.get("section_id") or "")
        section_title = str(section.get("section_title") or section_id).strip()
        content = str(section.get("content") or "").strip()
        blocks.append(f"## {section_title}\n\n{content}")
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
