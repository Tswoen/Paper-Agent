from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.paper_retrieval.models import PaperDocument


JsonObject = dict[str, Any]


@dataclass(slots=True)
class TextChunk:
    """保存一个供阅读和向量化使用的正文片段。

    中文注释：chunk_id 会写进 LLM 提示词，模型提取结论时必须带上它，例如
    “方法使用对比学习[paper_001:p0003]”。previous_chunk_id 和 next_chunk_id
    让后续节点能知道相邻片段是谁。
    """

    chunk_id: str
    paperId: str
    chunk_index: int
    content: str
    page_start: int | None = None
    page_end: int | None = None
    section: str = ""
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    metadata: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        """把切片转成可以写入 chunk.json 的普通字典。"""

        return {
            "chunkId": self.chunk_id,
            "paperId": self.paperId,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "previous_chunk_id": self.previous_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ChunkBuildResult:
    """保存分块结果和 chunk.json 的位置。"""

    chunks_path: Path
    chunks: list[TextChunk]


class BaseChunker(ABC):
    """正文分块基类。

    中文注释：不同切分方式只需要实现 chunk()。阅读节点不用关心它按页、按章节
    还是按其它规则切。
    """

    name = "base"

    @abstractmethod
    def chunk(self, paper: PaperDocument, markdown: str) -> list[TextChunk]:
        """把 Markdown 正文切成较小片段。"""


class PageChunker(BaseChunker):
    """按 PDF 页码切分 Markdown 的最简单策略。"""

    name = "page"
    max_chunk_characters = 1200

    def chunk(self, paper: PaperDocument, markdown: str) -> list[TextChunk]:
        """按 `<!-- page: N -->` 标记切分正文。

        中文注释：read_fulltext.py 转 PDF 时会把页码标记写进 Markdown。这里优先
        使用这些标记；如果遇到 HTML 或没有页码的文本，就退回成一个普通片段。
        """

        paper_id = str(paper.paperId or paper.id)
        parts = _split_by_page_marker(markdown)
        if not parts:
            parts = [{"page": None, "content": _remove_front_matter(markdown)}]
        chunks: list[TextChunk] = []
        for index, part in enumerate(parts):
            content = str(part.get("content") or "").strip()
            if not content:
                continue
            page = part.get("page")
            page_number = int(page) if isinstance(page, int) else None
            if len(content) > self.max_chunk_characters:
                for segment_index, page_chunk in enumerate(_split_long_content(content, self.max_chunk_characters), start=1):
                    page_prefix = f"{paper_id}:p{page_number:04d}" if page_number is not None else f"{paper_id}:c{index:04d}"
                    chunks.append(
                        TextChunk(
                            chunk_id=f"{page_prefix}:s{segment_index:04d}",
                            paperId=paper_id,
                            chunk_index=len(chunks),
                            content=page_chunk,
                            page_start=page_number,
                            page_end=page_number,
                            section=f"page_{page_number}" if page_number is not None else "姝ｆ枃",
                            metadata={"chunker": self.name},
                        )
                    )
                continue
            chunk_id = f"{paper_id}:p{page_number:04d}" if page_number is not None else f"{paper_id}:c{index:04d}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    paperId=paper_id,
                    chunk_index=len(chunks),
                    content=content,
                    page_start=page_number,
                    page_end=page_number,
                    section=f"page_{page_number}" if page_number is not None else "正文",
                    metadata={"chunker": self.name},
                )
            )
        _attach_neighbors(chunks)
        return chunks


def build_chunks_file(
    paper: PaperDocument,
    *,
    markdown_path: Path,
    chunks_path: Path | None = None,
    chunker: BaseChunker | None = None,
) -> ChunkBuildResult:
    """读取 Markdown，清理非正文内容后切分，并写入 chunk.json。

    中文注释：这个文件是后续全文提取和向量化共同使用的“同一份上下文”。这样
    extraction.json 里引用的 chunkId，和向量库里的 chunkId 可以保持一致。
    """

    output_path = chunks_path or markdown_path.parent / "chunk.json"
    cached = load_chunks_file(output_path)
    if cached and all(len(chunk.content) <= PageChunker.max_chunk_characters for chunk in cached):
        return ChunkBuildResult(chunks_path=output_path, chunks=cached)
    markdown = markdown_path.read_text(encoding="utf-8")
    resolved_chunker = chunker or PageChunker()
    # 中文注释：PDF 转换出的 Markdown 往往混有每页重复的页眉、页脚、摘要和参考文献。
    # 这些内容会干扰模型判断，所以切分前先只留下论文正文。
    body_markdown = preprocess_markdown_body(markdown)
    chunks = resolved_chunker.chunk(paper, body_markdown)
    payload = {
        "paperId": paper.paperId or paper.id,
        "chunker": resolved_chunker.name,
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return ChunkBuildResult(chunks_path=output_path, chunks=chunks)


def _split_long_content(content: str, max_characters: int) -> list[str]:
    """把过长的一页正文切成较小片段，避免一次请求超过服务限制。"""

    return [
        content[start : start + max_characters].strip()
        for start in range(0, len(content), max_characters)
        if content[start : start + max_characters].strip()
    ]


async def async_build_chunks_file(
    paper: PaperDocument,
    *,
    markdown_path: Path,
    chunks_path: Path | None = None,
    chunker: BaseChunker | None = None,
) -> ChunkBuildResult:
    """异步流程里的分块入口，把本地文件读写放到线程里执行。"""

    return await asyncio.to_thread(
        build_chunks_file,
        paper,
        markdown_path=markdown_path,
        chunks_path=chunks_path,
        chunker=chunker,
    )


def load_chunks_file(chunks_path: Path) -> list[TextChunk]:
    """读取已有 chunk.json。

    中文注释：缓存命中时直接复用，避免同一篇论文反复切分，也避免 chunk_id 在
    不同运行里发生变化。
    """

    try:
        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_chunks = payload.get("chunks") if isinstance(payload, dict) else payload
    if not isinstance(raw_chunks, list):
        return []
    chunks: list[TextChunk] = []
    for item in raw_chunks:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        chunk_id = str(item.get("chunkId") or "").strip()
        if not content or not chunk_id:
            continue
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                paperId=str(item.get("paperId") or item.get("paper_id") or ""),
                chunk_index=int(item.get("chunk_index") or len(chunks)),
                content=content,
                page_start=_optional_int(item.get("page_start")),
                page_end=_optional_int(item.get("page_end")),
                section=str(item.get("section") or ""),
                previous_chunk_id=_optional_text(item.get("previous_chunk_id")),
                next_chunk_id=_optional_text(item.get("next_chunk_id")),
                metadata=dict(item.get("metadata") or {}),
            )
        )
    return chunks


def preprocess_markdown_body(markdown: str) -> str:
    """删除页眉页脚、摘要和参考文献，只保留用于精读的正文。"""

    pages = _split_by_page_marker(markdown)
    if pages:
        # 中文注释：同一段文字反复出现在多页首尾时，通常就是页眉或页脚。
        # 先按页去掉这些重复文字，再拼回页码标记，后面的切分仍能保留页码信息。
        page_contents = _remove_repeated_page_margins([str(page.get("content") or "") for page in pages])
        body = "\n\n".join(
            f"<!-- page: {page['page']} -->\n{content.strip()}"
            for page, content in zip(pages, page_contents, strict=True)
            if content.strip()
        )
    else:
        body = _remove_front_matter(markdown)
    return _remove_abstract_and_references(body).strip()


def _remove_repeated_page_margins(page_contents: list[str]) -> list[str]:
    """识别多页中重复出现的首尾文字，并将它们从对应页面删除。"""

    if len(page_contents) < 2:
        return page_contents

    header_pages: dict[str, set[int]] = {}
    footer_pages: dict[str, set[int]] = {}
    for page_index, content in enumerate(page_contents):
        lines = _meaningful_lines(content)
        if not lines:
            continue
        header_key = _page_margin_key(lines[0])
        footer_key = _page_margin_key(lines[-1])
        if header_key:
            header_pages.setdefault(header_key, set()).add(page_index)
        if footer_key:
            footer_pages.setdefault(footer_key, set()).add(page_index)

    # 中文注释：至少出现在两页、且覆盖半数页面的首尾文字才删除，避免误删正文标题。
    minimum_pages = max(2, (len(page_contents) + 1) // 2)
    repeated_headers = {key for key, pages in header_pages.items() if len(pages) >= minimum_pages}
    repeated_footers = {key for key, pages in footer_pages.items() if len(pages) >= minimum_pages}
    cleaned_pages: list[str] = []
    for content in page_contents:
        lines = content.splitlines()
        meaningful_indexes = [index for index, line in enumerate(lines) if line.strip()]
        if meaningful_indexes and _page_margin_key(lines[meaningful_indexes[0]]) in repeated_headers:
            lines[meaningful_indexes[0]] = ""
        if meaningful_indexes and _page_margin_key(lines[meaningful_indexes[-1]]) in repeated_footers:
            lines[meaningful_indexes[-1]] = ""
        cleaned_pages.append("\n".join(lines).strip())
    return cleaned_pages


def _meaningful_lines(content: str) -> list[str]:
    """返回页面中非空的文字行，供页眉页脚判断使用。"""

    return [line.strip() for line in content.splitlines() if line.strip()]


def _page_margin_key(line: str) -> str:
    """把页码中的数字统一替换，识别“第 1 页”和“第 2 页”这类重复页脚。"""

    normalized = re.sub(r"\d+", "#", line.lower())
    normalized = re.sub(r"\s+", "", normalized)
    return normalized if 3 <= len(normalized) <= 160 else ""


def _remove_abstract_and_references(markdown: str) -> str:
    """删除摘要到引言之间的内容，并删除参考文献及其后的内容。"""

    lines = markdown.splitlines()
    reference_index = next((index for index, line in enumerate(lines) if _is_references_heading(line)), len(lines))
    body_lines = lines[:reference_index]
    abstract_index = next((index for index, line in enumerate(body_lines) if _is_abstract_heading(line)), None)
    if abstract_index is None:
        return "\n".join(body_lines)

    # 中文注释：只有找到引言这类正文起点才删除摘要，避免解析异常时误删后续正文。
    body_start = next(
        (index for index in range(abstract_index + 1, len(body_lines)) if _is_body_start_heading(body_lines[index])),
        None,
    )
    if body_start is None:
        return "\n".join(body_lines)
    return "\n".join(body_lines[:abstract_index] + body_lines[body_start:])


def _is_abstract_heading(line: str) -> bool:
    """判断一行是否是摘要标题，兼容英文和中文的常见写法。"""

    return bool(re.match(r"^\s*(?:#{1,6}\s*)?(?:abstract|摘要)\b[\s:：.\-—]*.*$", line, flags=re.IGNORECASE))


def _is_body_start_heading(line: str) -> bool:
    """判断一行是否标志着摘要结束后的正文开始。"""

    return bool(
        re.match(
            r"^\s*(?:#{1,6}\s*)?(?:(?:\d+|[ivxlcdm]+)[.)、]?\s*)?(?:introduction|引言)\b.*$",
            line,
            flags=re.IGNORECASE,
        )
    )


def _is_references_heading(line: str) -> bool:
    """判断一行是否是参考文献标题，命中后该行及后续内容都不参与精读。"""

    return bool(
        re.match(
            r"^\s*(?:#{1,6}\s*)?(?:(?:\d+|[ivxlcdm]+)[.)、]?\s*)?(?:references?|bibliography|参考文献)\s*$",
            line,
            flags=re.IGNORECASE,
        )
    )


def _split_by_page_marker(markdown: str) -> list[JsonObject]:
    """按 Markdown 里的页码标记切分内容。"""

    text = _remove_front_matter(markdown)
    pattern = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    parts: list[JsonObject] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        parts.append({"page": int(match.group(1)), "content": text[start:end].strip()})
    return parts


def _remove_front_matter(markdown: str) -> str:
    """去掉 Markdown 开头的元数据块，只保留正文。"""

    text = markdown.strip()
    if not text.startswith("---"):
        return markdown
    match = re.match(r"(?s)^---\s*.*?\s*---\s*", text)
    return text[match.end() :] if match else markdown


def _attach_neighbors(chunks: list[TextChunk]) -> None:
    """给每个 chunk 补上前后相邻 chunk 的编号。"""

    for index, chunk in enumerate(chunks):
        chunk.previous_chunk_id = chunks[index - 1].chunk_id if index > 0 else None
        chunk.next_chunk_id = chunks[index + 1].chunk_id if index + 1 < len(chunks) else None


def _optional_int(value: Any) -> int | None:
    """把可能为空的页码转成整数。"""

    try:
        return int(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    """把可能为空的字段转成字符串或 None。"""

    text = str(value).strip() if value is not None else ""
    return text or None
