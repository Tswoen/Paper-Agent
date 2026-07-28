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
        """把切片转成可以写入 chunks.json 的普通字典。"""

        return {
            "chunk_id": self.chunk_id,
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
    """保存分块结果和 chunks.json 的位置。"""

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
    """读取 Markdown，切分正文，并写入 chunks.json。

    中文注释：这个文件是后续全文提取和向量化共同使用的“同一份上下文”。这样
    extraction.json 里引用的 chunkId，和向量库里的 chunkId 可以保持一致。
    """

    output_path = chunks_path or markdown_path.parent / "chunks.json"
    cached = load_chunks_file(output_path)
    if cached:
        return ChunkBuildResult(chunks_path=output_path, chunks=cached)
    markdown = markdown_path.read_text(encoding="utf-8")
    resolved_chunker = chunker or PageChunker()
    chunks = resolved_chunker.chunk(paper, markdown)
    payload = {
        "paperId": paper.paperId or paper.id,
        "chunker": resolved_chunker.name,
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return ChunkBuildResult(chunks_path=output_path, chunks=chunks)


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
    """读取已有 chunks.json。

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
        chunk_id = str(item.get("chunk_id") or "").strip()
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
