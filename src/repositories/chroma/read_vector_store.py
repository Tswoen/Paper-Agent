from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.llm import LLMProvider
from src.paper_retrieval.models import PaperDocument
from src.repositories.chroma.vector_store import VectorUpsertItem, make_chroma_store, normalize_chroma_metadata


@dataclass(slots=True)
class ChunkIndexResult:
    """保存全文切片写入 Chroma 后的数量和向量库位置。"""

    chunk_count: int
    persist_path: Path
    collection_name: str


@dataclass(slots=True)
class EmbeddingConnection:
    """保存生成向量所需的 provider 和调用参数。"""

    provider: LLMProvider
    model_name: str
    dimensions: int | None = None
    batch_size: int = 32


def index_markdown_chunks(
    paper: PaperDocument,
    *,
    markdown_path: Path,
    source_url: str | None,
    vector_store_path: str | Path,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_connection: EmbeddingConnection,
) -> ChunkIndexResult:
    """按标题和段落切分 Markdown，生成向量并保存到 Chroma 向量数据库。"""

    markdown = markdown_path.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    chunks = _split_markdown(markdown, max(100, chunk_size), max(0, min(chunk_overlap, chunk_size // 2)))
    embeddings = _create_embeddings([str(chunk["content"]) for chunk in chunks], embedding_connection)
    if len(embeddings) != len(chunks):
        raise RuntimeError("embedding 服务返回的向量数量与正文片段数量不一致")

    # 中文注释：这里仍然沿用原来的切片和 embedding 生成方式，只把最后的保存方式
    # 从 jsonl 文件换成 Chroma。这样阅读节点和断点恢复逻辑不用重新设计。
    records = [
        _build_chunk_record(paper, chunk, embedding, index, markdown_path, source_url, content_hash)
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
    ]
    items = [_record_to_upsert_item(record) for record in records]
    store = None
    try:
        store = make_chroma_store(vector_store_path, collection_name)
        # 中文注释：同一篇论文重新索引时，先删旧切片再写新切片。
        # 如果这次切出来的片段更少，旧的多余片段也会被清掉，不会混在查询结果里。
        store.delete_by_paper_id(paper.id)
        store.upsert(items)
    except Exception as exc:
        raise ValueError(f"Chroma 向量库写入失败：{exc}") from exc
    finally:
        # 中文注释：阅读节点每次索引完一篇论文后就可以关闭客户端。
        # 下次索引或查询时会重新连接，这样能减少本地数据库文件被长期占用的情况。
        if store is not None:
            store.close()
    return ChunkIndexResult(chunk_count=len(records), persist_path=Path(vector_store_path), collection_name=collection_name)


def _create_embeddings(contents: list[str], connection: EmbeddingConnection) -> list[list[float]]:
    """分批调用 provider 的 embedding 方法，返回与正文片段一一对应的向量。"""

    if not contents:
        return []
    results: list[list[float]] = []
    for start in range(0, len(contents), max(1, connection.batch_size)):
        batch = contents[start : start + max(1, connection.batch_size)]
        try:
            # 具体怎么请求模型服务，交给 src.llm 里的 provider 处理；这里仍只关心拿到向量。
            response = connection.provider.embed_with_retry(batch, dimensions=connection.dimensions)
        except NotImplementedError as exc:
            raise RuntimeError(f"embedding provider 不支持当前模型向量化：{exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"embedding 服务调用失败：{exc}") from exc
        if not response.ok:
            detail = response.content.strip() or response.error_code or response.error_type or response.error_kind or "未知错误"
            raise RuntimeError(f"embedding 服务调用失败：{detail}")
        if len(response.embeddings) != len(batch):
            raise RuntimeError("embedding 服务返回的向量数量与请求文本数量不一致")
        for vector in response.embeddings:
            if not isinstance(vector, list) or not all(isinstance(value, int | float) for value in vector):
                raise RuntimeError("embedding 服务返回了无效向量")
            results.append([float(value) for value in vector])
    return results


def _split_markdown(markdown: str, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    """优先按 Markdown 小标题和段落分段，过长内容才按字符数量继续切开。"""

    lines = markdown.splitlines()
    blocks: list[dict[str, Any]] = []
    section = "正文"
    page: int | None = None
    paragraph: list[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        """把当前段落按最大长度拆开并追加到切片列表。"""

        text = "\n".join(paragraph).strip()
        if not text or text == "---":
            return
        for piece in _split_long_text(text, chunk_size, chunk_overlap):
            blocks.append({"content": piece, "section": section, "page": page, "line_start": start_line, "line_end": end_line})

    for number, line in enumerate(lines, start=1):
        heading = re.match(r"^#{1,6}\s+(.+)$", line)
        page_marker = re.match(r"^<!--\s*page:\s*(\d+)\s*-->$", line)
        if heading or page_marker or not line.strip():
            flush(number - 1)
            paragraph = []
            start_line = number + 1
            if heading:
                section = heading.group(1).strip()
            if page_marker:
                page = int(page_marker.group(1))
            continue
        paragraph.append(line)
    flush(len(lines))
    return blocks


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """将过长段落按句子边界尽量拆开，相邻片段保留少量上下文。"""

    if len(text) <= chunk_size:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = max(text.rfind("。", start, end), text.rfind(". ", start, end), text.rfind("\n", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return [piece for piece in pieces if piece]


def _build_chunk_record(
    paper: PaperDocument,
    chunk: dict[str, Any],
    embedding: list[float],
    index: int,
    markdown_path: Path,
    source_url: str | None,
    paper_content_hash: str,
) -> dict[str, Any]:
    """为一个正文片段补齐向量、来源和位置，供后续检索定位。"""

    content = str(chunk["content"])
    return {
        "chunk_id": f"{paper.id}:{index:04d}",
        "paper_id": paper.id,
        "doi": paper.doi,
        "title": paper.title,
        "source_url": source_url or paper.url,
        "pdf_url": paper.pdf_url,
        "markdown_path": str(markdown_path),
        "section": chunk["section"],
        "page_start": chunk["page"],
        "page_end": chunk["page"],
        "line_start": chunk["line_start"],
        "line_end": chunk["line_end"],
        "chunk_index": index,
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "paper_content_hash": paper_content_hash,
        "schema_version": 1,
        "content": content,
        "embedding": embedding,
    }


def _record_to_upsert_item(record: dict[str, Any]) -> VectorUpsertItem:
    """把内部切片记录转换成 Chroma upsert 需要的数据结构。"""

    # 中文注释：Chroma 会单独保存正文 document 和向量 embedding。
    # 因此 metadata 里不要再放正文和向量，避免数据重复，也避免 metadata 类型过于复杂。
    metadata = {key: value for key, value in record.items() if key not in {"chunk_id", "content", "embedding"}}
    embedding = record.get("embedding")
    if not isinstance(embedding, list) or not all(isinstance(value, int | float) for value in embedding):
        raise ValueError("切片记录缺少有效的 embedding 向量")
    return VectorUpsertItem(
        id=str(record["chunk_id"]),
        embedding=[float(value) for value in embedding],
        metadata=normalize_chroma_metadata(metadata),
        document=str(record.get("content") or ""),
    )
