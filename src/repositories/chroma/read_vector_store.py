from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from src.llm import LLMProvider
from src.llm.base import normalize_token_usage
from src.paper_retrieval.models import PaperDocument
from src.repositories.chroma.vector_store import VectorUpsertItem, make_chroma_store, normalize_chroma_metadata
from src.utils.read_utils.chunkers import PageChunker, TextChunk, load_chunks_file


if TYPE_CHECKING:
    from src.graph.runtime_resources import WorkflowRuntimeResources


_EMBEDDING_SEMAPHORE = asyncio.Semaphore(1)


@dataclass(slots=True)
class ChunkIndexResult:
    """保存向量库写入结果。"""

    chunk_count: int
    persist_path: Path
    collection_name: str


@dataclass(slots=True)
class EmbeddingConnection:
    """保存生成向量时需要的模型连接信息。"""

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
    runtime_resources: WorkflowRuntimeResources | None = None,
) -> ChunkIndexResult:
    """同步兼容入口，把 Markdown 切分后写入向量库。

    中文注释：新阅读节点会优先走 chunk.json 入库；这个函数保留给旧调用方。
    chunk_size 和 chunk_overlap 是旧参数，当前按页切分策略暂时不会使用它们。
    """

    del chunk_size, chunk_overlap
    awaitable = async_index_markdown_chunks(
        paper,
        markdown_path=markdown_path,
        source_url=source_url,
        vector_store_path=vector_store_path,
        collection_name=collection_name,
        chunk_size=1,
        chunk_overlap=0,
        embedding_connection=embedding_connection,
        runtime_resources=runtime_resources,
    )
    if runtime_resources is not None:
        return runtime_resources.run_from_sync(awaitable)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        with asyncio.Runner() as runner:
            return runner.run(awaitable)
    if hasattr(awaitable, "close"):
        awaitable.close()
    raise RuntimeError("同步向量入库入口不能在已有事件循环中调用，请改用 await async_index_markdown_chunks(...)")


async def async_index_markdown_chunks(
    paper: PaperDocument,
    *,
    markdown_path: Path,
    source_url: str | None,
    vector_store_path: str | Path,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_connection: EmbeddingConnection,
    runtime_resources: WorkflowRuntimeResources | None = None,
    usage_callback: Callable[[dict[str, int]], None] | None = None,
) -> ChunkIndexResult:
    """异步兼容入口，把 Markdown 临时按页切分后写入向量库。

    中文注释：主流程现在会先生成 chunk.json，再调用 async_index_chunk_file。
    这里继续存在，是为了让旧代码或脚本还能直接传 Markdown。
    """

    del chunk_size, chunk_overlap
    markdown = await asyncio.to_thread(markdown_path.read_text, encoding="utf-8")
    chunks = PageChunker().chunk(paper, markdown)
    return await _index_chunks(
        paper,
        chunks=chunks,
        source_url=source_url,
        source_path=markdown_path,
        vector_store_path=vector_store_path,
        collection_name=collection_name,
        embedding_connection=embedding_connection,
        runtime_resources=runtime_resources,
        usage_callback=usage_callback,
    )


async def async_index_chunk_file(
    paper: PaperDocument,
    *,
    chunks_path: Path,
    source_url: str | None,
    vector_store_path: str | Path,
    collection_name: str,
    embedding_connection: EmbeddingConnection,
    runtime_resources: WorkflowRuntimeResources | None = None,
    usage_callback: Callable[[dict[str, int]], None] | None = None,
) -> ChunkIndexResult:
    """把 chunk.json 写入向量库。

    中文注释：全文提取和向量化都使用同一个 chunk.json，所以提取结果里的
    [chunkId] 可以准确对应到向量库里的同一段正文。
    """

    chunks = await asyncio.to_thread(load_chunks_file, chunks_path)
    if not chunks:
        raise ValueError("chunk.json 中没有可写入向量库的正文片段")
    return await _index_chunks(
        paper,
        chunks=chunks,
        source_url=source_url,
        source_path=chunks_path,
        vector_store_path=vector_store_path,
        collection_name=collection_name,
        embedding_connection=embedding_connection,
        runtime_resources=runtime_resources,
        usage_callback=usage_callback,
    )


async def _index_chunks(
    paper: PaperDocument,
    *,
    chunks: list[TextChunk],
    source_url: str | None,
    source_path: Path,
    vector_store_path: str | Path,
    collection_name: str,
    embedding_connection: EmbeddingConnection,
    runtime_resources: WorkflowRuntimeResources | None,
    usage_callback: Callable[[dict[str, int]], None] | None = None,
) -> ChunkIndexResult:
    """给一组正文片段生成向量并写入 Chroma。"""

    content_hash = hashlib.sha256("\n\n".join(chunk.content for chunk in chunks).encode("utf-8")).hexdigest()
    embeddings = await _create_embeddings_async(
        [chunk.content for chunk in chunks],
        embedding_connection,
        runtime_resources=runtime_resources,
        usage_callback=usage_callback,
    )
    if len(embeddings) != len(chunks):
        raise RuntimeError("embedding 返回数量和正文片段数量不一致")
    records = [
        _build_chunk_record(paper, chunk, embedding, source_path, source_url, content_hash)
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    return await asyncio.to_thread(_write_records_to_store, paper, records, vector_store_path, collection_name)


async def _create_embeddings_async(
    contents: list[str],
    connection: EmbeddingConnection,
    *,
    runtime_resources: WorkflowRuntimeResources | None = None,
    usage_callback: Callable[[dict[str, int]], None] | None = None,
) -> list[list[float]]:
    """分批调用 embedding 服务。

    中文注释：一次论文可能有很多 chunk，按 batch_size 分批可以减少单次请求太大
    导致失败的概率。并发控制优先使用本次工作流的统一信号量。
    """

    if not contents:
        return []
    results: list[list[float]] = []
    semaphore = runtime_resources.embedding_semaphore if runtime_resources is not None else _EMBEDDING_SEMAPHORE
    for start in range(0, len(contents), max(1, connection.batch_size)):
        batch = contents[start : start + max(1, connection.batch_size)]
        try:
            async with semaphore:
                response = await connection.provider.embed(batch, dimensions=connection.dimensions)
        except NotImplementedError as exc:
            raise RuntimeError(f"embedding provider 不支持当前模型向量化：{exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"embedding 服务调用失败：{exc}") from exc
        if callable(usage_callback):
            # embedding 供应商也会在响应里返回 usage，这里直接使用，不根据文本长度估算。
            usage_callback(normalize_token_usage(getattr(response, "usage", None)))
        if not response.ok:
            detail = response.content.strip() or response.error_code or response.error_type or response.error_kind or "未知错误"
            raise RuntimeError(f"embedding 服务调用失败：{detail}")
        if len(response.embeddings) != len(batch):
            raise RuntimeError("embedding 返回数量和请求文本数量不一致")
        for vector in response.embeddings:
            if not isinstance(vector, list) or not all(isinstance(value, int | float) for value in vector):
                raise RuntimeError("embedding 服务返回了无效向量")
            results.append([float(value) for value in vector])
    return results


def _write_records_to_store(
    paper: PaperDocument,
    records: list[dict[str, Any]],
    vector_store_path: str | Path,
    collection_name: str,
) -> ChunkIndexResult:
    """把向量记录写入 Chroma。

    中文注释：同一篇论文重新写入前先删除旧 chunk，避免缓存更新后旧片段残留。
    """

    items = [_record_to_upsert_item(record) for record in records]
    store = None
    try:
        store = make_chroma_store(vector_store_path, collection_name)
        store.delete_by_paper_id(paper.id)
        store.upsert(items)
    except Exception as exc:
        raise ValueError(f"Chroma 向量库写入失败：{exc}") from exc
    finally:
        if store is not None:
            store.close()
    return ChunkIndexResult(chunk_count=len(records), persist_path=Path(vector_store_path), collection_name=collection_name)


def _build_chunk_record(
    paper: PaperDocument,
    chunk: TextChunk,
    embedding: list[float],
    source_path: Path,
    source_url: str | None,
    paper_content_hash: str,
) -> dict[str, Any]:
    """把一个正文片段整理成 Chroma 记录。"""

    content = chunk.content
    return {
        "chunk_id": chunk.chunk_id,
        "paper_id": paper.id,
        "paperId": paper.paperId or paper.id,
        "doi": paper.doi,
        "title": paper.title,
        "source_url": source_url or paper.url,
        "pdf_url": paper.pdf_url,
        "source_path": str(source_path),
        "section": chunk.section,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "chunk_index": chunk.chunk_index,
        "previous_chunk_id": chunk.previous_chunk_id,
        "next_chunk_id": chunk.next_chunk_id,
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "paper_content_hash": paper_content_hash,
        "schema_version": 2,
        "content": content,
        "embedding": embedding,
    }


def _record_to_upsert_item(record: dict[str, Any]) -> VectorUpsertItem:
    """把内部记录转成 Chroma upsert 需要的结构。"""

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
