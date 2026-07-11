from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from src.paper_retrieval.models import PaperDocument


@dataclass(slots=True)
class ChunkIndexResult:
    """保存全文切片写入索引后的数量和索引文件位置。"""

    chunk_count: int
    index_path: Path


@dataclass(slots=True)
class EmbeddingConnection:
    """保存调用 embedding 服务所需的最少连接信息。"""

    api_base: str
    api_key: str | None
    model_name: str
    extra_headers: dict[str, str]
    dimensions: int | None = None
    batch_size: int = 32
    timeout_seconds: int = 60


def index_markdown_chunks(
    paper: PaperDocument,
    *,
    markdown_path: Path,
    source_url: str | None,
    collection_path: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_connection: EmbeddingConnection,
) -> ChunkIndexResult:
    """按标题和段落切分 Markdown，生成向量并保存包含溯源信息的本地索引记录。"""

    markdown = markdown_path.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    chunks = _split_markdown(markdown, max(100, chunk_size), max(0, min(chunk_overlap, chunk_size // 2)))
    embeddings = _create_embeddings([str(chunk["content"]) for chunk in chunks], embedding_connection)
    if len(embeddings) != len(chunks):
        raise RuntimeError("embedding 服务返回的向量数量与正文片段数量不一致")
    base_dir = Path(collection_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    index_path = base_dir / "paper_chunks.jsonl"
    records = [
        _build_chunk_record(paper, chunk, embedding, index, markdown_path, source_url, content_hash)
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
    ]
    _replace_paper_records(index_path, paper.id, records)
    return ChunkIndexResult(chunk_count=len(records), index_path=index_path)


def _create_embeddings(contents: list[str], connection: EmbeddingConnection) -> list[list[float]]:
    """分批调用 OpenAI 兼容的 embedding 接口，返回与正文片段一一对应的向量。"""

    if not contents:
        return []
    headers = {"Content-Type": "application/json", **connection.extra_headers}
    if connection.api_key:
        headers.setdefault("Authorization", f"Bearer {connection.api_key}")
    results: list[list[float]] = []
    endpoint = connection.api_base.rstrip("/") + "/embeddings"
    with httpx.Client(timeout=float(max(1, connection.timeout_seconds))) as client:
        for start in range(0, len(contents), max(1, connection.batch_size)):
            batch = contents[start : start + max(1, connection.batch_size)]
            payload: dict[str, Any] = {"model": connection.model_name, "input": batch}
            if connection.dimensions is not None:
                payload["dimensions"] = connection.dimensions
            try:
                response = client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                raw_items = response.json().get("data")
            except (httpx.HTTPError, ValueError) as exc:
                raise RuntimeError(f"embedding 服务调用失败：{exc}") from exc
            if not isinstance(raw_items, list):
                raise RuntimeError("embedding 服务没有返回 data 列表")
            ordered = sorted(raw_items, key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0)
            for item in ordered:
                vector = item.get("embedding") if isinstance(item, dict) else None
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
        "embedding": embedding,
    }


def _replace_paper_records(index_path: Path, paper_id: str, records: list[dict[str, Any]]) -> None:
    """替换同一篇论文的旧索引记录，使重复运行不会产生重复片段。"""

    old_records: list[str] = []
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(payload.get("paper_id") or "") != paper_id:
                old_records.append(json.dumps(payload, ensure_ascii=False))
    new_records = [json.dumps(record, ensure_ascii=False) for record in records]
    index_path.write_text("\n".join([*old_records, *new_records]) + ("\n" if old_records or new_records else ""), encoding="utf-8")
