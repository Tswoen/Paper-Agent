from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from src.llm import ProviderSnapshot
from src.llm.base import LLMResponse
from src.paper_retrieval.models import PaperDocument
from src.utils.read_utils.chunkers import TextChunk, load_chunks_file


JsonObject = dict[str, Any]


EXTRACTION_SCHEMA: JsonObject = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "research_topic",
        "research_object",
        "methods",
        "conclusions",
        "contributions",
        "limitations",
    ],
    "properties": {
        "research_topic": {
            "type": "string",
            "description": "从全文中提取的研究主题，必须带来源，例如：研究了多智能体检索[paper:p0002]",
        },
        "research_object": {
            "type": "string",
            "description": "论文研究的对象、数据或任务，必须带来源 chunkId",
        },
        "methods": {
            "type": "string",
            "description": "关键方法名称和简要说明，必须带一个或多个来源 chunkId",
        },
        "conclusions": {
            "type": "string",
            "description": "核心结论，建议 2 到 3 句话，必须带来源 chunkId",
        },
        "contributions": {
            "type": "string",
            "description": "贡献点列表，可以用分号分隔，每个重要判断必须带来源 chunkId",
        },
        "limitations": {
            "type": "string",
            "description": "局限性列表，可以用分号分隔。全文没有明确说明时写空字符串",
        },
    },
}


async def async_extract_paper_from_chunks(
    paper: PaperDocument,
    *,
    chunks_path: Path,
    llm: ProviderSnapshot,
    runtime_resources: Any = None,
) -> JsonObject:
    """从 chunks.json 提取论文的结构化信息，并写入 extraction.json。

    中文注释：这里不重新解析 PDF，只读取已经缓存好的 chunks.json。模型必须按
    固定 JSON 字段回答；回答不合格时会抛错，让阅读节点记录失败原因。
    """

    chunks = await asyncio.to_thread(load_chunks_file, chunks_path)
    if not chunks:
        raise ValueError("chunks.json 中没有可用于全文提取的正文片段")
    output_path = chunks_path.parent / "extraction.json"
    cached = await asyncio.to_thread(_load_cached_extraction, output_path)
    if cached is not None:
        return cached
    response = await _call_model(
        llm,
        _extraction_messages(paper, chunks),
        runtime_resources=runtime_resources,
    )
    if not response.ok:
        detail = response.content.strip() or response.error_code or response.error_type or "未知错误"
        raise RuntimeError(f"全文提取模型调用失败：{detail}")
    payload = _parse_json_response(response)
    if payload is None:
        raise ValueError("全文提取模型没有返回合法 JSON")
    extraction = _validate_extraction(payload)
    record: JsonObject = {
        "schema_version": 1,
        "paperId": paper.paperId or paper.id,
        "schema": EXTRACTION_SCHEMA,
        "extraction": extraction,
        "chunks_used": [chunk.chunk_id for chunk in chunks],
    }
    await asyncio.to_thread(
        output_path.write_text,
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record


async def async_write_failed_extraction(
    paper: PaperDocument,
    *,
    chunks_path: Path,
    reason: str,
) -> JsonObject:
    """写入失败版 extraction.json。

    中文注释：模型返回格式不合格时，阅读流程仍然可以继续向量化。但缓存目录里
    也要留下 extraction.json，后续分析节点才能明确知道“提取失败”，而不是误以为
    还没处理过。
    """

    chunks = await asyncio.to_thread(load_chunks_file, chunks_path)
    record: JsonObject = {
        "schema_version": 1,
        "paperId": paper.paperId or paper.id,
        "schema": EXTRACTION_SCHEMA,
        "extraction": empty_extraction(),
        "chunks_used": [chunk.chunk_id for chunk in chunks],
        "status": "failed",
        "reason": reason,
    }
    output_path = chunks_path.parent / "extraction.json"
    await asyncio.to_thread(
        output_path.write_text,
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record


def empty_extraction() -> JsonObject:
    """返回空的全文提取结构。

    中文注释：下载或解析失败的论文也需要有稳定字段，方便汇总节点直接读取。
    """

    return {
        "research_topic": "",
        "research_object": "",
        "methods": "",
        "conclusions": "",
        "contributions": "",
        "limitations": "",
    }


def extraction_payload(record: JsonObject | None) -> JsonObject:
    """从 extraction.json 记录里取出真正给下游使用的 extraction 字段。"""

    if not isinstance(record, dict):
        return empty_extraction()
    extraction = record.get("extraction")
    return dict(extraction) if isinstance(extraction, dict) else empty_extraction()


async def _call_model(
    llm: ProviderSnapshot,
    messages: list[JsonObject],
    *,
    runtime_resources: Any,
) -> LLMResponse:
    """调用阅读模型。

    中文注释：如果工作流提供了 read_model_semaphore，就复用它，避免摘要阅读和全文
    提取同时把同一个模型打满。
    """

    semaphore = getattr(runtime_resources, "read_model_semaphore", None) if runtime_resources is not None else None
    if semaphore is None:
        return await llm.provider.chat(messages, temperature=0)
    async with semaphore:
        return await llm.provider.chat(messages, temperature=0)


def _extraction_messages(paper: PaperDocument, chunks: list[TextChunk]) -> list[JsonObject]:
    """构造全文提取提示词。

    中文注释：为了避免一次塞入太多内容，这里按顺序截取一个有限长度的上下文。
    每段前面都带 chunk_id，模型引用来源时只能使用这些编号。
    """

    context_parts: list[str] = []
    used_length = 0
    max_chars = 30000
    for chunk in chunks:
        text = f"[{chunk.chunk_id}]\n{chunk.content.strip()}"
        if used_length + len(text) > max_chars:
            break
        context_parts.append(text)
        used_length += len(text)
    payload = {
        "论文": {
            "paperId": paper.paperId or paper.id,
            "title": paper.title,
            "year": paper.year,
            "authors": paper.authors,
        },
        "JSON Schema": EXTRACTION_SCHEMA,
        "chunks": "\n\n".join(context_parts),
    }
    instruction = """你是论文全文阅读助手。只能依据用户提供的 chunks 内容回答，不能猜论文没有写明的信息。
请严格返回一个 JSON 对象，不要返回 Markdown，不要返回解释文字。
JSON 字段必须完全符合用户给出的 JSON Schema。
每个非空字段都必须在句末或判断后标注来源 chunkId，格式如 [paper:p0003]。
如果全文没有明确说明某个字段，请把该字段写成空字符串。"""
    return [{"role": "system", "content": instruction}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _load_cached_extraction(output_path: Path) -> JsonObject | None:
    """读取已经存在的 extraction.json。"""

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    extraction = payload.get("extraction")
    if not isinstance(extraction, dict):
        return None
    try:
        _validate_extraction(extraction)
    except ValueError:
        return None
    return payload


def _parse_json_response(response: LLMResponse) -> JsonObject | None:
    """从模型返回文本里取出 JSON 对象。"""

    text = response.content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _validate_extraction(payload: JsonObject) -> JsonObject:
    """检查全文提取结果是否符合固定字段和字符串类型。

    中文注释：项目暂时不额外引入 jsonschema 依赖，所以这里用手写校验完成当前
    Schema 的严格检查：字段不能多、不能少，每个值都必须是字符串。
    """

    required = list(EXTRACTION_SCHEMA["required"])
    allowed = set(required)
    keys = set(payload)
    missing = [key for key in required if key not in payload]
    extra = sorted(keys - allowed)
    if missing:
        raise ValueError(f"全文提取结果缺少字段：{', '.join(missing)}")
    if extra:
        raise ValueError(f"全文提取结果包含多余字段：{', '.join(extra)}")
    result: JsonObject = {}
    for key in required:
        value = payload.get(key)
        if not isinstance(value, str):
            raise ValueError(f"全文提取字段 {key} 必须是字符串")
        result[key] = value.strip()
    return result
