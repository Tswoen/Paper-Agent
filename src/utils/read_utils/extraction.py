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

_CHUNK_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


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
            "description": "从全文中提取的研究主题，必须带来源，引用输入 chunks 中提供的真实 chunkId（形如 <paperId>:p0001），例如：研究了多智能体检索[1901.00383v1:p0002]",
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
    topic: str = "",
    runtime_resources: Any = None,
    force: bool = False,
    feedback: str | None = None,
) -> JsonObject:
    """从 chunk.json 提取论文的结构化信息，并写入 extraction.json。

    中文注释：这里不重新解析 PDF，只读取已经缓存好的 chunk.json。模型必须按
    固定 JSON 字段回答；回答不合格时会抛错，让阅读节点记录失败原因。

    topic 是用户的研究主题：提取前模型会先通读全文判断这篇论文是否真的与
    主题相关，明显无关（例如跨领域同形词）时返回 status="irrelevant" 的记录，
    阅读节点据此止损，跳过本篇后续提取与核查。

    force=True 时跳过已缓存的 extraction.json，强制重新提取；
    feedback 会把上一次"哪些字段没通过核查"的提示追加给模型，让它自我修正。
    """

    chunks = await asyncio.to_thread(load_chunks_file, chunks_path)
    if not chunks:
        raise ValueError("chunk.json 中没有可用于全文提取的正文片段")
    output_path = chunks_path.parent / "extraction.json"
    valid_chunk_ids = {chunk.chunk_id for chunk in chunks}
    cached = await asyncio.to_thread(_load_cached_extraction, output_path, valid_chunk_ids)
    if cached is not None and not force:
        return cached
    response = await _call_model(
        llm,
        _extraction_messages(paper, chunks, topic=topic, feedback=feedback),
        runtime_resources=runtime_resources,
    )
    if not response.ok:
        detail = response.content.strip() or response.error_code or response.error_type or "未知错误"
        raise RuntimeError(f"全文提取模型调用失败：{detail}")
    payload = _parse_json_response(response)
    if payload is None:
        raise ValueError("全文提取模型没有返回合法 JSON")
    # 中文注释：⑤ 全文止损——模型通读全文后判定与主题无关，写入 irrelevant 记录，
    # 不做六字段提取校验，也不进入证据核查。
    if isinstance(payload.get("irrelevant"), bool) and payload["irrelevant"]:
        record: JsonObject = {
            "schema_version": 2,
            "paperId": paper.paperId or paper.id,
            "schema": EXTRACTION_SCHEMA,
            "extraction": empty_extraction(),
            "chunks_used": [],
            "status": "irrelevant",
            "reason": str(payload.get("reason") or "").strip() or "全文判定与主题无关",
        }
        await asyncio.to_thread(
            output_path.write_text,
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return record
    # 修复：模型常在相关时返回 {"irrelevant": false, ...}，这个 irrelevant（以及配套的
    # reason）不属于六个提取字段，会被 _validate_extraction 判为"多余字段"而打回整篇。
    # 这里在校验前剥离它们，只保留六个标准字段。
    payload.pop("irrelevant", None)
    payload.pop("reason", None)
    extraction = _validate_extraction(payload, valid_chunk_ids=valid_chunk_ids)
    record = {
        "schema_version": 2,
        "paperId": paper.paperId or paper.id,
        "schema": EXTRACTION_SCHEMA,
        "extraction": extraction,
        "chunks_used": _citation_ids_from_extraction(extraction, valid_chunk_ids=valid_chunk_ids),
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
        "schema_version": 2,
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
    try:
        if semaphore is None:
            return await llm.provider.chat(messages, temperature=0)
        async with semaphore:
            return await llm.provider.chat(messages, temperature=0)
    except Exception as exc:
        # 中文注释：把连接、鉴权等调用问题统一交给阅读节点处理，使它能保存当前
        # 论文已经生成的 Markdown 和 chunk.json，等模型恢复后再继续。
        raise RuntimeError(f"全文提取模型调用失败：{exc}") from exc


def _extraction_messages(paper: PaperDocument, chunks: list[TextChunk], *, topic: str = "", feedback: str | None = None) -> list[JsonObject]:
    """构造全文提取提示词。

    中文注释：精读必须阅读同一篇论文的全部正文块，不能只截取开头的一部分。
    发送给模型的每个块只保留 chunkId 和 content，避免页码、相邻块等无关字段
    干扰模型，也减少请求内容。

    topic 用于全文止损判断：模型通读后先确认论文真的与主题相关，再开始提取。
    feedback 是上一次提取结果没通过核查时的反馈，会追加到指令末尾，
    提示模型这次要修正哪些字段。
    """

    del paper
    payload = {
        "用户研究主题": topic,
        "chunks": [{"chunkId": chunk.chunk_id, "content": chunk.content.strip()} for chunk in chunks],
    }
    instruction = """你是论文全文阅读助手。只能依据用户提供的 chunks 内容回答，不能猜论文没有写明的信息。
第一步（相关性检查）：通读全文块，判断这篇论文的实际研究对象是否与"用户研究主题"真正相关。
主题中的产品名、型号、代号必须整体匹配；仅出现相同字符组合但属于不同领域术语的（例如主题是 "Kimi K3" 大模型产品，论文研究的是代数几何的 "K3 surfaces"），判为无关。
如果论文与主题明显无关，只输出一个 JSON 对象：{"irrelevant": true, "reason": "一句话说明为什么无关"}，不要输出其他字段。
第二步（提取）：论文与主题相关时，严格返回一个 JSON 对象，不要返回 Markdown，不要返回解释文字。
JSON 必须且仅包含 research_topic、research_object、methods、conclusions、contributions、limitations 六个字符串字段。
每个非空字段都必须在句末或判断后标注来源 chunkId，格式如 [chunkId]，并且只能引用输入中真实存在的 chunkId。
如果全文没有明确说明某个字段，请把该字段写成空字符串。"""
    if feedback:
        instruction += f"\n\n上一次提取结果未通过核查，请重点修正以下问题：{feedback}"
    return [{"role": "system", "content": instruction}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _load_cached_extraction(output_path: Path, valid_chunk_ids: set[str]) -> JsonObject | None:
    """读取已经存在的 extraction.json。"""

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != 2:
        return None
    if payload.get("status") == "failed":
        return None
    extraction = payload.get("extraction")
    if not isinstance(extraction, dict):
        return None
    try:
        _validate_extraction(extraction, valid_chunk_ids=valid_chunk_ids)
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


def _validate_extraction(payload: JsonObject, *, valid_chunk_ids: set[str]) -> JsonObject:
    """检查全文提取结果是否符合固定字段和字符串类型。

    中文注释：项目暂时不额外引入 jsonschema 依赖，所以这里用手写校验完成当前
    Schema 的严格检查：字段不能多、不能少，每个值都必须是字符串。非空内容还必须
    引用输入里真实存在的 chunkId，防止模型给出无法核对的结论。
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
        text = value.strip()
        if text:
            cited_chunk_ids = _chunk_citation_ids(text)
            # 修复：只把内容命中 valid_chunk_ids 的方括号当作有效引用。
            # 原逻辑对任意 [xxx] 都当 chunkId，导致 [2024]、[Transformer]、Markdown 链接
            # [see](url) 等被误判为"引用了不存在的 chunkId"，把本可成功的整篇提取打回。
            valid_cited = [chunk_id for chunk_id in cited_chunk_ids if chunk_id in valid_chunk_ids]
            if not valid_cited:
                raise ValueError(f"全文提取字段 {key} 缺少有效 chunkId 引用")
            # DEPRECATED: 2026-09-07
            # 原因：旧校验把"引用了不存在的 chunkId"当作致命错误，但方括号正则过宽，
            #       年份/方法名/链接都会命中，造成大量假阳性失败（见 EVAL_NOTES.md 第六节）。
            # 替代方案：改为上面的"只统计有效引用、无有效引用才报错"，非 chunkId 方括号直接忽略。
            # if not cited_chunk_ids:
            #     raise ValueError(f"全文提取字段 {key} 缺少 chunkId 引用")
            # unknown_chunk_ids = sorted(set(cited_chunk_ids) - valid_chunk_ids)
            # if unknown_chunk_ids:
            #     raise ValueError(f"全文提取字段 {key} 引用了不存在的 chunkId：{', '.join(unknown_chunk_ids)}")
        result[key] = text
    return result


def _chunk_citation_ids(text: str) -> list[str]:
    """从摘要文本中按出现顺序读取 [chunkId] 引用。"""

    return [value.strip() for value in _CHUNK_CITATION_PATTERN.findall(text) if value.strip()]


def _citation_ids_from_extraction(extraction: JsonObject, *, valid_chunk_ids: set[str] | None = None) -> list[str]:
    """整理结构化摘要实际用到的 chunkId，供后续写作按编号查找原文。

    中文注释：传入 valid_chunk_ids 时，只保留真实存在的 chunkId，过滤掉 [2024]、
    [Transformer] 这类被方括号正则误抓的非引用内容，避免污染下游写作的原文定位。
    """

    cited_chunk_ids: list[str] = []
    for value in extraction.values():
        if not isinstance(value, str):
            continue
        for chunk_id in _chunk_citation_ids(value):
            if valid_chunk_ids is not None and chunk_id not in valid_chunk_ids:
                continue
            if chunk_id not in cited_chunk_ids:
                cited_chunk_ids.append(chunk_id)
    return cited_chunk_ids
