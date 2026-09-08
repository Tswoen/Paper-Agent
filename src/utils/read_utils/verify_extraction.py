from __future__ import annotations

import asyncio
import inspect
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.Prompts import VERIFY_EXTRACTION_SYSTEM_PROMPT
from src.paper_retrieval.models import PaperDocument


JsonObject = dict[str, Any]

# 需要验证的字段，用的是全文提取结果里的字段名（和 EXTRACTION_SCHEMA 一致）。
DEFAULT_VERIFY_FIELDS = ("methods", "conclusions", "limitations")
# 可选字段：原文确实没写也不算提取错误。默认是 limitations。
DEFAULT_OPTIONAL_FIELDS = ("limitations",)


async def async_verify_extraction(
    paper: PaperDocument,
    *,
    extraction: JsonObject,
    markdown_path: Path,
    llm: Any,
    runtime_resources: Any = None,
    fields: tuple[str, ...] = DEFAULT_VERIFY_FIELDS,
    optional_fields: tuple[str, ...] = DEFAULT_OPTIONAL_FIELDS,
    max_chars: int = 120000,
) -> JsonObject:
    """把全文提取结果和原文逐字段核对，返回验证结论。

    中文注释：提取模型偶尔会编造方法、结论或局限，所以提取之后再用一次模型，
    对照原文判断每个字段是不是真的出现在论文里。验证不通过不等于论文失败，
    只会在结果里记录下来，供后续重试或人工核对。

    返回结构：
    {
        "paper_id": ...,
        "passed": bool,                 # 必填字段全部通过
        "status": "completed" | "unavailable",
        "verified_at": "...",
        "items": [{"field", "verified", "exact_quote", "reason", "original_claim"}, ...],
    }
    """

    paper_id = paper.paperId or paper.id

    # 1. 读取论文原文 Markdown，太长就保留开头和结尾，避免撑爆模型上下文。
    try:
        md_text = await asyncio.to_thread(markdown_path.read_text, encoding="utf-8")
    except OSError:
        return _unavailable_result(paper_id, "无法读取论文原文 Markdown，跳过验证")
    md_text = _truncate_middle(md_text, max_chars=max_chars)

    # 2. 组装待验证的三个字段；可选字段为空时统一写成“未提及”，不算错误。
    claims: JsonObject = {}
    for field in fields:
        value = extraction.get(field, "")
        if field in optional_fields and not str(value).strip():
            value = "未提及"
        claims[field] = str(value or "").strip()

    if not claims:
        return _passed_result(paper_id)

    # 3. 把原文和提取结果一起交给验证模型。
    user_prompt = (
        f"## 论文全文（Markdown）\n\n{md_text}\n\n"
        f"## 提取结果（待验证）\n\n"
        f"{json.dumps(claims, ensure_ascii=False, indent=2)}"
    )
    messages: list[JsonObject] = [
        {"role": "system", "content": VERIFY_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        parsed = await _call_verify_model(
            llm,
            messages,
            runtime_resources=runtime_resources,
        )
    except Exception as exc:
        # 验证模型调用失败时不阻塞论文，本轮先跳过验证。
        return _unavailable_result(paper_id, f"验证模型调用失败：{exc}")

    # 4. 整理每个字段的验证结论，必填字段全部通过才算 passed。
    items: list[JsonObject] = []
    passed = True
    for field in fields:
        field_result = parsed.get(field, {}) if isinstance(parsed, dict) else {}
        verified = bool(field_result.get("verified", False))
        items.append(
            {
                "field": field,
                "verified": verified,
                "exact_quote": str(field_result.get("exact_quote", "")),
                "reason": str(field_result.get("reason", "")),
                "original_claim": claims.get(field, ""),
            }
        )
        if not verified and field not in optional_fields:
            passed = False

    return {
        "paper_id": paper_id,
        "items": items,
        "passed": passed,
        "status": "completed",
        "verified_at": datetime.now().isoformat(),
    }


async def _call_verify_model(llm: Any, messages: list[JsonObject], *, runtime_resources: Any) -> JsonObject:
    """调用验证模型，并把模型返回文本解析成 JSON 字典。"""

    semaphore = getattr(runtime_resources, "read_model_semaphore", None) if runtime_resources is not None else None
    if semaphore is None:
        content = await _provider_chat(llm, messages)
    else:
        async with semaphore:
            content = await _provider_chat(llm, messages)
    parsed = _parse_json_response(content)
    if not isinstance(parsed, dict):
        raise ValueError("验证模型没有返回可解析的 JSON")
    return parsed


async def _provider_chat(llm: Any, messages: list[JsonObject]) -> str:
    """调用 provider 拿回文本内容，优先走异步 chat 接口。"""

    provider = llm.provider if hasattr(llm, "provider") else llm
    chat = getattr(provider, "chat", None)
    if callable(chat):
        result = chat(messages, temperature=0)
        if inspect.isawaitable(result):
            result = await result
        return getattr(result, "content", "") or ""
    result = await asyncio.to_thread(provider.chat_with_retry, messages, temperature=0)
    return getattr(result, "content", "") or ""


def _parse_json_response(text: str) -> JsonObject | None:
    """从模型返回文本里取出 JSON 对象，兼容 Markdown 代码块包裹。"""

    stripped = (text or "").strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _truncate_middle(text: str, *, max_chars: int) -> str:
    """超长原文保留开头和结尾各一半，中间用提示语代替。"""

    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + "\n\n...(中间内容过长已省略)...\n\n"
        + text[-half:]
    )


def _passed_result(paper_id: str) -> JsonObject:
    """没有可验证字段时直接返回通过。"""

    return {
        "paper_id": paper_id,
        "items": [],
        "passed": True,
        "status": "completed",
        "verified_at": datetime.now().isoformat(),
    }


def _unavailable_result(paper_id: str, reason: str) -> JsonObject:
    """验证无法进行时返回的结果，不参与通过/失败判定。"""

    return {
        "paper_id": paper_id,
        "items": [],
        "passed": True,
        "status": "unavailable",
        "reason": reason,
        "verified_at": datetime.now().isoformat(),
    }
