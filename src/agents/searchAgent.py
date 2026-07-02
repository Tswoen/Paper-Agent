from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.llm import ModelConfig, ProviderSnapshot, SystemConfig, make_provider

from .base import AgentContext, AgentSpec, BaseAgent
from .contracts import JsonObject


@dataclass(slots=True)
class SearchIntent:
    """描述搜索阶段的检索意图。

    这一层负责表达“我们想搜什么”，例如主题、关键词、年份和来源约束，
    但不负责生成具体的数据源查询语句。
    """

    topic: str
    keywords: list[str] = field(default_factory=list)
    excluded_terms: list[str] = field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    max_results: int = 30
    sources: list[str] = field(default_factory=list)


class SearchAgent(BaseAgent):
    """负责理解用户输入并生成搜索意图的 Agent。

    当前职责边界如下：
    1. 调用大模型提取主题关键词；
    2. 整理用户约束，形成结构化检索意图。

    真正的 query 拼接和来源适配由 paper_retrieval 层完成。
    """

    spec = AgentSpec(
        name="search_agent",
        role="search",
        description="根据主题与约束生成论文检索计划的代理。",
        llm_profile="default_agent",
        tools=(),
        skills=("query_rewrite", "source_routing"),
        input_keys=("request",),
    )

    def __init__(self, context: AgentContext):
        """初始化 SearchAgent，并保存依赖上下文。"""

        super().__init__(context)

    def _run(self, state: JsonObject) -> JsonObject:
        """从共享状态中生成搜索意图。

        中文注释：关键词完全依赖 LLM 产出；如果 LLM 不可用、调用失败或解析失败，
        当前搜索阶段直接终止，不再使用基于规则的关键词兜底。
        """

        keywords, raw_model_output, diagnostics = self._generate_keywords_with_llm(state)
        intent = self._build_search_intent(state, keywords or [])
        search_halted = keywords is None
        return {
            "search_intent": intent,
            "search_halted": search_halted,
            "diagnostics": {
                **diagnostics,
                "raw_model_output": raw_model_output,
            },
        }

    def _generate_keywords_with_llm(self, state: JsonObject) -> tuple[list[str] | None, str | None, JsonObject]:
        """调用大模型生成检索关键词。

        返回值包含：
        1. 解析成功的关键词列表；如果为 None，表示搜索阶段应直接终止；
        2. 模型原始输出，便于调试；
        3. 本次调用的诊断信息。
        """

        if self.context.llm is None:
            return None, None, {"used_llm": False, "status": "no_llm", "message": "未注入可用 LLM，搜索阶段已终止。"}
        messages = self._build_llm_messages(state)
        response = self.context.llm.provider.chat_with_retry(messages, max_tokens=800)
        raw_model_output = response.content or ""
        if not response.ok:
            return (
                None,
                raw_model_output,
                {
                    "used_llm": False,
                    "status": "llm_error",
                    "message": response.content or response.error_kind or "模型调用失败，搜索阶段已终止。",
                },
            )
        keywords = self._parse_llm_keywords(raw_model_output)
        if keywords is None:
            return (
                None,
                raw_model_output,
                {
                    "used_llm": False,
                    "status": "llm_parse_failed",
                    "message": "模型已有输出，但未能解析为约定 JSON，搜索阶段已终止。",
                },
            )
        return keywords, raw_model_output, {"used_llm": True, "status": "ok", "message": "已使用大模型生成检索关键词。"}

    def _build_llm_messages(self, state: JsonObject) -> list[JsonObject]:
        """构造给大模型的消息。

        这里刻意不出现任何具体搜索引擎名称，也不要求模型理解后端检索细节，
        只让它专注生成语义关键词和必要的检索辅助信息。
        """

        request = state["request"]
        system_prompt = """
作为一名论文查询助手，我将根据您的输入进行语义分析，提取查询条件，并将其转化为精确的英文检索条件。

例如，若您需要“近三年关于Transformer模型在机器翻译中的应用研究”，我将提取查询条件：Transformer, machine translation, 并限定年份为2023-2025，然后按照指定的格式输出。

请严格输出合法 JSON，字段必须包含：
- keywords: 英文检索条件数组
'JSON 结构为：{"keywords":["..."]}。'

"请只输出 JSON 对象，不要输出 Markdown、解释文本或代码块。"
"""
        user_prompt = json.dumps(
            {
                "topic": getattr(request, "topic", ""),
                "task": "请根据 topic 提炼适合学术检索的关键词。",
            },
            ensure_ascii=False,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_llm_keywords(self, raw_model_output: str) -> list[str] | None:
        """把模型输出解析成关键词列表。"""

        json_payload = self._extract_json_object(raw_model_output)
        if json_payload is None:
            return None
        try:
            data = json.loads(json_payload)
        except json.JSONDecodeError:
            return None
        keywords = self._clean_string_list(data.get("keywords"))
        if not keywords:
            return None
        return keywords

    def _build_search_intent(self, state: JsonObject, keywords: list[str]) -> SearchIntent:
        """根据请求约束和 LLM 关键词构建稳定的检索意图。

        中文注释：这里只接收 LLM 已解析出的关键词，不再把 topic 做规则分词，
        也不再把 LLM 关键词与启发式关键词做融合去重。
        """

        request = state["request"]
        constraints = dict(getattr(request, "constraints", {}) or {})
        topic = str(getattr(request, "topic", "")).strip()
        return SearchIntent(
            topic=topic,
            keywords=list(keywords),
            excluded_terms=self._normalize_string_list(constraints.get("excluded_terms")),
            year_from=self._coerce_optional_int(constraints.get("year_from")),
            year_to=self._coerce_optional_int(constraints.get("year_to")),
            max_results=self._coerce_positive_int(constraints.get("max_results"), default=30),
            sources=self._normalize_string_list(constraints.get("sources")),
        )

    def _extract_json_object(self, text: str) -> str | None:
        """从模型输出中提取第一个 JSON 对象。"""

        stripped = text.strip()
        if not stripped:
            return None
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = stripped[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            return None

    def _deduplicate_terms(self, terms: list[str]) -> list[str]:
        """保持原顺序去重。"""

        seen: set[str] = set()
        unique: list[str] = []
        for term in terms:
            normalized = term.strip()
            normalized_key = normalized.lower()
            if not normalized or normalized_key in seen:
                continue
            seen.add(normalized_key)
            unique.append(normalized)
        return unique

    def _clean_string_list(self, values: Any) -> list[str]:
        """把字符串或字符串数组规整为字符串列表，并保留模型原始顺序。

        中文注释：LLM 关键词不在这里去重，避免把“模型输出”悄悄改造成另一套规则结果。
        """

        if values is None:
            return []
        if isinstance(values, str):
            values = [values]
        cleaned: list[str] = []
        for value in values:
            text = str(value).strip()
            if text:
                cleaned.append(text)
        return cleaned

    def _normalize_string_list(self, values: Any) -> list[str]:
        """把字符串或字符串数组统一规整为去重后的字符串列表。"""

        if values is None:
            return []
        if isinstance(values, str):
            values = [values]
        normalized: list[str] = []
        for value in values:
            text = str(value).strip()
            if text:
                normalized.append(text)
        return self._deduplicate_terms(normalized)

    def _coerce_positive_int(self, value: Any, default: int) -> int:
        """把输入安全转换为正整数，失败时回退默认值。"""

        try:
            resolved = int(value)
        except (TypeError, ValueError):
            return default
        return resolved if resolved > 0 else default

    def _coerce_optional_int(self, value: Any) -> int | None:
        """把输入安全转换为可选整数，失败时返回 None。"""

        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


def load_search_agent_llm(
    agent_name: str | None = None,
    model_config_path: str | Path = "config/model.json",
    system_config_path: str | Path = "config/system.yaml",
    *,
    client: Any | None = None,
) -> ProviderSnapshot | None:
    """根据本地模型配置装配 SearchAgent 使用的默认 LLM。"""

    model_path = Path(model_config_path)
    if not model_path.exists():
        return None
    try:
        data = json.loads(model_path.read_text(encoding="utf-8"))
        system = SystemConfig.load(system_config_path)
        config = ModelConfig.from_dict(data, system)
        resolved_agent_name = agent_name or SearchAgent.spec.llm_profile
        return make_provider(config, resolved_agent_name, client=client)
    except Exception:
        # 中文注释：配置读取、Provider 装配或 SDK 初始化失败时返回 None，由 Agent 直接终止搜索。
        return None


def build_search_agent(llm: ProviderSnapshot | None | str = "auto") -> SearchAgent:
    """构建一个最小可用的 SearchAgent。"""

    resolved_llm = load_search_agent_llm() if llm == "auto" else llm
    context = AgentContext(spec=SearchAgent.spec, llm=resolved_llm)
    return SearchAgent(context)
