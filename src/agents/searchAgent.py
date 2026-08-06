from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.llm import ModelConfig, ProviderSnapshot, SystemConfig, make_provider

from .base import AgentContext, AgentSpec, BaseAgent
from .contracts import JsonObject
from .Prompts import SEARCH_AGENT_SYSTEM_PROMPT


@dataclass(slots=True)
class SearchSubtopic:
    """单个细分研究方向，以及它对应的检索关键词表达式。

    中文注释：`subtopic` 是给人看的研究方向，`keyword` 是交给论文数据库检索的关键词串。
    关键词串统一使用形如 `(k1 and k2) or (k3 and k4)` 的格式，后续每个数据源再自己转换成适合自己的查询写法。
    """

    subtopic: str
    keyword: str


@dataclass(slots=True)
class SearchPlan:
    """保存模型生成的检索计划，避免把“子主题”和“关键词列表”拆散传递。"""

    keywords: list[str] = field(default_factory=list)
    subtopics: list[SearchSubtopic] = field(default_factory=list)


@dataclass(slots=True)
class SearchIntent:
    """描述搜索阶段的检索意图。

    这一层负责表达“我们想搜什么”，例如主题、细分方向、关键词、年份和来源约束。
    具体到每个网站如何写查询参数，仍然交给 paper_retrieval 里的 connector 自己处理。
    """

    topic: str
    keywords: list[str] = field(default_factory=list)
    subtopics: list[SearchSubtopic] = field(default_factory=list)
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

    # 通过指定AgentSpec.llm_profile来确定使用的配置文件中的哪个Agent配置
    spec = AgentSpec(
        name="search_agent",
        role="search",
        description="根据主题与约束生成论文检索计划的代理。",
        llm_profile="luna_agent",
        skills=(),
        input_keys=("request",),  # 当前 Agent 所必须的输入字段
    )

    def __init__(self, context: AgentContext):
        """初始化 SearchAgent，并保存依赖上下文。"""
        context.spec = self.spec  # 确保上下文里有当前 Agent 的 spec
        super().__init__(context)

    def _run(self, state: JsonObject) -> JsonObject:
        """从共享状态中生成搜索意图。

        中文注释：关键词完全依赖 LLM 产出；如果 LLM 不可用、调用失败或解析失败，
        当前搜索阶段直接终止，不再使用基于规则的关键词兜底。
        """

        plan, raw_model_output, diagnostics = self._generate_keywords_with_llm(state)
        intent = self._build_search_intent(state, plan or SearchPlan())
        search_halted = plan is None
        return {
            "search_intent": intent,
            "search_halted": search_halted,
            "diagnostics": {
                **diagnostics,
                "raw_model_output": raw_model_output,
            },
        }

    async def async_run(self, state: JsonObject) -> JsonObject:
        """异步执行搜索 Agent，给异步工作流直接使用。"""

        self._validate_state(state)
        plan, raw_model_output, diagnostics = await self._generate_keywords_with_llm_async(state)
        intent = self._build_search_intent(state, plan or SearchPlan())
        search_halted = plan is None
        return {
            "search_intent": intent,
            "search_halted": search_halted,
            "diagnostics": {
                **diagnostics,
                "raw_model_output": raw_model_output,
            },
        }

    def _generate_keywords_with_llm(self, state: JsonObject, usage_callback: Any | None = None) -> tuple[SearchPlan | None, str | None, JsonObject]:
        """调用大模型生成检索计划。

        返回值包含：
        1. 解析成功的检索计划；如果为 None，表示搜索阶段应直接终止；
        2. 模型原始输出，便于调试；
        3. 本次调用的诊断信息。
        """

        if self.context.llm is None:
            return None, None, {"used_llm": False, "status": "no_llm", "message": "未注入可用 LLM，搜索阶段已终止。"}
        messages = self._build_llm_messages(state)
        # 搜索关键词需要先理解研究主题，再拆分出多个检索方向，所以固定开启中等强度的思考模式。
        # 这里显式传参，避免注入的模型配置把思考强度覆盖成 none。
        response = self.context.llm.provider.chat_with_retry(
            messages,
            reasoning_effort="medium",
        )
        self.report_usage(response, usage_callback)
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
        plan = self._parse_llm_search_plan(raw_model_output)
        if plan is None:
            return (
                None,
                raw_model_output,
                {
                    "used_llm": False,
                    "status": "llm_parse_failed",
                    "message": "模型已有输出，但未能解析为约定 JSON，搜索阶段已终止。",
                },
            )
        return plan, raw_model_output, {"used_llm": True, "status": "ok", "message": "已使用大模型生成检索关键词。"}

    async def _generate_keywords_with_llm_async(self, state: JsonObject, usage_callback: Any | None = None) -> tuple[SearchPlan | None, str | None, JsonObject]:
        """异步调用大模型生成检索计划。"""

        if self.context.llm is None:
            return None, None, {"used_llm": False, "status": "no_llm", "message": "未注入可用 LLM，搜索阶段已终止。"}
        messages = self._build_llm_messages(state)
        response = await self._call_provider_chat_async(messages)
        self.report_usage(response, usage_callback)
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
        plan = self._parse_llm_search_plan(raw_model_output)
        if plan is None:
            return (
                None,
                raw_model_output,
                {
                    "used_llm": False,
                    "status": "llm_parse_failed",
                    "message": "模型已有输出，但未能解析为约定 JSON，搜索阶段已终止。",
                },
            )
        return plan, raw_model_output, {"used_llm": True, "status": "ok", "message": "已使用大模型生成检索关键词。"}

    async def _call_provider_chat_async(self, messages: list[JsonObject]):
        """优先走真正的异步 provider；只有老接口缺失时才退回同步兼容入口。"""

        provider = self.context.llm.provider
        chat = getattr(provider, "chat", None)
        if callable(chat):
            # 异步调用与同步调用保持相同的思考设置，确保工作流入口不同也不会改变搜索质量。
            response = chat(messages, reasoning_effort="medium")
            if inspect.isawaitable(response):
                return await response
            return response
        # 中文注释：这一步只是给极少数还没补齐 async 接口的旧 provider 兜底。
        # 新链路正常情况下会直接走上面的 await provider.chat(...)。
        return await asyncio.to_thread(
            provider.chat_with_retry,
            messages,
            reasoning_effort="medium",
        )

    def _build_llm_messages(self, state: JsonObject) -> list[JsonObject]:
        """构造给大模型的消息。

        这里刻意不出现任何具体搜索引擎名称，也不要求模型理解后端检索细节，
        只让它专注把主题拆成几个适合检索的研究方向，并给每个方向生成关键词表达式。
        """

        request = state["request"]
        # 中文说明：系统规则统一放在 Prompts.py，当前函数只负责填充用户主题。
        system_prompt = SEARCH_AGENT_SYSTEM_PROMPT
        user_prompt = json.dumps(
            {
                "topic": getattr(request, "topic", ""),
                "task": "请根据 topic 拆分研究方向，并为每个方向生成英文检索关键词表达式。",
            },
            ensure_ascii=False,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_llm_search_plan(self, raw_model_output: str) -> SearchPlan | None:
        """把模型输出解析成检索计划。

        中文注释：新版模型会返回 subtopics；旧版测试或旧提示词可能只返回 keywords。
        这里兼容两种格式，避免因为模型输出还没升级就让整个检索节点停掉。
        """

        json_payload = self._extract_json_object(raw_model_output)
        if json_payload is None:
            return None
        try:
            data = json.loads(json_payload)
        except json.JSONDecodeError:
            return None
        subtopics = self._parse_subtopics(data.get("subtopics"))
        keywords = self._clean_string_list(data.get("keywords"))
        if not subtopics and keywords:
            subtopics = [SearchSubtopic(subtopic="综合检索", keyword=self._keyword_expression_from_terms(keywords))]
        if not subtopics:
            return None
        if not keywords:
            keywords = self._keywords_from_subtopics(subtopics)
        return SearchPlan(keywords=keywords, subtopics=subtopics)

    def _build_search_intent(self, state: JsonObject, plan: SearchPlan) -> SearchIntent:
        """根据请求约束和 LLM 关键词构建稳定的检索意图。

        中文注释：这里只接收 LLM 已解析出的子主题和关键词，不再把 topic 做规则分词，
        这样检索方向完全来自模型明确给出的计划，排查问题时也更容易看懂。
        """

        request = state["request"]
        constraints = dict(getattr(request, "constraints", {}) or {})
        topic = str(getattr(request, "topic", "")).strip()
        return SearchIntent(
            topic=topic,
            keywords=list(plan.keywords),
            subtopics=list(plan.subtopics),
            excluded_terms=self._normalize_string_list(constraints.get("excluded_terms")),
            year_from=self._coerce_optional_int(constraints.get("year_from")),
            year_to=self._coerce_optional_int(constraints.get("year_to")),
            max_results=self._coerce_positive_int(constraints.get("max_results"), default=60),
            sources=self._normalize_string_list(constraints.get("sources")),
        )

    def _parse_subtopics(self, values: Any) -> list[SearchSubtopic]:
        """解析模型返回的子主题列表。"""

        if not isinstance(values, list):
            return []
        subtopics: list[SearchSubtopic] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            subtopic = str(item.get("subtopic") or "").strip()
            keyword = str(item.get("keyword") or "").strip()
            if not subtopic or not keyword:
                continue
            subtopics.append(SearchSubtopic(subtopic=subtopic, keyword=keyword))
        return subtopics

    def _keywords_from_subtopics(self, subtopics: list[SearchSubtopic]) -> list[str]:
        """从子主题检索式里整理出旧字段 keywords 需要的关键词列表。"""

        return self._deduplicate_terms([subtopic.keyword for subtopic in subtopics])

    def _keyword_expression_from_terms(self, keywords: list[str]) -> str:
        """把旧版关键词列表包装成 `(k1 and k2) or ...` 形式。"""

        cleaned = self._deduplicate_terms(keywords)
        groups: list[str] = []
        for index in range(0, len(cleaned), 2):
            pair = cleaned[index : index + 2]
            if len(pair) == 1:
                groups.append(f"({pair[0]})")
            else:
                groups.append(f"({pair[0]} and {pair[1]})")
        return " or ".join(groups)

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
