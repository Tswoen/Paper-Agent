from __future__ import annotations

import asyncio
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.llm import ModelConfig, ProviderSnapshot, SystemConfig, make_provider
from src.llm.base import LLMResponse
from src.models.read_models import ReadNote, ReadRelevance, normalize_match_levels
from src.paper_retrieval.models import PaperDocument

from .base import AgentContext, AgentSpec, BaseAgent
from .contracts import JsonObject
from .Prompts import READ_AGENT_SYSTEM_PROMPT


@dataclass(slots=True)
class AbstractReadResult:
    """保存摘要阅读 Agent 对单篇论文整理出的结果。

    中文注释：这里把笔记、相关性判断和提醒信息放在一个对象里，调用方不用记住
    tuple 里每个位置分别代表什么，阅读起来会直观一些。
    """

    note: ReadNote
    relevance: ReadRelevance
    warnings: list[str]

    def as_tuple(self) -> tuple[ReadNote, ReadRelevance, list[str]]:
        """兼容阅读节点原来的返回格式，减少节点侧改动。"""

        return self.note, self.relevance, list(self.warnings)


class ReadAgentModelUnavailableError(RuntimeError):
    """表示阅读 Agent 调模型失败，需要由阅读节点保存现场后暂停。"""

    def __init__(self, message: str, *, response: LLMResponse | None = None):
        """保存模型响应，方便节点层后续扩展更细的诊断信息。"""

        super().__init__(message)
        self.response = response


class ReadAgent(BaseAgent):
    """负责阅读摘要并给出论文与用户主题的三维匹配程度。

    中文注释：这个 Agent 只管“理解论文摘要”这一件事，不管下载全文、转 Markdown、
    建向量库和保存 checkpoint。那些步骤属于阅读节点的流程安排，留在 graph 层更清楚。
    """

    spec = AgentSpec(
        name="read_agent",
        role="read",
        description="根据论文标题和摘要整理阅读笔记，并给出三维匹配程度。",
        llm_profile="default_agent",
        input_keys=("paper", "topic"),
    )

    def __init__(self, context: AgentContext):
        """初始化 ReadAgent，并确保上下文里记录当前 Agent 的说明信息。"""

        context.spec = self.spec
        super().__init__(context)

    def _run(self, state: JsonObject) -> JsonObject:
        """同步读取一篇论文摘要，主要给旧同步流程或调试入口使用。"""

        paper = state["paper"]
        if not isinstance(paper, PaperDocument):
            raise ValueError("read_agent 需要传入 PaperDocument 类型的 paper")
        topic = str(state.get("topic") or "")
        constraints = dict(state.get("constraints") or {})
        result = self.read_abstract(paper, topic=topic, constraints=constraints)
        return {
            "note": result.note,
            "relevance": result.relevance,
            "warnings": result.warnings,
        }

    def read_abstract(self, paper: PaperDocument, *, topic: str, constraints: JsonObject, usage_callback: Any | None = None) -> AbstractReadResult:
        """同步阅读摘要，返回结构化笔记和相关性判断。"""

        if not (paper.abstract or "").strip():
            return self._metadata_only_result(paper)
        if self.context.llm is None:
            raise ReadAgentModelUnavailableError("未配置可用的阅读模型，请在模型设置中验证可用后继续执行。")
        try:
            response = self.context.llm.provider.chat_with_retry(
                self._abstract_messages(paper, topic, constraints),
                temperature=0,
            )
        except Exception as exc:
            raise ReadAgentModelUnavailableError(f"阅读模型调用失败，请验证当前模型可用后继续执行：{exc}") from exc
        self.report_usage(response, usage_callback)
        return self._result_from_response(paper, response)

    async def async_read_abstract(
        self,
        paper: PaperDocument,
        *,
        topic: str,
        constraints: JsonObject,
        semaphore: asyncio.Semaphore | None = None,
        usage_callback: Any | None = None,
    ) -> AbstractReadResult:
        """异步阅读摘要，供新版并发阅读节点直接调用。"""

        if not (paper.abstract or "").strip():
            return self._metadata_only_result(paper)
        if self.context.llm is None:
            raise ReadAgentModelUnavailableError("未配置可用的阅读模型，请在模型设置中验证可用后继续执行。")
        try:
            response = await self._call_provider_chat_async(
                self._abstract_messages(paper, topic, constraints),
                semaphore=semaphore,
            )
        except Exception as exc:
            raise ReadAgentModelUnavailableError(f"阅读模型调用失败，请验证当前模型可用后继续执行：{exc}") from exc
        self.report_usage(response, usage_callback)
        return self._result_from_response(paper, response)

    async def _call_provider_chat_async(
        self,
        messages: list[JsonObject],
        *,
        semaphore: asyncio.Semaphore | None,
    ) -> LLMResponse:
        """统一控制阅读模型调用，避免并发论文一起把同一个模型打满。"""

        if self.context.llm is None:
            raise ReadAgentModelUnavailableError("未配置可用的阅读模型，请在模型设置中验证可用后继续执行。")
        if semaphore is None:
            return await self._call_provider_chat_once(messages)
        async with semaphore:
            return await self._call_provider_chat_once(messages)

    async def _call_provider_chat_once(self, messages: list[JsonObject]) -> LLMResponse:
        """执行一次阅读模型调用，具体重试规则仍由 provider 自己统一处理。"""

        provider = self.context.llm.provider
        chat = getattr(provider, "chat", None)
        if callable(chat):
            response = chat(messages, temperature=0)
            if inspect.isawaitable(response):
                return await response
            return response
        return await asyncio.to_thread(provider.chat_with_retry, messages, temperature=0)

    def _result_from_response(self, paper: PaperDocument, response: LLMResponse) -> AbstractReadResult:
        """把模型返回整理成摘要阅读结果。"""

        if not response.ok:
            raise ReadAgentModelUnavailableError(self._model_unavailable_message(response), response=response)
        parsed = self._parse_model_result(response)
        if parsed is None:
            return self._fallback_abstract_note(paper, warning="阅读模型返回内容格式不符合要求，已改用保守摘要笔记")
        try:
            note, relevance, warnings = self._note_from_model(parsed)
            return AbstractReadResult(note=note, relevance=relevance, warnings=warnings)
        except Exception as exc:
            return self._fallback_abstract_note(paper, warning=f"阅读模型返回内容无法整理，已改用保守摘要笔记：{exc}")

    def _metadata_only_result(self, paper: PaperDocument) -> AbstractReadResult:
        """论文没有摘要时，只根据标题和基本信息生成保守结果。"""

        note = ReadNote(
            short_summary=f"《{paper.title}》只有标题和基本信息，暂无摘要可供整理。",
            evidence_level="metadata",
        )
        relevance = ReadRelevance()
        return AbstractReadResult(note=note, relevance=relevance, warnings=[])

    def _fallback_abstract_note(self, paper: PaperDocument, *, warning: str) -> AbstractReadResult:
        """模型返回内容不可靠时，生成一份保守笔记继续处理当前论文。"""

        abstract = (paper.abstract or "").strip()
        title = paper.title.strip() or "未命名论文"
        summary_source = abstract or title
        short_summary = summary_source[:500]
        if len(summary_source) > 500:
            short_summary += "..."
        note = ReadNote(
            short_summary=f"《{title}》的模型阅读结果不可用，以下仅保留论文原始摘要片段：{short_summary}",
            evidence_level="abstract" if abstract else "metadata",
        )
        relevance = ReadRelevance()
        return AbstractReadResult(note=note, relevance=relevance, warnings=[warning])

    def _abstract_messages(self, paper: PaperDocument, topic: str, constraints: JsonObject) -> list[JsonObject]:
        """构造摘要阅读提示，要求模型只使用提供内容并返回固定 JSON。"""

        payload = {
            "用户主题": topic,
            "用户要求": constraints,
            "论文": {
                "标题": paper.title,
                "摘要": paper.abstract,
            },
        }
        # 中文说明：阅读规则集中管理，确保同步和异步阅读使用完全相同的提示词。
        instruction = READ_AGENT_SYSTEM_PROMPT
        return [{"role": "system", "content": instruction}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]

    def _parse_model_result(self, response: LLMResponse) -> JsonObject | None:
        """从模型文本中取出 JSON 对象，格式不合格时返回空值。"""

        if not response.ok or not response.content.strip():
            return None
        text = response.content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _note_from_model(self, payload: JsonObject) -> tuple[ReadNote, ReadRelevance, list[str]]:
        """把模型 JSON 整理成项目内部使用的阅读笔记和相关性判断。"""

        note = ReadNote(
            main_question=self._text_value(payload.get("main_question")),
            methods=self._string_list(payload.get("methods")),
            datasets=self._string_list(payload.get("datasets")),
            contributions=self._string_list(payload.get("contributions")),
            limitations=self._string_list(payload.get("limitations")),
            main_results=self._string_list(payload.get("main_results")),
            short_summary=self._text_value(payload.get("short_summary"))[:800],
            evidence_level="abstract",
        )
        relevance = ReadRelevance(match_levels=normalize_match_levels(payload.get("match_levels")))
        return note, relevance, []

    def _model_unavailable_message(self, response: LLMResponse) -> str:
        """把 provider 返回的错误整理成人能看懂的阅读模型不可用提示。"""

        status = response.error_status_code or response.error_kind or response.finish_reason or "unknown"
        detail = response.content or response.error_code or "模型接口没有返回可用内容"
        return f"阅读模型当前不可用（{status}）：{detail}。请在模型设置中验证可用后继续执行。"

    def _text_value(self, value: Any) -> str:
        """把模型字段安全整理成字符串。"""

        return str(value).strip() if isinstance(value, str) else ""

    def _string_list(self, value: Any) -> list[str]:
        """把模型字段安全整理成字符串列表。"""

        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                result.append(text)
        return result

def load_read_agent_llm(
    agent_name: str | None = None,
    model_config_path: str | Path = "config/model.json",
    system_config_path: str | Path = "config/system.yaml",
    *,
    client: Any | None = None,
) -> ProviderSnapshot | None:
    """读取模型配置并装配 ReadAgent 使用的默认 LLM。"""

    model_path = Path(model_config_path)
    if not model_path.exists():
        return None
    try:
        data = json.loads(model_path.read_text(encoding="utf-8"))
        system = SystemConfig.load(system_config_path)
        config = ModelConfig.from_dict(data, system)
        resolved_agent_name = agent_name or ReadAgent.spec.llm_profile
        return make_provider(config, resolved_agent_name, client=client)
    except Exception:
        return None


def build_read_agent(llm: ProviderSnapshot | None | str = "auto", usage_callback: Any | None = None) -> ReadAgent:
    """构建一个最小可用的 ReadAgent。"""

    resolved_llm = load_read_agent_llm() if llm == "auto" else llm
    context = AgentContext(spec=ReadAgent.spec, llm=resolved_llm, usage_callback=usage_callback)
    return ReadAgent(context)
