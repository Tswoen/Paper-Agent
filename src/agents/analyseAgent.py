from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.llm import ModelConfig, ProviderSnapshot, SystemConfig, make_provider

from .base import AgentContext, AgentSpec, BaseAgent
from .contracts import JsonObject


@dataclass(slots=True)
class AnalyseModelResult:
    """保存一次分析模型调用的结果。

    中文说明：
    parsed 是已经解析出来的 JSON；如果为 None，说明模型不可用、调用失败或格式不对。
    reason 用简单中文说明失败原因，方便分析节点生成兜底报告。
    """

    parsed: JsonObject | None = None
    raw_model_output: str = ""
    reason: str = ""


class AnalyseAgent(BaseAgent):
    """负责让大模型完成论文分析的 Agent。

    中文说明：
    这个 Agent 只关心“怎么提示模型、怎么解析模型结果”。
    至于论文怎么分组、报告怎么保存，仍然放在 analyse_node 里处理。
    """

    spec = AgentSpec(
        name="analyse_agent",
        role="analyze",
        description="根据阅读节点产出的结构化摘要，分析子主题和全局研究现状。",
        llm_profile="analyse_agent",
        skills=(),
        input_keys=("request",),
    )

    def __init__(self, context: AgentContext):
        """初始化 AnalyseAgent，并保存当前 Agent 的配置说明。"""

        context.spec = self.spec
        super().__init__(context)

    def _run(self, state: JsonObject) -> JsonObject:
        """BaseAgent 要求实现同步入口，但分析节点当前只使用异步方法。"""

        raise NotImplementedError("AnalyseAgent 请使用 async_analyse_subtopic 或 async_analyse_overall")

    async def async_analyse_subtopic(self, *, topic: str, group: JsonObject) -> AnalyseModelResult:
        """分析一个子主题，并返回解析后的 JSON。"""

        if self.context.llm is None:
            return AnalyseModelResult(reason="未配置可用分析模型")
        try:
            response = await self.context.llm.provider.chat(
                _subtopic_messages(topic=topic, group=group),
                temperature=0.2,
                max_tokens=4000,
                reasoning_effort="medium",
            )
        except Exception as exc:
            return AnalyseModelResult(reason=f"分析模型调用失败：{exc}")
        return _parse_response(response)

    async def async_analyse_overall(self, *, topic: str, subtopic_analyses: list[JsonObject]) -> AnalyseModelResult:
        """综合所有子主题分析，并返回解析后的 JSON。"""

        if self.context.llm is None:
            return AnalyseModelResult(reason="未配置可用分析模型")
        try:
            response = await self.context.llm.provider.chat(
                _overall_messages(topic=topic, subtopic_analyses=subtopic_analyses),
                temperature=0.2,
                max_tokens=4000,
                reasoning_effort="medium",
            )
        except Exception as exc:
            return AnalyseModelResult(reason=f"分析模型调用失败：{exc}")
        return _parse_response(response)


def load_analyse_agent_llm(
    agent_name: str | None = None,
    model_config_path: str | Path = "config/model.json",
    system_config_path: str | Path = "config/system.yaml",
    *,
    client: Any | None = None,
) -> ProviderSnapshot | None:
    """根据本地模型配置装配 AnalyseAgent 使用的默认 LLM。"""

    model_path = Path(model_config_path)
    if not model_path.exists():
        return None
    try:
        data = json.loads(model_path.read_text(encoding="utf-8"))
        system = SystemConfig.load(system_config_path)
        config = ModelConfig.from_dict(data, system)
        resolved_agent_name = agent_name or AnalyseAgent.spec.llm_profile
        # 中文说明：分析节点必须使用自己的模型配置。
        # 如果配置里没有 analyse_agent，就不要悄悄退回 default_agent，避免用户误以为已经分开配置。
        if resolved_agent_name not in config.agents:
            return None
        return make_provider(config, resolved_agent_name, client=client)
    except Exception:
        # 中文说明：分析模型配置不可用时返回 None，让分析节点生成结构稳定的兜底报告。
        return None


def build_analyse_agent(llm: ProviderSnapshot | None | str = "auto") -> AnalyseAgent:
    """构建一个最小可用的 AnalyseAgent。"""

    resolved_llm = load_analyse_agent_llm() if llm == "auto" else llm
    context = AgentContext(spec=AnalyseAgent.spec, llm=resolved_llm)
    return AnalyseAgent(context)


def _parse_response(response: Any) -> AnalyseModelResult:
    """把模型响应解析为 JSON。"""

    raw_model_output = str(getattr(response, "content", "") or "")
    if not getattr(response, "ok", False):
        return AnalyseModelResult(
            raw_model_output=raw_model_output,
            reason=raw_model_output or str(getattr(response, "error_kind", "") or "分析模型调用失败"),
        )
    parsed = _extract_json_object(raw_model_output)
    if parsed is None:
        return AnalyseModelResult(raw_model_output=raw_model_output, reason="模型没有返回可解析的 JSON")
    return AnalyseModelResult(parsed=parsed, raw_model_output=raw_model_output)


def _subtopic_messages(*, topic: str, group: JsonObject) -> list[JsonObject]:
    """为单个子主题生成提示词。"""

    paper_ids = [paper["paperId"] for paper in group["papers"]]
    return [
        {
            "role": "system",
            "content": (
                "你是论文综述分析助手。请先认真思考不同论文之间的关系，但最终只输出 JSON，"
                "不要输出 Markdown、解释文字或代码块。所有判断都必须带论文引用，引用格式使用 [paperId]。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "任务": "按子主题分析阅读节点产出的论文结构化摘要",
                    "用户综述主题": topic,
                    "子主题": group["subtopic"],
                    "检索关键词": group["search_keyword"],
                    "允许引用的paperId": paper_ids,
                    "输出要求": _analysis_schema_hint(),
                    "论文结构化摘要": group["papers"],
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]


def _overall_messages(*, topic: str, subtopic_analyses: list[JsonObject]) -> list[JsonObject]:
    """为全局综合分析生成提示词。"""

    summaries = [
        {
            "subtopic": item.get("subtopic"),
            "paper_count": item.get("paper_count"),
            "subtopic_summary": item.get("subtopic_summary") or item.get("研究现状"),
            "一致点": item.get("一致点", []),
            "矛盾点": item.get("矛盾点", []),
            "研究空白": item.get("研究空白", []),
            "时间线演化": item.get("时间线演化", []),
            "研究热度趋势": item.get("研究热度趋势", ""),
        }
        for item in subtopic_analyses
    ]
    return [
        {
            "role": "system",
            "content": (
                "你是论文综述分析助手。请先认真比较各子主题之间的联系和差异，最终只输出 JSON。"
                "所有结论都尽量保留来自子主题分析中的 [paperId] 引用。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "任务": "根据各子主题分析摘要做全局综合分析",
                    "用户综述主题": topic,
                    "输出要求": _analysis_schema_hint(),
                    "各子主题分析摘要": summaries,
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]


def _analysis_schema_hint() -> JsonObject:
    """给模型看的输出格式说明。"""

    return {
        "subtopic": "子主题名称，全局综合分析时填“全局综合分析”",
        "search_keyword": "检索关键词，没有就填空字符串",
        "paper_count": "本次分析覆盖的论文数量",
        "paperIds": ["本次分析实际使用的 paperId 列表"],
        "subtopic_summary": "一段话概括研究现状，必须出现 [paperId]",
        "研究现状": "和 subtopic_summary 含义相同，可以写得稍详细，必须出现 [paperId]",
        "一致点": [{"point": "共识点", "paperIds": ["paperId1"], "evidence": "一句话说明，必须出现 [paperId]"}],
        "矛盾点": [
            {
                "title": "矛盾点标题",
                "positions": [{"claim": "某一方观点", "paperIds": ["paperId1"], "evidence": "必须出现 [paperId]"}],
                "explanation": "这个矛盾目前怎么解释，必须出现 [paperId]",
            }
        ],
        "研究空白": [{"type": "地区/时间/视角/方法/其他", "gap": "空白说明，必须出现 [paperId]", "paperIds": ["paperId1"]}],
        "时间线演化": [{"stage": "阶段名称", "years": "年份范围", "feature": "阶段特征，必须出现 [paperId]", "paperIds": ["paperId1"]}],
        "技术方法栈演变": "方法从早期到近期怎么变化，必须出现 [paperId]",
        "研究方法趋势": "主流研究方法怎么变化，必须出现 [paperId]",
        "研究热度趋势": "近2-3年论文数量变化趋势和原因，必须出现 [paperId]",
        "relationships": [{"relation": "与其他主题或变量的关系", "paperIds": ["paperId1"], "evidence": "必须出现 [paperId]"}],
    }


def _extract_json_object(text: str) -> JsonObject | None:
    """从模型输出里用正则提取 JSON 对象。"""

    candidates: list[str] = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    loose = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if loose:
        candidates.append(loose.group(1))
    candidates.append(text)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
