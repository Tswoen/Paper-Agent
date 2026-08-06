from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.llm import ModelConfig, ProviderSnapshot, SystemConfig, make_provider

from .base import AgentContext, AgentSpec, BaseAgent
from .contracts import JsonObject
from .Prompts import ANALYSE_OVERALL_SYSTEM_PROMPT, ANALYSE_SUBTOPIC_SYSTEM_PROMPT


@dataclass(slots=True)
class AnalyseModelResult:
    """保存一次分析模型调用的结果。

    中文说明：
    parsed 是已经解析出来的 JSON；如果为 None，说明模型不可用、调用失败或格式不对。
    reason 用简单中文说明失败原因，供分析节点决定当前阶段是保守处理还是直接中止。
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
        # 分析任务需要更强的模型；如果用户没有配置 solar_agent，模型工厂会回退到 default_agent。
        llm_profile="solar_agent",
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

    async def async_analyse_subtopic(self, *, topic: str, group: JsonObject, usage_callback: Any | None = None) -> AnalyseModelResult:
        """分析一个子主题，并返回解析后的 JSON。"""

        if self.context.llm is None:
            return AnalyseModelResult(reason="未配置可用分析模型")
        try:
            response = await self.context.llm.provider.chat(
                _subtopic_messages(topic=topic, group=group),
                temperature=0.2,
                reasoning_effort="medium",
            )
        except Exception as exc:
            return AnalyseModelResult(reason=f"分析模型调用失败：{exc}")
        self.report_usage(response, usage_callback)
        return _parse_response(response)

    async def async_analyse_overall(self, *, topic: str, subtopic_analyses: list[JsonObject], usage_callback: Any | None = None) -> AnalyseModelResult:
        """综合所有子主题分析，并返回解析后的 JSON。"""

        if self.context.llm is None:
            return AnalyseModelResult(reason="未配置可用分析模型")
        try:
            response = await self.context.llm.provider.chat(
                _overall_messages(topic=topic, subtopic_analyses=subtopic_analyses),
                temperature=0.2,
                reasoning_effort="medium",
            )
        except Exception as exc:
            return AnalyseModelResult(reason=f"分析模型调用失败：{exc}")
        self.report_usage(response, usage_callback)
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
        # 模型工厂会把未配置的 luna_agent 或 solar_agent 自动替换为 default_agent。
        return make_provider(config, resolved_agent_name, client=client)
    except Exception:
        # 中文说明：分析模型配置不可用时返回 None，让分析节点生成结构稳定的兜底报告。
        return None


def build_analyse_agent(llm: ProviderSnapshot | None | str = "auto", usage_callback: Any | None = None) -> AnalyseAgent:
    """构建一个最小可用的 AnalyseAgent。"""

    resolved_llm = load_analyse_agent_llm() if llm == "auto" else llm
    context = AgentContext(spec=AnalyseAgent.spec, llm=resolved_llm, usage_callback=usage_callback)
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
            # 中文说明：子主题分析的系统规则统一放在 Prompts.py。
            "content": ANALYSE_SUBTOPIC_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "任务": "按子主题分析阅读节点产出的论文结构化摘要",
                    "用户综述主题": topic,
                    "子主题": group["subtopic"],
                    # "检索关键词": group["search_keyword"],
                    # "允许引用的paperId": paper_ids,
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
            "paperIds": item.get("paperIds", []),
            "研究现状": item.get("研究现状", ""),
            "一致点": item.get("一致点", []),
            "矛盾点": item.get("矛盾点", ""),
            "研究空白": item.get("研究空白", ""),
            "时间线演化": item.get("时间线演化", ""),
            "技术方法栈演变": item.get("技术方法栈演变", ""),
        }
        for item in subtopic_analyses
    ]
    return [
        {
            "role": "system",
            # 中文说明：全局分析单独使用专门提示词，强调跨主题归纳和证据可追溯。
            "content": ANALYSE_OVERALL_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "任务": "根据各子主题分析摘要做全局综合分析",
                    "用户综述主题": topic,
                    "输出要求": _overall_analysis_schema_hint(),
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
        "研究现状": "详细说明当前研究进展、主要发现和代表性工作；相关句子必须使用 [paperId] 引用",
        "一致点": ["一个一致点用一整段文字说明，并使用 [paperId] 引用"],
        "矛盾点": "用一整段文字说明不同论文的观点、结果或适用条件为何不同；没有明确矛盾时如实说明；必须使用 [paperId] 引用",
        "研究空白": "用一整段文字说明尚未解决的问题、数据或方法不足；必须使用 [paperId] 引用",
        "时间线演化": "用一整段文字按时间说明研究如何演变；必须使用 [paperId] 引用",
        "技术方法栈演变": "方法从早期到近期怎么变化，必须出现 [paperId]",
    }


def _overall_analysis_schema_hint() -> JsonObject:
    """给综合分析模型的八部分独立输出格式说明。"""

    citation_rule = "使用 Markdown 文本详细作答；每个关键结论都要在句子中引用输入中的 [paperId]，没有证据时明确说明证据不足"
    return {
        "领域整体研究概况": f"{citation_rule}；概括整体研究热度、覆盖范围、核心方向、成熟度、核心命题和争议总览",
        "领域全域共性研究共识": f"{citation_rule}；只保留跨多个子主题或全场景通用的共识，并按研究价值、应用特征、运行规律、基础认知等维度归纳",
        "领域核心研究争议与矛盾体系": f"{citation_rule}；说明对立观点、支撑文献、争议底层逻辑，以及场景、约束、视角或方法差异造成的适用边界",
        "领域系统性研究空白与局限": f"{citation_rule}；必须从研究地域、研究视角、研究时长、研究方法、研究内容、研究场景六个维度整理全域缺口",
        "领域研究时序演化脉络": f"{citation_rule}；依据论文发表时间划分阶段，说明每阶段的整体特征、关注重点、突破和不足",
        "领域技术与研究方法迭代脉络": f"{citation_rule}；梳理主流技术和方法的更替路径、适配场景、迭代动因、优势与局限",
        "各子主题横向差异对比分析": f"{citation_rule}；比较各子主题的成熟度、共识统一度、争议集中度、研究缺口体量、技术应用深度和研究丰富度，区分强弱板块",
        "领域整体总结与研究展望": f"{citation_rule}；总结核心结论、整体价值和约束边界，给出通用实践启示及与研究空白对应的可落地未来方向",
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
