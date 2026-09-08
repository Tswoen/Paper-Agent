from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.paper_retrieval.models import PaperDocument


JsonObject = dict[str, Any]


@dataclass(slots=True)
class ReadNote:
    """保存仅根据标题和摘要整理出的论文笔记。"""

    main_question: str = ""
    methods: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    contributions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    main_results: list[str] = field(default_factory=list)
    short_summary: str = ""
    # 中文说明：证据等级由程序根据是否提供摘要确定，不再要求模型额外判断。
    evidence_level: str = "metadata"

    def to_dict(self) -> JsonObject:
        """把笔记转换为普通字典，方便写入状态和 JSON 文件。"""

        return asdict(self)

MATCH_LEVEL_FIELDS = (
    "research_question",
    "research_object_or_scene",
    "method_or_technical_route",
)

# DEPRECATED: 2026-09-07
# 原因：三维等级 + 固定评分表制度僵化（同一篇论文跨领域同形词也能拿高分），
#       相关性判断已改为模型直接输出二元结论 {relevant, reason}，程序不再算分。
# 替代方案：ReadRelevance.relevant / ReadRelevance.reason
# 计划移除：阅读阶段持续过滤机制稳定运行一段时间后清理
# # 中文说明：模型只能从这三个文字中选择，其他值统一按不匹配处理，避免错误内容影响名额分配。
# MATCH_LEVEL_VALUES = {"match", "partial_match", "not_match"}
#
# # 中文说明：总分只能由下面这张固定表计算，模型不会也不允许直接给总分。
# MATCH_LEVEL_SCORES = {
#     "research_question": {"match": 50, "partial_match": 25, "not_match": 0},
#     "research_object_or_scene": {"match": 30, "partial_match": 15, "not_match": 0},
#     "method_or_technical_route": {"match": 20, "partial_match": 10, "not_match": 0},
# }
#
#
# def normalize_match_levels(value: Any) -> dict[str, str]:
#     """整理模型返回的三个匹配程度，缺失或错误值都当作不匹配。"""
#
#     payload = value if isinstance(value, dict) else {}
#     return {
#         field_name: str(payload.get(field_name) or "not_match")
#         if str(payload.get(field_name) or "not_match") in MATCH_LEVEL_VALUES
#         else "not_match"
#         for field_name in MATCH_LEVEL_FIELDS
#     }
#
#
# def calculate_relevance_score(match_levels: Any) -> int:
#     """按固定表计算总分，保证全文筛选不受模型主观分数影响。"""
#
#     normalized = normalize_match_levels(match_levels)
#     return sum(MATCH_LEVEL_SCORES[field_name][normalized[field_name]] for field_name in MATCH_LEVEL_FIELDS)


# 中文说明：精读名额状态。discarded 表示被 AI 判定与主题无关而软丢弃，
# 不进下游分析/写作，但笔记照常落盘可审计。
READ_RELEVANCE_STATUSES = {
    "pending",
    "selected_for_deep_read",
    "not_selected_due_to_limit",
    "discarded",
}


@dataclass(slots=True)
class ReadRelevance:
    """保存模型对单篇论文的二元相关性判断与精读名额状态。

    中文注释：relevant 是模型"内部三维分析后给出的唯一结论"；reason 是判断依据，
    用于前端展示和事后审计。discarded 标记软丢弃（不物理删除）；uncertain 标记
    保底回捞回来的"相关性存疑"论文。
    """

    relevant: bool = True
    reason: str = ""
    discarded: bool = False
    uncertain: bool = False
    status: str = "pending"

    def to_dict(self) -> JsonObject:
        """把相关性判断转换为普通字典。"""

        return asdict(self)


@dataclass(slots=True)
class FullTextStatus:
    """保存全文下载、转换和索引的当前结果。"""

    status: str = "not_requested"
    reason: str = ""
    source_url: str | None = None
    source_path: str | None = None
    markdown_path: str | None = None
    page_count: int | None = None
    chunk_count: int = 0

    def to_dict(self) -> JsonObject:
        """把全文处理状态转换为普通字典。"""

        return asdict(self)


@dataclass(slots=True)
class PaperReadResult:
    """保存一篇论文从摘要阅读到全文入库的完整结果。"""

    paper: PaperDocument
    note: ReadNote = field(default_factory=ReadNote)
    relevance: ReadRelevance = field(default_factory=ReadRelevance)
    full_text: FullTextStatus = field(default_factory=FullTextStatus)
    extraction: JsonObject = field(default_factory=dict)
    # 中文说明：保存“全文提取结果的证据验证”结论，例如每个字段是否忠于原文、
    # 对应的原文引用和失败原因。没有启用验证时保持空字典。
    verification: JsonObject = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> JsonObject:
        """把单篇阅读结果转成可序列化字典。"""

        return {
            "paper": self.paper.to_dict(),
            "note": self.note.to_dict(),
            "relevance": self.relevance.to_dict(),
            "full_text": self.full_text.to_dict(),
            "extraction": dict(self.extraction),
            "verification": dict(self.verification),
            "warnings": list(self.warnings),
        }
