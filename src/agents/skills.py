from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SkillSpec:
    """描述 Agent 可加载的领域技能，例如检索式改写、证据分级、学术写作风格等。"""

    name: str
    description: str
    prompt_path: str | None = None
    metadata: dict[str, str] | None = None


class SkillRegistry:
    """轻量技能注册表；当前只声明能力，后续可接 prompt 模板或外部 skill 包。"""

    def __init__(self, skills: dict[str, SkillSpec] | None = None):
        self._skills = dict(skills or {})

    def register(self, skill: SkillSpec) -> None:
        self._skills[skill.name] = skill

    def require(self, name: str) -> SkillSpec:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise ValueError(f"unknown skill: {name}") from exc

    def select(self, names: tuple[str, ...]) -> "SkillRegistry":
        return SkillRegistry({name: self.require(name) for name in names})

    def list(self) -> list[SkillSpec]:
        return list(self._skills.values())
