"""数据模型：字段冻结，伪代码与 trace 三处必须同名。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# TTL 默认值置于模型层，使 CacheEntry.ttl_remaining() 可被单独调用
DEFAULT_TTL_MAX = 10


class SlotRole(str, Enum):
    """槽位角色的封闭枚举。无法归类者不入缓存。"""

    SUBJECT = "subject"
    OBJECT = "object"
    QUANTITY = "quantity"
    RECIPIENT = "recipient"
    ATTRIBUTE = "attribute"
    CONSTRAINT = "constraint"
    TIME = "time"
    LOCATION = "location"


class HitKind(str, Enum):
    """三级命中。只有前两级递增 hit_count。"""

    LITERAL = "literal"
    NORMALIZED = "normalized"
    ASSOCIATIVE = "associative"


class UpsertKind(str, Enum):
    NEW = "new"
    OVERWRITE = "overwrite"
    HIT = "hit"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class CacheEntry:
    """关键词缓存条目。

    存活回合的**唯一事实源是 last_hit_turn**，ttl 由 ttl_remaining() 派生，
    不单独存储——双计数在长会话中必然漂移，且会出现负值。
    """

    keyword: str
    surface: str
    role: SlotRole
    slot: str
    hit_count: int
    first_turn: int
    last_hit_turn: int
    summary: str
    summary_turn: int = 0
    superseded: tuple[str, ...] = ()

    def ttl_remaining(self, current_turn: int, ttl_max: int = DEFAULT_TTL_MAX) -> int:
        return max(0, ttl_max - (current_turn - self.last_hit_turn))

    def to_dict(self, current_turn: int, ttl_max: int = DEFAULT_TTL_MAX) -> dict[str, Any]:
        return {
            "keyword": self.keyword,
            "surface": self.surface,
            "role": self.role.value,
            "slot": self.slot,
            "hit_count": self.hit_count,
            "first_turn": self.first_turn,
            "last_hit_turn": self.last_hit_turn,
            "ttl": self.ttl_remaining(current_turn, ttl_max),
            "summary": self.summary,
            "summary_turn": self.summary_turn,
            "superseded": list(self.superseded),
        }


@dataclass(frozen=True)
class Candidate:
    """提取阶段的候选词，尚未写入缓存。"""

    keyword: str
    surface: str
    role: SlotRole
    slot: str
    source: str  # number | noun | diff
    start: int = -1
    end: int = -1


@dataclass(frozen=True)
class UpsertRecord:
    kind: UpsertKind
    keyword: str
    slot: str
    role: SlotRole
    reason: str = ""
    superseded: tuple[str, ...] = ()


@dataclass
class SweepReport:
    expired: list[str] = field(default_factory=list)
    evicted: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractReport:
    rule: tuple[Candidate, ...]
    diff: tuple[str, ...]
    anaphora: bool = False
