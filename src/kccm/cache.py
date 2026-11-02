"""关键词缓存：dict[keyword, CacheEntry]，TTL 为主、LFU 为辅。"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Iterator

from .config import Config
from .models import (
    CacheEntry,
    Candidate,
    HitKind,
    SweepReport,
    UpsertKind,
    UpsertRecord,
)


class KeywordCache:
    def __init__(self, cfg: Config | None = None) -> None:
        self._cfg = cfg or Config()
        self._entries: dict[str, CacheEntry] = {}
        self._turn = 0
        self._pinned: set[str] = set()
        self.events: list[dict] = []

    # ---------- 基础访问 ----------
    @property
    def turn(self) -> int:
        return self._turn

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, keyword: str) -> bool:
        return keyword in self._entries

    def __iter__(self) -> Iterator[CacheEntry]:
        return iter(self._entries.values())

    def get(self, keyword: str) -> CacheEntry | None:
        return self._entries.get(keyword)

    def keys(self) -> set[str]:
        return set(self._entries)

    def ttl_of(self, entry: CacheEntry) -> int:
        return entry.ttl_remaining(self._turn, self._cfg.ttl_max)

    def find_by_slot(self, slot: str) -> CacheEntry | None:
        for entry in self._entries.values():
            if entry.slot == slot:
                return entry
        return None

    def entries_where(self, predicate) -> list[CacheEntry]:
        return [e for e in self._entries.values() if predicate(e)]

    # ---------- 回合推进 ----------
    def begin_turn(self, turn_id: int) -> SweepReport:
        """回合开始：更新回合号 → 清 TTL → 清空上一轮 pin。"""
        self._turn = turn_id
        self._pinned = set()
        return self.sweep_ttl()

    def set_pinned(self, keywords: Iterable[str]) -> None:
        # 允许包含尚未插入的新关键词：pin 是按名字生效的，插入时必须立刻受保护
        self._pinned = set(keywords)

    def sweep_ttl(self) -> SweepReport:
        report = SweepReport()
        for keyword, entry in list(self._entries.items()):
            if keyword in self._pinned:
                continue
            if self.ttl_of(entry) == 0:
                del self._entries[keyword]
                report.expired.append(keyword)
                report.details.append(
                    {"keyword": keyword, "reason": "TTL_EXPIRED", "ttl": 0,
                     "hit": entry.hit_count, "slot": entry.slot}
                )
                self.events.append(report.details[-1])
        return report

    # ---------- 写入 ----------
    def touch(self, keyword: str, kind: HitKind) -> bool:
        entry = self._entries.get(keyword)
        if entry is None:
            return False
        # 关联命中只续命不加频：防止泛化词借事件框架刷高 LFU
        new_hits = entry.hit_count if kind is HitKind.ASSOCIATIVE else entry.hit_count + 1
        self._entries[keyword] = replace(
            entry, last_hit_turn=self._turn, hit_count=new_hits
        )
        self.events.append(
            {"keyword": keyword, "event": "HIT", "kind": kind.value,
             "ttl": self.ttl_of(self._entries[keyword]), "hit": new_hits, "turn": self._turn}
        )
        return True

    def upsert_new(self, cand: Candidate, summary: str) -> UpsertRecord:
        keyword = cand.keyword
        slot = cand.slot
        if self.find_by_slot(slot) is not None:
            # 违反 I2 的最后一道闸：宁可降级为独立槽位，也不覆盖未确认的关系
            slot = f"{slot}#{keyword}"
        entry = CacheEntry(
            keyword=keyword,
            surface=cand.surface,
            role=cand.role,
            slot=slot,
            hit_count=1,
            first_turn=self._turn,
            last_hit_turn=self._turn,
            summary=summary[: self._cfg.summary_max_chars],
            summary_turn=self._turn,
        )
        self._entries[keyword] = entry
        self.events.append(
            {"keyword": keyword, "event": "UPSERT_NEW", "slot": slot,
             "role": cand.role.value, "ttl": self._cfg.ttl_max, "hit": 1, "turn": self._turn}
        )
        return UpsertRecord(UpsertKind.NEW, keyword, slot, cand.role, "NEW")

    def upsert_overwrite(
        self, old_keyword: str, cand: Candidate, summary: str, reason: str
    ) -> UpsertRecord:
        old = self._entries.pop(old_keyword, None)
        if old is None:
            return self.upsert_new(cand, summary)

        entry = CacheEntry(
            keyword=cand.keyword,
            surface=cand.surface,
            role=cand.role,
            slot=old.slot,
            hit_count=old.hit_count + 1,
            first_turn=old.first_turn,
            last_hit_turn=self._turn,
            summary=summary[: self._cfg.summary_max_chars],
            summary_turn=self._turn,
            superseded=old.superseded + (old.keyword,),
        )
        self._entries[cand.keyword] = entry
        self.events.append(
            {"keyword": cand.keyword, "event": "OVERWRITE", "old": old_keyword,
             "slot": old.slot, "reason": reason, "ttl": self._cfg.ttl_max,
             "hit": entry.hit_count, "superseded": list(entry.superseded), "turn": self._turn}
        )
        return UpsertRecord(
            UpsertKind.OVERWRITE, cand.keyword, old.slot, cand.role, reason, entry.superseded
        )

    # ---------- 容量淘汰（LFU） ----------
    def enforce_capacity(self) -> SweepReport:
        report = SweepReport()
        while len(self._entries) > self._cfg.max_entries:
            candidates = [e for e in self._entries.values() if e.keyword not in self._pinned]
            if not candidates:
                # I4：pin 集不可删。宁可临时超限，也不破坏本轮语义。
                self.events.append(
                    {"event": "CAPACITY_SKIPPED", "reason": "ALL_PINNED",
                     "len": len(self._entries), "turn": self._turn}
                )
                break

            min_ttl = min(self.ttl_of(e) for e in candidates)
            bucket = [e for e in candidates if self.ttl_of(e) <= min_ttl + self._cfg.ttl_tolerance]
            victim = min(
                bucket,
                key=lambda e: (e.hit_count, e.last_hit_turn, e.first_turn, e.keyword),
            )
            # 淘汰原因的三种真实形态：TTL 独占 / LFU 决胜 / LFU 内部再平局
            min_hit = min(e.hit_count for e in bucket)
            tied = [e for e in bucket if e.hit_count == min_hit]
            if len(bucket) == 1:
                reason = "TTL_DOMINANT"
            elif len(tied) == 1:
                reason = "LFU_EVICT"
            else:
                reason = "LFU_TIEBREAK"
            del self._entries[victim.keyword]
            report.evicted.append(victim.keyword)
            report.details.append(
                {"keyword": victim.keyword, "reason": reason,
                 "ttl": self.ttl_of(victim), "hit": victim.hit_count,
                 "bucket": [e.keyword for e in bucket], "turn": self._turn}
            )
            self.events.append(report.details[-1])
        return report

    # ---------- 读取 ----------
    def snapshot(self) -> list[CacheEntry]:
        return sorted(
            self._entries.values(),
            key=lambda e: (-self.ttl_of(e), -e.hit_count, e.keyword),
        )

    def export(self) -> list[dict]:
        return [e.to_dict(self._turn, self._cfg.ttl_max) for e in self.snapshot()]

    def state(self) -> dict[str, tuple[int, int]]:
        """(ttl, hit_count) 快照，供测试断言。"""
        return {e.keyword: (self.ttl_of(e), e.hit_count) for e in self.snapshot()}
