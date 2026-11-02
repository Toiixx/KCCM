"""回合编排：begin_turn → extract → upsert/hit → sweep → compose → call。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cache import KeywordCache
from .composer import ComposeResult, Composer
from .config import Config
from .differ import diff_extract
from .extractor import REPLACE_VERBS, Extractor
from .llm import LLMClient, get_llm
from .models import (
    Candidate,
    HitKind,
    SlotRole,
    SweepReport,
    UpsertKind,
    UpsertRecord,
)
from .tagger import get_tagger
from .trace import Tracer

ASSOC_TEXT_TRIGGERS = ("各", "每人", "分别", "均分", "平均分")


@dataclass(frozen=True)
class TurnResult:
    turn: int
    question: str
    extracted: tuple[str, ...]
    diff_tokens: tuple[str, ...]
    upserts: tuple[UpsertRecord, ...]
    expired: tuple[str, ...]
    evicted: tuple[str, ...]
    prompt: str
    answer: str
    tokens: int
    level: str
    snapshot: tuple[dict, ...]
    mode: str = "kccm"


@dataclass
class Session:
    cfg: Config = field(default_factory=Config)
    llm: LLMClient | None = None
    tagger: object | None = None
    tracer: Tracer | None = None

    def __post_init__(self) -> None:
        self.cache = KeywordCache(self.cfg)
        self.composer = Composer(self.cfg)
        self.extractor = Extractor(self.cfg, tagger=self.tagger or get_tagger())
        self.llm = self.llm or get_llm()
        self._turn = 0
        self._last_question = ""
        self._last_mode: str | None = None
        self._last_raw_tokens: int | None = None
        self._dialogue: list[tuple[str, str]] = []
        self.history: list[TurnResult] = []

    # ---------- 主流程 ----------
    def ask(self, question: str) -> TurnResult:
        self._turn += 1
        turn = self._turn

        ttl_sweep = self.cache.begin_turn(turn)

        # 短会话旁路：fixed 按轮次，auto 每轮按实测 token 择优
        bypass = self.cfg.hybrid_mode == "fixed" and turn <= self.cfg.hybrid_turns
        # 命中旁路时，是否连缓存也不建（字面版 `if turn < N: 直接存入`）
        build_cache = not (bypass and self.cfg.hybrid_skip_cache)

        rule = self.extractor.rule_extract(question) if build_cache else []
        rule_keywords = {c.keyword for c in rule} if build_cache else set()

        upserts: list[UpsertRecord] = []
        delta: set[str] = set()
        anaphora = False
        capacity = SweepReport()

        if build_cache:
            self.cache.set_pinned(rule_keywords)
            delta, anaphora = diff_extract(
                question,
                self._last_question,
                self.cache.keys(),
                rule_keywords,
                self.extractor.tagger,
            )
            for cand in rule:
                upserts.append(self._apply(cand, question))
            if delta:
                diff_text = " ".join(sorted(delta))
                for cand in self.extractor.rule_extract(diff_text, relaxed=True):
                    if cand.keyword in rule_keywords:
                        continue
                    upserts.append(self._apply(cand, question, source="diff"))
            self._apply_associative(question, upserts, anaphora)
            capacity = self.cache.enforce_capacity()

        composed, mode = self._select_prompt(
            turn, question, rule_keywords, bypass
        )

        answer = self.llm.complete(composed.prompt)
        self._dialogue.append((question, answer))

        result = TurnResult(
            turn=turn,
            question=question,
            extracted=tuple(c.keyword for c in rule),
            diff_tokens=tuple(sorted(delta)),
            upserts=tuple(upserts),
            expired=tuple(ttl_sweep.expired),
            evicted=tuple(capacity.evicted),
            prompt=composed.prompt,
            answer=answer,
            tokens=composed.tokens,
            level=composed.level,
            mode=mode,
            snapshot=tuple(self.cache.export()),
        )
        self.history.append(result)
        self._trace(result, composed, ttl_sweep, capacity)
        self._last_question = question
        self._last_mode = mode.split("+")[0]
        return result

    def _dialogue_slice(self) -> list[tuple[str, str]]:
        recent = self.cfg.hybrid_recent
        return self._dialogue if recent == 0 else self._dialogue[-recent:]

    def _select_prompt(
        self, turn: int, question: str, pinned: set[str], bypass: bool
    ) -> tuple[ComposeResult, str]:
        """决定本轮走原文还是走缓存，并在切回时可选地补一次桥接。"""
        mode_cfg = self.cfg.hybrid_mode
        if not bypass and mode_cfg != "auto":
            return self._cached_prompt(question, pinned)

        raw = self.composer.compose_raw(question, self._dialogue_slice())
        previous_raw, self._last_raw_tokens = self._last_raw_tokens, raw.tokens
        if bypass:
            return raw, "raw"

        cached = self.composer.compose(question, self.cache, pinned)
        if not self._prefer_cached(raw.tokens, cached.tokens, previous_raw):
            return raw, "raw"
        return self._cached_prompt(question, pinned, raw_tokens=raw.tokens)

    def _cached_prompt(
        self, question: str, pinned: set[str], raw_tokens: int | None = None
    ) -> tuple[ComposeResult, str]:
        """切回缓存。若刚离开原文模式，可选地补一次上一轮原文。"""
        bridge = None
        if self._last_mode == "raw" and self.cfg.bridge_once and self._dialogue:
            bridge = self._dialogue[-1]

        if bridge is not None:
            composed = self.composer.compose(question, self.cache, pinned, bridge=bridge)
            # 桥接不得把省下的 token 又吃回去：比原文还贵就不加
            if raw_tokens is None or composed.tokens <= raw_tokens:
                return composed, "kccm+bridge"

        return self.composer.compose(question, self.cache, pinned), "kccm"

    def _prefer_cached(
        self, raw_tokens: int, cached_tokens: int, previous_raw: int | None = None
    ) -> bool:
        """择优：带死区 + 一步预测。

        只看当下会在临界点上踩晚一拍——本轮只贵 1%，下一轮就贵 14%。
        用上一轮的增量外推下一轮，能提前一轮切换。
        """
        previous = self._last_mode
        margin = 1.0 - self.cfg.switch_hysteresis

        projected = raw_tokens
        if self.cfg.hybrid_lookahead and previous_raw is not None:
            projected = raw_tokens + (raw_tokens - previous_raw)
        # 取两者较大者：若趋势向上，就按下一轮会更贵的预期提前切
        reference = max(raw_tokens, projected)

        if previous is None:
            return cached_tokens <= raw_tokens
        if previous == "raw":
            return cached_tokens < reference * margin
        return not (raw_tokens < cached_tokens * margin)

    # ---------- 写入策略 ----------
    def _apply(self, cand: Candidate, question: str, source: str = "rule") -> UpsertRecord:
        summary = self.extractor.summary_for(question, cand.keyword)

        if cand.keyword in self.cache:
            self.cache.touch(cand.keyword, HitKind.LITERAL)
            entry = self.cache.get(cand.keyword)
            return UpsertRecord(
                UpsertKind.HIT, cand.keyword, entry.slot if entry else cand.slot, cand.role, "LITERAL"
            )

        existing = self.cache.find_by_slot(cand.slot)
        if existing is not None and self._should_overwrite(cand, existing, question):
            reason = "REPLACE_VERB" if any(v in question for v in REPLACE_VERBS) else "SAME_COUNT_SLOT"
            return self.cache.upsert_overwrite(existing.keyword, cand, summary, reason)

        return self.cache.upsert_new(cand, summary)

    @staticmethod
    def _should_overwrite(cand: Candidate, existing, question: str) -> bool:
        """覆盖的两个合法入口：替换动词，或同一计数槽位的数值改写。

        两者都不是时不得覆盖——否则"加香蕉"会误杀"牛奶"。
        """
        if any(v in question for v in REPLACE_VERBS):
            return True
        return (
            cand.role is SlotRole.QUANTITY
            and existing.role is SlotRole.QUANTITY
            and cand.slot.endswith(".count")
        )

    def _apply_associative(
        self, question: str, upserts: list[UpsertRecord], anaphora: bool
    ) -> None:
        count_overwrite = any(
            r.kind is UpsertKind.OVERWRITE and r.slot.endswith(".count") for r in upserts
        )
        if count_overwrite:
            for entry in self.cache.entries_where(
                lambda e: e.slot == "item"
                or e.slot == "share.per_person"
                or e.slot.startswith("list.item")
            ):
                self.cache.touch(entry.keyword, HitKind.ASSOCIATIVE)

        if any(w in question for w in ASSOC_TEXT_TRIGGERS):
            for entry in self.cache.entries_where(lambda e: e.slot == "share.per_person"):
                self.cache.touch(entry.keyword, HitKind.ASSOCIATIVE)

        if anaphora:
            for entry in self.cache.snapshot():
                self.cache.touch(entry.keyword, HitKind.ASSOCIATIVE)

    # ---------- 追踪 ----------
    def _trace(self, result: TurnResult, composed: ComposeResult, ttl_sweep, capacity) -> None:
        tracer = self.tracer
        if tracer is None:
            return
        tracer.emit(result.turn, "TURN_START", question=result.question, mode=result.mode)
        tracer.emit(result.turn, "EXTRACT", rule=list(result.extracted), diff=list(result.diff_tokens))
        for record in result.upserts:
            tracer.emit(
                result.turn,
                record.kind.value.upper(),
                keyword=record.keyword,
                slot=record.slot,
                reason=record.reason,
                superseded=list(record.superseded),
            )
        for detail in ttl_sweep.details + capacity.details:
            payload = {k: v for k, v in detail.items() if k != "turn"}
            tracer.emit(result.turn, "EVICT", **payload)
        tracer.emit(
            result.turn,
            "COMPOSE",
            level=composed.level,
            tokens=composed.tokens,
            kept=list(composed.kept),
            dropped=list(composed.dropped),
            prompt=composed.prompt,
        )
        tracer.emit(result.turn, "ANSWER", text=result.answer, snapshot=list(result.snapshot))


__all__ = ["Session", "TurnResult"]
