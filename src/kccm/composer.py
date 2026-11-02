"""上下文拼接：三段式 + 五级裁剪。

关键分离：本模块**只读快照、只改视图**，绝不删除缓存条目。
存储层的淘汰由 KeywordCache.sweep_ttl / enforce_capacity 负责。
"""

from __future__ import annotations

from dataclasses import dataclass

from .cache import KeywordCache
from .config import Config
from .models import CacheEntry

SYSTEM_PROMPT = """你是多轮对话助手。工作记忆由系统以「关键词缓存」形式提供，不是完整对话历史。遵守三条：
1. 只使用 MEMORY 与 SLOTS 中的事实；缺失时直接说明缺哪个值，不要编造。
2. 数值计算必须写出算式。
3. 若当前问题是省略表达（如「四个人呢」），先用 SLOTS 还原完整问题，首行固定输出 `还原：<完整问题>`。"""

# 精简模板：把槽位并回记忆行、去掉 role（可由 slot 推断）、压缩系统指令。
# 实测可把固定开销从约 134 tokens 降到约 50，代价是提示的结构化程度下降。
LITE_SYSTEM = """你是多轮对话助手。MEMORY 是关键词缓存，不是完整历史：只使用其中事实，缺值就直说。
省略式提问先用槽位还原成完整句，首行输出「还原：…」；计算写出算式。"""

# 短会话旁路：前几轮直接发原文，不构造关键词缓存
RAW_SYSTEM = """你是多轮对话助手。下面是最近的对话记录，请基于它回答当前问题。
缺信息时直说，不要编造；计算写出算式。"""


class ContextBudgetExceeded(RuntimeError):
    """L5：预算裁剪到极限仍放不下。禁止静默截断用户当前问题。"""

    def __init__(self, message: str, detail: dict) -> None:
        super().__init__(message)
        self.detail = detail


@dataclass(frozen=True)
class ComposeResult:
    prompt: str
    level: str
    tokens: int
    kept: tuple[str, ...]
    dropped: tuple[str, ...]
    pinned: tuple[str, ...]


def estimate_tokens(text: str) -> int:
    """中文按字计、ASCII 按 4 字符计。误差由 config 中的安全边际吸收。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + (other + 3) // 4


class Composer:
    def __init__(self, cfg: Config | None = None) -> None:
        self._cfg = cfg or Config()

    def compose(
        self,
        question: str,
        cache: KeywordCache,
        pinned: set[str],
        bridge: tuple[str, str] | None = None,
    ) -> ComposeResult:
        budget = self._cfg.prompt_budget
        entries = cache.snapshot()
        dropped: list[str] = []

        prompt, tokens = self._render(entries, question, cache, bridge=bridge)
        if tokens > budget and bridge is not None:
            # 桥接块只是锦上添花，超预算时先丢它，不动事实段
            prompt, tokens = self._render(entries, question, cache)
        if tokens <= budget:
            return self._ok(prompt, "L0_FULL", tokens, entries, dropped, pinned)

        # L1 时效裁剪：本轮不用、下轮就消失的条目不占预算
        view = [e for e in entries if not (cache.ttl_of(e) == 1 and e.keyword not in pinned)]
        dropped += [e.keyword for e in entries if e not in view]
        prompt, tokens = self._render(view, question, cache)
        if tokens <= budget:
            return self._ok(prompt, "L1_TTL", tokens, view, dropped, pinned)

        # L2 摘要截断：120 → 48 → 24
        for width in (48, 24):
            prompt, tokens = self._render(view, question, cache, summary_width=width)
            if tokens <= budget:
                return self._ok(prompt, f"L2_SUMMARY_{width}", tokens, view, dropped, pinned)

        # L3 频次裁剪：逐条移出视图（不删缓存）
        while True:
            candidates = [e for e in view if e.keyword not in pinned]
            if not candidates:
                break
            victim = min(candidates, key=lambda e: (cache.ttl_of(e), e.hit_count, e.keyword))
            view = [e for e in view if e.keyword != victim.keyword]
            dropped.append(victim.keyword)
            prompt, tokens = self._render(view, question, cache)
            if tokens <= budget:
                return self._ok(prompt, "L3_FREQ", tokens, view, dropped, pinned)

        # L4 格式降级：单行紧凑串
        prompt, tokens = self._render(view, question, cache, compact=True)
        if tokens <= budget:
            return self._ok(prompt, "L4_COMPACT", tokens, view, dropped, pinned)

        # L5 拒绝静默截断
        raise ContextBudgetExceeded(
            f"上下文超出预算 {budget} tokens（当前 {tokens}），请拆分问题或提高 prompt_budget",
            {
                "tokens": tokens,
                "budget": budget,
                "kept": [e.keyword for e in view],
                "dropped": dropped,
                "question_tokens": estimate_tokens(question),
            },
        )

    def compose_raw(
        self, question: str, turns: list[tuple[str, str]]
    ) -> ComposeResult:
        """旁路模式：把原始对话直接拼进 prompt，不读缓存。

        仍受同一预算约束：超限时从最旧的一轮开始丢，而不是静默截断当前问题。
        """
        budget = self._cfg.prompt_budget
        view = list(turns)
        dropped: list[str] = []
        while True:
            prompt = self._render_raw(view, question)
            tokens = estimate_tokens(prompt)
            if tokens <= budget or not view:
                break
            oldest = view.pop(0)
            dropped.append(oldest[0])

        if tokens > budget:
            raise ContextBudgetExceeded(
                f"原始对话超出预算 {budget} tokens（当前 {tokens}），请拆分问题",
                {"tokens": tokens, "budget": budget, "dropped": dropped},
            )
        return ComposeResult(
            prompt=prompt,
            level="RAW",
            tokens=tokens,
            kept=tuple(q for q, _ in view),
            dropped=tuple(dropped),
            pinned=(),
        )

    def _render_raw(self, turns: list[tuple[str, str]], question: str) -> str:
        lines = ["# 对话"]
        if turns:
            for user, assistant in turns:
                lines.append(f"用户: {user}")
                lines.append(f"助手: {assistant}")
        else:
            lines.append("（空）")
        parts = [f"# 系统指令\n{RAW_SYSTEM}", "\n".join(lines), f"# 当前问题\n{question}"]
        return "\n\n".join(parts)

    # ---------- 渲染 ----------
    def _render(
        self,
        entries: list[CacheEntry],
        question: str,
        cache: KeywordCache,
        summary_width: int | None = None,
        compact: bool = False,
        style: str | None = None,
        bridge: tuple[str, str] | None = None,
    ) -> tuple[str, int]:
        cfg = self._cfg
        width = summary_width or cfg.summary_max_chars
        style = style or cfg.prompt_style

        if compact:
            memory = "# 工作记忆\n" + (
                "·".join(f"{e.keyword}({_short_role(e)})" for e in entries) or "（空）"
            )
            slots = ""
        elif style == "lite":
            lines = [f"# 记忆（ttl 降序，{len(entries)}/{cfg.max_entries}）"]
            if entries:
                for e in entries:
                    history = f"(原{'、'.join(e.superseded)})" if e.superseded else ""
                    turn_label = e.summary_turn or e.first_turn
                    lines.append(
                        f"- {e.keyword}·{e.slot}·ttl{cache.ttl_of(e)}·h{e.hit_count}"
                        f"·T{turn_label}「{e.summary[:width]}」{history}"
                    )
            else:
                lines.append("（空）")
            memory, slots = "\n".join(lines), ""
        else:
            lines = [f"# 工作记忆（按 ttl 降序，{len(entries)} / {cfg.max_entries} 条）"]
            if entries:
                for e in entries:
                    summary = e.summary[:width]
                    turn_label = e.summary_turn or e.first_turn
                    lines.append(
                        f"- {e.keyword} | {e.role.value} | {e.slot} | "
                        f"ttl={cache.ttl_of(e)} | hit={e.hit_count} | T{turn_label}「{summary}」"
                    )
            else:
                lines.append("（空）")
            memory = "\n".join(lines)

            slot_lines = ["# 槽位"]
            if entries:
                for e in entries:
                    history = f"（原 {'、'.join(e.superseded)}）" if e.superseded else ""
                    slot_lines.append(f"- {e.slot} = {e.keyword}{history}")
            else:
                slot_lines.append("（空）")
            slots = "\n".join(slot_lines)

        system = LITE_SYSTEM if style == "lite" else SYSTEM_PROMPT
        parts = [f"# 系统指令\n{system}", memory]
        if slots:
            parts.append(slots)
        if bridge is not None:
            parts.append(f"# 上一轮原文\n用户: {bridge[0]}\n助手: {bridge[1]}")
        parts.append(f"# 当前问题\n{question}")
        prompt = "\n\n".join(parts)
        return prompt, estimate_tokens(prompt)

    def _ok(self, prompt, level, tokens, entries, dropped, pinned) -> ComposeResult:
        return ComposeResult(
            prompt=prompt,
            level=level,
            tokens=tokens,
            kept=tuple(e.keyword for e in entries),
            dropped=tuple(dropped),
            pinned=tuple(sorted(pinned)),
        )


def _short_role(entry: CacheEntry) -> str:
    mapping = {
        "quantity": "qty",
        "object": "obj",
        "recipient": "rcp",
        "attribute": "attr",
        "subject": "sub",
        "constraint": "con",
        "time": "tim",
        "location": "loc",
    }
    return mapping.get(entry.role.value, entry.role.value[:3])
