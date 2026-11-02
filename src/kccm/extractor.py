"""规则提取：数字 + 名词，两类之外一律不入缓存。"""

from __future__ import annotations

import re

from .config import Config
from .models import Candidate, SlotRole
from .normalize import (
    cn_false_positive,
    cn_to_int,
    is_cn_numeral_char,
    normalize_keyword,
    normalize_text,
)
from .tagger import NounHit, get_tagger

ARABIC_RE = re.compile(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)(%?)")
CN_NUM_RE = re.compile(r"[零〇一二三四五六七八九十百千万亿两壹贰叁肆伍陆柒捌玖拾佰仟]+")
UNIT_CHARS = ("元", "块", "斤", "公斤", "克", "%", "个", "人", "位", "只", "份", "条")
MONEY_UNITS = ("元", "块")
CLAUSE_SPLIT = re.compile(r"[，。；！？,.!?；\n]")

REPLACE_VERBS = ("换成", "改成", "变成", "换为", "改为", "而不是", "代替")
ASSOC_RECIPIENT_WORDS = ("各", "每人", "分别", "均分", "平均分")


class Extractor:
    def __init__(self, cfg: Config, tagger=None) -> None:
        self.cfg = cfg
        self.tagger = tagger or get_tagger()

    # ---------- 数字 ----------
    def _number_candidates(self, text: str, noun_hits: list[NounHit]) -> list[Candidate]:
        out: list[Candidate] = []

        for m in ARABIC_RE.finditer(text):
            raw, pct = m.group(1), m.group(2)
            start, end = m.start(1), m.end(1)
            unit = pct or ""
            if not unit and end < len(text):
                for u in UNIT_CHARS:
                    if text.startswith(u, end):
                        unit = u
                        break
            keyword = _clean_number(raw)
            role, slot = _resolve_number_slot(text, start, end, unit, noun_hits)
            out.append(Candidate(keyword, text[start:end] + unit, role, slot, "number", start, end))

        for m in CN_NUM_RE.finditer(text):
            start, end = m.start(), m.end()
            if start > 0 and (text[start - 1].isdigit() or is_cn_numeral_char(text[start - 1])):
                continue
            if cn_false_positive(text, end):
                continue
            value = cn_to_int(m.group())
            if value is None or value == 0:
                continue
            unit = ""
            if end < len(text):
                for u in UNIT_CHARS:
                    if text.startswith(u, end):
                        unit = u
                        break
            role, slot = _resolve_number_slot(text, start, end, unit, noun_hits)
            out.append(
                Candidate(str(value), m.group() + unit, role, slot, "number", start, end)
            )
        return out

    # ---------- 名词 ----------
    def _noun_candidates(self, text: str, relaxed: bool = False) -> list[NounHit]:
        hits = self.tagger.nouns(text, relaxed=relaxed)
        return hits

    def rule_extract(self, text: str, relaxed: bool = False) -> list[Candidate]:
        norm = normalize_text(text)
        hits = self._noun_candidates(norm, relaxed=relaxed)
        candidates = self._number_candidates(norm, hits)

        for hit in hits:
            keyword = normalize_keyword(hit.word)
            slot = hit.slot
            # list.* 家族必须按关键词取后缀，否则两个采购项会争同一槽位（违反 I2）
            if slot.startswith("list.") and not slot.endswith(f".{keyword}"):
                slot = f"{slot}.{keyword}"
            candidates.append(
                Candidate(keyword, hit.word, hit.role, slot, "noun", hit.start, hit.end)
            )
        return dedup_candidates(candidates)

    def summary_for(self, text: str, keyword: str) -> str:
        norm = normalize_text(text)
        for clause in CLAUSE_SPLIT.split(norm):
            if keyword in clause:
                return clause.strip()[: self.cfg.summary_max_chars]
        return norm.strip()[: self.cfg.summary_max_chars]


def _clean_number(raw: str) -> str:
    if "." in raw:
        return raw.rstrip("0").rstrip(".")
    return raw


def _resolve_number_slot(
    text: str, start: int, end: int, unit: str, noun_hits: list[NounHit]
) -> tuple[SlotRole, str]:
    """数量槽位判定。顺序即优先级，越具体的上下文越先判定。"""
    before = text[max(0, start - 8):start]
    after = text[end:end + 8]

    if unit in MONEY_UNITS or after[:1] in MONEY_UNITS:
        return SlotRole.ATTRIBUTE, "budget.amount"
    if unit == "%" or after[:1] == "%":
        return SlotRole.ATTRIBUTE, "ratio.percent"
    if "单价" in before:
        return SlotRole.ATTRIBUTE, "price.unit"
    if "预算" in before or "总价" in before or "总共" in before:
        return SlotRole.ATTRIBUTE, "budget.amount"

    if any(unit.startswith(k) or after[:2].find(k) >= 0 for k in ("人", "位")):
        return SlotRole.QUANTITY, "person.count"
    if any(k in before for k in ("还剩", "剩", "余")):
        return SlotRole.QUANTITY, "remainder.count"
    if any(k in before for k in ("买", "采购", "加", "添")):
        return SlotRole.QUANTITY, "purchase.count"

    nearest = _nearest_noun_after(noun_hits, end, max_gap=4)
    if nearest is not None:
        hint = nearest.num_hint
        if hint == "person":
            return SlotRole.QUANTITY, "person.count"
        return SlotRole.QUANTITY, f"{hint}.count"

    return SlotRole.QUANTITY, "quantity.generic"


def _nearest_noun_after(hits: list[NounHit], end: int, max_gap: int) -> NounHit | None:
    best: NounHit | None = None
    for hit in hits:
        if hit.start < end:
            continue
        if hit.start - end > max_gap:
            break
        if best is None or hit.start < best.start:
            best = hit
    return best


def dedup_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """按 (keyword, slot) 有序去重；禁止用 set 破坏顺序（否则不可复现）。"""
    seen: set[tuple[str, str]] = set()
    out: list[Candidate] = []
    for cand in candidates:
        key = (cand.keyword, cand.slot)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out
