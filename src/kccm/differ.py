"""差分补充：用新旧提问的差异补提取遗漏关键词，并识别纯指代。"""

from __future__ import annotations

import re

ANAPHORA_RE = re.compile(r"[这那它其该]|呢|还是|同样|一样|刚才|上面|继续|如上")
ANAPHORA_SENTINEL = "__ANAPHORA__"


def token_set(text: str, tagger) -> set[str]:
    """词级 token 集：词典词 + 数字 + 英文串。仅用于差分比较，不参与提取。"""
    from .extractor import ARABIC_RE, CN_NUM_RE
    from .normalize import cn_to_int, normalize_keyword, normalize_text

    norm = normalize_text(text)
    tokens: set[str] = set()
    for hit in tagger.nouns(norm):
        tokens.add(normalize_keyword(hit.word))
    for m in ARABIC_RE.finditer(norm):
        tokens.add(m.group(1))
    for m in CN_NUM_RE.finditer(norm):
        value = cn_to_int(m.group())
        if value is not None:
            tokens.add(str(value))
    return tokens


def diff_extract(
    question_now: str,
    question_prev: str,
    cache_keys: set[str],
    rule_keywords: set[str],
    tagger,
) -> tuple[set[str], bool]:
    """返回 (差集词, 是否纯指代)。

    ΔQ = tokens(Q_now) - tokens(Q_prev) - rule_hits - cache_keys
    差集只负责“规则没提到、但确实是新信息”的词；已命中的词不再重复。
    """
    now = token_set(question_now, tagger)
    prev = token_set(question_prev, tagger)
    delta = now - prev - set(rule_keywords) - set(cache_keys)

    if not delta and not rule_keywords and ANAPHORA_RE.search(question_now):
        return set(), True
    return delta, False
