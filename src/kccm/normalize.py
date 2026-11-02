"""归一化：全角、中文数字、单位剥离。所有 keyword 的唯一入口。"""

from __future__ import annotations

import re
import unicodedata

_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "壹": 1, "二": 2, "贰": 2, "两": 2,
    "三": 3, "叁": 3, "四": 4, "肆": 4, "五": 5, "伍": 5,
    "六": 6, "陆": 6, "七": 7, "柒": 7, "八": 8, "捌": 8, "九": 9, "玖": 9,
}
_CN_UNITS = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000,
             "万": 10000, "亿": 100000000}

# 这些字符跟在中文数字后面时，它不是数量（“一起”“一般”“片刻”）——用于负向过滤
_CN_FALSE_POSITIVE = ("些", "样", "般", "起", "旦", "会儿", "直", "向", "边", "定", "共", "齐")


def to_halfwidth(text: str) -> str:
    """全角转半角，含表意空格。"""
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def normalize_text(text: str) -> str:
    """文本级归一化：全角折叠 + NFC。"""
    return unicodedata.normalize("NFC", to_halfwidth(text))


def normalize_keyword(word: str) -> str:
    """关键词级归一化：去空白、折叠大小写。数字已在提取阶段转阿拉伯。"""
    return normalize_text(word).strip().lower()


def cn_to_int(text: str) -> int | None:
    """中文数字转整数，支持“十二 / 二十三 / 一百零五 / 两千 / 一万”。"""
    if not text:
        return None
    total = 0
    section = 0
    number = 0
    seen = False
    for ch in text:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
            seen = True
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            seen = True
            if unit >= 10000:
                section = (section + number) * unit
                total += section
                section = 0
            else:
                if number == 0:
                    number = 1
                section += number * unit
            number = 0
        else:
            return None
    if not seen:
        return None
    return total + section + number


def is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def is_cn_numeral_char(ch: str) -> bool:
    return ch in _CN_DIGITS or ch in _CN_UNITS


def cn_false_positive(text: str, end: int) -> bool:
    """判断中文数字是否只是构词成分。"""
    tail = text[end:end + 2]
    return any(tail.startswith(tok) for tok in _CN_FALSE_POSITIVE)


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
