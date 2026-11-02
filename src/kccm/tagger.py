"""分词后端：jieba 优先，缺失时退化为内置词典标注器。

设计动机：Demo 必须在**无网络、无第三方依赖**的环境下可运行、可测试。
因此把分词抽象成可插拔后端，而不是硬绑定 jieba。

- JiebaTagger：词性标注准确，但依赖 jieba 与词典加载。
- LexiconTagger：纯标准库，长词优先匹配 + 停用/动词/量词三段屏蔽，
  relaxed 模式下把未登录的连续汉字串作为 OOV 候选，供差分层兜底。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import SlotRole
from .normalize import is_cjk, normalize_keyword

_DATA = Path(__file__).parent / "data" / "lexicon.txt"

# 功能词：代词、连词、助词、疑问词、时间副词
STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "把", "被", "给", "对", "从", "到",
    "吗", "呢", "吧", "啊", "呀", "哦", "嗯", "这", "那", "这个", "那个", "这么",
    "那么", "这样", "那样", "如果", "要是", "但是", "然后", "还有", "以及", "我",
    "你", "他", "她", "它", "我们", "你们", "他们", "大家", "自己", "什么", "怎么",
    "为什么", "多少", "几", "如何", "哪些", "哪个", "时候", "东西", "问题", "情况",
    "办法", "样子", "地方", "现在", "可以", "应该", "还", "也", "都", "就", "又",
}

# 动词：仅提取数字与名词，动词一律不入缓存
VERBS = {
    "有", "要", "想", "能", "会", "得到", "分", "分给", "平均", "列", "加", "买",
    "换", "剩", "还剩", "算", "帮", "做", "来", "去", "说", "看", "知道", "需要",
    "采购", "统计", "包括", "用", "让", "请", "告诉", "计算", "求", "换成", "改成",
    "变成", "加上", "添", "再", "各", "用", "按", "照", "称", "叫", "花", "付",
}

# 量词/单位：锚定数量，自身不是关键词
QUANTIFIERS = {
    "个", "只", "位", "份", "条", "件", "斤", "公斤", "克", "元", "块", "角", "张",
    "本", "台", "辆", "杯", "瓶", "袋", "箱", "包", "双", "组", "批", "次", "遍",
    "趟", "米", "千米", "升", "毫升", "小时", "天", "周", "月", "年", "人", "打",
}

# 泛化名词：出现频率高但不携带事实，进缓存只会挤占预算
GENERIC_NOUNS = {"清单", "数量", "结果", "方法", "方式", "部分", "内容", "事项", "东西"}

# 被数量吸收的描述词：不独立成条，只作为数量槽位的命名依据
ABSORBED = {"预算", "人数", "份数", "总量", "总共", "一共", "合计"}


@dataclass(frozen=True)
class NounHit:
    word: str
    start: int
    end: int
    role: SlotRole
    slot: str
    num_hint: str


class LexiconTagger:
    """长词优先的词典标注器（纯标准库实现）。"""

    def __init__(self, lexicon_path: Path | str = _DATA) -> None:
        self._lexicon: dict[str, tuple[SlotRole, str, str]] = {}
        for line in Path(lexicon_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) != 4:
                continue
            word, role, slot, hint = parts
            self._lexicon[word] = (SlotRole(role), slot, hint)
        self._skip: set[str] = set(STOPWORDS) | set(VERBS) | set(QUANTIFIERS) | set(GENERIC_NOUNS) | set(ABSORBED)
        self._max_len = max((len(w) for w in set(self._lexicon) | self._skip), default=1)
        self.name = "lexicon"

    def _match(self, text: str, i: int) -> tuple[str, str] | tuple[None, None]:
        upper = min(self._max_len, len(text) - i)
        for length in range(upper, 0, -1):
            piece = text[i:i + length]
            if piece in self._lexicon:
                return "lex", piece
            if piece in self._skip:
                return "skip", piece
        return None, None

    def nouns(self, text: str, relaxed: bool = False) -> list[NounHit]:
        hits: list[NounHit] = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if not is_cjk(ch):
                i += 1
                continue
            kind, piece = self._match(text, i)
            if kind == "lex":
                role, slot, hint = self._lexicon[piece]
                hits.append(NounHit(piece, i, i + len(piece), role, slot, hint))
                i += len(piece)
                continue
            if kind == "skip":
                i += len(piece)
                continue
            if relaxed:
                j = i
                while j < n and is_cjk(text[j]) and self._match(text, j)[0] is None:
                    j += 1
                if j - i >= 2:
                    word = text[i:j]
                    hits.append(NounHit(word, i, j, SlotRole.OBJECT, f"entity.{word}", "entity"))
                    i = j
                    continue
            i += 1
        return hits


class JiebaTagger:
    """词性标注后端。需 ``pip install jieba``。"""

    def __init__(self, lexicon_path: Path | str = _DATA) -> None:
        import jieba.posseg as pseg  # noqa: F401  (延迟导入，缺失时由 get_tagger 兜底)

        self._pseg = pseg
        self._hints: dict[str, tuple[SlotRole, str, str]] = {}
        for line in Path(lexicon_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) == 4:
                word, role, slot, hint = parts
                self._hints[word] = (SlotRole(role), slot, hint)
        self.name = "jieba"

    def nouns(self, text: str, relaxed: bool = False) -> list[NounHit]:
        hits: list[NounHit] = []
        cursor = 0
        keep = ("n", "vn", "eng")
        for word, flag in self._pseg.cut(text):
            start = text.find(word, cursor)
            if start < 0:
                start = cursor
            end = start + len(word)
            cursor = end
            if word in GENERIC_NOUNS or word in STOPWORDS or word in ABSORBED:
                continue
            if word in self._hints:
                role, slot, hint = self._hints[word]
            elif flag.startswith(keep):
                role, slot, hint = SlotRole.OBJECT, f"entity.{word}", "entity"
            else:
                continue
            if len(word) == 1 and word not in self._hints:
                continue
            hits.append(NounHit(word, start, end, role, slot, hint))
        return hits


def get_tagger(prefer_jieba: bool = True) -> LexiconTagger | JiebaTagger:
    if prefer_jieba:
        try:
            return JiebaTagger()
        except Exception:  # noqa: BLE001 - 分词后端不可用时必须能降级
            pass
    return LexiconTagger()


def dedup_hits(hits: list[NounHit]) -> list[NounHit]:
    """同词多现只保留首次，保证顺序可复现（禁用 set 去重）。"""
    seen: dict[str, NounHit] = {}
    out: list[NounHit] = []
    for hit in hits:
        key = normalize_keyword(hit.word)
        if key in seen:
            continue
        seen[key] = hit
        out.append(hit)
    return out
