import unittest

from kccm import Config, Extractor, SlotRole, get_tagger
from kccm.extractor import dedup_candidates
from kccm.models import Candidate


class ExtractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ex = Extractor(Config(), tagger=get_tagger(prefer_jieba=False))

    def test_apple_case(self) -> None:
        """示例第 1 回合：必须提取 12、苹果、3、每个人 四个关键词。"""
        cands = self.ex.rule_extract("我有12个苹果，3个人每个人得到多少")
        keywords = [c.keyword for c in cands]
        self.assertEqual(keywords, ["12", "3", "苹果", "每个人"])

        by_kw = {c.keyword: c for c in cands}
        self.assertEqual(by_kw["12"].role, SlotRole.QUANTITY)
        self.assertEqual(by_kw["12"].slot, "apple.count")
        self.assertEqual(by_kw["3"].role, SlotRole.QUANTITY)
        self.assertEqual(by_kw["3"].slot, "person.count")
        self.assertEqual(by_kw["苹果"].role, SlotRole.OBJECT)
        self.assertEqual(by_kw["苹果"].slot, "item")
        self.assertEqual(by_kw["每个人"].role, SlotRole.RECIPIENT)
        self.assertEqual(by_kw["每个人"].slot, "share.per_person")

    def test_chinese_numeral(self) -> None:
        cands = self.ex.rule_extract("那么四个人呢")
        self.assertEqual([c.keyword for c in cands], ["4"])
        self.assertEqual(cands[0].slot, "person.count")

    def test_multi_digit_chinese_numeral(self) -> None:
        cands = self.ex.rule_extract("有十二个苹果")
        self.assertIn("12", [c.keyword for c in cands])

    def test_verbs_and_generics_dropped(self) -> None:
        """仅提取数字与名词：动词、泛化名词不得入缓存。"""
        cands = self.ex.rule_extract("帮我列个采购清单，加香蕉和牛奶")
        keywords = [c.keyword for c in cands]
        self.assertEqual(keywords, ["香蕉", "牛奶"])
        self.assertNotIn("清单", keywords)
        self.assertNotIn("采购", keywords)

    def test_money_slot(self) -> None:
        cands = self.ex.rule_extract("预算50元")
        self.assertEqual(cands[0].keyword, "50")
        self.assertEqual(cands[0].slot, "budget.amount")

    def test_dedup_keeps_order(self) -> None:
        """回归：曾用 set 去重导致顺序不稳定，不可复现。"""
        left = Candidate("a", "a", SlotRole.OBJECT, "s", "noun")
        right = Candidate("b", "b", SlotRole.OBJECT, "s", "noun")
        self.assertEqual([c.keyword for c in dedup_candidates([left, right, left])], ["a", "b"])


class TaggerFallbackTest(unittest.TestCase):
    def test_fallback_without_jieba(self) -> None:
        tagger = get_tagger(prefer_jieba=False)
        self.assertEqual(tagger.name, "lexicon")
        words = [h.word for h in tagger.nouns("我有12个苹果，3个人每个人得到多少")]
        self.assertIn("苹果", words)
        self.assertIn("每个人", words)


if __name__ == "__main__":
    unittest.main()
