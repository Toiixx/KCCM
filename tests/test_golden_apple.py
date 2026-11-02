import unittest

from kccm import Config, Session

QUESTIONS = [
    "我有12个苹果，3个人每个人得到多少",
    "那么四个人呢",
    "如果换成梨，还剩6个呢",
    "帮我列个采购清单，加香蕉和牛奶",
    "再各买5个",
    "预算50元",
]

# 与 docs/DESIGN.md §2.7 的表格逐格一致：keyword → (ttl, hit_count)
EXPECTED = [
    {"12": (10, 1), "3": (10, 1), "苹果": (10, 1), "每个人": (10, 1)},
    {"4": (10, 2), "苹果": (10, 1), "每个人": (10, 1), "12": (9, 1)},
    {"梨": (10, 2), "6": (10, 1), "4": (9, 2), "每个人": (9, 1), "12": (8, 1)},
    {"香蕉": (10, 1), "牛奶": (10, 1), "梨": (9, 2), "6": (9, 1), "4": (8, 2),
     "每个人": (8, 1), "12": (7, 1)},
    {"5": (10, 1), "每个人": (10, 1), "香蕉": (9, 1), "牛奶": (9, 1),
     "梨": (8, 2), "6": (8, 1), "4": (7, 2), "12": (6, 1)},
    {"50": (10, 1), "5": (9, 1), "每个人": (9, 1), "牛奶": (8, 1), "香蕉": (8, 1),
     "梨": (7, 2), "6": (7, 1), "4": (6, 2)},
]


class GoldenAppleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.session = Session(cfg=Config(max_entries=8))

    def test_turn_two_answers_three(self) -> None:
        self.session.ask(QUESTIONS[0])
        result = self.session.ask(QUESTIONS[1])
        self.assertIn("12 ÷ 4 = 3", result.answer)
        self.assertIn("还原：", result.answer)

    def test_overwrite_no_coexist(self) -> None:
        """回归 A1：覆盖后旧值必须消失，不能被拼成 '4(替换3)' 之类的新词。"""
        self.session.ask(QUESTIONS[0])
        self.session.ask(QUESTIONS[1])
        cache = self.session.cache
        self.assertNotIn("3", cache)
        self.assertIn("4", cache)
        entry = cache.get("4")
        self.assertEqual(entry.hit_count, 2)
        self.assertEqual(entry.first_turn, 1)
        self.assertEqual(entry.superseded, ("3",))

    def test_associative_hit_keeps_frequency(self) -> None:
        """关联命中只续命不加频：苹果/每个人停在 hit=1，ttl 被重置。"""
        self.session.ask(QUESTIONS[0])
        self.session.ask(QUESTIONS[1])
        cache = self.session.cache
        self.assertEqual(cache.get("苹果").hit_count, 1)
        self.assertEqual(cache.ttl_of(cache.get("苹果")), 10)
        self.assertEqual(cache.ttl_of(cache.get("12")), 9)

    def test_six_turn_trace_matches_spec(self) -> None:
        for index, question in enumerate(QUESTIONS):
            self.session.ask(question)
            self.assertEqual(
                self.session.cache.state(),
                EXPECTED[index],
                msg=f"第 {index + 1} 回合快照与规格不一致",
            )

    def test_lfu_eviction_at_turn_six(self) -> None:
        for question in QUESTIONS[:5]:
            self.session.ask(question)
        result = self.session.ask(QUESTIONS[5])
        self.assertEqual(result.evicted, ("12",))
        detail = [e for e in self.session.cache.events if e.get("keyword") == "12"][-1]
        self.assertEqual(detail["reason"], "LFU_TIEBREAK")

    def test_prompt_contains_ttl_and_slots(self) -> None:
        self.session.ask(QUESTIONS[0])
        result = self.session.ask(QUESTIONS[1])
        self.assertIn("ttl=10", result.prompt)
        self.assertIn("person.count = 4（原 3）", result.prompt)
        self.assertIn("# 当前问题\n那么四个人呢", result.prompt)

    def test_lite_style_still_answers(self) -> None:
        """精简模板不得损伤正确性：省的是格式，不是事实。"""
        session = Session(cfg=Config(prompt_style="lite"))
        session.ask(QUESTIONS[0])
        result = session.ask(QUESTIONS[1])
        self.assertIn("12 ÷ 4 = 3", result.answer)
        self.assertNotIn("# 槽位", result.prompt)


if __name__ == "__main__":
    unittest.main()
