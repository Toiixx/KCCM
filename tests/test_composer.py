import unittest

from kccm import Config, KeywordCache, SlotRole
from kccm.composer import Composer, ContextBudgetExceeded, estimate_tokens
from kccm.models import Candidate, CacheEntry


def fill(cache: KeywordCache, count: int, summary_len: int = 60) -> None:
    for i in range(count):
        cache.upsert_new(
            Candidate(f"词{i}", f"词{i}", SlotRole.OBJECT, f"slot.{i}", "noun"),
            "前缀" + "测" * summary_len,
        )


class ComposerTest(unittest.TestCase):
    def test_budget_never_exceeded(self) -> None:
        cfg = Config(prompt_budget=600)
        cache = KeywordCache(cfg)
        cache.begin_turn(1)
        fill(cache, 30)

        result = Composer(cfg).compose("这些词里哪些有用", cache, pinned=set())
        self.assertLessEqual(result.tokens, cfg.prompt_budget)
        self.assertNotEqual(result.level, "L0_FULL")
        self.assertLessEqual(estimate_tokens(result.prompt), cfg.prompt_budget)

    def test_pinned_survives_trimming(self) -> None:
        cfg = Config(prompt_budget=420)
        cache = KeywordCache(cfg)
        cache.begin_turn(1)
        fill(cache, 40)
        pinned = {"词0", "词1"}

        result = Composer(cfg).compose("词0 和 词1 呢", cache, pinned=pinned)
        self.assertLessEqual(result.tokens, cfg.prompt_budget)
        for keyword in pinned:
            self.assertIn(keyword, result.kept)
            self.assertNotIn(keyword, result.dropped)

    def test_trimming_does_not_mutate_cache(self) -> None:
        """裁剪只改视图：缓存条目数在 compose 前后必须一致。"""
        cfg = Config(prompt_budget=420)
        cache = KeywordCache(cfg)
        cache.begin_turn(1)
        fill(cache, 40)
        before = len(cache)
        Composer(cfg).compose("随便问问", cache, pinned=set())
        self.assertEqual(len(cache), before)

    def test_budget_exceeded_raises(self) -> None:
        cfg = Config(prompt_budget=256)
        cache = KeywordCache(cfg)
        cache.begin_turn(1)
        with self.assertRaises(ContextBudgetExceeded):
            Composer(cfg).compose("问" * 2000, cache, pinned=set())

    def test_slots_section_present(self) -> None:
        cfg = Config()
        cache = KeywordCache(cfg)
        cache.begin_turn(1)
        cache.upsert_new(Candidate("12", "12", SlotRole.QUANTITY, "apple.count", "number"), "12个苹果")
        result = Composer(cfg).compose("多少", cache, pinned=set())
        self.assertIn("# 槽位", result.prompt)
        self.assertIn("apple.count = 12", result.prompt)

    def test_lite_style_is_cheaper(self) -> None:
        """精简模板：去掉独立槽位段与 role 字段，必须显著小于 full。"""
        full_cfg = Config(prompt_style="full")
        lite_cfg = Config(prompt_style="lite")

        full_cache = KeywordCache(full_cfg)
        full_cache.begin_turn(1)
        fill(full_cache, 8, summary_len=12)
        full_tokens = Composer(full_cfg).compose("问题", full_cache, pinned=set()).tokens

        lite_cache = KeywordCache(lite_cfg)
        lite_cache.begin_turn(1)
        fill(lite_cache, 8, summary_len=12)
        lite = Composer(lite_cfg).compose("问题", lite_cache, pinned=set())

        self.assertLess(lite.tokens, full_tokens)
        self.assertIn("# 记忆", lite.prompt)
        self.assertNotIn("# 槽位", lite.prompt)
        self.assertNotIn("| quantity |", lite.prompt)


if __name__ == "__main__":
    unittest.main()
