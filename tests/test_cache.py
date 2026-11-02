import unittest

from kccm import Config, KeywordCache, SlotRole
from kccm.models import CacheEntry


def seed(cache: KeywordCache, keyword: str, slot: str, ttl: int, hit: int,
         turn: int, role: SlotRole = SlotRole.OBJECT) -> None:
    """直接注入指定 ttl/hit 的条目，用于构造淘汰场景。

    ttl = ttl_max - (turn - last_hit_turn) → 反解 last_hit_turn。
    """
    last_hit = turn - (cache._cfg.ttl_max - ttl)
    cache._entries[keyword] = CacheEntry(
        keyword=keyword, surface=keyword, role=role, slot=slot,
        hit_count=hit, first_turn=1, last_hit_turn=last_hit, summary=f"{keyword} 摘要",
        summary_turn=last_hit,
    )


class TTLTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = KeywordCache(Config(ttl_max=10, max_entries=99))

    def test_boundary_turn_10_keeps_turn_11_deletes(self) -> None:
        seed(self.cache, "苹果", "item", ttl=10, hit=1, turn=1)
        self.cache.begin_turn(10)
        self.assertIn("苹果", self.cache)
        self.assertEqual(self.cache.ttl_of(self.cache.get("苹果")), 1)

        report = self.cache.begin_turn(11)
        self.assertNotIn("苹果", self.cache)
        self.assertEqual(report.expired, ["苹果"])

    def test_ten_silent_turns(self) -> None:
        seed(self.cache, "苹果", "item", ttl=10, hit=1, turn=1)
        self.cache.begin_turn(2)
        self.assertEqual(self.cache.ttl_of(self.cache.get("苹果")), 9)
        for turn in range(3, 11):
            self.cache.begin_turn(turn)
        self.assertIn("苹果", self.cache)  # 第 10 回合仍存活，ttl=1
        self.cache.begin_turn(11)
        self.assertNotIn("苹果", self.cache)

    def test_pinned_cleared_each_turn(self) -> None:
        """pin 是回合级状态，begin_turn 必须清空，否则上一轮的豁免会泄漏。"""
        seed(self.cache, "苹果", "item", ttl=10, hit=1, turn=1)
        self.cache.begin_turn(2)
        self.cache.set_pinned({"苹果"})
        self.cache.begin_turn(3)
        self.assertEqual(self.cache._pinned, set())  # noqa: SLF001 - 白盒断言


class LFUTest(unittest.TestCase):
    def test_case_a_ttl_gap_wins(self) -> None:
        """场景 A：TTL 差 5 > tolerance，即使高频也不救。"""
        cache = KeywordCache(Config(ttl_max=10, max_entries=3, ttl_tolerance=2))
        cache.begin_turn(8)
        seed(cache, "苹果", "item", ttl=4, hit=5, turn=8)
        seed(cache, "橘子", "item2", ttl=9, hit=1, turn=8)
        seed(cache, "预算", "budget.amount", ttl=8, hit=2, turn=8)
        seed(cache, "香蕉", "list.item.香蕉", ttl=10, hit=1, turn=8)

        report = cache.enforce_capacity()
        self.assertEqual(report.evicted, ["苹果"])
        self.assertEqual(report.details[0]["reason"], "TTL_DOMINANT")

    def test_case_b_lfu_saves_frequent(self) -> None:
        """场景 B：TTL 相近（差 1 ≤ tolerance），淘汰低频者。"""
        cache = KeywordCache(Config(ttl_max=10, max_entries=3, ttl_tolerance=2))
        cache.begin_turn(8)
        seed(cache, "苹果", "item", ttl=6, hit=5, turn=8)
        seed(cache, "橘子", "item2", ttl=7, hit=1, turn=8)
        seed(cache, "预算", "budget.amount", ttl=9, hit=2, turn=8)
        seed(cache, "香蕉", "list.item.香蕉", ttl=10, hit=1, turn=8)

        report = cache.enforce_capacity()
        self.assertEqual(report.evicted, ["橘子"])
        self.assertIn("苹果", cache)

    def test_tiebreak_is_deterministic(self) -> None:
        cache = KeywordCache(Config(ttl_max=10, max_entries=2, ttl_tolerance=2))
        cache.begin_turn(6)
        seed(cache, "b", "s2", ttl=7, hit=1, turn=6)
        seed(cache, "a", "s1", ttl=7, hit=1, turn=6)
        seed(cache, "c", "s3", ttl=10, hit=1, turn=6)
        report = cache.enforce_capacity()
        # hit 与 last_hit 全相同 → 字典序兜底，a 先走
        self.assertEqual(report.evicted, ["a"])

    def test_pinned_never_evicted(self) -> None:
        cache = KeywordCache(Config(ttl_max=10, max_entries=2, ttl_tolerance=2))
        cache.begin_turn(3)
        seed(cache, "苹果", "item", ttl=5, hit=1, turn=3)
        seed(cache, "梨", "item2", ttl=6, hit=1, turn=3)
        cache.set_pinned({"苹果", "梨", "香蕉"})
        seed(cache, "香蕉", "list.item.香蕉", ttl=10, hit=1, turn=3)

        report = cache.enforce_capacity()
        self.assertEqual(report.evicted, [])
        self.assertEqual(len(cache), 3)  # 宁可临时超限，也不破坏 pin 语义


if __name__ == "__main__":
    unittest.main()
