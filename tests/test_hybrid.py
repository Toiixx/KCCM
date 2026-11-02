import unittest

from kccm import Config, Session

Q1 = "我有12个苹果，3个人每个人得到多少"
Q2 = "那么四个人呢"
Q3 = "如果换成梨，还剩6个呢"
Q4 = "帮我列个采购清单，加香蕉和牛奶"
Q5 = "再各买5个"
Q6 = "预算50元"
Q7 = "那按最便宜的算"
Q8 = "总共要花多少钱"
ALL = [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8]


class FixedBranchTest(unittest.TestCase):
    def test_bypass_uses_raw_prompt(self) -> None:
        session = Session(cfg=Config(hybrid_mode="fixed", hybrid_turns=3))
        result = session.ask(Q1)
        self.assertEqual(result.mode, "raw")
        self.assertIn("# 对话", result.prompt)
        self.assertNotIn("# 记忆", result.prompt)

    def test_warm_cache_during_bypass(self) -> None:
        """旁路只改渲染，不改记账：切回时缓存必须已经攒好。"""
        session = Session(cfg=Config(hybrid_mode="fixed", hybrid_turns=2))
        for question in (Q1, Q2):
            session.ask(question)
        self.assertGreater(len(session.cache), 0)

        result = session.ask(Q3)
        self.assertEqual(result.mode, "kccm+bridge")  # 切回首轮带桥接
        self.assertIn("12", [e["keyword"] for e in result.snapshot])

    def test_cold_start_loses_everything(self) -> None:
        """字面实现 `if turn < N: 直接存入` 的代价：切换那一轮缓存是空的。"""
        cfg = Config(hybrid_mode="fixed", hybrid_turns=2, hybrid_skip_cache=True, bridge_once=False)
        session = Session(cfg=cfg)
        session.ask(Q1)
        result = session.ask(Q2)
        self.assertEqual(len(session.cache), 0)
        self.assertEqual(result.mode, "raw")

        switched = session.ask(Q3)
        self.assertEqual(switched.mode, "kccm")
        self.assertNotIn("12", [e["keyword"] for e in switched.snapshot])

    def test_bypass_recent_window(self) -> None:
        cfg = Config(hybrid_mode="fixed", hybrid_turns=99, hybrid_recent=1)
        session = Session(cfg=cfg)
        for question in (Q1, Q2, Q3):
            session.ask(question)
        third = session.history[-1]
        self.assertIn(Q2, third.prompt)      # 只带最近 1 轮
        self.assertNotIn(Q1, third.prompt)   # 更早的不带


class AutoBranchTest(unittest.TestCase):
    def test_auto_picks_cheaper_prompt(self) -> None:
        session = Session(cfg=Config(hybrid_mode="auto"))
        result = session.ask(Q1)
        self.assertEqual(result.mode, "raw")  # 第 1 轮原文 76 < 缓存 269
        self.assertLess(result.tokens, 200)

    def test_hysteresis_with_lookahead_is_not_worse(self) -> None:
        """死区 + 外推：不会比"见便宜就切"更贵，切换也不更早。"""
        eager = Session(cfg=Config(max_entries=8, hybrid_mode="auto",
                                   switch_hysteresis=0.0, bridge_once=False))
        cautious = Session(cfg=Config(max_entries=8, hybrid_mode="auto",
                                      switch_hysteresis=0.10, bridge_once=False))
        for question in ALL:
            eager.ask(question)
            cautious.ask(question)

        eager_switch = next(i + 1 for i, r in enumerate(eager.history) if r.mode.startswith("kccm"))
        cautious_switch = next(i + 1 for i, r in enumerate(cautious.history) if r.mode.startswith("kccm"))
        self.assertGreaterEqual(cautious_switch, eager_switch)
        self.assertLessEqual(
            sum(r.tokens for r in cautious.history), sum(r.tokens for r in eager.history)
        )

    def test_hysteresis_alone_delays_switch(self) -> None:
        """关掉外推后，10% 死区会把切换推迟到第 8 轮之后。"""
        cautious = Session(cfg=Config(max_entries=8, hybrid_mode="auto", hybrid_lookahead=False,
                                      switch_hysteresis=0.10, bridge_once=False))
        for question in ALL:
            cautious.ask(question)
        self.assertTrue(all(r.mode == "raw" for r in cautious.history))

    def test_bridge_appears_on_switch(self) -> None:
        """白盒：切回缓存且原文并不更便宜时，才允许挂桥接块。"""
        session = Session(cfg=Config(hybrid_mode="auto", switch_hysteresis=0.0))
        session.ask(Q1)
        session._last_mode = "raw"  # noqa: SLF001 - 构造切换瞬间
        session._dialogue = [(Q1, "答：每人 4 个苹果。")]  # noqa: SLF001

        composed, mode = session._cached_prompt(Q2, set(), raw_tokens=10_000)  # noqa: SLF001
        self.assertEqual(mode, "kccm+bridge")
        self.assertIn("# 上一轮原文", composed.prompt)

    def test_bridge_dropped_when_it_costs_more(self) -> None:
        """桥接不得把省下的 token 吃回去。"""
        session = Session(cfg=Config(hybrid_mode="auto", switch_hysteresis=0.0))
        session.ask(Q1)
        session._last_mode = "raw"  # noqa: SLF001
        session._dialogue = [(Q1, "答：每人 4 个苹果。")]  # noqa: SLF001

        _, mode = session._cached_prompt(Q2, set(), raw_tokens=10)  # noqa: SLF001
        self.assertEqual(mode, "kccm")

    def test_bridge_can_be_disabled(self) -> None:
        session = Session(cfg=Config(hybrid_mode="auto", switch_hysteresis=0.0, bridge_once=False))
        session.ask(Q1)
        session._last_mode = "raw"  # noqa: SLF001
        session._dialogue = [(Q1, "答：每人 4 个苹果。")]  # noqa: SLF001
        _, mode = session._cached_prompt(Q2, set(), raw_tokens=10_000)  # noqa: SLF001
        self.assertEqual(mode, "kccm")


if __name__ == "__main__":
    unittest.main()
