"""旁路策略对比：固定轮次 vs 自动择优 vs 近窗。

两套场景：
- normal：短问答，每轮增量约 40 tokens
- dense ：每轮多带一句背景，增量约 90 tokens（模拟长回答/长上下文会话）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kccm import Config, Session  # noqa: E402

QUESTIONS = [
    "我有12个苹果，3个人每个人得到多少",
    "那么四个人呢",
    "如果换成梨，还剩6个呢",
    "帮我列个采购清单，加香蕉和牛奶",
    "再各买5个",
    "预算50元",
    "那按最便宜的算",
    "总共要花多少钱",
]

PAD = "（背景：沿用上次确认的采购口径与库存盘点规则，预算按人天折算，不必重复确认）"

CONFIGS = [
    ("A 全KCCM", Config(max_entries=8)),
    ("B 固定7轮", Config(max_entries=8, hybrid_mode="fixed",
                       hybrid_turns=7, bridge_once=False)),
    ("C 自动h=0", Config(max_entries=8, hybrid_mode="auto",
                        switch_hysteresis=0.0, bridge_once=True)),
    ("D 自动h=.1", Config(max_entries=8, hybrid_mode="auto",
                         switch_hysteresis=0.10, bridge_once=True)),
    ("E 近2轮", Config(max_entries=8, hybrid_mode="fixed",
                       hybrid_turns=7, hybrid_recent=2)),
]


def run(cfg: Config, questions: list[str]) -> dict:
    session = Session(cfg=cfg)
    tokens, modes = [], []
    for question in questions:
        result = session.ask(question)
        tokens.append(result.tokens)
        modes.append(result.mode)
    return {
        "tokens": tokens,
        "modes": modes,
        "total": sum(tokens),
        "entries": len(session.cache),
        "switch": next((i + 1 for i, m in enumerate(modes) if m.startswith("kccm")), None),
        "ok": "12 ÷ 4 = 3" in session.history[1].answer,
    }


def report(title: str, questions: list[str]) -> None:
    results = {name: run(cfg, questions) for name, cfg in CONFIGS}
    names = [name for name, _ in CONFIGS]
    print(f"\n=== {title} ===")
    head = f"{'轮次':<5}" + "".join(f"{n:>14}" for n in names)
    print(head)
    print("-" * len(head))
    for i in range(len(questions)):
        cells = ""
        for name in names:
            mark = "*" if results[name]["modes"][i] == "raw" else (
                "+" if results[name]["modes"][i] == "kccm+bridge" else " ")
            cells += f"{results[name]['tokens'][i]:>10}{mark:>4}"
        print(f"T{i + 1:<4}{cells}")
    print("-" * len(head))
    print(f"{'累计':<4}" + "".join(f"{results[n]['total']:>14}" for n in names))
    print(f"{'切换轮':<3}" + "".join(f"{str(results[n]['switch']):>14}" for n in names))
    print(f"{'T2正确':<4}" + "".join(f"{str(results[n]['ok']):>14}" for n in names))
    print("* 原文模式    + 缓存+桥接   空格 = 纯缓存")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="旁路策略对比")
    parser.add_argument("--only", choices=["normal", "dense"], default=None)
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.only in (None, "normal"):
        report("normal：短问答（每轮增量 ≈40）", QUESTIONS)
    if args.only in (None, "dense"):
        report("dense：长上下文（每轮增量 ≈90）", [q + PAD for q in QUESTIONS])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
