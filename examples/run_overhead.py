"""短会话 token 开销分析：固定成本、边际成本、交叉点。

回答的问题是「短会话里 KCCM 到底贵在哪」，因此按段拆解，而不是只报总数。
用法：python examples/run_overhead.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kccm import Config, Session, estimate_tokens  # noqa: E402
from kccm.composer import LITE_SYSTEM, SYSTEM_PROMPT  # noqa: E402

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

DENSE = "我有12个苹果、6个梨、3个香蕉、5个橘子、2个桃，3个人每个人得到多少"
SPARSE = "12个苹果分给3个人，每人多少"

BASE_SYSTEM = "你是一个乐于助人的助手。"  # 全量回放方的固定系统提示


def breakdown(prompt: str) -> dict[str, int]:
    """按 '# ' 段切分并计 token。"""
    out: dict[str, int] = {}
    for block in prompt.split("\n\n"):
        head = block.splitlines()[0] if block else ""
        key = head.replace("# ", "").split("（")[0].strip() or "空"
        out[key] = out.get(key, 0) + estimate_tokens(block)
    return out


def run(style: str, turns: int) -> list[tuple[int, int, int, dict[str, int], int]]:
    session = Session(cfg=Config(prompt_style=style))
    history: list[str] = []
    rows = []
    for question in QUESTIONS[:turns]:
        baseline = estimate_tokens(BASE_SYSTEM + "\n" + "\n".join(history + [question]))
        result = session.ask(question)
        history += [question, result.answer]
        rows.append((result.turn, result.tokens, baseline, breakdown(result.prompt),
                     len(result.snapshot)))
    return rows


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 78)
    print("[1] 固定开销：与缓存无关、每轮都要付")
    print("=" * 78)
    for name, text in (("full", SYSTEM_PROMPT), ("lite", LITE_SYSTEM)):
        print(f"  系统指令 {name:<5} {estimate_tokens(text):>4} tokens（{len(text)} 字符）")
    base = Session(cfg=Config())
    _, empty_tokens = base.composer._render([], "问题", base.cache)  # noqa: SLF001
    print(f"  空缓存整包（full）      {empty_tokens:>4} tokens  ← 第 1 轮的不可压缩部分")
    print(f"  空缓存整包（lite）      "
          f"{base.composer._render([], '问题', base.cache, style='lite')[1]:>4} tokens")

    print()
    print("=" * 78)
    print("[2] 逐轮对比：全量回放 vs KCCM(full) vs KCCM(lite)")
    print("=" * 78)
    full_rows = run("full", len(QUESTIONS))
    lite_rows = run("lite", len(QUESTIONS))
    print(f"{'轮次':<5}{'条目':>5}{'基线':>7}{'full':>7}{'lite':>7}"
          f"{'full/基线':>11}{'lite/基线':>11}")
    print("-" * 60)
    for (t, ft, b, _, n), (_, lt, _, _, _) in zip(full_rows, lite_rows):
        print(f"{t:<6}{n:>6}{b:>8}{ft:>8}{lt:>8}{ft / b:>10.2f}x{lt / b:>10.2f}x")

    print()
    print("=" * 78)
    print("[3] 第 2 轮的钱花在哪（full）")
    print("=" * 78)
    _, total, _, parts, _ = full_rows[1]
    for name, tokens in parts.items():
        print(f"  {name:<14}{tokens:>6} tokens{tokens / total * 100:>7.1f}%")
    print(f"  {'合计':<14}{total:>6} tokens")

    print()
    print("=" * 78)
    print("[4] 边际成本：每多一条缓存条目要付多少")
    print("=" * 78)
    for label, question in (("稀疏", SPARSE), ("标准", QUESTIONS[0]), ("密集", DENSE)):
        session = Session(cfg=Config())
        result = session.ask(question)
        n = len(result.snapshot)
        parts = breakdown(result.prompt)
        render_cost = parts.get("工作记忆", 0) + parts.get("槽位", 0)
        per_entry = render_cost / n if n else 0
        print(f"  {label:<6} 条目 {n:>2}  总计 {result.tokens:>4}  "
              f"条目渲染 {render_cost:>4}  单条约 {per_entry:>5.1f} tokens")

    print()
    print("=" * 78)
    print("[5] 交叉点：第几轮起 KCCM 更省")
    print("=" * 78)
    for style, rows in (("full", full_rows), ("lite", lite_rows)):
        cross = next((t for t, k, b, _, _ in rows if k < b), None)
        print(f"  {style:<5} 交叉点 {'第 ' + str(cross) + ' 轮' if cross else '本轮区间内未出现'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
