"""对比实验：累积式全量回放 vs KCCM 关键词缓存。

基线 = 每轮把「所有历史问答」完整拼进 prompt（滚雪球）。
KCCM = 每轮只拼「当前问题 + 缓存条目 + 槽位」。
两者使用同一个回答器，因此差异只来自上下文构造方式。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kccm import Config, Session, Tracer, estimate_tokens  # noqa: E402

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="全量回放 vs KCCM 的 token 对比")
    parser.add_argument("--turns", type=int, default=len(QUESTIONS))
    parser.add_argument("--max-entries", type=int, default=8)
    parser.add_argument("--style", default="full", choices=["full", "lite"])
    parser.add_argument("--trace", default=str(ROOT / "examples" / "traces" / "apple.jsonl"))
    parser.add_argument("--verbose", action="store_true", help="打印每轮回答")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    trace_path = Path(args.trace)
    session = Session(
        cfg=Config(max_entries=args.max_entries, prompt_style=args.style),
        tracer=Tracer(trace_path),
    )

    history: list[str] = []
    rows: list[tuple[int, int, int]] = []

    for question in QUESTIONS[: args.turns]:
        # 基线取"本轮输入"：此前全部问答 + 当前问题。当前回答尚未产生，不能计入。
        baseline = estimate_tokens("\n".join(history + [f"用户: {question}"]))
        result = session.ask(question)
        history.append(f"用户: {question}")
        history.append(f"助手: {result.answer}")
        rows.append((result.turn, result.tokens, baseline))
        if args.verbose:
            print(f"T{result.turn} {question}\n   {result.answer.splitlines()[0]}")

    print(f"{'轮次':<6}{'全量回放':>10}{'KCCM':>10}{'压缩比':>10}")
    print("-" * 38)
    for turn, kccm_tokens, baseline_tokens in rows:
        ratio = baseline_tokens / kccm_tokens if kccm_tokens else 0
        print(f"{turn:<8}{baseline_tokens:>10}{kccm_tokens:>10}{ratio:>9.2f}x")

    first, last = rows[0], rows[-1]
    crossover = next((t for t, k, b in rows if k < b), None)
    print("-" * 38)
    print(f"全量回放：{first[2]} → {last[2]} tokens（{last[2] / first[2]:.2f}x，随轮次线性上升）")
    print(f"KCCM   ：{first[1]} → {last[1]} tokens（{last[1] / first[1]:.2f}x，近似常数）")
    if crossover:
        print(f"交叉点：第 {crossover} 轮起 KCCM 更省；短会话下固定开销（系统指令 + 格式）占主导")
    print(f"轨迹已写入: {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
