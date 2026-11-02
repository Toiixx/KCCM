"""命令行入口：交互式多轮，逐回合打印缓存状态。"""

from __future__ import annotations

import argparse
import sys

from .composer import ContextBudgetExceeded
from .config import load_config
from .llm import get_llm
from .session import Session
from .trace import Tracer

DEMO_SCRIPT = [
    "我有12个苹果，3个人每个人得到多少",
    "那么四个人呢",
]


def _ensure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - 某些终端不支持重配置
                pass


def render_turn(result) -> str:
    lines = [
        "=" * 66,
        f"T{result.turn}  USER: {result.question}",
        f"提取: {list(result.extracted)}   差分: {list(result.diff_tokens)}",
        f"写入: {[(u.kind.value, u.keyword, u.slot) for u in result.upserts]}",
    ]
    if result.expired:
        lines.append(f"TTL 过期: {list(result.expired)}")
    if result.evicted:
        lines.append(f"LFU 淘汰: {list(result.evicted)}")
    lines.append(f"缓存快照（ttl/hit）: {_snapshot_text(result.snapshot)}")
    lines.append(f"prompt: {result.tokens} tokens / 裁剪级别 {result.level}")
    lines.append("-" * 66)
    lines.append(result.answer)
    return "\n".join(lines)


def _snapshot_text(snapshot) -> str:
    return "  ".join(f"{e['keyword']}={e['ttl']}/{e['hit_count']}" for e in snapshot)


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8()
    parser = argparse.ArgumentParser(prog="kccm", description="关键词缓存多轮对话 Demo")
    parser.add_argument("--config", default=None, help="config.yaml 路径")
    parser.add_argument("--trace", default=None, help="输出 JSONL 轨迹的路径")
    parser.add_argument("--real", action="store_true", help="使用真实 LLM（需 OPENAI_API_KEY）")
    parser.add_argument("--demo", action="store_true", help="跑内置两回合示例后退出")
    parser.add_argument("--show-prompt", action="store_true", help="打印完整 prompt")
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()
    tracer = Tracer(args.trace) if args.trace else None
    session = Session(cfg=cfg, llm=get_llm(use_real=args.real), tracer=tracer)

    def run(text: str) -> None:
        try:
            result = session.ask(text)
        except ContextBudgetExceeded as exc:
            print(f"[预算超限] {exc}")
            print(f"[明细] {exc.detail}")
            return
        print(render_turn(result))
        if args.show_prompt:
            print("-" * 66)
            print(result.prompt)

    if args.demo:
        for text in DEMO_SCRIPT:
            run(text)
        return 0

    print("KCCM 交互模式（空行退出）")
    while True:
        try:
            text = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        run(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
