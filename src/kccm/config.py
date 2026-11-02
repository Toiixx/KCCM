"""参数装载与校验。越界即刻报错，不做静默兜底。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import DEFAULT_TTL_MAX


@dataclass(frozen=True)
class Config:
    ttl_max: int = DEFAULT_TTL_MAX
    max_entries: int = 32
    ttl_tolerance: int = 2
    prompt_budget: int = 6912
    summary_max_chars: int = 120
    template_version: str = "v2"
    prompt_style: str = "full"  # full | lite
    # 短会话旁路
    hybrid_mode: str = "off"    # off | fixed | auto
    hybrid_turns: int = 6       # fixed 模式：前 N 轮走原文
    hybrid_recent: int = 0      # 旁路带几轮原文；0 = 全量历史
    hybrid_skip_cache: bool = False  # True = 旁路期完全不建缓存（会冷启动，不建议）
    switch_hysteresis: float = 0.10  # auto 模式的死区：便宜不足 10% 不切换
    hybrid_lookahead: bool = True    # auto 模式：按下一轮预测长度决定，而不是只看当下
    bridge_once: bool = True    # 切回 KCCM 的首轮补一次上一轮原文

    def __post_init__(self) -> None:
        if self.ttl_max < 1:
            raise ValueError("ttl_max 必须 >= 1")
        if self.max_entries < 1:
            raise ValueError("max_entries 必须 >= 1")
        if self.ttl_tolerance < 0:
            raise ValueError("ttl_tolerance 必须 >= 0")
        if self.prompt_budget < 256:
            raise ValueError("prompt_budget 过小，无法容纳系统指令与当前问题")
        if self.summary_max_chars < 8:
            raise ValueError("summary_max_chars 过小")
        if self.prompt_style not in ("full", "lite"):
            raise ValueError("prompt_style 只支持 full / lite")
        if self.hybrid_turns < 0:
            raise ValueError("hybrid_turns 必须 >= 0")
        if self.hybrid_recent < 0:
            raise ValueError("hybrid_recent 必须 >= 0")
        if self.hybrid_mode not in ("off", "fixed", "auto"):
            raise ValueError("hybrid_mode 只支持 off / fixed / auto")
        if not 0.0 <= self.switch_hysteresis < 1.0:
            raise ValueError("switch_hysteresis 必须落在 [0, 1)")

    # 预算三段：当前问题 / 缓存条目 / 槽位
    @property
    def budget_question(self) -> int:
        return int(0.20 * self.prompt_budget)

    @property
    def budget_memory(self) -> int:
        return int(0.55 * self.prompt_budget)

    @property
    def budget_slots(self) -> int:
        return int(0.25 * self.prompt_budget)


_FIELDS = {
    "ttl_max": int,
    "max_entries": int,
    "ttl_tolerance": int,
    "prompt_budget": int,
    "summary_max_chars": int,
    "template_version": str,
    "prompt_style": str,
    "hybrid_mode": str,
    "hybrid_turns": int,
    "hybrid_recent": int,
    "hybrid_skip_cache": bool,
    "switch_hysteresis": float,
    "hybrid_lookahead": bool,
    "bridge_once": bool,
}


def _cast(raw: str, kind: type):
    if kind is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if kind is float:
        return float(raw)
    return kind(raw)


def load_config(path: str | Path | None = None) -> Config:
    """读取 config.yaml。

    只支持 ``key: value`` 的扁平结构，不引入 PyYAML 依赖：
    本项目的配置项是标量，为一个标量解析器装一个 YAML 引擎不划算。
    """
    if path is None:
        return Config()

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")

    data: dict[str, Any] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key not in _FIELDS:
            continue
        cast = _FIELDS[key]
        data[key] = _cast(value, cast) if cast is not str else value
    return Config(**data)
