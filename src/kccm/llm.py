"""LLM 适配层：离线 RuleLLM（默认）与任意 OpenAI 兼容端点。"""

from __future__ import annotations

import os
import re
from typing import Protocol


class LLMClient(Protocol):
    name: str

    def complete(self, prompt: str) -> str:  # pragma: no cover - 协议定义
        ...


def _section(prompt: str, header: str) -> str:
    lines = prompt.splitlines()
    out: list[str] = []
    active = False
    for line in lines:
        if line.startswith("# "):
            active = line.strip().startswith(header)
            continue
        if active:
            out.append(line)
    return "\n".join(out).strip()


def parse_slots(prompt: str) -> dict[str, str]:
    slots: dict[str, str] = {}
    for line in _section(prompt, "# 槽位").splitlines():
        line = line.strip()
        if not line.startswith("- ") or "=" not in line:
            continue
        slot, value = line[2:].split("=", 1)
        value = value.split("（原")[0].strip()
        slots[slot.strip()] = value

    if slots:
        return slots

    # 精简模板没有独立槽位段：从「- 4·person.count·ttl10·h2·…」还原
    for line in _section(prompt, "# 记忆").splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        parts = line[2:].split("·")
        if len(parts) >= 2:
            slots[parts[1].strip()] = parts[0].strip()
    return slots


def parse_question(prompt: str) -> str:
    return _section(prompt, "# 当前问题")


class RuleLLM:
    """离线确定性 LLM 替身。

    它不是"假装会说话"——而是从 prompt 的 SLOTS 段做真实数值推理，
    因此可以用来验证「上下文是否足以支撑正确作答」这一核心命题。
    能力边界：只处理"总量 ÷ 人数"这一类问题；其余场景会明确说明缺什么。
    """

    name = "rule"
    ASK_MARKERS = ("多少", "每人", "几", "分", "呢")

    def complete(self, prompt: str) -> str:
        from .extractor import CN_NUM_RE
        from .normalize import cn_to_int

        slots = parse_slots(prompt)
        question = parse_question(prompt)
        if not slots:
            # 旁路模式：没有槽位段，退化为从原文里抽取「数量 + 被分配物」
            slots = _slots_from_raw(prompt)
        person = slots.get("person.count")

        count_slots = [
            s for s in slots if s.endswith(".count") and not s.startswith("person.")
        ]
        asked_numbers = set(re.findall(r"\d+", question))
        for m in CN_NUM_RE.finditer(question):
            value = cn_to_int(m.group())
            if value is not None:
                asked_numbers.add(str(value))

        total_slot = next((s for s in count_slots if slots[s] in asked_numbers), None)
        if total_slot is None and len(count_slots) == 1:
            total_slot = count_slots[0]
        asks = any(k in question for k in self.ASK_MARKERS)

        known = "、".join(f"{k}={v}" for k, v in slots.items()) or "（空）"

        if not asks:
            return f"本轮未提出计算请求。已记录槽位：{known}。"

        if total_slot is None or person is None:
            if len(count_slots) > 1:
                options = "、".join(f"{s}={slots[s]}" for s in count_slots)
                why = f"检测到多个数量（{options}），无法确定被统计对象"
            else:
                why = "缺少可计算的数量"
            return f"{why}。当前槽位：{known}。\n如需计算，请说明「总量」与「人数」。"

        total = _as_int(slots[total_slot])
        people = _as_int(person)
        if total is None or people in (None, 0):
            return "无法作答：槽位中的数值不可解析或人数为 0。"

        # 出现了无法归属的新数量时宁可说明，也不要拿旧总量硬算
        unassigned = asked_numbers - {str(total), str(people)}
        if unassigned:
            return (
                f"本轮出现新数量 {'、'.join(sorted(unassigned))}，但无法确定它对应哪个槽位。"
                f"当前槽位：{known}。"
            )

        obj = slots.get("item", "物品")
        quotient, remainder = divmod(total, people)
        restate = f"把 {total} 个{obj}平均分给 {people} 个人，每人分得多少？"

        lines = [f"还原：{restate}", f"{total} ÷ {people} = {quotient}"]
        if remainder:
            lines.append(f"余数 {remainder}，即每人 {quotient} 个，还剩 {remainder} 个。")
        lines.append(f"答：每人分得 {quotient} 个{obj}。")
        return "\n".join(lines)


def _slots_from_raw(prompt: str) -> dict[str, str]:
    """从原文里粗略还原「数量 / 被分配物 / 人数」，仅供离线替身在旁路模式使用。

    优先读当前问题：旁路时最新一轮的问题才是操作数来源，历史只作补充。
    """
    from .normalize import cn_to_int

    def find_person(text: str) -> str | None:
        m = re.search(r"(\d+)\s*(?:个)?\s*人", text)
        if m:
            return m.group(1)
        m = re.search(r"([一二三四五六七八九十两]+)\s*(?:个)?\s*人", text)
        if m:
            value = cn_to_int(m.group(1))
            return str(value) if value else None
        return None

    question = _section(prompt, "# 当前问题")
    # 只扫用户行：助手回答里的「4 个苹果」会把操作数带偏
    user_lines = [
        ln for ln in _section(prompt, "# 对话").splitlines() if ln.strip().startswith("用户:")
    ]
    slots: dict[str, str] = {}

    # 当前问题优先，其次从最近一轮往前找：旁路时最新提及才是有效操作数
    person = find_person(question) or next(
        (p for p in (find_person(ln) for ln in reversed(user_lines)) if p), None
    )
    if person:
        slots["person.count"] = person

    item_pattern = re.compile(r"(\d+)\s*个\s*([\u4e00-\u9fa5]{1,4})")
    bad_head = "呢吗吧啊的了个"

    def usable(match):
        return match is not None and match.group(2)[0] not in bad_head

    item = item_pattern.search(question)
    if not usable(item):
        item = next(
            (m for m in (item_pattern.search(ln) for ln in reversed(user_lines)) if usable(m)),
            None,
        )
    if item:
        slots["counting.count"] = item.group(1)
        slots["item"] = item.group(2)

    return slots


def _as_int(value: str) -> int | None:
    try:
        return int(float(value.strip()))
    except (TypeError, ValueError):
        return None


class OpenAICompatibleLLM:
    """OpenAI 协议客户端。DeepSeek / vLLM / Ollama / OpenAI 均可。"""

    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "deepseek-chat")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.temperature = temperature
        if not self.api_key:
            raise RuntimeError("缺少 OPENAI_API_KEY，请复制 .env.example 为 .env 并填写")

    def complete(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError("使用真实 LLM 需先安装：pip install -e '.[llm]'") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()


def get_llm(use_real: bool | None = None) -> LLMClient:
    if use_real is None:
        use_real = os.getenv("KCCM_USE_REAL_LLM", "0") == "1"
    if use_real:
        return OpenAICompatibleLLM()
    return RuleLLM()
