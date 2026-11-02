"""DeepSeek Flash 调用示例（requests 实现，Python 3.9+，默认流式）。

两个用途：
1. 单独运行：一条 user 消息，验证密钥、网络与返回结构。
       python examples/deepseek_flash_demo.py
2. 作为 KCCM 的 LLM 后端：实现 ``LLMClient`` 协议（``complete()`` + ``name``），
   可直接注入 ``Session(llm=...)``，与 ``RuleLLM`` / ``OpenAICompatibleLLM`` 平级。
       python examples/deepseek_flash_demo.py --kccm

默认 ``stream=True``：模型每吐一小段就打印一段（打字机效果）。
加 ``--no-stream`` 退回一次性返回，便于对比两种模式的差异。

依赖：requests（``pip install requests``）
密钥：环境变量 DEEPSEEK_API_KEY，不在代码里硬编码。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_TIMEOUT = 15          # 建立连接的上限（秒）
DEFAULT_READ_TIMEOUT = 120    # 流式时"相邻两个分片之间"的最长等待（秒）
DEMO_PROMPT = "你好，请简要介绍你自己"

SSE_DATA_PREFIX = "data:"
SSE_DONE = "[DONE]"


class DeepSeekAPIError(RuntimeError):
    """携带状态码与响应体的异常，便于上层打印可诊断的错误信息。

    ``partial`` 只在流式传输中途失败时填充：已经吐出来的那部分文字。
    流式一旦中断，前面的内容无法从网络重取，只能靠这里带出来。
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        body: Optional[str] = None,
        partial: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.partial = partial

    def describe(self) -> str:
        parts = [str(self)]
        if self.status_code is not None:
            parts.append(f"HTTP 状态码: {self.status_code}")
        if self.partial:
            parts.append(f"已收到 {len(self.partial)} 字（内容不完整，需重试）")
        if self.body:
            parts.append(f"响应体: {self.body[:800]}")
        return "\n".join(parts)


@dataclass(frozen=True)
class ChatResult:
    """一次成功调用的结果：正文 + 用量 + 实际模型名 + 结束原因。

    ``reasoning`` 只有推理模型（如 deepseek-flash）才有：先想后说，
    思考过程走 ``reasoning_content``，正文走 ``content``，两者是分开的两条流。
    """

    content: str
    usage: Dict[str, Any]
    model: str
    finish_reason: str = ""
    reasoning: str = ""


def get_api_key() -> str:
    """读取并校验密钥。缺失时立刻抛错，避免带着空 key 发请求。"""
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise DeepSeekAPIError(
            "未找到环境变量 DEEPSEEK_API_KEY。\n"
            "  Windows: setx DEEPSEEK_API_KEY \"sk-...\"（重开终端生效）\n"
            "  macOS/Linux: export DEEPSEEK_API_KEY=\"sk-...\""
        )
    return key


def build_messages(content: str) -> List[Dict[str, str]]:
    """构造对话消息。这里是最小单轮；多轮时按 role/content 追加即可。"""
    return [{"role": "user", "content": content}]


def build_payload(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 512,
    stream: bool = False,
    include_usage: bool = True,
) -> Dict[str, Any]:
    """组装请求体。五个参数显式写出，避免默认值随服务端漂移。

    流式时多带一个 ``stream_options``：不声明它，服务端**不会**在流里回 usage，
    你就只能看到文字、看不到 token 消耗。
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if stream and include_usage:
        payload["stream_options"] = {"include_usage": True}
    return payload


def parse_response(response: "requests.Response") -> ChatResult:
    """非流式解析：校验状态码 → 解析 JSON → 取 choices[0].message.content。"""
    if response.status_code != 200:
        raise DeepSeekAPIError(
            "接口返回非 200",
            status_code=response.status_code,
            body=response.text,
        )

    try:
        data = response.json()
    except ValueError as exc:  # 非 JSON 响应（网关错误页、空响应等）
        raise DeepSeekAPIError(
            "响应不是合法 JSON", status_code=response.status_code, body=response.text
        ) from exc

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise DeepSeekAPIError(
            "响应缺少 choices 或为空",
            status_code=response.status_code,
            body=json.dumps(data, ensure_ascii=False),
        )

    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise DeepSeekAPIError(
            "choices[0].message.content 缺失或为空",
            status_code=response.status_code,
            body=json.dumps(data, ensure_ascii=False),
        )

    finish = first.get("finish_reason") if isinstance(first, dict) else None
    reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
    return ChatResult(
        content=content,
        usage=data.get("usage") or {},
        model=data.get("model") or "",
        finish_reason=finish or "",
        reasoning=reasoning if isinstance(reasoning, str) else "",
    )


# ---------------------------------------------------------------------------
# 流式解析（纯函数，不碰网络，可离线测试）
# ---------------------------------------------------------------------------


def consume_stream(
    lines: Iterable[str],
    on_delta: Optional[Callable[[str], None]] = None,
    on_reasoning: Optional[Callable[[str], None]] = None,
) -> ChatResult:
    """把 SSE 行流拼成完整结果。

    流式响应长这样（每个分片一行，空行分隔）：

        data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}

        data: {"choices":[{"delta":{"reasoning_content":"让我算算"}}]}

        data: {"choices":[{"delta":{"content":"你好"}}]}

        data: {"choices":[{"delta":{"content":"呀"},"finish_reason":"stop"}]}

        data: {"choices":[],"usage":{"prompt_tokens":9,...}}

        data: [DONE]

    要点：
    - ``delta.content`` 是**增量**，不是全量。要自己累加，不能覆盖。
    - 推理模型先吐 ``reasoning_content``（思考），再吐 ``content``（回答）。
      两者是两条独立的流，同一帧里通常只有一个非空。
    - 每帧只有一小段，可能是半个词。放心，JSON 的一个字符串不会跨行拆开。
    - 最后一帧 ``choices`` 是空数组，只带 usage，不能当成错误。
    - ``[DONE]`` 是结束哨兵，不是 JSON，直接跳出。
    """
    pieces: List[str] = []
    thoughts: List[str] = []
    usage: Dict[str, Any] = {}
    model_name = ""
    finish_reason = ""
    saw_data = False

    for raw in lines:
        if isinstance(raw, bytes):  # 防御：万一调用方没开 decode_unicode
            raw = raw.decode("utf-8", errors="replace")
        line = raw.strip()

        if not line:
            continue                      # 空行：SSE 用它分隔事件，跳过
        if line.startswith(":"):
            continue                      # 以冒号开头是注释/心跳，跳过
        if not line.startswith(SSE_DATA_PREFIX):
            continue                      # event: / id: / retry: 等字段，本接口不用

        payload_text = line[len(SSE_DATA_PREFIX):].strip()
        if payload_text == SSE_DONE:
            saw_data = True
            break                         # 正常收尾

        saw_data = True
        try:
            event = json.loads(payload_text)
        except ValueError as exc:
            raise DeepSeekAPIError(
                "流式分片不是合法 JSON", body=payload_text
            ) from exc

        if not isinstance(event, dict):
            raise DeepSeekAPIError("流式分片不是 JSON 对象", body=payload_text)

        # 有的网关会把错误塞进流里，HTTP 状态码却是 200
        if "error" in event:
            raise DeepSeekAPIError(
                "流式返回错误", body=json.dumps(event, ensure_ascii=False)
            )

        model_name = model_name or event.get("model") or ""
        if event.get("usage"):
            usage = event["usage"]

        choices = event.get("choices") or []
        if not choices:
            continue                      # usage-only 帧，继续读

        first = choices[0]
        if not isinstance(first, dict):
            continue

        if first.get("finish_reason"):
            finish_reason = first["finish_reason"]

        delta = first.get("delta") or {}
        if not isinstance(delta, dict):
            continue

        thought = delta.get("reasoning_content")
        if isinstance(thought, str) and thought:
            thoughts.append(thought)
            if on_reasoning is not None:
                on_reasoning(thought)

        piece = delta.get("content")
        if isinstance(piece, str) and piece:
            pieces.append(piece)
            if on_delta is not None:
                on_delta(piece)

    if not saw_data:
        raise DeepSeekAPIError("流式响应为空：一行 data 都没收到")

    content = "".join(pieces)
    if not content:
        raise DeepSeekAPIError(
            f"模型没有返回正文（finish_reason={finish_reason or '空'}）。"
            "推理模型的 max_tokens 含思考部分，预算太小会把回答挤没，试着调大。"
        )

    return ChatResult(
        content=content,
        usage=usage,
        model=model_name,
        finish_reason=finish_reason or "stop",
        reasoning="".join(thoughts),
    )


@dataclass
class DeepSeekFlashLLM:
    """KCCM 的 LLM 后端：实现 ``complete(prompt) -> str``。

    与 ``RuleLLM`` 的差别只在"谁来回答"，接口保持一致，
    因此可以直接 ``Session(cfg=Config(), llm=DeepSeekFlashLLM())``。
    """

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: int = DEFAULT_TIMEOUT
    read_timeout: int = DEFAULT_READ_TIMEOUT
    api_key: Optional[str] = None
    name: str = "deepseek-flash"
    stream: bool = True                  # 默认流式
    include_usage: bool = True           # 流式时也要 token 用量
    on_delta: Optional[Callable[[str], None]] = None       # 正文分片的去处
    on_reasoning: Optional[Callable[[str], None]] = None   # 思考分片的去处
    last_usage: Dict[str, Any] = field(default_factory=dict)
    last_finish_reason: str = ""
    last_reasoning: str = ""

    # ---------- 协议方法 ----------
    def complete(
        self,
        prompt: str,
        on_delta: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
    ) -> str:
        """KCCM 每轮把拼好的 prompt 交进来，这里包成单条 user 消息。

        回调都是可选的：不传就用实例上的 ``self.on_delta`` / ``self.on_reasoning``。
        ``Session.ask()`` 只传一个参数，所以想在 KCCM 里看到打字机效果，
        在创建实例后设置 ``llm.on_delta = ...`` 即可，不用改 Session。
        """
        return self.chat(
            build_messages(prompt), on_delta=on_delta, on_reasoning=on_reasoning
        )

    # ---------- HTTP ----------
    def endpoint(self) -> str:
        """拼接地址，兼容 base_url 是否已带 /v1。"""
        base = self.base_url.rstrip("/")
        return f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"

    def chat(
        self,
        messages: List[Dict[str, str]],
        on_delta: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
    ) -> str:
        """按 ``self.stream`` 分派到流式或一次性。"""
        sink = on_delta if on_delta is not None else self.on_delta
        thinker = on_reasoning if on_reasoning is not None else self.on_reasoning
        if self.stream:
            return self._chat_stream(messages, sink, thinker)
        return self._chat_once(messages)

    def _headers(self, key: str, stream: bool) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {key}",
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
        }

    def _post(self, messages: List[Dict[str, str]], stream: bool) -> "requests.Response":
        """发请求，把网络层异常统一包成 DeepSeekAPIError。"""
        key = self.api_key or get_api_key()
        payload = build_payload(
            self.model,
            messages,
            self.temperature,
            self.max_tokens,
            stream=stream,
            include_usage=self.include_usage,
        )
        try:
            return requests.post(
                self.endpoint(),
                headers=self._headers(key, stream),
                json=payload,
                # 元组 = (连接超时, 读取超时)。流式时读取超时管的是"两个分片之间"，
                # 不能沿用总时长，否则生成到一半被掐断。
                timeout=(self.timeout, self.read_timeout),
                stream=stream,
            )
        except requests.Timeout as exc:
            raise DeepSeekAPIError(f"请求超时（连接 {self.timeout}s）") from exc
        except requests.RequestException as exc:
            raise DeepSeekAPIError(f"网络错误: {exc}") from exc

    def _chat_once(self, messages: List[Dict[str, str]]) -> str:
        """非流式：一次性拿完整 JSON。"""
        response = self._post(messages, stream=False)
        result = parse_response(response)
        self.last_usage = result.usage
        self.last_finish_reason = result.finish_reason
        return result.content

    def _chat_stream(
        self,
        messages: List[Dict[str, str]],
        on_delta: Optional[Callable[[str], None]],
        on_reasoning: Optional[Callable[[str], None]] = None,
    ) -> str:
        """流式：边收边吐，最后仍返回完整字符串（KCCM 需要整段）。"""
        response = self._post(messages, stream=True)

        buffer: List[str] = []      # 已吐出的片段，用于中途失败时保住内容

        def sink(piece: str) -> None:
            buffer.append(piece)
            if on_delta is not None:
                on_delta(piece)

        try:
            if response.status_code != 200:
                raise DeepSeekAPIError(
                    "接口返回非 200",
                    status_code=response.status_code,
                    body=response.text,
                )
            # 必须显式指定编码：流式时 requests 拿不到完整 body，
            # 猜编码会退化成 latin-1，中文全变乱码。
            response.encoding = "utf-8"

            try:
                result = consume_stream(
                    response.iter_lines(decode_unicode=True), sink, on_reasoning
                )
            except requests.Timeout as exc:
                raise DeepSeekAPIError(
                    f"流式读取超时（相邻分片间隔超过 {self.read_timeout}s）",
                    partial="".join(buffer),
                ) from exc
            except requests.RequestException as exc:
                raise DeepSeekAPIError(
                    f"流式传输中断: {exc}", partial="".join(buffer)
                ) from exc
        finally:
            response.close()        # 不关会漏连接，长时间跑必炸

        self.last_usage = result.usage
        self.last_finish_reason = result.finish_reason
        self.last_reasoning = result.reasoning
        return result.content


# ---------------------------------------------------------------------------
# 命令行示例
# ---------------------------------------------------------------------------


def _make_printers(show_reasoning: bool) -> Tuple[Callable[[str], None], Callable[[str], None]]:
    """返回 (正文回调, 思考回调)。

    推理模型会先想后说：思考阶段可能持续数秒，正文一个字都不出。
    如果只盯着 content，用户会以为卡死了。所以：
    - 默认：思考一开始就打一行提示到 stderr，正文仍干干净净走 stdout
    - ``--show-reasoning``：思考过程实时打到 stderr，正文照常走 stdout
    分流的妙处是 ``python x.py > out.txt`` 时，out.txt 里只有答案，没有过程。
    """
    state = {"thought": False, "content": False}

    def on_content(piece: str) -> None:
        if not state["content"]:
            state["content"] = True
            if state["thought"] and show_reasoning:
                sys.stdout.flush()
                print(file=sys.stderr)      # 思考块结束，换行
        print(piece, end="", flush=True)    # flush 不写就等于没有流式效果

    def on_reasoning(piece: str) -> None:
        if not state["thought"]:
            state["thought"] = True
            sys.stdout.flush()              # 先把已缓冲的正文吐出来，别让 stderr 插队
            if show_reasoning:
                print("[思考] ", end="", file=sys.stderr, flush=True)
            else:
                print(
                    "[模型正在思考…（加 --show-reasoning 可查看推理过程）]",
                    file=sys.stderr,
                    flush=True,
                )
        if show_reasoning:
            print(piece, end="", file=sys.stderr, flush=True)

    return on_content, on_reasoning


def _usage_line(llm: "DeepSeekFlashLLM") -> str:
    """把用量拼成一行。推理模型的 completion 里含 reasoning_tokens。"""
    usage = llm.last_usage or {}
    if not usage:
        return ""
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens")
    extra = f" reasoning={reasoning}" if reasoning else ""
    return (
        f"[用量] prompt={usage.get('prompt_tokens')} "
        f"completion={usage.get('completion_tokens')}{extra} "
        f"total={usage.get('total_tokens')} finish={llm.last_finish_reason}"
    )


def run_single_turn(prompt: str, stream: bool = True, show_reasoning: bool = False) -> int:
    """最小示例：一条消息、一次请求、逐字打印结果。"""
    llm = DeepSeekFlashLLM(stream=stream)
    print(f"[模型] {llm.model}")
    print(f"[请求] {prompt}")
    print(f"[模式] stream={stream}")
    print("-" * 60)

    on_content, on_reasoning = _make_printers(show_reasoning)
    try:
        if stream:
            content = llm.complete(prompt, on_delta=on_content, on_reasoning=on_reasoning)
        else:
            content = llm.complete(prompt)
    except DeepSeekAPIError as exc:
        print(file=sys.stderr)
        print("[失败]", file=sys.stderr)
        print(exc.describe(), file=sys.stderr)
        return 1

    if not stream:
        print(content)
    print()
    print("-" * 60)

    line = _usage_line(llm)
    if line:
        print(line)
    return 0


def run_with_kccm(stream: bool = True, show_reasoning: bool = False) -> int:
    """把同一个后端接到 KCCM 上，验证两轮指代是否成立。"""
    sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "src")))
    try:
        from kccm import Config, Session
    except ImportError:
        print("[失败] 未找到 kccm 包，请先 pip install -e .", file=sys.stderr)
        return 1

    llm = DeepSeekFlashLLM(stream=stream)
    on_content, on_reasoning = _make_printers(show_reasoning)
    if stream:
        # Session 只调 complete(prompt)，拿不到回调；挂到实例上即可生效
        llm.on_delta = on_content
        llm.on_reasoning = on_reasoning

    session = Session(cfg=Config(), llm=llm)
    for question in ("我有12个苹果，3个人每个人得到多少", "那么四个人呢"):
        print(f"\n[第 {len(session.history) + 1} 轮] {question}")
        try:
            result = session.ask(question)
        except DeepSeekAPIError as exc:
            print(file=sys.stderr)
            print("[失败]", file=sys.stderr)
            print(exc.describe(), file=sys.stderr)
            return 1

        if stream:
            print()                      # 流式内容没有换行，这里补一个
        print(f"[完成] 模式={result.mode} prompt={result.tokens} tokens")
        if not stream:
            print(result.answer)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DeepSeek Flash 调用示例（默认流式）")
    parser.add_argument("--prompt", default=DEMO_PROMPT, help="单轮示例的用户输入")
    parser.add_argument("--kccm", action="store_true", help="接入 KCCM 跑两轮对话")
    parser.add_argument("--no-stream", action="store_true", help="关闭流式，一次性返回")
    parser.add_argument(
        "--show-reasoning", action="store_true", help="实时显示模型的思考过程（走 stderr）"
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # 入口处先校验密钥，缺了就直说，不浪费一次请求
    try:
        get_api_key()
    except DeepSeekAPIError as exc:
        print("[失败]", file=sys.stderr)
        print(exc.describe(), file=sys.stderr)
        return 1

    stream = not args.no_stream
    if args.kccm:
        return run_with_kccm(stream, args.show_reasoning)
    return run_single_turn(args.prompt, stream, args.show_reasoning)


if __name__ == "__main__":
    raise SystemExit(main())
