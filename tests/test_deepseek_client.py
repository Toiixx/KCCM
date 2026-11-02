import importlib.util
import json
import sys
import unittest
import unittest.mock
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "deepseek_flash_demo.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("deepseek_flash_demo", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["deepseek_flash_demo"] = module
    spec.loader.exec_module(module)
    return module


try:
    demo = _load_module()
    HAS_REQUESTS = True
except ImportError:  # 未安装 requests 时跳过，不阻塞其余测试
    demo = None
    HAS_REQUESTS = False


class FakeResponse:
    def __init__(self, status_code, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload, ensure_ascii=False)

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


@unittest.skipUnless(HAS_REQUESTS, "未安装 requests")
class DeepSeekClientTest(unittest.TestCase):
    def test_payload_contract(self):
        payload = demo.build_payload(
            "deepseek-flash", demo.build_messages("你好"), 0.7, 512, False
        )
        self.assertEqual(payload["model"], "deepseek-flash")
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["max_tokens"], 512)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"], [{"role": "user", "content": "你好"}])

    def test_endpoint_join(self):
        self.assertEqual(
            demo.DeepSeekFlashLLM().endpoint(),
            "https://api.deepseek.com/v1/chat/completions",
        )
        self.assertEqual(
            demo.DeepSeekFlashLLM(base_url="https://api.deepseek.com/v1").endpoint(),
            "https://api.deepseek.com/v1/chat/completions",
        )

    def test_parse_success(self):
        response = FakeResponse(
            200,
            {
                "model": "deepseek-flash",
                "choices": [{"message": {"role": "assistant", "content": "你好呀"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
            },
        )
        result = demo.parse_response(response)
        self.assertEqual(result.content, "你好呀")
        self.assertEqual(result.usage["total_tokens"], 12)

    def test_http_error_carries_status_and_body(self):
        response = FakeResponse(401, {"error": {"message": "Authentication Fails"}})
        with self.assertRaises(demo.DeepSeekAPIError) as ctx:
            demo.parse_response(response)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Authentication Fails", ctx.exception.body)
        self.assertIn("HTTP 状态码: 401", ctx.exception.describe())

    def test_invalid_json(self):
        response = FakeResponse(502, payload=None, text="<html>Bad Gateway</html>")
        with self.assertRaises(demo.DeepSeekAPIError) as ctx:
            demo.parse_response(response)
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("Bad Gateway", ctx.exception.body)

    def test_missing_choices(self):
        response = FakeResponse(200, {"model": "deepseek-flash"})
        with self.assertRaises(demo.DeepSeekAPIError) as ctx:
            demo.parse_response(response)
        self.assertIn("choices", str(ctx.exception))

    def test_empty_content(self):
        response = FakeResponse(200, {"choices": [{"message": {"content": "   "}}]})
        with self.assertRaises(demo.DeepSeekAPIError):
            demo.parse_response(response)

    def test_missing_key_raises(self):
        import os

        original = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            with self.assertRaises(demo.DeepSeekAPIError) as ctx:
                demo.get_api_key()
            self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))
        finally:
            if original is not None:
                os.environ["DEEPSEEK_API_KEY"] = original


# ---------------------------------------------------------------------------
# 流式：分片拼装与异常路径
# ---------------------------------------------------------------------------


def sse(payload) -> str:
    """把字典包成一行 SSE data。"""
    if isinstance(payload, str):
        return f"data: {payload}"
    return "data: " + json.dumps(payload, ensure_ascii=False)


def chunk(**kwargs):
    """构造一个最小的流式分片。"""
    base = {"id": "x", "model": "deepseek-flash", "choices": [{"index": 0, "delta": {}}]}
    delta = kwargs.pop("delta", None)
    if delta is not None:
        base["choices"][0]["delta"] = delta
    base["choices"][0].update(kwargs)
    return base


@unittest.skipUnless(HAS_REQUESTS, "未安装 requests")
class DeepSeekStreamTest(unittest.TestCase):
    def test_deltas_are_accumulated(self):
        lines = [
            sse(chunk(delta={"role": "assistant", "content": ""})),
            "",
            sse(chunk(delta={"content": "12 ÷ 4"})),
            sse(chunk(delta={"content": " = 3"})),
            "",
            sse("[DONE]"),
        ]
        seen = []
        result = demo.consume_stream(lines, on_delta=seen.append)
        self.assertEqual(result.content, "12 ÷ 4 = 3")
        self.assertEqual(seen, ["12 ÷ 4", " = 3"])   # 回调按到达顺序触发
        self.assertEqual(result.finish_reason, "stop")

    def test_usage_only_frame_is_not_an_error(self):
        lines = [
            sse(chunk(delta={"content": "好"}, finish_reason="stop")),
            sse({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                                          "total_tokens": 7}}),
            sse("[DONE]"),
        ]
        result = demo.consume_stream(lines)
        self.assertEqual(result.content, "好")
        self.assertEqual(result.usage["total_tokens"], 7)   # usage 来自最后那帧

    def test_done_sentinel_stops_reading(self):
        lines = [
            sse(chunk(delta={"content": "A"})),
            sse("[DONE]"),
            sse(chunk(delta={"content": "不该被读到"})),
        ]
        self.assertEqual(demo.consume_stream(lines).content, "A")

    def test_reasoning_kept_separate_from_content(self):
        lines = [
            sse(chunk(delta={"reasoning_content": "先想"})),
            sse(chunk(delta={"reasoning_content": "一下"})),
            sse(chunk(delta={"content": "答案"})),
        ]
        thoughts = []
        result = demo.consume_stream(lines, on_reasoning=thoughts.append)
        self.assertEqual(result.content, "答案")          # 正文不含思考
        self.assertEqual(result.reasoning, "先想一下")
        self.assertEqual(thoughts, ["先想", "一下"])

    def test_error_event_inside_stream(self):
        lines = [sse({"error": {"message": "quota exceeded"}})]
        with self.assertRaises(demo.DeepSeekAPIError) as ctx:
            demo.consume_stream(lines)
        self.assertIn("quota exceeded", ctx.exception.body)

    def test_invalid_json_line(self):
        with self.assertRaises(demo.DeepSeekAPIError) as ctx:
            demo.consume_stream(["data: {oops"])
        self.assertIn("{oops", ctx.exception.body)

    def test_empty_stream(self):
        with self.assertRaises(demo.DeepSeekAPIError) as ctx:
            demo.consume_stream(["", ": keep-alive"])
        self.assertIn("空", str(ctx.exception))

    def test_comment_and_blank_lines_ignored(self):
        lines = [": ping", "", sse(chunk(delta={"content": "ok"})), ""]
        self.assertEqual(demo.consume_stream(lines).content, "ok")

    def test_no_content_blames_max_tokens(self):
        lines = [sse(chunk(finish_reason="length"))]
        with self.assertRaises(demo.DeepSeekAPIError) as ctx:
            demo.consume_stream(lines)
        self.assertIn("max_tokens", str(ctx.exception))

    def test_payload_carries_stream_options(self):
        streamed = demo.build_payload("deepseek-flash", [], stream=True)
        oneshot = demo.build_payload("deepseek-flash", [], stream=False)
        self.assertTrue(streamed["stream"])
        self.assertEqual(streamed["stream_options"], {"include_usage": True})
        self.assertNotIn("stream_options", oneshot)

    def test_chat_stream_closes_response(self):
        fake = FakeStreamResponse([sse(chunk(delta={"content": "hi"})), sse("[DONE]")])
        with unittest.mock.patch.object(demo.requests, "post", return_value=fake):
            llm = demo.DeepSeekFlashLLM(api_key="sk-test")
            self.assertEqual(llm.complete("问"), "hi")
        self.assertTrue(fake.closed)
        self.assertEqual(llm.last_finish_reason, "stop")

    def test_partial_text_preserved_on_interrupt(self):
        def broken():
            yield sse(chunk(delta={"content": "前半段"}))
            raise demo.requests.exceptions.ChunkedEncodingError("连接断了")

        fake = FakeStreamResponse(broken())
        with unittest.mock.patch.object(demo.requests, "post", return_value=fake):
            llm = demo.DeepSeekFlashLLM(api_key="sk-test")
            with self.assertRaises(demo.DeepSeekAPIError) as ctx:
                llm.complete("问")

        self.assertEqual(ctx.exception.partial, "前半段")   # 已吐出的字没丢
        self.assertIn("已收到 3 字", ctx.exception.describe())
        self.assertTrue(fake.closed)

    def test_single_run_no_stream_still_works(self):
        fake = FakeResponse(
            200,
            {"model": "deepseek-flash",
             "choices": [{"message": {"content": "一次性"}, "finish_reason": "stop"}]},
        )
        with unittest.mock.patch.object(demo.requests, "post", return_value=fake):
            llm = demo.DeepSeekFlashLLM(api_key="sk-test", stream=False)
            self.assertEqual(llm.complete("问"), "一次性")


class FakeStreamResponse:
    """够用的假流式响应：只实现 _chat_stream 用到的那几个成员。"""

    def __init__(self, lines, status_code=200):
        self.status_code = status_code
        self.encoding = None
        self.text = ""
        self.closed = False
        self._lines = lines

    def iter_lines(self, decode_unicode=False):
        yield from self._lines

    def close(self):
        self.closed = True


if __name__ == "__main__":
    unittest.main()
