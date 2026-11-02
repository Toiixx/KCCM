# 从一次性返回到流式输出：`stream=True` 的改造说明

> 对应文件：`examples/deepseek_flash_demo.py`（原 258 行 → 现 596 行）
> 前置阅读：`docs/walkthrough-deepseek-flash.md`（逐行讲解的是 v1 非流式版本，行号已变，概念仍适用）

---

## 0. 一句话结论

**只把 `"stream": False` 改成 `True`，程序会当场崩掉。**

因为流式返回的不再是一个 JSON，而是一串 `data: {...}` 文本片段。原来那段
`response.json()["choices"][0]["message"]["content"]` 的路径整条失效。
所以这次改的不是一个参数，而是**传输层 + 解析层 + 打印层**三处。

---

## 1. 为什么必须重写解析

### 1.1 两种响应的形态差异

**非流式**（v1）：服务器算完，一次性给你一整个对象。

```json
{
  "choices": [{
    "message": { "content": "12 ÷ 4 = 3" },   ← 完整答案，一次到位
    "finish_reason": "stop"
  }],
  "usage": { "prompt_tokens": 9, "completion_tokens": 3 }
}
```

**流式**（v2）：服务器一边算一边吐，每次只给一小段。

```text
data: {"choices":[{"delta":{"role":"assistant","content":null}}]}

data: {"choices":[{"delta":{"reasoning_content":"用户问"}}]}

data: {"choices":[{"delta":{"content":"12"}}]}

data: {"choices":[{"delta":{"content":" ÷ 4"}}]}

data: {"choices":[{"delta":{"content":" = 3"},"finish_reason":"stop"}]}

data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":3}}

data: [DONE]
```

三个关键差别，每一个都能让 v1 的代码挂掉：

| 差异 | v1 的假设 | 流式的现实 |
|---|---|---|
| 字段名 | `message.content` | `delta.content` |
| 语义 | **全量**答案 | **增量**片段，要自己拼 |
| 结束标志 | 读到 JSON 就结束 | 没有 `choices` 的 usage 帧 + `[DONE]` 哨兵 |

**最容易踩的坑是第二行**：`delta` 是"增量"。如果你写成 `content = piece`（覆盖），
最后只会剩下最后一个字。必须 `pieces.append(piece)` 再 `"".join()`。

### 1.2 改动后的数据流

```
requests.post(stream=True)
        │  立即返回，只拿到响应头
        ▼
response.iter_lines()          ← 阻塞式逐行读，来一行给一行
        │
        ▼
consume_stream(lines, on_delta, on_reasoning)   ← 纯函数，可离线测
        │
        ├── 每收到一个 content 片段 → on_delta(片段) → 立刻打印
        ├── 每收到一个 reasoning 片段 → on_reasoning(片段)
        └── 读完 [DONE] → ChatResult(完整正文, usage, finish_reason)
        │
        ▼
complete() 返回**完整字符串**
        │
        ▼
Session / KCCM 照旧使用        ← 上层完全不用改
```

**核心设计**：流式只是"传输与展示"的方式变了，`complete()` 的契约没变——
**仍然返回整段字符串**。所以 KCCM 一行代码都不用动。

---

## 2. 逐段讲新增与改动

### 2.1 `build_payload`（L100–L122）：多带一个 `stream_options`

```python
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
```

**为什么需要 `stream_options`？**

流式时，服务端默认**不返回** `usage`。你就只能看到文字，不知道花了多少 token。
加上这个声明，服务端会在 `[DONE]` 之前补一帧只带 `usage`、`choices` 为空数组的包。

**实测**：加了它之后，最后一帧是

```json
{"choices":[],"usage":{"prompt_tokens":36,"completion_tokens":361,
 "total_tokens":397,"completion_tokens_details":{"reasoning_tokens":310}}}
```

不加的话这一帧根本不会出现，`llm.last_usage` 永远是空字典。

> 这是 OpenAI 兼容协议的扩展字段。接入不支持的网关时，把 `include_usage=False`
> 就能退回"只看文字不看用量"。

### 2.2 `consume_stream`（L175–L292）：新的解析核心

函数签名三个参数：

```python
def consume_stream(
    lines: Iterable[str],                          # 行流，可以是生成器
    on_delta: Optional[Callable[[str], None]] = None,      # 正文片段去处
    on_reasoning: Optional[Callable[[str], None]] = None,  # 思考片段去处
) -> ChatResult:
```

**为什么参数是"行流"而不是 Response 对象？**

因为这样它就是**纯函数**——不碰网络。测试时直接喂一个列表
`["data: {...}", "", "data: [DONE]"]` 就能验证全部逻辑，13 个流式用例零网络跑完。

逐层过滤（L211–L221）：

```python
    for raw in lines:
        if isinstance(raw, bytes):          # 防御：调用方没开 decode_unicode
            raw = raw.decode("utf-8", errors="replace")
        line = raw.strip()

        if not line:
            continue                        # 空行：SSE 用它分隔事件
        if line.startswith(":"):
            continue                        # 冒号开头是注释/心跳
        if not line.startswith(SSE_DATA_PREFIX):
            continue                        # event: / id: / retry: 等字段
```

SSE 协议里有四类行，只有第三类对我们是数据：

| 行 | 例子 | 处理 |
|---|---|---|
| 空行 | `` | 跳过，它是事件分隔符 |
| 注释 | `: ping` | 跳过，很多网关用它保活 |
| 其他字段 | `event: message` | 跳过，本接口不用 |
| 数据 | `data: {...}` | **解析** |

取数据部分（L223–L226）：

```python
        payload_text = line[len(SSE_DATA_PREFIX):].strip()
        if payload_text == SSE_DONE:
            saw_data = True
            break
```

- `line[5:]` 就是切掉 `"data:"` 这五个字符
- `[DONE]` **不是 JSON**，`json.loads` 会直接抛异常，必须先拦
- `break` 而不是 `continue`：哨兵之后不会再有数据，继续读只会读到垃圾

解析与校验（L228–L243）：

```python
        saw_data = True
        try:
            event = json.loads(payload_text)
        except ValueError as exc:
            raise DeepSeekAPIError("流式分片不是合法 JSON", body=payload_text) from exc

        if not isinstance(event, dict):
            raise DeepSeekAPIError("流式分片不是 JSON 对象", body=payload_text)

        if "error" in event:
            raise DeepSeekAPIError("流式返回错误", body=json.dumps(event, ensure_ascii=False))
```

`"error" in event` 这一条**是流式独有的**：有的网关在 HTTP 200 的流里塞
`data: {"error": {"message": "quota exceeded"}}`。状态码是 200，前面所有检查都过，
但内容是个错误对象。不单独判，就会当成"模型只回了一个空对象"。

空 `choices` 必须放过（L249–L251）：

```python
        choices = event.get("choices") or []
        if not choices:
            continue                        # usage-only 帧，继续读
```

**这是从 v1 搬过来最容易出错的地方**。v1 里"`choices` 为空"是错误；
流式里它是**正常现象**——那就是最后一帧 usage 包。照抄旧校验会把成功的请求判成失败。

增量累加（L264–L274）：

```python
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
```

- `isinstance(piece, str)`：首帧的 `content` 是 `null`，直接拼会 `TypeError`
- `and piece`：空字符串不拼，也不回调（否则前端会收到一堆空事件）
- **先 append 再回调**：顺序很重要。回调如果抛异常，buffer 里已经有这段字了，
  上层才能报"已收到 N 字"

### 2.3 `_post`（L362–L386）：超时改成元组

```python
                timeout=(self.timeout, self.read_timeout),
                stream=stream,
```

**这是流式改造里最容易被忽略的一处。**

v1 写的是 `timeout=60`，含义是"整个请求最多 60 秒"。
流式下这个语义是错的——一个 800 token 的回答可能要 2 分钟，
但**任意两个分片之间的间隔**通常不超过 3 秒。

所以拆成两个数：

| 位置 | 值 | 管什么 |
|---|---|---|
| 第一个 | 15s | 建立连接（TCP + TLS） |
| 第二个 | 120s | **相邻两个分片之间**的最长等待 |

沿用 `timeout=60` 的后果：生成到一半被 `Timeout` 掐断，用户看到半句话。

`stream=stream` 让 requests **不等 body 就返回**，这是能"边收边处理"的前提。
不加这个参数，`requests.post` 会阻塞到整个响应下载完——那就完全没流式效果了。

### 2.4 `_chat_stream`（L396–L442）：三段式防护

```python
        buffer: List[str] = []

        def sink(piece: str) -> None:
            buffer.append(piece)
            if on_delta is not None:
                on_delta(piece)
```

`buffer` 和 `sink` 的作用不是打印，而是**保命**：流式中断时，已经吐出去的字
无法从网络重取（服务器不会重发），只能靠这个 buffer 带出来。

```python
            response.encoding = "utf-8"
```

**必须显式设置。** 非流式时 requests 拿到完整 body 会自己猜编码；
流式时它拿不到完整内容，猜编码会退化成 `latin-1`，中文全部变乱码。
不写这一行，你会看到 `ä½ å¥½` 这种东西。

```python
            try:
                result = consume_stream(response.iter_lines(decode_unicode=True), sink, on_reasoning)
            except requests.Timeout as exc:
                raise DeepSeekAPIError(..., partial="".join(buffer)) from exc
            except requests.RequestException as exc:
                raise DeepSeekAPIError(..., partial="".join(buffer)) from exc
        finally:
            response.close()
```

三个细节：

1. **异常是在迭代过程中抛的**，不在 `requests.post` 那一行。所以 try 必须包住
   `consume_stream`，而不是只包 `post`。这是从 v1 搬过来最容易漏的一处。
2. `requests.Timeout` 写在 `RequestException` 前面——子类必须在前，否则永远匹配不到。
3. `finally: response.close()` 无条件执行。流式连接不关会一直占着 socket，
   跑几百轮就耗尽连接池。

### 2.5 `_make_printers`（L450–L485）：为什么要两个回调

推理模型会**先想后说**。抓包看到的真实顺序是：

```
data: {"delta":{"reasoning_content":"我们需要"}}     ← 思考，可能持续好几秒
data: {"delta":{"reasoning_content":"回答中文"}}
...
data: {"delta":{"content":"12 ÷ 4 = 3"}}            ← 正文才开始
```

如果只监听 `content`，用户在那几秒里**一个字符都看不到**，会以为程序卡死。

所以两个回调分开：

| 回调 | 去向 | 默认行为 |
|---|---|---|
| `on_content` | **stdout** | 直接打印，flush |
| `on_reasoning` | **stderr** | 默认只打一行提示；`--show-reasoning` 时实时打印 |

**为什么思考走 stderr？** 因为这样可以 `python x.py > answer.txt`——
文件里只有答案，思考过程不污染。这是 Unix 的常规做法：数据走 stdout，诊断走 stderr。

`flush=True` 是流式的命门。不写它，输出会攒在缓冲区里，
程序结束时"唰"地一次性全出来——看起来和非流式一模一样。

---

## 3. 推理模型的两个隐藏坑

### 3.1 `max_tokens` 包含思考

`deepseek-flash` 是推理模型，`completion_tokens = 正文 + 思考`。
实测一次"1+1 等于几"：

```
prompt=35  completion=29  reasoning=22  total=64
```

29 个 completion token 里有 22 个是思考。也就是说 `max_tokens=512` 实际只有
不到 500 留给答案，复杂问题光思考就可能吃掉大半。

**表现**：`finish_reason="length"` 且 `content` 为空，程序报
"模型没有返回正文"。这不是 bug，是预算被思考耗光了。

代码里已经把这个提示写进错误信息（L280–L284）。

### 3.2 思考阶段不产生正文，但不是卡住

判断"是否还在工作"不能靠"有没有新字符"，要看 `iter_lines` 是否还在返回数据。
真要加超时监控，监控的应该是**分片间隔**（`read_timeout` 已经在做这件事）。

---

## 4. 实测数据

### 4.1 分片确实是分批到的

```
问题：数一下从1到20，每个数字之间用空格隔开
分片数：39
首片到达：1.07s   总耗时：1.17s
前 8 个分片时间点：[1.07, 1.07, 1.07, 1.07, 1.07, 1.07, 1.09, 1.09]
累计字数：50 | 最终长度：50
```

**累计字数 == 最终长度**，说明 39 个增量拼起来正好是完整答案，没多没少。
这是流式解析正确性的最强证据——少拼一个片段就会对不上。

### 4.2 用量能拿到

```
[用量] prompt=36 completion=361 total=397 finish=stop
```

`completion=361` 明显大于正文长度，因为里面含思考 token。

### 4.3 首字延迟的意义

首片 1.07s、总 1.17s——这个例子里几乎同时到，因为答案短。
换成需要长思考的问题，首字延迟会拉长到数秒，但**一旦开始，就是逐字出现的**。
流式优化的是"感知延迟"，不是"总耗时"。总耗时反而可能略增（分片有额外开销）。

---

## 5. 失败模式对照

| 现象 | 原因 | 代码里的处理 |
|---|---|---|
| 中文全是 `ä½ å¥½` | 没设 `response.encoding` | L421 强制 utf-8 |
| 最后只剩一个字 | 用覆盖代替累加 | L272 `pieces.append` |
| 正常回答被报"choices 为空" | 把 usage 帧当成错误 | L250 `continue` 放过 |
| `json.loads` 在 `[DONE]` 上崩 | 哨兵不是 JSON | L224 先拦 |
| 生成到一半断掉，内容全丢 | 没有 buffer | L405 `buffer` + `partial` |
| 长回答被 60s 掐断 | 用了总超时 | L380 拆成 (connect, read) |
| 连接越跑越少，最后卡死 | 没 close | L437 `finally` |
| 看不到任何输出，最后一次性出现 | 忘了 `flush=True` | L467 每个片段都 flush |
| 思考几秒钟一片空白 | 只监听 `content` | L450 双回调 |

---

## 6. 与 KCCM 的配合

**上层完全没改。** `Session.ask()` 仍然只做一件事：`self.llm.complete(prompt)`。

想在 KCCM 里也看到打字机效果，只需要在创建实例后挂两个回调：

```python
llm = DeepSeekFlashLLM(stream=True)
llm.on_delta = on_content          # Session 传不进回调，就挂在实例上
llm.on_reasoning = on_reasoning
session = Session(cfg=Config(), llm=llm)
```

`complete()` 的签名是 `complete(prompt, on_delta=None, on_reasoning=None)`——
不传就用实例上的。这样既满足 `LLMClient` 协议（一个位置参数），
又保留了传回调的能力。

实测两轮：

```
[第 1 轮] 我有12个苹果，3个人每个人得到多少
每人得到：12 ÷ 3 = 4 个苹果。
[完成] 模式=kccm prompt=270 tokens

[第 2 轮] 那么四个人呢
还原：12个苹果分给4个人，每个人得到多少？
算式：12 ÷ 4 = 3
所以，每个人得到 3 个苹果。
[完成] 模式=kccm prompt=259 tokens
```

第二轮能正确用上第一轮缓存里的 `12`，说明流式改造没有破坏记忆链路。

---

## 7. 两个必须承认的限制

**① 流式不能重试。**

非流式失败了可以无脑重发一次。流式不行——用户已经看到半句话，
重发会从头再来一遍，体验更糟。所以 `_chat_stream` 里**没有任何重试逻辑**，
出错就带着 `partial` 报上去，由调用方决定是"补一句抱歉"还是"整段重来"。

**② 流式不省 token，反而略贵。**

分片本身有协议开销，首字快不等于总账便宜。这次改造买的是**感知延迟**，
不是费用。要省钱还是得靠 KCCM 那套上下文裁剪。

---

## 8. 复现命令

```bash
# 默认流式
python examples/deepseek_flash_demo.py

# 看思考过程
python examples/deepseek_flash_demo.py --show-reasoning

# 退回非流式，对比首字延迟
python examples/deepseek_flash_demo.py --no-stream

# 接 KCCM，两轮流式
python examples/deepseek_flash_demo.py --kccm

# 只跑流式解析的 13 个离线用例
python -m unittest tests.test_deepseek_client -v
```
