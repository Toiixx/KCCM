# 逐行读懂 `examples/deepseek_flash_demo.py`

> ⚠️ **版本说明**：本文讲解的是 **v1（`stream=False`，258 行）**。
> 该文件已改为默认流式（v2，596 行），**行号全部变化**，
> 但 v1 的骨架（密钥、消息、请求体、异常、入口）在 v2 中原样保留，
> 概念部分仍然完全适用。
> 流式改动的逐段说明见 **`docs/streaming.md`**。

> 目标读者：会一点点 Python（知道 `def`、`if`、`for` 就行），但没调过大模型 API。
> 全文按**行号**组织，258 行一行不落。遇到新语法会当场解释，第二次出现就不再啰嗦。

---

## 0. 先建立三个直觉

在看代码之前，先把"调 API"这件事翻译成生活场景。代码只是把这个场景写下来而已。

### 直觉一：调用大模型 = 寄一封信，收一封回信

```
你的程序  ──[ 一个 JSON 包裹 ]──▶  DeepSeek 服务器
                                      │
         ◀──[ 一个 JSON 包裹 ]──────┘
```

没有"连接保持"，没有"对话状态"。**服务器不记得你**。所谓"多轮对话"，是你每次寄信时把之前的内容重新附上——这也正是 KCCM 想省掉的东西。

### 直觉二：API Key 是信封上的门禁章

服务器只认这个章，不认你是谁。所以密钥要放在**请求头**（HTTP Header）里，不能放在正文里（正文会被日志记录）。

```
Authorization: Bearer sk-xxxxxxxx
                └── 注意这个空格，少了就是 401
```

### 直觉三：回信是一棵"套娃"

服务器返回的 JSON 长这样，"正文"埋在第三层：

```json
{
  "id": "chatcmpl-xxx",
  "model": "deepseek-flash",
  "choices": [                          ← 候选答案列表
    {
      "index": 0,
      "message": {                      ← 第一条候选
        "role": "assistant",
        "content": "你好！我是……"        ← 你要的正文在这里
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {                            ← 计费信息
    "prompt_tokens": 36,
    "completion_tokens": 369,
    "total_tokens": 405
  }
}
```

所以代码里一定会出现 `choices[0].message.content` 这个路径。`[0]` 是因为 `choices` 是数组——理论上服务端可以一次给多个候选，实际通常只有 1 个。

### 这份文件在整个项目里的位置

```
                 ┌───────────────┐
                 │   Session     │  ← 记忆机制（KCCM 本体）
                 │  .ask(问题)    │
                 └───────┬───────┘
                         │ 只要满足两个条件：
                         │   ① 有 complete(prompt) -> str
                         │   ② 有 name
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   RuleLLM        OpenAICompatibleLLM   DeepSeekFlashLLM
   （离线假人）      （通用 OpenAI 格式）    （本文件，走 requests 手写）
```

**关键点**：`DeepSeekFlashLLM` 不是"另一个程序"，它是可替换的**零件**。换掉它，记忆机制完全不变。这就是"依赖注入"。

---

## 1. 文件说明书（L1–L12）

```python
1   """DeepSeek Flash 调用示例（requests 实现，Python 3.9+）。
...
12  """
```

| 行 | 解释 |
|---|---|
| L1 | 三个双引号 `"""` 开启**多行字符串**。它出现在文件最开头，就叫 **docstring（文档字符串）**。Python 不会执行它，但 IDE 悬浮提示、`help(模块)` 会读它。 |
| L3–L8 | 说明两个用途：单独跑 / 当 KCCM 后端。写清楚用法是给"三个月后的自己"看的。 |
| L10 | 依赖只有 `requests`，标准库之外仅此一个。 |
| L11 | 明确写"密钥走环境变量，不硬编码"——这是安全底线，也是 Code Review 第一条。 |
| L12 | `"""` 闭合。 |

> **小白常见疑问**：注释用什么？
> `#` 是单行注释；`"""..."""` 是多行字符串，但因为没赋给变量，实际起到"多行注释"的作用。函数内部用它还会被 `help()` 读到。

---

## 2. 导入区（L14–L23）

```python
14  from __future__ import annotations
16  import argparse
17  import json
18  import os
19  import sys
20  from dataclasses import dataclass, field
21  from typing import Any, Dict, List, Optional
23  import requests
```

**L14 `from __future__ import annotations`**

这条必须放在文件最前面（docstring 之后）。作用是：**所有类型注解延迟求值**，先当字符串存着，不在导入时真的去算。

为什么要它？因为 Python 3.9 不认 `list[str]` 这种写法（3.10 才支持），加了这行就都认了。同时能避免启动时因为类型引用顺序问题报错。

> 一句话：告诉 Python「类型标注先别较真，等我真用到再说」。

**L16 `import argparse`** — 标准库，解析命令行参数。就是让 `--prompt`、`--kccm` 这种开关能用。

**L17 `import json`** — JSON 和 Python 字典互转。本文件用它把字典变回字符串（L117、L127），好打印给人看。

**L18 `import os`** — 操作系统接口。这里只用一件事：`os.getenv` 读环境变量。顺带用了 `os.path` 拼路径（L216）。

**L19 `import sys`** — 三处用处：
- `sys.stderr`：把错误打到"标准错误流"，而不是正常输出（L197）
- `sys.stdout.reconfigure`：改控制台编码（L244）
- `sys.path`：模块搜索路径（L216）

**L20 `from dataclasses import dataclass, field`**

- `dataclass` 是**装饰器**：给类自动生成 `__init__`、`__repr__`、`__eq__`。手写这些很啰嗦，写错还容易出 bug。
- `field` 用来处理"可变默认值"这个坑（L152 会讲）。

**L21 `from typing import Any, Dict, List, Optional`**

四个类型工具。**它们只影响提示，不影响运行**——你写 `Dict[str, Any]` 却塞一个字符串进去，Python 也不会拦你（那是 mypy 这类静态检查工具的事）。

| 写法 | 含义 | 等价的白话 |
|---|---|---|
| `str` / `int` / `float` / `bool` | 基本类型 | 文字 / 整数 / 小数 / 真假 |
| `List[str]` | 字符串组成的列表 | `["a", "b"]` |
| `Dict[str, Any]` | 键是字符串、值随便什么 | `{"a": 1, "b": "x"}` |
| `Optional[int]` | `int` 或 `None` | 可能没有 |
| `Any` | 任意类型 | 不检查 |

**L23 `import requests`** — 第三方 HTTP 库。整个文件唯一的非标准库依赖。没装会直接 `ModuleNotFoundError`，所以 README 里写了 `pip install requests`。

---

## 3. 常量区（L25–L28）

```python
25  DEFAULT_BASE_URL = "https://api.deepseek.com"
26  DEFAULT_MODEL = "deepseek-flash"
27  DEFAULT_TIMEOUT = 60
28  DEMO_PROMPT = "你好，请简要介绍你自己"
```

| 行 | 解释 |
|---|---|
| L25 | 服务地址。注意**不带** `/v1`——拼接逻辑在 L163 处理，两种写法都兼容。 |
| L26 | 模型名。**这是全文件唯一出现模型名的地方**，改模型只改这一行。 |
| L27 | 超时秒数。网络卡住时不能让程序永远等下去。60 秒对生成 512 token 来说很宽松了。 |
| L28 | 示例问题。放在常量里，方便 `--prompt` 的默认值引用（L239）。 |

> **命名约定**：全大写 + 下划线 = 常量。Python **不会**强制你不能改它，这纯粹是"约定即契约"——看到全大写就知道别动。

---

## 4. 自定义异常（L31–L50）

```python
31  class DeepSeekAPIError(RuntimeError):
32      """携带状态码与响应体的异常，便于上层打印可诊断的错误信息。"""
34      def __init__(
35          self,
36          message: str,
37          status_code: Optional[int] = None,
38          body: Optional[str] = None,
39      ) -> None:
40          super().__init__(message)
41          self.status_code = status_code
42          self.body = body
```

**L31 `class DeepSeekAPIError(RuntimeError):`**

定义一个**自己的异常类型**。括号里是父类，表示"我是一种运行时错误"。

为什么不用内置的 `Exception`？三个理由：
1. 上层可以精确地 `except DeepSeekAPIError`，不会误捕别的错误
2. 名字本身就是文档——看到这个异常就知道是 API 层出问题
3. 可以挂载额外信息（状态码、响应体）

**L32** docstring，说明这个异常存在的意义。

**L34–L39 函数签名**

```python
def __init__(self, message: str, status_code: Optional[int] = None, body: Optional[str] = None) -> None:
```

- `__init__` 是**构造方法**，`DeepSeekAPIError("xxx")` 时自动调用
- `self` 是"这个对象自己"，所有方法第一个参数都必须是它
- `message: str` — 必填，错误描述
- `status_code: Optional[int] = None` — 可填可不填，不填就是 `None`
- `-> None` — 表示这个函数不返回值（构造函数本来就只负责初始化）
- 每个参数一行、末尾加逗号，这叫**垂直对齐**，参数多了更好读，diff 也更干净

**L40 `super().__init__(message)`**

调用父类 `RuntimeError` 的构造函数，把 `message` 存进去。

**为什么必须有这行？** 因为 `print(exc)` 或 `str(exc)` 打出来的就是父类存的那个 message。不调用它，异常打印出来是空的。

**L41 `self.status_code = status_code`**

把参数存到对象上。`self.xxx = ...` 就是"给这个对象挂一个属性"。之后在任何地方都能 `exc.status_code` 取到。

> **小白坑点**：如果只写 `status_code = status_code`（不加 self），那只是创建了一个局部变量，函数结束就没了。

**L42** 同理，存响应体。

```python
44      def describe(self) -> str:
45          parts = [str(self)]
46          if self.status_code is not None:
47              parts.append(f"HTTP 状态码: {self.status_code}")
48          if self.body:
49              parts.append(f"响应体: {self.body[:800]}")
50          return "\n".join(parts)
```

**L44** 定义一个方法，返回拼好的诊断文本。

**L45 `parts = [str(self)]`**

建一个列表，第一项是异常自身的消息。`str(self)` 会调用 `__str__`，也就是 L40 存进去的那句话。

**L46 `if self.status_code is not None:`**

判断"有没有状态码"。

**为什么用 `is not None` 而不是 `if self.status_code:`？** 因为状态码可能是 `0`，而 `0` 在布尔判断里是"假"，会被误判成"没有"。`is not None` 只判断"是不是真的没有"，更严谨。

**L47 `parts.append(f"HTTP 状态码: {self.status_code}")`**

- `f"..."` 是 **f-string**，花括号 `{}` 里可以放任意表达式，会被求值后转成文字嵌进去
- `.append()` 往列表末尾加一项

**L48 `if self.body:`** — 这里可以直接用 `if`，因为 `body` 只可能是字符串或 `None`，空字符串和 `None` 都属于"没有"，语义正确。

**L49 `self.body[:800]`**

**切片**：取前 800 个字符。为什么截断？因为有的网关会把整个 HTML 错误页塞进 body，几千字打满屏幕。

> 切片语法 `[起:止]`，含头不含尾。`[0:3]` 取第 0、1、2 个。省略起点就是从头开始。

**L50 `return "\n".join(parts)`**

`"\n".join(列表)` 用换行符把列表里的字符串粘成一个。结果形如：

```
接口返回非 200
HTTP 状态码: 401
响应体: {"error": ...}
```

**为什么单独写 `describe()`，而不在构造时就把字符串拼好？**

因为异常可能被捕获后选择不打印（比如上层要重试）。拼字符串是白做功。而且 `str(exc)` 保持简短，日志里更清爽；需要详细信息时再显式调 `describe()`。

---

## 5. 结果容器（L53–L59）

```python
53  @dataclass(frozen=True)
54  class ChatResult:
55      """一次成功调用的结果：正文 + 用量 + 实际模型名。"""
57      content: str
58      usage: Dict[str, Any]
59      model: str
```

**L53 `@dataclass(frozen=True)`**

- `@dataclass` 装饰器：自动生成 `__init__`，所以你不用写 `def __init__(self, content, usage, model)` 那一堆
- `frozen=True`：**冻结**，创建后不能改字段。`result.content = "x"` 会抛 `FrozenInstanceError`

**为什么冻结？** 两个实际好处：
1. 防止有人中途偷偷改结果，导致"日志里是一个值、用的是另一个值"
2. 冻结的对象可以算哈希，能当字典的 key、能放进 `set`

**L54** 类名用大驼峰 `ChatResult`——Python 的类都这么写（`KeywordCache`、`Session` 同理）。

**L55** docstring。

**L57–L59 三个字段**

| 字段 | 类型 | 说明 |
|---|---|---|
| `content` | `str` | 模型生成的正文 |
| `usage` | `Dict[str, Any]` | token 用量，键是 `prompt_tokens` 等 |
| `model` | `str` | **服务端实际使用的**模型名 |

注意最后一条：你请求 `deepseek-flash`，服务端可能回 `deepseek-flash-20250101`。记录真实值对排查问题很重要。

> **为什么用 `@dataclass` 而不是普通 `dict`？**
> `dict` 的键名写错要到运行时才发现（`result["contnet"]` → KeyError）；dataclass 写错属性名，IDE 当场标红，mypy 也能查出来。

---

## 6. 读密钥（L62–L71）

```python
62  def get_api_key() -> str:
63      """读取并校验密钥。缺失时立刻抛错，避免带着空 key 发请求。"""
64      key = os.getenv("DEEPSEEK_API_KEY", "").strip()
65      if not key:
66          raise DeepSeekAPIError(
67              "未找到环境变量 DEEPSEEK_API_KEY。\n"
68              "  Windows: setx DEEPSEEK_API_KEY \"sk-...\"（重开终端生效）\n"
69              "  macOS/Linux: export DEEPSEEK_API_KEY=\"sk-...\""
70          )
71      return key
```

**L62 `def get_api_key() -> str:`** — 无参数，返回字符串。

**L64 `key = os.getenv("DEEPSEEK_API_KEY", "").strip()`**

拆开看，从左到右执行：

1. `os.getenv("DEEPSEEK_API_KEY", "")` — 读环境变量。**第一个参数是变量名，第二个是"读不到时的默认值"**。这里给 `""`，所以永远不会返回 `None`。
2. `.strip()` — 去掉首尾的空白字符（空格、换行、制表符）

**为什么必须 `.strip()`？** 因为很多人复制密钥时会带上换行。`sk-abc\n` 和 `sk-abc` 在服务器看来完全不同，会返回 401，而你在屏幕上根本看不出差别。这是真实世界最常见的坑之一。

**L65 `if not key:`**

空字符串 `""` 在 Python 里是**假值**（falsy）。所以这行等价于"如果 key 是空的"。

Python 的假值清单（记一下，很常用）：`False`、`None`、`0`、`0.0`、`""`、`[]`、`{}`、`()`。除此之外都是真。

**L66–L70 `raise DeepSeekAPIError(...)`**

`raise` = 主动抛出异常，函数立即中断，不会执行到 L71。

括号里是三个字符串，**没有用 `+` 连接**——这是 Python 的特性：**相邻的字符串字面量会自动拼接**。所以三个字符串会变成一整段带换行的文字。

为什么要写三行？因为一行太长读不下去。这是格式技巧，不是语法必需。

**注意 L68 里的 `\"`**：因为整个字符串用双引号包裹，里面的双引号必须转义。这就是为什么有人偏爱单引号包字符串——里面就能随便写双引号了。

**L71 `return key`** — 走到这里说明有密钥，返回它。

> **为什么不在模块导入时就检查？**
> 因为导入应该是"无副作用"的。而且测试时要能临时改环境变量，如果导入时就锁死，测试没法跑。

---

## 7. 组装消息（L74–L76）

```python
74  def build_messages(content: str) -> List[Dict[str, str]]:
75      """构造对话消息。这里是最小单轮；多轮时按 role/content 追加即可。"""
76      return [{"role": "user", "content": content}]
```

**L74 返回类型 `List[Dict[str, str]]`** — 一个列表，里面每个元素是"键和值都是字符串"的字典。读法：从里往外——`Dict[str,str]` 是一条消息，`List[...]` 是消息列表。

**L76** 返回列表字面量，只有一个元素。

`role` 只有三种合法值：

| role | 谁说的 | 用途 |
|---|---|---|
| `system` | 系统 | 设定人设与规则，一般放第一条 |
| `user` | 用户 | 你的问题 |
| `assistant` | 模型 | 模型之前的回答（多轮时把历史答案也放进来） |

**为什么抽成函数？** 因为以后要加 `system` 提示词、要支持多轮，改这一处就行，不用去翻 `chat()`。

---

## 8. 组装请求体（L79–L93）

```python
79  def build_payload(
80      model: str,
81      messages: List[Dict[str, str]],
82      temperature: float = 0.7,
83      max_tokens: int = 512,
84      stream: bool = False,
85  ) -> Dict[str, Any]:
86      """组装请求体。四个参数显式写出，避免默认值随服务端漂移。"""
87      return {
88          "model": model,
89          "messages": messages,
90          "temperature": temperature,
91          "max_tokens": max_tokens,
92          "stream": stream,
93      }
```

**L79–L85 签名**

- 前两个必填，后三个有默认值 → 调用时 `build_payload("m", msgs)` 也能用
- `float = 0.7`、`int = 512`、`bool = False` 都是默认参数

**L87–L93 字典字面量**

`{"键": 值, ...}`。键必须是字符串，值随便什么。

每个字段的作用：

| 字段 | 值 | 作用 | 调它会发生什么 |
|---|---|---|---|
| `model` | `"deepseek-flash"` | 用哪个模型 | 写错 → 404 或 400 |
| `messages` | 消息列表 | 对话内容 | 空列表 → 400 |
| `temperature` | `0.7` | 随机性，0–2 | 调到 0 → 每次答案几乎一样；调到 1.5 → 开始胡说 |
| `max_tokens` | `512` | **最多生成**多少 token | 调太小 → 答案被拦腰截断，`finish_reason` 会变成 `length` |
| `stream` | `False` | 是否流式 | 改 `True` 后 `response.json()` 会直接崩（见下方警告） |

> ⚠️ **`stream=True` 时整个解析逻辑都要重写**
> 流式返回的是 `data: {...}\n\n` 这样一行行的 SSE 文本，不是单个 JSON。本文件固定 `stream=False`（L168 写死了），所以 L96 的解析才成立。
> **这是刻意的**——先保证正确，再谈体验。

**L86 那句 docstring 的门道**：「避免默认值随服务端漂移」。

如果请求体里不写 `temperature`，服务端会用**它自己的**默认值。今天可能是 1.0，明天改成 0.8，你的输出就莫名其妙变了，还很难查。**显式写出所有影响输出的参数，是可复现性的前提。**

---

## 9. 解析响应（L96–L134）★ 全文件最核心

```python
96  def parse_response(response: "requests.Response") -> ChatResult:
97      """校验状态码 → 解析 JSON → 取 choices[0].message.content。"""
```

**L96**

- 参数注解 `"requests.Response"` **带引号**：表示这是"字符串形式的注解"，Python 不会去解析它。加了 `from __future__ import annotations`（L14）后其实不加引号也一样，这里保留是习惯。
- 返回值是上一节定义的 `ChatResult`

**为什么把解析独立成函数？** 因为它**不碰网络**。只要构造一个假的 response 对象就能测——`tests/test_deepseek_client.py` 正是这么做的，7 种异常情况全部离线覆盖，不发一次真实请求。

### 9.1 第一关：状态码（L98–L103）

```python
98      if response.status_code != 200:
99          raise DeepSeekAPIError(
100             "接口返回非 200",
101             status_code=response.status_code,
102             body=response.text,
103         )
```

**L98** `status_code` 是 HTTP 状态码。`200` = 成功。`!=` 是不等于。

常见的失败码：

| 码 | 含义 | 这里通常是什么原因 |
|---|---|---|
| 400 | 请求格式错 | `messages` 为空、参数名拼错 |
| 401 | 未认证 | key 错、过期、多了空格 |
| 402 | 余额不足 | 账户没钱了 |
| 429 | 限流 | 请求太快 |
| 500 / 503 | 服务端故障 | 过一会儿再试 |

**L99–L103**

`raise` 时用了**关键字参数**（`status_code=...`），好处是不用记参数顺序，也不会传错位。

`body=response.text` — `.text` 是原始响应文本（字符串）。**即使状态码是 500，也要把 body 带上**，因为服务端往往会在里面写清楚原因。

> **反面教材**：只写 `raise Exception(f"请求失败: {response.status_code}")`。
> 你只知道"401"，不知道是"key 无效"还是"key 过期"。加上 body，一句 `Authentication Fails, Your api key: ****test is invalid` 就定位了。

### 9.2 第二关：是不是 JSON（L105–L110）

```python
105     try:
106         data = response.json()
107     except ValueError as exc:  # 非 JSON 响应（网关错误页、空响应等）
108         raise DeepSeekAPIError(
109             "响应不是合法 JSON", status_code=response.status_code, body=response.text
110         ) from exc
```

**L105–L106** `try` 里放"可能出错的代码"。`response.json()` 会把响应文本解析成 Python 字典。

**L107 `except ValueError as exc:`**

`response.json()` 在解析失败时抛的是 `ValueError` 的子类。这里捕获它，`as exc` 把异常对象存到变量 `exc`，供 L110 使用。

**什么情况下会解析失败？**
- 公司网关拦截了请求，返回一个 HTML 登录页
- 代理返回了纯文本的"Blocked"
- 响应体是空的

这三种情况状态码**可能都是 200**，所以第一关拦不住，必须单独处理。

**L108–L110**

`from exc` 是 **异常链**：把原始异常挂在 `__cause__` 上。

好处：`traceback` 会打印成

```
ValueError: Expecting value: line 1 column 1 (char 0)

The above exception was the direct cause of the following exception:

DeepSeekAPIError: 响应不是合法 JSON
```

**你能同时看到"最终报什么"和"底层为什么"**。不加 `from exc` 就会丢失底层信息。

### 9.3 第三关：有没有 choices（L112–L118）

```python
112     choices = data.get("choices")
113     if not isinstance(choices, list) or not choices:
114         raise DeepSeekAPIError(
115             "响应缺少 choices 或为空",
116             status_code=response.status_code,
117             body=json.dumps(data, ensure_ascii=False),
118         )
```

**L112 `data.get("choices")`**

`.get()` 和 `data["choices"]` 的区别：**取不到时 `.get()` 返回 `None`，方括号会抛 `KeyError`**。这里优先用 `.get()`，因为我们打算自己给出更好的错误信息。

**L113 `if not isinstance(choices, list) or not choices:`**

两个检查用 `or` 连起来，任一成立就报错：

1. `not isinstance(choices, list)` — 是不是列表？如果服务端返回 `"choices": null` 或 `"choices": {}`，这关拦住
2. `not choices` — 空列表？`[]` 是假值，所以 `not []` 是 `True`

**为什么要检查两次？** 因为 `[]` 通过第一关（它确实是 list），但取 `choices[0]` 会 `IndexError`。这种"看起来合法但取不到东西"的情况最难查。

**L117 `json.dumps(data, ensure_ascii=False)`**

- `json.dumps` = 把 Python 对象**序列化**回 JSON 字符串
- `ensure_ascii=False` = **不要**把中文转成 `\u4f60\u597d` 这种转义形式，直接输出中文

不加这个参数，错误信息里全是 `\uXXXX`，等于没打。

### 9.4 第四关：正文到底在不在（L120–L128）

```python
120     first = choices[0]
121     message = first.get("message") if isinstance(first, dict) else None
122     content = message.get("content") if isinstance(message, dict) else None
123     if not isinstance(content, str) or not content.strip():
124         raise DeepSeekAPIError(
125             "choices[0].message.content 缺失或为空",
126             status_code=response.status_code,
127             body=json.dumps(data, ensure_ascii=False),
128         )
```

**L120 `first = choices[0]`** — 取第一个候选。`[0]` 是索引，Python 从 0 开始数。

**L121 三元表达式**

```python
message = first.get("message") if isinstance(first, dict) else None
```

读法：`A if 条件 else B` — 条件真就取 A，假就取 B。

**为什么要先 `isinstance(first, dict)`？** 因为如果 `first` 是个字符串或数字，`.get()` 这个方法根本不存在，会抛 `AttributeError`。**先确认类型，再调方法**——这是写健壮代码的固定套路。

**L122** 同样套路，再往里钻一层。

**L123 `if not isinstance(content, str) or not content.strip():`**

- `not isinstance(content, str)` — 必须是字符串。如果服务端返回 `"content": null`（触发了内容安全策略时很常见），这里拦住
- `not content.strip()` — 去掉空白后不能为空。防止返回一串空格当答案

**这一步的意义**：如果 `content` 是 `null`，后面 `return ChatResult(content=None)`，上层打印时是 `None`，然后 `len(answer)` 就崩在某个完全无关的地方。**在入口处拦住，比在十层之后崩掉好查一百倍。**

### 9.5 组装结果（L130–L134）

```python
130     return ChatResult(
131         content=content,
132         usage=data.get("usage") or {},
133         model=data.get("model") or "",
134     )
```

**L132 `data.get("usage") or {}`**

`or` 的**短路求值**：左边为假（`None` 或 `{}`），就取右边 `{}`。

这样保证 `usage` 永远是字典，上层可以放心 `.get("prompt_tokens")`，不用先判空。

**L133 `data.get("model") or ""`** — 同理，取不到就给空字符串而不是 `None`。**"永远不返回 None"比"有时返回 None"好维护。**

---

## 10. LLM 后端类（L137–L152）

```python
137 @dataclass
138 class DeepSeekFlashLLM:
139     """KCCM 的 LLM 后端：实现 ``complete(prompt) -> str``。
140
141     与 ``RuleLLM`` 的差别只在"谁来回答"，接口保持一致，
142     因此可以直接 ``Session(cfg=Config(), llm=DeepSeekFlashLLM())``。
143     """
145     model: str = DEFAULT_MODEL
146     base_url: str = DEFAULT_BASE_URL
147     temperature: float = 0.7
148     max_tokens: int = 512
149     timeout: int = DEFAULT_TIMEOUT
150     api_key: Optional[str] = None
151     name: str = "deepseek-flash"
152     last_usage: Dict[str, Any] = field(default_factory=dict)
```

**L137 `@dataclass`（这次没有 frozen）**

为什么不冻结？因为 L185 要中途改 `self.last_usage`——每次调用后更新用量。**冻结的对象不能改字段，所以这里必须放开。**

**L139–L143** docstring 里的 `` `` `` 是 reStructuredText 的行内代码标记，Sphinx 文档工具会渲染成等宽字体。

**L145–L151 字段即配置**

dataclass 的每个带默认值的字段，都可以在创建时覆盖：

```python
DeepSeekFlashLLM()                        # 全用默认
DeepSeekFlashLLM(model="其他模型")          # 只改模型
DeepSeekFlashLLM(temperature=0, max_tokens=64)
```

| 字段 | 说明 |
|---|---|
| `model` | 模型名，默认取 L26 的常量 |
| `base_url` | 服务地址，改它就能指向代理或自建网关 |
| `temperature` / `max_tokens` | 直接透传给 L167 的 `build_payload` |
| `timeout` | 透传给 L177 |
| `api_key` | **`None` 表示"去读环境变量"**；显式传值就优先用它 |
| `name` | 协议要求的标识，用于日志区分是哪个后端回答的 |

**L152 `last_usage: Dict[str, Any] = field(default_factory=dict)`** ★ 经典坑

**为什么不能写 `= {}`？**

```python
# 错误示范
@dataclass
class A:
    data: dict = {}     # ← Python 会直接报错 ValueError
```

即使不报错（用别的方式写的），可变默认值也是**所有实例共享同一个对象**：

```python
a1 = A(); a2 = A()
a1.data["x"] = 1
print(a2.data)          # {'x': 1}  ← 见鬼了
```

`field(default_factory=dict)` 的意思是：**每次创建实例时，调用一次 `dict()` 生成新的空字典**。各自独立，互不影响。

> 记忆口诀：**默认值是"可变对象"（list / dict / set）时，一律用 `field(default_factory=...)`。**

---

## 11. 协议方法（L155–L157）

```python
154     # ---------- 协议方法 ----------
155     def complete(self, prompt: str) -> str:
156         """KCCM 每轮把拼好的 prompt 交进来，这里包成单条 user 消息。"""
157         return self.chat(build_messages(prompt))
```

**L155** 这就是 `LLMClient` 协议要求的唯一方法：**吃一个字符串，吐一个字符串**。

**L157** 一行做三件事，从里往外读：

1. `build_messages(prompt)` → `[{"role":"user","content":prompt}]`
2. `self.chat(...)` → 发 HTTP，拿回正文
3. `return` → 直接把正文返回

**为什么 KCCM 只要这么简单的接口？**

因为记忆机制**不应该关心**底层是 GPT、DeepSeek 还是一个 `if-else` 假人。`Session` 只做一件事：把拼好的 prompt 交给后端。**接口越窄，替换成本越低**——这就是为什么 `RuleLLM`（离线假人）和 `DeepSeekFlashLLM` 能无缝互换，也是为什么 58 个测试能在零网络环境下跑完。

---

## 12. 拼地址（L160–L163）

```python
160     def endpoint(self) -> str:
161         """拼接地址，兼容 base_url 是否已带 /v1。"""
162         base = self.base_url.rstrip("/")
163         return f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
```

**L162 `self.base_url.rstrip("/")`**

`rstrip("/")` = 去掉**右侧**所有 `/` 字符。

- `"https://api.deepseek.com/"` → `"https://api.deepseek.com"`
- `"https://api.deepseek.com///"` → 同样干净

**为什么要去？** 否则会拼出 `https://api.deepseek.com//v1/chat/completions`，双斜杠。虽然多数服务器能容忍，但不该赌。

**L163 一行三元**

```python
f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
```

| 输入 `base` | 判断 | 输出 |
|---|---|---|
| `https://api.deepseek.com` | 不以 `/v1` 结尾 | `.../v1/chat/completions` |
| `https://api.deepseek.com/v1` | 以 `/v1` 结尾 | `.../v1/chat/completions` |
| `http://localhost:8000/v1` | 以 `/v1` 结尾 | `http://localhost:8000/v1/chat/completions` |

**这就是"防御式拼接"**：用户从文档里复制地址，带不带 `/v1` 都有可能。与其写"请务必去掉 /v1"，不如代码自己兼容。

---

## 13. 发请求（L165–L186）

```python
165     def chat(self, messages: List[Dict[str, str]]) -> str:
166         key = self.api_key or get_api_key()
```

**L166** `or` 短路：`self.api_key` 非空就用它，否则调 `get_api_key()` 去读环境变量。

**设计意图**：既支持"从环境变量读"（生产环境），也支持"显式传进来"（测试、多账号切换）。

### 13.1 准备请求（L167–L173）

```python
167         payload = build_payload(
168             self.model, messages, self.temperature, self.max_tokens, stream=False
169         )
170         headers = {
171             "Authorization": f"Bearer {key}",
172             "Content-Type": "application/json",
173         }
```

**L167–L169** 位置参数依次对应 L80–L84 的五个形参。

**`stream=False` 写死**，不暴露成参数——因为解析器（L96）只支持非流式。**如果允许传 `True`，就会制造一个"能发出去但一定解析失败"的配置**。宁可暂时不支持，也不要给一个坏开关。

**L170–L173 headers**

| 头 | 值 | 作用 |
|---|---|---|
| `Authorization` | `Bearer sk-xxx` | 认证。`Bearer` 是 HTTP 规范的方案名，**中间那个空格不能少** |
| `Content-Type` | `application/json` | 告诉服务端"正文是 JSON" |

> **诚实说明**：用了 `json=payload`（L177）后，`requests` 会**自动**设置 `Content-Type: application/json`。所以 L172 是冗余的。
> 保留它有两个理由：① 读代码时一眼知道发的是什么格式；② 万一以后改成 `data=` 手写序列化，这里不用补。
> 但它确实是"可以不写的一行"，Code Review 时值得知道。

### 13.2 发出去，分三类接住（L175–L182）

```python
175         try:
176             response = requests.post(
177                 self.endpoint(), headers=headers, json=payload, timeout=self.timeout
178             )
179         except requests.Timeout as exc:
180             raise DeepSeekAPIError(f"请求超时（{self.timeout}s）") from exc
181         except requests.RequestException as exc:
182             raise DeepSeekAPIError(f"网络错误: {exc}") from exc
```

**L176–L178 `requests.post(url, headers=..., json=..., timeout=...)`**

- `post` = HTTP 的 POST 方法（提交数据）；GET 是"获取"，POST 是"提交"
- `json=payload` — **传入字典，requests 自动做三件事**：序列化成 JSON 字符串、设置 `Content-Type`、编码成 UTF-8 字节。比手动 `data=json.dumps(payload)` 省事且不易错
- `timeout=60` — **单个数字表示"连接超时"和"读取超时"都是 60 秒"**

**为什么要设 timeout？** `requests` 默认**永不超时**。服务端不响应，你的程序就永远挂在那里。生产事故里这一条极其常见。

**L179 `except requests.Timeout`**

**顺序至关重要**：`Timeout` 是 `RequestException` 的**子类**。Python 按书写顺序匹配，如果把 `RequestException` 写在前面，`Timeout` 就永远轮不到——所有超时都会被归成"网络错误"，日志里就分不清了。

**规则：子类异常写在父类前面。**

**L180 `from exc`** — 保留原始异常链（同 L110）。

**L181 `except requests.RequestException`**

这是 requests 所有网络异常的**基类**，覆盖：
- DNS 解析失败（域名拼错）
- 连接被拒（服务没起、端口错）
- SSL 证书错误
- 代理配置错误
- 连接被重置

**为什么要包一层自定义异常？** 让上层只需要 `except DeepSeekAPIError` 这一种，不必知道底层用的是 requests 还是 httpx。**换 HTTP 库时上层代码不用改。**

### 13.3 解析 + 记用量（L184–L186）

```python
184         result = parse_response(response)
185         self.last_usage = result.usage
186         return result.content
```

**L184** 交给纯函数处理（第 9 节）。**HTTP 归 HTTP，解析归解析**，职责单一。

**L185** 把用量挂到实例上。

**为什么用这种"副作用"而不是从 `complete()` 返回一个对象？**
因为 `complete()` 的签名必须满足 `LLMClient` 协议（返回 `str`）。如果返回 `ChatResult`，`Session` 就用不了了。**在协议约束下，用属性带出附加信息**是标准解法。

代价：它不是线程安全的。多线程共用一个实例会互相覆盖。`docs/limits.md` 里应该记这一条。

**L186** 只把正文交出去。

---

## 14. 单轮示例（L189–L211）

```python
189 def run_single_turn(prompt: str) -> int:
190     """最小示例：一条消息、一次请求、打印结果。"""
191     print(f"[模型] {DEFAULT_MODEL}")
192     print(f"[请求] {prompt}")
```

**L189 返回 `int`** — 这是**退出码**（exit code）。`0` 表示成功，非 `0` 表示失败。Shell、CI、`&&` 链都靠它判断。

**L191** 打印模型名。注意这里用的是常量 `DEFAULT_MODEL`，而不是 `llm.model`——如果创建实例时改了模型，这行会显示错的。**这是个小瑕疵**，下面「诚实清单」会列出来。

**L192** `{prompt}` 把变量值嵌进字符串。

```python
193     try:
194         llm = DeepSeekFlashLLM()
195         content = llm.complete(prompt)
196     except DeepSeekAPIError as exc:
197         print("[失败]", file=sys.stderr)
198         print(exc.describe(), file=sys.stderr)
199         return 1
```

**L194** 创建实例，全用默认参数。

**L195** 一条语句完成"发请求 + 解析 + 取正文"。前面所有铺垫都是为了这一行的干净。

**L196** 只捕获我们自己的异常。**没有裸 `except:`**——那会连 `KeyboardInterrupt`（Ctrl+C）都吞掉，用户按 Ctrl+C 都停不下来。

**L197 `file=sys.stderr`**

`print` 默认输出到**标准输出**（stdout）。加 `file=sys.stderr` 改到**标准错误**（stderr）。

**为什么要分开？** 这样用户可以把正常输出重定向到文件（`> out.txt`），错误仍然显示在屏幕上。`python x.py > result.txt` 时，错误不会被写进结果文件污染数据。

**L199 `return 1`** — 非零退出码，脚本调用方（CI、shell 的 `$?`）能感知失败。

```python
201     usage = llm.last_usage or {}
202     if usage:
203         print(
204             f"[用量] prompt={usage.get('prompt_tokens')} "
205             f"completion={usage.get('completion_tokens')} "
206             f"total={usage.get('total_tokens')}"
207         )
208     print("-" * 60)
209     print(content)
210     print("-" * 60)
211     return 0
```

**L201** 取用量，`or {}` 兜底（同 L132 套路）。

**L202** 有用量才打印。有些兼容网关不回 `usage` 字段，这时候静默跳过，不要打印一排 `None`。

**L204–L206 三个 f-string 换行**

注意这是**一个** `print`，三个字符串靠"相邻字面量自动拼接"连起来（同 L67 的技巧）。所以输出是**一行**，不是三行。

**L208 / L210 `"-" * 60`**

字符串乘法 = 重复。`"-" * 60` 得到 60 个减号，当分隔线用。

**L211 `return 0`** — 成功。

---

## 15. 接 KCCM（L214–L234）

```python
214 def run_with_kccm() -> int:
215     """把同一个后端接到 KCCM 上，验证两轮指代是否成立。"""
216     sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "src")))
```

**L216 这一行在干什么？** 拆开看：

| 片段 | 结果（假设文件在 `examples/` 下） |
|---|---|
| `__file__` | `.../kccm-demo/examples/deepseek_flash_demo.py` |
| `os.path.dirname(__file__)` | `.../kccm-demo/examples` |
| `os.path.join(..., "..", "src")` | `.../kccm-demo/examples/../src` |
| `str(...)` | 转成字符串（`Path` 对象才需要，这里已确保是 str） |
| `sys.path.insert(0, ...)` | 插到模块搜索路径**最前面** |

**为什么要这么做？**

`src/kccm/` 是源码目录，没有 `pip install -e .` 时 Python 找不到它。手动把 `src` 加进 `sys.path`，`import kccm` 就能工作了。

**`insert(0, ...)` 的 `0`** 是位置——放最前面，优先级最高。如果用 `append`，系统里已安装的旧版 `kccm` 会抢先被导入。

> **这是"脚本能直接跑"的常见技巧**，但生产代码不推荐——正确做法是 `pip install -e .`。这里为了"下载即可运行"做了妥协。

```python
217     try:
218         from kccm import Config, Session
219     except ImportError:
220         print("[失败] 未找到 kccm 包，请先 pip install -e .", file=sys.stderr)
221         return 1
```

**L218 在函数内部 import**

放在函数里而不是文件顶部，是为了**单轮模式不依赖 kccm**。用户只跑 `python examples/deepseek_flash_demo.py` 时，即使没装本项目也能用。

这叫**延迟导入**（lazy import）：真正需要时才加载。

**L217–L221** 导入失败给明确指引，而不是让 traceback 糊一脸。

```python
223     session = Session(cfg=Config(), llm=DeepSeekFlashLLM())
224     for question in ("我有12个苹果，3个人每个人得到多少", "那么四个人呢"):
225         try:
226             result = session.ask(question)
227         except DeepSeekAPIError as exc:
228             print("[失败]", file=sys.stderr)
229             print(exc.describe(), file=sys.stderr)
230             return 1
```

**L223 `Session(cfg=Config(), llm=DeepSeekFlashLLM())`**

**这一行是整个示例的目的所在。**

- `cfg=Config()` — 用默认配置（`ttl_max=10`、`max_entries=32` 等）
- `llm=DeepSeekFlashLLM()` — **把"谁来回答"作为参数传进去**

同样的写法换成 `llm=RuleLLM()` 就是离线假人，换成 `llm=OpenAICompatibleLLM()` 就是别的服务商。`Session` 一行都不用改。

这就是**依赖注入**（Dependency Injection）：不在内部 `new` 一个具体实现，而是从外部接收。

**L224 元组遍历** — `for question in (a, b)` 依次取两个元素。

**L226 `session.ask(question)`** — 完整跑一轮：提取关键词 → 更新缓存 → 拼 prompt → 调 `complete()` → 返回 `TurnResult`。

**L227–L230** 网络错误直接退出。注意这里返回 `1` 用的是 `return`，在循环里会中断整个函数——符合预期。

```python
231         print(f"\n[第 {result.turn} 轮] {question}")
232         print(f"[模式] {result.mode} / [prompt] {result.tokens} tokens")
233         print(result.answer)
234     return 0
```

**L231 开头的 `\n`** — 在字符串前面加个换行，让输出之间有间隔。

**L232 `result.mode`** 是 hybrid 分支的产物，取值：

| 值 | 含义 |
|---|---|
| `raw` | 走旁路，直接发对话原文 |
| `kccm` | 走缓存，发关键词 + 槽位 |
| `kccm+bridge` | 切回缓存的第一轮，额外附了上一轮原文 |

**`result.tokens`** 是拼好的 prompt 的 token 估算值——不是服务端返回的真实值。要看真实值得取 `llm.last_usage["prompt_tokens"]`。

> **这是一个容易搞混的点，值得记住**：`result.tokens` 是**本地估算**（1 个中文字 ≈ 1 token），用来做策略决策；`usage.prompt_tokens` 是**服务端实测**，用来算钱。两者通常差 5%–15%。

---

## 16. 命令行入口（L237–L254）

```python
237 def main(argv: Optional[List[str]] = None) -> int:
238     parser = argparse.ArgumentParser(description="DeepSeek Flash 调用示例")
239     parser.add_argument("--prompt", default=DEMO_PROMPT, help="单轮示例的用户输入")
240     parser.add_argument("--kccm", action="store_true", help="接入 KCCM 跑两轮对话")
241     args = parser.parse_args(argv)
```

**L237 `argv: Optional[List[str]] = None`**

**为什么参数默认 `None` 而不是直接读 `sys.argv`？**

为了**可测试**。测试时写 `main(["--kccm"])` 就能模拟命令行，不用真的改 `sys.argv`。`None` 时 `argparse` 会自动退回 `sys.argv[1:]`。

**L238** 创建解析器。`description` 会显示在 `--help` 里。

**L239 `--prompt`**

- 有 `--` 前缀 = **选项参数**，用法 `--prompt "问题"`
- `default=DEMO_PROMPT` — 不传就用 L28 的默认问题
- `help=` — `--help` 时显示的说明

**L240 `--kccm` + `action="store_true"`**

- 这是一种**开关**（flag），不需要值
- 出现 → `args.kccm = True`
- 不出现 → `args.kccm = False`

**L241 `args = parser.parse_args(argv)`**

解析后返回一个对象，用 `args.prompt`、`args.kccm` 访问。`--prompt` 的横杠会自动变成下划线。

```python
243     if hasattr(sys.stdout, "reconfigure"):
244         sys.stdout.reconfigure(encoding="utf-8", errors="replace")
```

**L243 `hasattr(obj, "名字")`** — 判断对象有没有这个属性或方法，返回 `True`/`False`。

**为什么判断？** `reconfigure` 是 Python 3.7 才有的方法。某些环境（被重定向到文件、被 pytest 捕获）的 `sys.stdout` 可能没有它。**先问再做**，避免 `AttributeError`。

**L244**

- `encoding="utf-8"` — 强制用 UTF-8 输出
- `errors="replace"` — 遇到编不出来的字符，用 `?` 代替而**不是抛异常**

**为什么需要？** Windows 控制台默认编码常常是 `gbk`，打印 emoji（模型很爱用 👋）或某些符号时直接 `UnicodeEncodeError` 崩掉。加这两句就稳了。

**这是 Windows 上跑中文脚本的必备三行**（判断 + reconfigure）。

```python
246     # 入口处先校验密钥，缺了就直说，不浪费一次请求
247     try:
248         get_api_key()
249     except DeepSeekAPIError as exc:
250         print("[失败]", file=sys.stderr)
251         print(exc.describe(), file=sys.stderr)
252         return 1
253
254     return run_with_kccm() if args.kccm else run_single_turn(args.prompt)
```

**L247–L252** 先查密钥。

**为什么不直接让它在该报错的地方报？** 因为 `run_with_kccm` 里会先建 `Session`、做分词、拼 prompt，全都白做，最后才在发请求时失败。**在入口拦住，用户 0.1 秒看到错误，而不是等了半秒。**

**L254** 三元选择分支，一行搞定 if-else 两条路。

**注意这里没有检查返回值**——直接把 `run_xxx()` 的结果当 `main()` 的结果返回，因为那两者本来就返回 0/1。

---

## 17. 程序入口（L257–L258）

```python
257 if __name__ == "__main__":
258     raise SystemExit(main())
```

**L257 `if __name__ == "__main__":`**

**这是 Python 最重要的惯用法之一。**

每个文件都有一个内置变量 `__name__`：

| 场景 | `__name__` 的值 |
|---|---|
| `python x.py`（直接运行） | `"__main__"` |
| `import x`（被别人导入） | `"x"` |

所以这行的意思是：**"只有直接运行我这个文件时，才执行下面的代码；被别人 import 时什么都不做。"**

**为什么需要？** 这样文件既能当脚本跑，又能被当模块导入（L218 的 `from kccm import ...` 就是导入场景）。**一份代码，两种用法。**

**L258 `raise SystemExit(main())`**

- `main()` 先执行，返回 0 或 1
- `SystemExit(0)` 是一个特殊的异常，作用是"让解释器退出，退出码 0"

**为什么用 `raise SystemExit(...)` 而不是 `sys.exit(...)`？**

两者效果相同。`raise` 的写法更明确地表达"这里要终止程序"，也避免有人以为 `sys.exit` 是个普通函数可以随便调。

**为什么不像老教材那样直接写 `main()`？**

因为那样**退出码永远是 0**，即使内部失败了。CI 会认为"构建成功"，shell 的 `&&` 也会继续执行下一步。**把退出码传出去，是脚本能被自动化的前提。**

---

## 18. 一次调用到底发生了什么（完整时序）

以 `python examples/deepseek_flash_demo.py` 为例：

```
① python 解释器启动
   └─ 执行 docstring、import、定义常量和类（不执行任何函数体）

② 走到 L257，__name__ == "__main__" 成立
   └─ 调用 main()

③ main()
   ├─ argparse 解析参数 → args.prompt = "你好，请简要介绍你自己"
   ├─ reconfigure 输出编码（Windows 防乱码）
   ├─ get_api_key() → 读环境变量 → "sk-..."
   └─ 调 run_single_turn("你好，请简要介绍你自己")

④ run_single_turn()
   ├─ 打印 [模型] [请求]
   ├─ llm = DeepSeekFlashLLM()          ← 全默认
   └─ llm.complete(prompt)

⑤ complete()
   └─ build_messages() → [{"role":"user","content":"你好..."}]
   └─ self.chat(消息列表)

⑥ chat()
   ├─ key = api_key or get_api_key()
   ├─ build_payload() → 5 个字段的字典
   ├─ headers = {Authorization, Content-Type}
   ├─ endpoint() → "https://api.deepseek.com/v1/chat/completions"
   └─ requests.post(...)  ←──── 网络往返 3~30 秒 ────┐
                                                      │
⑦ 服务器返回 200 + JSON                               │
   └─ parse_response()                                │
       ├─ 状态码 200？ ✓                               │
       ├─ 是合法 JSON？ ✓                              │
       ├─ choices 是非空列表？ ✓                       │
       ├─ content 是非空字符串？ ✓                      │
       └─ 返回 ChatResult(content, usage, model) ──────┘

⑧ 回到 chat()
   ├─ self.last_usage = usage
   └─ return content

⑨ 回到 run_single_turn()
   ├─ 打印用量
   ├─ 打印分隔线和正文
   └─ return 0

⑩ raise SystemExit(0) → 解释器退出，退出码 0
```

**看懂这张图，就看懂了整个文件。** 其余所有代码都是在为这 10 步做防护和格式化。

---

## 19. 常见报错对照表

| 你看到的信息 | 真正的原因 | 怎么办 |
|---|---|---|
| `未找到环境变量 DEEPSEEK_API_KEY` | 没设环境变量 | Windows 用 `setx` 后**重开终端**；macOS/Linux 用 `export` |
| `HTTP 状态码: 401` + `Authentication Fails` | key 错 / 过期 / 多了空格 | 重新复制，确认没有首尾空白 |
| `HTTP 状态码: 402` | 余额不足 | 充值 |
| `HTTP 状态码: 404` | 路径错 | `base_url` 里别带 `/v1`，让 L163 自己加 |
| `HTTP 状态码: 429` | 请求太快 | 加延迟、加重试 |
| `响应不是合法 JSON` | 网关/代理返回了 HTML | 检查是否被公司网络拦截，或 `base_url` 指向了错误地址 |
| `响应缺少 choices 或为空` | 参数错，或内容被风控拦截 | 看 `body` 里的 `error.message` |
| `choices[0].message.content 缺失或为空` | 内容被安全策略清空 | 换个问法，或看 `finish_reason` |
| `请求超时（60s）` | 网络慢 / 问题太长 | 调大 `timeout`，或缩短输入 |
| `网络错误: ...` | DNS / 连接 / SSL / 代理 | 先用 `ping api.deepseek.com` 和 `curl` 排查 |
| `ModuleNotFoundError: requests` | 没装依赖 | `pip install requests` |
| `未找到 kccm 包` | `--kccm` 模式下缺项目 | `pip install -e .` |

---

## 20. 诚实清单：这文件还能改的地方

代码能跑，但不等于完美。以下是真实存在的四个小问题：

**① `get_api_key()` 被调用了两次**

`main()` 里一次（L248），`chat()` 里一次（L166）。第二次是"兜底"，但严格说多余——如果环境变量在两次调用之间被改了，行为会不一致。

*改法*：在 `main()` 里读到 key 后传给实例——`DeepSeekFlashLLM(api_key=get_api_key())`。这样只读一次，且 `chat()` 走 `self.api_key` 分支。

**② `run_single_turn` 打印的是常量，不是实例字段**

L191 打 `DEFAULT_MODEL`，但实例可能是用别的模型创建的。改成 `llm.model` 更准。不过 `llm` 在 L194 才创建，得把打印挪到创建之后。

**③ `Content-Type` 头是冗余的**

L172 与 `json=payload`（L177）重复。删掉不影响功能。

**④ `parse_response` 只认 200**

严格的写法是 `if not 200 <= response.status_code < 300`。目前 DeepSeek 的成功码固定 200，所以没暴露问题，但这属于"写死了别人的实现细节"。

**这四条都不影响本示例的用途**（演示 + 当后端），所以保留了。代码不是越"正确"越好，是越**匹配当前用途**越好。

---

## 21. 三个动手练习（真的动手，别只看）

**练习 1 — 感受 temperature**

```bash
python examples/deepseek_flash_demo.py --prompt "用一句话夸我"
```

跑三次，看回答是否不同。然后把 L82 的 `0.7` 改成 `0.0`，再跑三次——应该几乎一模一样。

**为什么？** `temperature=0` 时模型总选概率最高的词，是确定性的（严格说服务端仍有微小随机，但差异极小）。

**练习 2 — 看答案被截断的样子**

把 L83 的 `max_tokens: int = 512` 改成 `8`，跑一次。

你会看到一句说到一半的话。此时响应里 `finish_reason` 是 `"length"` 而不是 `"stop"`——**这是判断"是否被截断"的唯一可靠信号**，不能靠"看起来像没说完"来猜。

**练习 3 — 亲手制造 401**

把环境变量改成 `sk-wrong`，跑一次，观察 `describe()` 的输出结构：

```
接口返回非 200
HTTP 状态码: 401
响应体: {"error":{"message":"Authentication Fails, Your api key: ****rong is invalid",...}}
```

**注意响应体里 key 是被打码的**（`****rong`）——服务端不会回显完整密钥，这是安全设计。也正因如此，你没法从错误信息里看出"是不是多了个空格"，只能自己检查。

---

## 附录：本篇出现的 Python 语法速查

| 语法 | 出现位置 | 一句话解释 |
|---|---|---|
| `"""..."""` | L1 | 多行字符串，放文件头就是文档 |
| `from __future__ import ...` | L14 | 启用未来版本的特性，必须放最前 |
| `@decorator` | L53, L137 | 装饰器，在不改函数体的前提下加工它 |
| `class A(B):` | L31 | 定义类 A，继承 B |
| `def f(x: int = 1) -> str:` | 多处 | 定义函数；`:` 后是类型提示，`=` 后是默认值 |
| `self` | 所有方法 | 指向"当前这个对象" |
| `super().__init__()` | L40 | 调用父类的构造方法 |
| `f"{var}"` | 多处 | f-string，花括号里求值后嵌入 |
| `"a" "b"` | L67 | 相邻字符串字面量自动拼接 |
| `x if c else y` | L121, L254 | 三元表达式 |
| `a or b` | L132, L166 | 短路求值：a 为假则取 b |
| `is not None` | L46 | 判断"不是 None"（比 `!= None` 更规范） |
| `not x` | L65, L113 | 逻辑取反；空值也算"假" |
| `[a:b]` | L49 | 切片，含头不含尾 |
| `"x" * 60` | L208 | 字符串重复 |
| `"\n".join(list)` | L50 | 用换行把列表粘成字符串 |
| `x.get(k)` | 多处 | 字典取值，取不到给 None 而不报错 |
| `isinstance(x, T)` | L113 等 | 判断 x 是不是 T 类型 |
| `try/except/raise ... from` | 多处 | 异常捕获、抛出、链接原始异常 |
| `print(..., file=sys.stderr)` | 多处 | 输出到标准错误流 |
| `@dataclass(frozen=True)` | L53 | 自动生成构造/比较方法，且不可修改 |
| `field(default_factory=dict)` | L152 | 每次创建实例时生成新的可变默认值 |
| `sys.path.insert(0, p)` | L216 | 把目录加入模块搜索路径最前 |
| `hasattr(o, "n")` | L243 | 判断对象有没有某属性 |
| `if __name__ == "__main__":` | L257 | 只在直接运行时执行的代码块 |
| `raise SystemExit(code)` | L258 | 带退出码终止程序 |
