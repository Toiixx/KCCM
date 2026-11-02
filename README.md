# KCCM — 关键词缓存上下文管理

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-58%20passed-brightgreen)
[![CI](https://github.com/Toiixx/kccm/actions/workflows/ci.yml/badge.svg)](https://github.com/Toiixx/kccm/actions/workflows/ci.yml)

**用 10 回合滑动窗口 + 关键词级 TTL/LFU 的显式记忆字典，替代多轮对话中"每轮把全部历史重新塞进 prompt"的累积式回放。**

多轮对话的默认做法是把历史消息全量拼接：第 20 轮时，prompt 里塞着 20 轮原文，其中大部分内容早已与当前问题无关。KCCM 换了一种记账方式——不存对话，存**事实**：每个数字、每个名词，连同它的槽位、存活回合、来源摘要一起进字典，过期自动退场。

```
用户: 我有12个苹果，3个人每个人得到多少
用户: 那么四个人呢          ← 缓存命中 12/苹果/每个人，4 覆盖 3
助手: 12 ÷ 4 = 3
```

第二轮的 prompt 里没有一个字来自第一轮的原文，模型仍然答对。这就是本项目的全部主张。

---

## 为什么不用全量历史

`python examples/run_compare.py` 的实测结果（同一个回答器，只改上下文构造方式）：

| 轮次 | 全量回放 | KCCM | 压缩比 |
|---:|---:|---:|---:|
| 1 | 18 | 269 | 0.07x |
| 4 | 172 | 352 | 0.49x |
| 6 | 296 | 363 | 0.82x |
| 8 | 431 | 367 | 1.17x |

- 全量回放随轮次**线性增长**，8 轮涨了 23.9 倍。
- KCCM 近似**常数**，8 轮涨 1.36 倍，上界由缓存容量与模板长度决定，与历史长度无关。
- **交叉点在第 8 轮**。短会话下 KCCM 反而更贵——系统指令与结构化格式有固定开销。这是真实权衡，README 不打算掩饰它。

短会话为什么更贵、贵多少、怎么治，见 [`docs/overhead.md`](docs/overhead.md)：第 1 轮贵约 15 倍；到第 2 轮，光系统指令一项就占 KCCM prompt 的 47.9%；切到 `prompt_style: lite` 后交叉点提前到第 5 轮。

---

## 适用场景

KCCM 不是通用记忆层，它的收益高度依赖「对话里是否存在可枚举的事实槽位」。选型时按这张表判断：

| 场景 | 结论 | 原因 |
|---|---|---|
| 任务型长会话（订单、工单、配置、参数协商、教学追问） | **适合** | 事实可枚举、槽位稳定，8 轮后 token 收益显著 |
| 需要审计与回放的生产链路 | **适合** | 每轮决策写入 JSONL，淘汰确定性可重放 |
| 成本敏感的批量对话 | **适合** | prompt 近似常数，不随轮次膨胀 |
| 开放式闲聊 | 不适合 | 无稳定事实槽位，提取层产不出有效条目 |
| 短会话（< 7 轮） | 不适合 | 固定开销占主导，第 1 轮约贵 15 倍 |
| 长文档问答 | 不适合 | 属于检索问题（RAG），不是记忆问题 |
| 多值不确定陈述（"可能 3 也可能 4"） | 受限 | 单槽位只保留一个 active 值 |
| 跨会话记忆、多用户并发 | 未实现 | 当前仅进程内 `dict`，重启即清空 |

判断标准可以归结为一句话：**如果这段对话的下一轮只依赖少量已确定的事实，而不是依赖前面的叙述细节，KCCM 就适用。**

---

## 优势

1. **上下文近似常数。** 8 轮实测从 269 涨到 367 tokens（1.36x），全量回放同期从 18 涨到 431（23.94x）。轮次越多，差距越大。
2. **只留事实，不留叙述。** 进 prompt 的是 `person.count = 4` 这样的槽位与来源摘要，不是"那么四个人呢"这样的整句原文。冗余被结构性消除，而不是靠截断。
3. **确定性可验证。** 淘汰采用四级 tie-break，同一状态必得同一 victim；每轮 HIT/OVERWRITE/EVICT/COMPOSE 写入 JSONL，可逐行重放。58 个测试在无网络、无密钥环境下跑完。
4. **核心零依赖。** `src/kccm` 只用标准库；离线 `RuleLLM` 会真的从 SLOTS 段读取数值做除法，因此"上下文是否足以支撑正确作答"可以被自动验证，而不是靠人眼看。
5. **后端无关。** `LLMClient` 协议只有一个方法：吃一个字符串，吐一个字符串。`RuleLLM`、`OpenAICompatibleLLM`、`DeepSeekFlashLLM` 可直接互换，`Session` 不需要改动。
6. **权衡公开。** 短会话更贵、槽位靠手写规则、代词消解只做兜底——这些限制写在 [`docs/limits.md`](docs/limits.md)，不藏在脚注里。

---

## 机制

### 三个参数，不是两个

| 参数 | 默认 | 作用 |
|---|---|---|
| `ttl_max` | 10 | 时间维度：连续 10 回合未被提及即删除（硬约束） |
| `max_entries` | 32 | 空间维度：字典条目上限，触顶启动 LFU 腾挪 |
| `ttl_tolerance` | 2 | TTL 差 ≤2 视为"相近"，进入同一 LFU 比较桶 |

把"缓存 10 回合"和"字典 10 条"混成一个数字是常见误解：一轮对话产出 4–8 个关键词，10 回合对应 30–50 条候选，`max_entries=10` 会在第 2 回合就误伤语义。

### 一条缓存长这样

```python
CacheEntry(
    keyword="4", surface="四", role=SlotRole.QUANTITY,
    slot="person.count",            # 槽位地址：覆盖判定靠它，不靠 role
    hit_count=2, first_turn=1, last_hit_turn=2,
    summary="那么四个人呢", summary_turn=2, superseded=("3",),
)
```

- **TTL 不单独存储**：`ttl_remaining = ttl_max - (current_turn - last_hit_turn)`。双计数在长会话里必然漂移，还会出现负数。
- **`role` 是类型，`slot` 是地址**。只有角色时，"3 个人"和"12 个苹果"同为 `QUANTITY`，无法判断"四个人呢"该覆盖谁。

### 提取：规则 + 差分

1. **规则层**：数字（阿拉伯 + 中文数字"十二"→12 + 单位）、名词（词性标注或词典匹配，剔除代词/量词/泛化词）。
2. **差分层**：`ΔQ = tokens(Q_now) − tokens(Q_prev) − 已命中词`，对增量再跑一次放宽的提取，补规则漏掉的词；纯指代（"那呢"）时兜底复用上一轮活跃集。
3. **覆盖判定**：新词与旧词同槽位，且（本轮有替换动词，或二者同为计数类数值）→ `OVERWRITE`，旧条目**删除**并记入 `superseded`。

### 淘汰：TTL 优先，LFU 次之

```
① pin 集（本轮显式提及的词）不进候选
② ttl_remaining == 0 → 删除
③ 仍超限：mn = 最小 ttl，bucket = {ttl ≤ mn + tolerance}
   victim = min(bucket, key=(hit_count, last_hit_turn, first_turn, keyword))
```

TTL 管**正确性**（这条记忆还有效吗），LFU 管**空间**（不够时牺牲谁）。过期条目在回合开始就被物理删除，根本进不了 LFU 候选集，所以不存在"因高频而保留过期记忆"。四级 tie-break 保证同一状态必得同一 victim——测试能断言，轨迹能重放。

### 拼接：三段 + 五级裁剪

优先级（预算保护顺序）：**当前问题 > 缓存条目及摘要 > 槽位**；文本顺序相反，当前问题放最后。

裁剪分五级：`L1 剔除 ttl=1` → `L2 摘要 120→48→24` → `L3 按 (ttl, hit) 逐条移出视图` → `L4 单行紧凑格式` → `L5 抛异常`。

> **裁剪只改视图，不删缓存。** 两者混在一个方法里，会出现"一次超长输入清空记忆"。

---

## 快速开始

### 零安装直接跑

示例脚本自行把 `src/` 加入 `sys.path`，克隆后即可运行，不需要装包、不需要密钥：

```bash
git clone https://github.com/Toiixx/kccm.git && cd kccm

python examples/run_cli.py --demo        # 两回合示例，打印缓存快照
python examples/run_cli.py --show-prompt # 查看真正发给模型的 prompt
python examples/run_compare.py           # 全量回放 vs KCCM 的 token 对比
python examples/run_compare.py --style lite   # 精简模板，交叉点提前到第 5 轮
python examples/run_overhead.py          # 短会话开销拆解（固定成本 / 边际成本 / 交叉点）
python examples/run_hybrid.py            # 旁路策略对比（固定轮次 vs 自动择优）
python -m unittest discover -s tests -t .     # 58 个测试，无需网络
```

**默认零依赖、零密钥**：`RuleLLM` 会真的从 prompt 的 SLOTS 段读取数值做除法，因此离线即可验证"上下文是否足以支撑正确作答"。

### 安装为包（可选）

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

kccm --demo                        # 等价于 python -m kccm --demo
```

### 接真实模型

任意 OpenAI 兼容端点均可（DeepSeek / vLLM / Ollama / OpenAI）：

```bash
cp .env.example .env      # 填 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
pip install -e ".[llm]"
python examples/run_cli.py --real
```

若不想装 openai SDK，`examples/deepseek_flash_demo.py` 用 `requests` 直连
`/v1/chat/completions`（模型 `deepseek-flash`，密钥读 `DEEPSEEK_API_KEY`），
**默认流式**（`stream=True`，逐字打印），既可单独跑，也能作为 `LLMClient` 注入 Session：

```bash
python examples/deepseek_flash_demo.py                  # 单轮流式
python examples/deepseek_flash_demo.py --show-reasoning # 连思考过程一起看
python examples/deepseek_flash_demo.py --no-stream      # 退回一次性返回
python examples/deepseek_flash_demo.py --kccm           # 用真实模型跑 KCCM 两轮
```

流式改造的逐段说明在 [`docs/streaming.md`](docs/streaming.md)。

> 密钥只从环境变量读取，仓库内不含任何凭据；`.env` 已在 `.gitignore` 中。

---

## 示例走查

配置 `max_entries=8`（便于演示淘汰），`ttl_max=10`。

| 轮次 | 输入 | 事件 | 活跃条目（keyword → ttl/hit） |
|---|---|---|---|
| 1 | 我有12个苹果，3个人每个人得到多少 | 新增 4 条 | `12→10/1` `苹果→10/1` `3→10/1` `每个人→10/1` |
| 2 | 那么四个人呢 | `3` 被 `4` 覆盖；两条关联命中 | `4→10/2` `苹果→10/1` `每个人→10/1` `12→9/1` |
| 3 | 换成梨，还剩6个 | `苹果` 被 `梨` 覆盖；新增 `6` | `梨→10/2` `6→10/1` `4→9/2` `每个人→9/1` `12→8/1` |
| 4 | 加香蕉和牛奶 | 新增 2 条（`清单` 泛化词，剔除） | 7 条；`12→7/1` |
| 5 | 再各买5个 | 新增 `5`；"各"触发关联命中 | 8 条触顶；`每个人→10/1`；`12→6/1` |
| 6 | 预算50元 | 新增 `50` → 9 条超限，触发 LFU | 淘汰 `12`，回到 8 条 |

第 6 回合的淘汰决策：

```
候选：12(5,1) 4(6,2) 梨(7,2) 6(7,1) 香蕉(8,1) 牛奶(8,1) 5(9,1) 每个人(9,1) 50(10,1)
mn = 5 → bucket = ttl ≤ 7 → [12(5,1), 4(6,2), 梨(7,2), 6(7,1)]
min hit = 1 → 平局 {12, 6} → tie-break last_hit_turn: 12(1) < 6(3) → 淘汰 12
```

完整逐事件轨迹见 `examples/traces/apple.jsonl`，说明见 [`docs/TRACE.md`](docs/TRACE.md)。

---

## 配置

`config.yaml` 中的任何一项都会改变行为：

| 改什么 | 后果 |
|---|---|
| `prompt_style: lite` | 空缓存固定开销 151→86，交叉点从第 8 轮提前到第 5 轮（第 2 轮 −42%） |
| `hybrid_mode: auto` | 每轮渲染两条 prompt 按成本择优，拐点自适应；配 `switch_hysteresis` 防抖 |
| `hybrid_mode: fixed` | 前 `hybrid_turns` 轮直接发原文，之后切 KCCM（固定阈值，需自己调 N） |
| 调大 `ttl_max` | 记忆更久，但过期条目挤占预算，prompt 变长 |
| 调小 `max_entries` | 淘汰更频繁，长尾事实更早丢失 |
| 调大 `ttl_tolerance` | LFU 权重上升，"旧但高频"更易获救，"新但低频"更易被牺牲 |
| 调小 `prompt_budget` | 更早触发 L2/L3 裁剪，摘要变短、条目变少 |
| 调小 `summary_max_chars` | **对短会话基本无效**：摘要本身只有十几字，从未触顶 |

库的默认值是 `hybrid_mode=off`（纯机制行为可复现）；`config.yaml` 中给的是推荐配置 `auto`。

---

## 仓库结构

```
kccm/
├── README.md
├── LICENSE                     MIT
├── pyproject.toml              打包元数据（PEP 621），核心零依赖
├── requirements.txt            可选运行时依赖（requests / openai / jieba）
├── requirements-dev.txt        开发依赖（pytest / ruff / mypy）
├── config.yaml                 推荐参数（唯一真值表）
├── .env.example                真实模型所需环境变量样例
│
├── src/kccm/
│   ├── config.py               参数加载与校验（越界即报错）
│   ├── models.py               CacheEntry / SlotRole / Candidate（字段冻结）
│   ├── normalize.py            全角、中文数字、单位
│   ├── tagger.py               分词后端：jieba 优先，缺失时降级为词典标注器
│   ├── extractor.py            规则提取 + 槽位判定
│   ├── differ.py               差分补充 + 纯指代判定
│   ├── cache.py                dict[keyword, CacheEntry]：hit / upsert / sweep
│   ├── composer.py             三段拼接 + 五级裁剪（只读快照）
│   ├── llm.py                  RuleLLM（离线）/ OpenAI 兼容客户端
│   ├── session.py              回合编排
│   ├── trace.py                JSONL 事件流
│   ├── cli.py                  命令行入口
│   └── __main__.py             支持 python -m kccm
│
├── examples/
│   ├── run_cli.py              交互 / demo
│   ├── run_compare.py          全量回放 vs KCCM
│   ├── run_overhead.py         短会话开销拆解
│   ├── run_hybrid.py           旁路策略对比
│   ├── deepseek_flash_demo.py  requests 直连 DeepSeek（流式）
│   └── traces/                 可重放的 JSONL 轨迹
│
├── tests/                      58 个用例，全部离线
├── docs/                       设计、开销、限制、轨迹、旁路、流式
└── .github/workflows/ci.yml    Python 3.10–3.13 × Ubuntu/Windows + ruff/mypy
```

模块边界：`extractor` 只产候选不写缓存，`cache` 只改状态不碰文本，`composer` 只读快照不改字典。三者可独立替换。

---

## 已知限制

1. **槽位是手写规则**，不是模型推断。换领域需扩 `src/kccm/data/lexicon.txt` 与槽位规则表。
2. **单槽位覆盖**：同一槽位只保留一个 active 值，不支持"同时持有多个候选值再择优"。
3. **代词跨段消解**弱：只做"纯指代兜底"（整轮回放活跃集），不做真正的指代链解析。
4. **中文为主**：英文仅按专名处理。
5. **不适合长文档问答**：那是检索问题，不是记忆问题。
6. **短会话（< 7 轮）**下因固定开销，token 反而高于全量回放。

完整边界与路线图见 [`docs/limits.md`](docs/limits.md)。

---

## 测试

```
tests/test_extract.py         提取与槽位                7
tests/test_cache.py           TTL 边界、LFU、pin 保护    7
tests/test_composer.py        预算不越界、裁剪不改缓存   6
tests/test_golden_apple.py    六回合数值逐格比对         7
tests/test_hybrid.py          旁路与择优分支            10
tests/test_deepseek_client.py SSE 解析、异常与重试       21
```

**58 个用例全部通过**，`unittest` 与 `pytest` 均可运行，无需网络与 API Key。

---

## English Abstract

KCCM (Keyword-Cache Context Management) replaces cumulative full-history replay in multi-turn LLM dialogue with an explicit memory dictionary. Instead of appending every prior turn to the prompt, it extracts numbers and nouns, binds each to a slot (e.g. `person.count`), and keeps them alive under a 10-turn TTL with LFU eviction when the dictionary is full. Prompt size stays near-constant: over 8 turns it grows 1.36× versus 23.94× for full-history replay, with the crossover at turn 8 (turn 5 with the `lite` template). The reference implementation is pure standard library, deterministic, trace-replayable, and ships with an offline `RuleLLM` so correctness can be tested without an API key. It fits task-oriented, fact-dense conversations; it is not a RAG system, a vector memory, or an agent framework.

---

## License

[MIT](LICENSE)。定位是机制参考实现，价值在于被复制与改造；MIT 的义务最少、法务成本最低。若实现中含可能涉及专利的算法，或需显式专利授权条款，应改用 Apache-2.0。
