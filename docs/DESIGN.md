# 设计说明

本文只讲**为什么这样设计**。字段与算法以代码为准，`README.md` 讲怎么用，`TRACE.md` 给可核对的数值。

---

## 1. 问题定义

多轮对话的上下文构造有两种极端：

| 方式 | 做法 | 代价 |
|---|---|---|
| 累积式全量回放 | 每轮拼接全部历史 | prompt 随轮次线性增长，绝大部分内容与当前问题无关 |
| 无状态 | 每轮只发当前问题 | 无法消解省略与指代，第二轮就答不了 |

KCCM 取中间态：**保留事实，丢弃叙述**。每个关键词连同槽位、存活时间、来源摘要一起记账，过期自动退场。

非目标：向量检索、跨会话持久化、多用户隔离、工具调用。这是机制参考实现，可读性与可验证性优先于召回率。

---

## 2. 参数解耦

```
ttl_max         = 10     # 时间：连续 10 回合未被提及即删除
max_entries     = 32     # 空间：字典条目上限
ttl_tolerance   = 2      # "TTL 相近"的判定阈值
prompt_budget   = 6912   # 8192 - 1024(输出) - 256(安全边际)
```

**为什么不能共用一个 10**：一轮对话产出 4–8 个关键词，10 回合对应 30–50 条候选。`max_entries=10` 会在第 2 回合因容量淘汰而误伤语义。时间与空间必须分开计价。

---

## 3. 数据结构

```python
@dataclass(frozen=True)
class CacheEntry:
    keyword: str
    surface: str
    role: SlotRole          # 8 类封闭枚举
    slot: str               # 事实位地址，如 person.count
    hit_count: int          # 仅 LITERAL / NORMALIZED 命中递增
    first_turn: int         # 槽位首次出现的回合（覆盖时继承）
    last_hit_turn: int      # TTL 的唯一事实源
    summary: str            # ≤ summary_max_chars
    summary_turn: int       # 摘要取自哪一回合（覆盖后与 first_turn 不同）
    superseded: tuple[str, ...] = ()
```

三条不可动摇的约定：

1. **TTL 不落库**，由 `last_hit_turn` 派生。双计数在长会话中必然漂移，还会出现负值。
2. **`role` 是类型，`slot` 是地址**。只有角色时，"3 个人"与"12 个苹果"同为 `QUANTITY`，无法判断"四个人呢"该覆盖谁。
3. **同 slot 至多一个 active 条目**。违反即覆盖或降级为带后缀的独立槽位，二者必居其一。

---

## 4. 提取

### 4.1 规则层

| 类别 | 规则 | 例 |
|---|---|---|
| 阿拉伯数字 | `(?<![A-Za-z0-9_.])\d+(\.\d+)?%?` | `12`、`3.5%` |
| 中文数字 | `[一二三四五六七八九十百千万两]+` 查表 | `十二` → 12 |
| 单位 | 数字后紧跟 `元/斤/个/人/%` | `50元` → keyword `50` |
| 名词 | 词性标注（`n/nr/ns/nz/vn/eng`）或词典长词优先匹配 | `苹果`、`每个人` |

负向过滤：代词、量词、泛化名词（`清单/东西/情况`）、动词、疑问词一律丢弃。

### 4.2 槽位判定（数量）

按优先级短路：

```
单位=元/块        → budget.amount
单位=%            → ratio.percent
前文含"单价"      → price.unit
后文首字含"人/位" → person.count
前文含"还剩/剩"   → remainder.count
前文含"买/采购"   → purchase.count
后文 4 字内有名词 → f"{noun.num_hint}.count"
兜底              → quantity.generic
```

槽位规则是**硬编码表**，不让模型生成——同一输入必须得到同一 slot，否则测试与轨迹重放都失效。

### 4.3 差分层

```
ΔQ = tokens(Q_now) − tokens(Q_prev) − rule_hits − cache_keys
```

对 `ΔQ` 再跑一次放宽的提取（允许未登录词），补规则漏掉的词。若本轮既无规则命中、又无差集、且含指代标记，则判定为**纯指代**，回放上一轮活跃集作为兜底——宁可持续续命，也不空手作答。

### 4.4 覆盖判定

```
同 slot 且（替换动词 ∈ {换成, 改成, 变成, …} 或 新旧同为计数类数值）
→ OVERWRITE：new.hit_count = old.hit_count + 1
             new.first_turn = old.first_turn
             new.superseded = old.superseded + (old.keyword,)
             del cache[old.keyword]
```

删除旧条目是关键。若新旧并存，prompt 里会出现 `person.count = 3` 与 `person.count = 4`，模型无从判断，示例必然答错。

`hit_count` 继承的理由：LFU 奖励的是"这个事实位被反复使用"，不是"这个字符串出现过几次"。

---

## 5. 淘汰

| 策略 | 回答 | 性质 | 触发 |
|---|---|---|---|
| TTL | 这条记忆还有效吗 | 正确性 | 每回合开始，无条件 |
| LFU | 空间不够时牺牲谁 | 空间 | 仅超限时 |

```
while len > max_entries:
    cands  = 活跃条目 − pin
    mn     = min(ttl)
    bucket = {c | ttl(c) ≤ mn + tolerance}
    victim = min(bucket, key=(hit_count, last_hit_turn, first_turn, keyword))
```

用桶而非全局 `(ttl, hit)` 排序：全局排序时 TTL 几乎总不同，LFU 形同虚设；桶内比较才等价于"在 TTL 相近的条目中淘汰频次最低者"。

四级 tie-break 的意义不是美观，而是**确定性**：同一状态必得同一 victim，测试可断言，trace 可重放。

淘汰原因分三种，避免日志说谎：

| reason | 含义 |
|---|---|
| `TTL_DOMINANT` | 桶内只有一条，纯时效决定 |
| `LFU_EVICT` | 桶内多条，频次唯一分出胜负 |
| `LFU_TIEBREAK` | 桶内多条且最低频次并列，由回合号/字典序兜底 |

---

## 6. 拼接

### 6.1 优先级与位置分离

"优先级"指**预算保护顺序**，不指文本位置：

| 优先级 | 段 | 保护 | 位置 |
|---|---|---|---|
| P1 | 当前问题 + 系统指令 | 硬下限，只允许抛异常 | 最后 |
| P2 | 缓存条目与摘要 | 可逐条裁剪 | 中间 |
| P3 | 槽位 | 可压缩或整段省略 | P2 之后 |

当前问题放最后，是因为长上下文末尾的指令遵循度更高。把它放在最后不等于它优先级低。

### 6.2 五级裁剪

```
L0 全量
L1 剔除 ttl == 1（本轮不用，下轮就消失）
L2 摘要 120 → 48 → 24
L3 按 (ttl, hit) 升序逐条移出视图
L4 单行紧凑：12(qty)·苹果(obj)·4(qty)
L5 抛 ContextBudgetExceeded
```

**裁剪只作用于视图。** `Composer.compose()` 读快照、改本地列表；`KeywordCache.enforce_capacity()` 才改字典。两者混用会出现"一次超长输入清空记忆"。

### 6.3 正文格式

```
- 4 | quantity | person.count | ttl=10 | hit=2 | T1「那么四个人呢」
```

四类信号各司其职：`ttl` 表示新鲜度、`hit` 表示重要度、`slot` 表示身份、`summary` 提供原文语境。缺任何一项，模型都会退化成"看到一堆词"。

---

## 7. 可验证性

| 手段 | 保证 |
|---|---|
| 字段冻结 | 伪代码 / dataclass / trace JSON 三处同名 |
| 确定性淘汰 | 四级 tie-break，无随机因素 |
| 有序去重 | 提取结果用 list，禁止 set 打乱顺序 |
| 轨迹重放 | JSONL 记录每次 HIT/OVERWRITE/EVICT/COMPOSE |
| 离线回答器 | `RuleLLM` 从 SLOTS 段做真实计算，无网络也能验证 |
| golden test | 六回合快照与规格表逐格比对 |
