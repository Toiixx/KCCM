# 轨迹说明

仓库内置两条 JSONL 轨迹，每行一个事件，可逐行重放（不变量 I6）。

| 文件 | 生成命令 | 内容 |
|---|---|---|
| `examples/traces/apple.jsonl` | `python examples/run_compare.py` | 8 轮，`full` 模板，含 T6 淘汰 |
| `examples/traces/apple_lite.jsonl` | `python examples/run_compare.py --style lite` | 同上，精简模板 |
| `examples/traces/eviction.jsonl` | `python examples/run_compare.py --turns 6` | 6 轮，与本文数值表严格对应 |

## 第 1–6 回合数值表

`max_entries=8`、`ttl_max=10`、`ttl_tolerance=2`。

| 轮次 | 输入 | 事件 | 活跃条目（keyword → ttl/hit） |
|---|---|---|---|
| 1 | 我有12个苹果，3个人每个人得到多少 | 新增 `12` `苹果` `3` `每个人` | `12→10/1` `苹果→10/1` `3→10/1` `每个人→10/1` |
| 2 | 那么四个人呢 | `3` 被 `4` OVERWRITE；`苹果` `每个人` 关联命中 | `4→10/2` `苹果→10/1` `每个人→10/1` `12→9/1` |
| 3 | 如果换成梨，还剩6个呢 | `苹果` 被 `梨` OVERWRITE；新增 `6` | `梨→10/2` `6→10/1` `4→9/2` `每个人→9/1` `12→8/1` |
| 4 | 帮我列个采购清单，加香蕉和牛奶 | 新增 `香蕉` `牛奶`（`清单` 被泛化词表剔除） | 7 条；`12→7/1` `4→8/2` `梨→9/2` |
| 5 | 再各买5个 | 新增 `5`；"各"触发 `每个人` 关联命中 | 8 条触顶；`每个人→10/1`；`12→6/1` |
| 6 | 预算50元 | 新增 `50` → 9 条超限，触发 LFU | 淘汰 `12`，回到 8 条 |

## 关键事件逐条解释

### T2：覆盖而不是新增

```json
{"turn": 2, "event": "OVERWRITE", "keyword": "4", "old": "3",
 "slot": "person.count", "reason": "SAME_COUNT_SLOT",
 "hit": 2, "superseded": ["3"]}
```

`4` 与 `3` 同为 `person.count` 槽位上的计数、且都是数值 → 判定为同槽位改写。
`hit_count` 继承旧值再 +1（1→2），`first_turn` 保持 1。

如果只做字符串层面的"新增"，缓存里会同时存在 `3` 与 `4`，prompt 变成 `person.count = 3` / `person.count = 4`，模型无从判断——这正是本项目要避免的失败模式。

### T2：关联命中不加频

```json
{"turn": 2, "event": "HIT", "keyword": "苹果", "kind": "associative", "ttl": 10, "hit": 1}
```

"四个人呢"没有字面提及苹果，但它预设了"分东西"的事件结构 → `苹果` 与 `每个人` 续命（ttl 重置为 10），`hit_count` 保持 1。
若关联命中也计频，泛化词会借事件框架反复刷高 LFU，淘汰决策随之失真。

### T6：LFU 真正出手

```
候选：12(5,1) 4(6,2) 梨(7,2) 6(7,1) 香蕉(8,1) 牛奶(8,1) 5(9,1) 每个人(9,1) 50(10,1)
mn = 5 → bucket = ttl ≤ 5+2 = 7 → [12(5,1), 4(6,2), 梨(7,2), 6(7,1)]
min hit_count = 1 → 平局 {12, 6}
tie-break last_hit_turn: 12(1) < 6(3) → 淘汰 12
reason = LFU_TIEBREAK
```

`12` 是任务锚点却被淘汰，这是 TTL 语义的必然结果：它已沉默 5 回合，第 11 回合无论如何都会归零。
若产品要求锚点常驻，应显式开启 `anchor_pin` 并把代价写进文档，而不是偷偷调大 `ttl_max`。

## 第 2 回合实际发给模型的 prompt

```text
# 系统指令
你是多轮对话助手。工作记忆由系统以「关键词缓存」形式提供，不是完整对话历史。遵守三条：
1. 只使用 MEMORY 与 SLOTS 中的事实；缺失时直接说明缺哪个值，不要编造。
2. 数值计算必须写出算式。
3. 若当前问题是省略表达（如「四个人呢」），先用 SLOTS 还原完整问题，首行固定输出 `还原：<完整问题>`。

# 工作记忆（按 ttl 降序，4 / 32 条）
- 4 | quantity | person.count | ttl=10 | hit=2 | T2「那么四个人呢」
- 每个人 | recipient | share.per_person | ttl=10 | hit=1 | T1「3个人每个人得到多少」
- 苹果 | object | item | ttl=10 | hit=1 | T1「我有12个苹果」
- 12 | quantity | apple.count | ttl=9 | hit=1 | T1「我有12个苹果」

# 槽位
- person.count = 4（原 3）
- share.per_person = 每个人
- item = 苹果
- apple.count = 12

# 当前问题
那么四个人呢
```

模型输出：

```text
还原：把 12 个苹果平均分给 4 个人，每人分得多少？
12 ÷ 4 = 3
答：每人分得 3 个苹果。
```

第一轮的原文一句话都没有进 prompt，`12` 只以 `apple.count = 12` 的形式存在。
