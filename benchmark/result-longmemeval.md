# LongMemEval 数据集测试结果

## cleaned-s

**basic settings**

1. 使用修改后的auto-memory prompt，关闭auto-dream机制
2. reme-memory中的全部session的时间一定早于question的时间

**results **

1. Agentic answer框架回答，每次最多调用5次search

| Category | Total | Correct | Wrong | Accuracy |
|---|---|---|---|---|
| single-session-user | 70 | 66 | 4 | 94.3% |
| single-session-assistant | 56 | 52 | 4 | 92.9% |
| knowledge-update | 78 | 60 | 18 | 76.9% |
| multi-session | 133 | 93 | 40 | 69.9% |
| temporal-reasoning | 133 | 78 | 55 | 58.6% |
| single-session-preference | 30 | 8 | 22 | 26.7% |
| **Overall** | **500** | **357** | **143** | **71.4%** |

2. prompted-based amswer，每次固定使用原始query召回10个fileChunk

| Category | Total | Correct | Wrong | Accuracy |
|---|---|---|---|---|
| single-session-assistant | 56 | 56 | 0 | 100.0% |
| single-session-user | 70 | 67 | 3 | 95.7% |
| knowledge-update | 78 | 69 | 9 | 88.5% |
| multi-session | 133 | 99 | 34 | 74.4% |
| temporal-reasoning | 133 | 83 | 50 | 62.4% |
| single-session-preference | 30 | 16 | 14 | 53.3% |
| **Overall** | **500** | **390** | **110** | **78.0%** |

3. golden session。 使用与prompt-based answer相似的方法，唯一区别是，输入的chunk是longMemEval提供的golden session。

| Category | Total | Correct | Wrong | Accuracy |
|---|---|---|---|---|
| single-session-assistant | 56 | 56 | 0 | 100.0% |
| single-session-user | 70 | 69 | 1 | 98.6% |
| knowledge-update | 78 | 74 | 4 | 94.9% |
| temporal-reasoning | 133 | 124 | 9 | 93.2% |
| multi-session | 133 | 117 | 16 | 88.0% |
| single-session-preference | 30 | 17 | 13 | 56.7% |
| **Overall** | **500** | **457** | **43** | **91.4%** |

4. golden session + time filter. 和上面一个实验的区别是，输入的golden被过滤了一次，要求输入session的时间戳必须早于question的时间才行。

一共被过滤掉了75个session，44个question受到了影响。temperal reasoning类型受影响最大。有20个case不包含任何一个groundtruth session。 根据golden session回答正确并且golden session非空，一共有424个case。

| Category | Total | Correct | Wrong | Accuracy |
|---|---|---|---|---|
| knowledge-update | 78 | 75 | 3 | 96.2% |
| single-session-user | 70 | 67 | 3 | 95.7% |
| multi-session | 133 | 122 | 11 | 91.7% |
| single-session-assistant | 56 | 55 | 1 | 98.2% |
| temporal-reasoning | 133 | 91 | 42 | 68.4% |
| single-session-preference | 30 | 16 | 14 | 53.3% |
| **Overall** | **500** | **426** | **74** | **85.2%** |

5. 关闭auto-memory机制，根据原始query一次性混合检索召回原始session，计算recall.

| Category | Total  | yes-judge | recall@5 / yes | recall@10 / yes |
|---|---|---|---|---|
| knowledge-update | 78 | 75 | 99.3% | 100% |
| single-session-user | 70 | 67 | 100% | 100% |
| multi-session | 133 | 122 | 91.8% | 95.8% |
| single-session-assistant | 56 | 55 | 100% | 100% |
| temporal-reasoning | 133 | 91 | 87.6% | 94.2% |
| single-session-preference | 30 | 16 | 100% | 100% |
| **Overall** | **500** | **426** | **87.6%** | **94.2%** |


## 最终groundtruth

### agentic + prompted（最终GT，2026-07-16）


| Category | Total | Agentic | Prompted limit=15 |
|---|---|---|---|
| single-session-assistant | 56 | 56/56 (100.0%) | 54/56 (96.4%) |
| single-session-user | 70 | 66/70 (94.3%) | 62/70 (88.6%) |
| knowledge-update | 78 | 75/78 (96.2%) | 67/78 (85.9%) |
| temporal-reasoning | 133 | 122/133 (91.7%) | 117/133 (88.0%) |
| multi-session | 133 | 115/133 (86.5%) | 101/133 (75.9%) |
| single-session-preference | 30 | 21/30 (70.0%) | 10/30 (33.3%) |
| **Overall** | **500** | **455/500 (91.0%)** | **411/500 (82.2%)** |

Prompted token 消耗：总 input 13,111,421 (平均 26,275/题)，总 output 313,370 (平均 628/题)。
平均 sessions_ingested: 44.8，dreams_triggered: 0。


### 消融实验

| 实验条件 |	总准确率 |	耗时 |
|---|---|---|
| Baseline（全量）|	88.6% (443/500) |	8.8 min |
| 禁用 session file chunks |	85.2% (426/500) |	7.4 min |
| 禁用 note file chunks |	86.2% (431/500) |	8.9 min |

### 压缩session

三种压缩策略对比(eval-only 复用同一批 memory workspace,32 并发,2026-07-28/29):

| 实验条件 | 总准确率 | 耗时 | search 总调用 |
|---|---|---|---|
| 无压缩 | 89.6% (448/500) | 7.5 min | 1621 |
| query-aware 压缩 | 87.8% (439/500) | 39.8 min | 1644 |
| query-independent 压缩 | 87.0% (435/500) | 42.9 min | 1699 |

**search 工具调用次数统计**(由日志按时间窗+词面重叠归属到 item,长尾数字含少量归属误差):

| 指标 | 无压缩 | query-aware | query-independent |
|---|---|---|---|
| 均值/题 | 3.24 | 3.29 | 3.40 |
| 中位数 | 2 | 2 | 2 |
| P90 | 6 | 6 | 7 |
| 最大 | 27 | 50 | 37 |
| 0 次调用题数 | 14 | 9 | 10 |

**search 调用次数分布**(searches/题 → 题数):

| searches/题 | 无压缩 | query-aware | query-independent |
|---|---|---|---|
| 0 | 14 | 9 | 10 |
| 1 | 85 | 85 | 89 |
| 2 | 194 | 217 | 201 |
| 3 | 71 | 65 | 67 |
| 4 | 41 | 34 | 35 |
| 5 | 23 | 28 | 23 |
| 6 | 27 | 17 | 25 |
| 7 | 15 | 14 | 12 |
| 8 | 5 | 5 | 12 |
| 9 | 8 | 8 | 4 |
| 10-14 | 9 | 8 | 13 |
| 15-19 | 2 | 5 | 1 |
| 20+ | 6 | 5 | 8 |

观察:

1. 三种策略的分布形态基本一致:峰值都在 2 次(38.8%~43.4%),1~3 次覆盖约 70~73% 的题目,压缩策略不改变 agent 的搜索习惯。
2. `max_iters=10` 限制的是推理轮数而非工具调用数:LLM 单轮可输出多个 search tool_call(ReMe 侧 job 工具标记 `is_concurrency_safe=False`,批内串行执行),因此长尾可达 20~50 次;真正触发 10 轮上限的极少(无压缩 0 题、query-aware 0 题、query-independent 2 题)。
3. query-independent 长尾最重(8 次及以上 38 题 vs 无压缩 30 题、query-aware 31 题):通用压缩保留内容与问题相关性弱,难题上需要更多轮改写查询。
4. 压缩为净损耗:两种压缩策略总准确率均低于无压缩约 2~2.6%,且因 compressor LLM 调用耗时增加 5 倍以上;受损最重的题型为 single-session-preference 与 single-session-user。
