# beam result

## longmemeval版本的prompt

### 100K

eval-only 模式，32 并发，20 个 case，每 case 20 题，共 400 题。耗时 72.4 分钟。

| 题型 | Prompted(limit=15) | Agentic |
|---|---|---|
| abstention | 0.525 | 0.575 |
| contradiction_resolution | 0.100 | 0.384 |
| event_ordering | 0.403 | 0.465 |
| information_extraction | 0.618 | 0.884 |
| instruction_following | 0.481 | 0.719 |
| knowledge_update | 0.637 | 0.650 |
| multi_session_reasoning | 0.444 | 0.633 |
| preference_following | 0.706 | 0.829 |
| summarization | 0.423 | 0.617 |
| temporal_reasoning | 0.344 | 0.550 |
| **OVERALL** | **0.468** | **0.631** |