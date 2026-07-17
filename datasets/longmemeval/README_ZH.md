# LongMemEval Cleaned-S 数据集说明

本项目在 [LongMemEval](https://github.com/xiaowu0162/LongMemEval)  **cleaned-S** 基础上重新制作的高质量 ground truth 文件
[`final_groundtruth_cleaned_s.json`](./final_groundtruth_cleaned_s.json)。

## 1. 我们做了什么

原始 cleaned-S 数据集在 ground truth 层面存在一些问题，我们针对性地进行了修正，主要改动有两点：

1. **修复"未来 session 泄漏"的时序 bug**
   原始数据集中，部分问题的证据（evidence）session 时间**晚于** `question_time`，
   即回答问题时用到了提问时刻之后才发生的会话。这不符合"基于历史记忆回答问题"的评测设定。
   我们对所有 case 的证据 session 做了时序校验，剔除/修正了发生在提问时间之后的证据。

2. **提供更准确的 ground truth**
   我们对答案（`answer`）与证据 session 列表（`evidence_session_ids`）进行了重新核对与清洗，
   使标准答案更精确、更贴合原始会话内容。

## 2. 文件结构


评测时还需原始 cleaned-S 会话数据（haystack），从此处获取：
[huggingface.co/datasets/xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)（`longmemeval_s_cleaned.json`），


## 3. Ground truth 字段说明

`final_groundtruth_cleaned_s.json` 是一个 JSON 数组，共 500 个 case。每个元素字段如下：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `question_id` | string | 问题唯一标识，与原始数据集cleaned-s的 `question_id` 对应 |
| `question_type` | string | 问题类型 |
| `question_time` | string | 提问时间，格式 `YYYY/MM/DD (Ddd) HH:MM`，对应原始数据的 `question_date` |
| `answer` | string | 修正后的标准答案；弃答类以 `"The information provided is not enough."` 开头 |
| `evidence_session_ids` | string[] | 支撑答案的证据 session id 列表 |

`evidence_session_ids` 中的 id 对应原始数据集 `longmemeval_s_cleaned.json` 中每个 case 的
`answer_session_ids`（进而映射到 `haystack_session_ids` / `haystack_sessions`）。


## 4. 引用

原始 LongMemEval 数据集来自
[xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval)，
清洗版本托管于
[huggingface.co/datasets/xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)。

```
Wu D, Wang H, Yu W, et al. Longmemeval: Benchmarking chat assistants on long-term interactive memory[J]. arXiv preprint arXiv:2410.10813, 2024.
```

## 5. Citation

If you use this dataset, please cite the original authors:

