# LongMemEval Cleaned-S Dataset

*[中文版 / Chinese version](./README_ZH.md)*

This is a high-quality ground truth file
[`final_groundtruth_cleaned_s.json`](./final_groundtruth_cleaned_s.json),
rebuilt by this project on top of the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) **cleaned-S** split.

## 1. What We Changed

The original cleaned-S dataset had some issues at the ground truth level. We made targeted
fixes, with two main changes:

1. **Fixed the "future session leakage" temporal bug**
   In the original dataset, the evidence sessions of some questions occurred **later** than
   `question_time`, meaning the answer relied on conversations that happened after the moment
   the question was asked. This violates the "answer questions based on past memory" evaluation
   setting. We performed a temporal validation on the evidence sessions of every case and
   removed/corrected any evidence that occurred after the question time.

2. **Provided more accurate ground truth**
   We re-verified and cleaned the answers (`answer`) and the evidence session lists
   (`evidence_session_ids`), making the ground truth answers more precise and better aligned
   with the original conversation content.

## 2. File Structure


For evaluation you also need the original cleaned-S conversation data (haystack), available at:
[huggingface.co/datasets/xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) (`longmemeval_s_cleaned.json`).


## 3. Ground Truth Field Reference

`final_groundtruth_cleaned_s.json` is a JSON array of 500 cases. Each element has the following fields:

| Field | Type | Description |
| --- | --- | --- |
| `question_id` | string | Unique question identifier, matching the `question_id` in the original cleaned-S dataset |
| `question_type` | string | Question type |
| `question_time` | string | Question time, formatted as `YYYY/MM/DD (Ddd) HH:MM`, corresponding to `question_date` in the original data |
| `answer` | string | The corrected reference answer; abstention cases start with `"The information provided is not enough."` |
| `evidence_session_ids` | string[] | The list of evidence session ids supporting the answer |

The ids in `evidence_session_ids` correspond to the `answer_session_ids` of each case in the
original dataset `longmemeval_s_cleaned.json` (which in turn map to `haystack_session_ids` /
`haystack_sessions`).


## 4. References

The original LongMemEval dataset comes from
[xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval),
and the cleaned version is hosted at
[huggingface.co/datasets/xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).

```txt
Wu D, Wang H, Yu W, et al. Longmemeval: Benchmarking chat assistants on long-term interactive memory[J]. arXiv preprint arXiv:2410.10813, 2024.
```

## 5. Citation

If you use this dataset, please cite the original authors:

