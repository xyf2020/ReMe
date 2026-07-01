"""Evaluation metrics for LongMemEval.

Implements Exact Match (EM), F1, and LLM-as-judge evaluation.
"""

import re
import string
import unicodedata
from collections import Counter
from typing import Optional

import numpy as np


def normalize_answer(s: str) -> str:
    """Normalize text for EM/F1 comparison.

    Lowercase, remove articles and punctuation, normalize whitespace.
    """
    s = s.replace(',', "")

    def remove_articles(text):
        return re.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Check if prediction exactly matches ground truth after normalization.

    Args:
        prediction: Model's predicted answer
        ground_truth: Ground truth answer

    Returns:
        True if exact match, False otherwise
    """
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)

    # Compare as sets to handle word order differences
    return set(prediction.split()) == set(ground_truth.split())


def ems(prediction: str, ground_truths: list[str]) -> float:
    """Exact Match Score - max EM across multiple ground truths.

    Args:
        prediction: Model's predicted answer
        ground_truths: List of acceptable ground truth answers

    Returns:
        1.0 if any ground truth matches, 0.0 otherwise
    """
    return float(max([exact_match_score(prediction, gt) for gt in ground_truths]))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth.

    Args:
        prediction: Model's predicted answer
        ground_truth: Ground truth answer

    Returns:
        F1 score (0.0 to 1.0)
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score, handling comma-separated multi-answers.

    Args:
        prediction: Model's predicted answer (may be comma-separated)
        ground_truth: Ground truth answer (may be comma-separated)

    Returns:
        Mean F1 score across all ground truth parts
    """
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]

    return float(np.mean([
        max([f1_score(pred, gt) for pred in predictions])
        for gt in ground_truths
    ]))


async def llm_as_judge_binary(
    question: str,
    prediction: str,
    ground_truth: str,
    llm_client,
    model_name: str = "qwen3-max",
) -> dict:
    """LLM-as-judge for binary correct/incorrect classification.

    Args:
        question: The original question
        prediction: Model's predicted answer
        ground_truth: Ground truth answer
        llm_client: LLM client instance
        model_name: Model to use for judging

    Returns:
        Dict with 'correct' (bool) and 'reasoning' (str)
    """
    prompt = f"""You are evaluating an answer to a question. Determine if the predicted answer is correct.

Question: {question}

Ground Truth: {ground_truth}

Predicted Answer: {prediction}

Is the predicted answer correct? Consider:
- The answer should contain the key information from the ground truth
- Minor wording differences are acceptable
- The answer should not contradict the ground truth

Respond with:
CORRECT or INCORRECT
Reasoning: <brief explanation>"""

    try:
        response = await llm_client.chat_completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        result_text = response["choices"][0]["message"]["content"].strip()

        is_correct = result_text.upper().startswith("CORRECT")
        reasoning = result_text

        return {"correct": is_correct, "reasoning": reasoning}
    except Exception as e:
        return {"correct": False, "reasoning": f"LLM judge error: {str(e)}"}


async def llm_as_judge_score(
    question: str,
    prediction: str,
    ground_truth: str,
    llm_client,
    model_name: str = "qwen3-max",
) -> dict:
    """LLM-as-judge for 0-5 scoring.

    Args:
        question: The original question
        prediction: Model's predicted answer
        ground_truth: Ground truth answer
        llm_client: LLM client instance
        model_name: Model to use for judging

    Returns:
        Dict with 'score' (int 0-5) and 'reasoning' (str)
    """
    prompt = f"""You are evaluating an answer to a question. Rate the answer quality on a scale of 0-5.

Question: {question}

Ground Truth: {ground_truth}

Predicted Answer: {prediction}

Rate the predicted answer:
- 5: Perfect match, contains all key information from ground truth
- 4: Very good, contains most key information with minor omissions
- 3: Good, contains some key information but misses important details
- 2: Poor, contains little relevant information
- 1: Very poor, mostly incorrect or irrelevant
- 0: Completely wrong or no answer

Respond with:
Score: <0-5>
Reasoning: <brief explanation>"""

    try:
        response = await llm_client.chat_completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        result_text = response["choices"][0]["message"]["content"].strip()

        # Extract score
        score_match = re.search(r'Score:\s*(\d)', result_text, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 0
        score = max(0, min(5, score))  # Clamp to 0-5

        return {"score": score, "reasoning": result_text}
    except Exception as e:
        return {"score": 0, "reasoning": f"LLM judge error: {str(e)}"}


def evaluate_single(
    prediction: str,
    ground_truth: str,
) -> dict:
    """Compute EM and F1 for a single prediction.

    Args:
        prediction: Model's predicted answer
        ground_truth: Ground truth answer

    Returns:
        Dict with 'em' (bool) and 'f1' (float)
    """
    return {
        "em": exact_match_score(prediction, ground_truth),
        "f1": f1_score(prediction, ground_truth),
    }


def evaluate_batch(results: list[dict]) -> dict:
    """Aggregate metrics across a batch of results.

    Args:
        results: List of dicts, each with 'em', 'f1', and optionally 'llm_judge_binary', 'llm_judge_score'

    Returns:
        Dict with aggregated metrics
    """
    if not results:
        return {"em": 0.0, "f1": 0.0, "count": 0}

    em_scores = [r["em"] for r in results]
    f1_scores = [r["f1"] for r in results]

    metrics = {
        "em": float(np.mean(em_scores)),
        "f1": float(np.mean(f1_scores)),
        "count": len(results),
    }

    # Add LLM judge metrics if available
    if any("llm_judge_binary" in r for r in results):
        binary_scores = [r.get("llm_judge_binary", {}).get("correct", False) for r in results]
        metrics["llm_judge_accuracy"] = float(np.mean(binary_scores))

    if any("llm_judge_score" in r for r in results):
        score_values = [r.get("llm_judge_score", {}).get("score", 0) for r in results]
        metrics["llm_judge_avg_score"] = float(np.mean(score_values))

    return metrics
