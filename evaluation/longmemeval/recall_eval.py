"""Session-level hybrid retrieval recall evaluation for LongMemEval.

For each case in LongMemEval cleaned-s:
  1. Filter sessions to only those before question_date.
  2. Treat each session as one chunk (concatenate all messages).
  3. Run hybrid retrieval (vector + BM25, RRF fusion) aligned with search.py.
  4. Compute recall@5 and recall@10 against ground-truth sessions (also time-filtered).

Usage:
    python evaluation/longmemeval/recall_eval.py
    python evaluation/longmemeval/recall_eval.py --config evaluation/longmemeval/config.yaml
    python evaluation/longmemeval/recall_eval.py --num_items 50   # only first 50 items
    python evaluation/longmemeval/recall_eval.py --start_index 100
"""

import argparse
import asyncio
import json
import logging
import math
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("recall_eval")

# Suppress noisy loggers
for _name in ("httpx", "httpcore", "openai"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Date parsing (aligned with run.py)
# ---------------------------------------------------------------------------
_HAYSTACK_DATE_RE = re.compile(r"(\d{4}/\d{2}/\d{2})\s+\(\w+\)\s+(\d{2}:\d{2})")


def parse_haystack_date(date_str: str) -> datetime:
    """Parse haystack date format to datetime."""
    m = _HAYSTACK_DATE_RE.match(date_str)
    if not m:
        raise ValueError(f"Cannot parse haystack date: {date_str!r}")
    return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y/%m/%d %H:%M")


# ---------------------------------------------------------------------------
# Tokenizer (aligned with reme RegexTokenizer)
# ---------------------------------------------------------------------------
_WORD_PATTERN = re.compile(r"(?u)\b\w\w+\b")
_CHINESE_PATTERN = re.compile(r"[一-鿿]")


def tokenize(text: str) -> list[str]:
    """Tokenize text into word and character tokens."""
    tokens = _CHINESE_PATTERN.findall(text)
    tokens.extend(_WORD_PATTERN.findall(_CHINESE_PATTERN.sub(" ", text)))
    return tokens


# ---------------------------------------------------------------------------
# BM25 (aligned with reme BM25Index scoring)
# ---------------------------------------------------------------------------
class SimpleBM25:
    """Minimal BM25 index matching reme's BM25Index scoring formula."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_lens: list[int] = []
        # inverted index: token -> {doc_idx: tf}
        self.postings: dict[str, dict[int, int]] = {}
        self.avg_len: float = 0.0

    def add_docs(self, docs: list[tuple[str, str]]) -> None:
        """Add (doc_id, content) pairs."""
        for doc_id, content in docs:
            tokens = tokenize(content)
            if not tokens:
                continue
            idx = len(self.doc_ids)
            self.doc_ids.append(doc_id)
            self.doc_lens.append(len(tokens))
            counts = Counter(tokens)
            for tok, tf in counts.items():
                if tok not in self.postings:
                    self.postings[tok] = {}
                self.postings[tok][idx] = tf
        n = len(self.doc_ids)
        self.avg_len = sum(self.doc_lens) / n if n > 0 else 0.0

    def retrieve(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Return [(doc_id, score)] sorted by score descending."""
        n_docs = len(self.doc_ids)
        if n_docs == 0:
            return []

        query_tokens = list(dict.fromkeys(t for t in tokenize(query)))
        if not query_tokens:
            return []

        k1, b = self.k1, self.b
        avg_len = self.avg_len
        denom_base = k1 * (1.0 - b)
        denom_norm = k1 * b / avg_len if avg_len > 0 else 0.0

        scores = np.zeros(n_docs, dtype=np.float64)
        for tok in query_tokens:
            posting = self.postings.get(tok)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            if idf == 0.0:
                continue
            for idx, tf in posting.items():
                d_len = self.doc_lens[idx]
                scores[idx] += idf * tf * (k1 + 1.0) / (tf + denom_base + denom_norm * d_len)

        # Top-k
        positive_mask = scores > 0
        if not positive_mask.any():
            return []
        k = min(limit, int(positive_mask.sum()))
        top_idxs = np.argpartition(-scores, k - 1)[:k]
        top_idxs = top_idxs[np.argsort(-scores[top_idxs])]

        return [(self.doc_ids[int(i)], float(scores[int(i)])) for i in top_idxs]


# ---------------------------------------------------------------------------
# Embedding API
# ---------------------------------------------------------------------------
def _get_embedding_client() -> AsyncOpenAI:
    api_key = os.environ.get("EMBEDDING_API_KEY", "")
    base_url = os.environ.get("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-v4")
_EMBEDDING_BATCH_SIZE = 20  # dashscope limit per request


async def batch_embed(client: AsyncOpenAI, texts: list[str], max_retries: int = 5) -> np.ndarray:
    """Embed a list of texts in batches with exponential backoff on 429/5xx."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + _EMBEDDING_BATCH_SIZE]
        for attempt in range(max_retries + 1):
            try:
                resp = await client.embeddings.create(
                    model=_EMBEDDING_MODEL,
                    input=batch,
                    dimensions=1024,
                )
                # Sort by index to preserve input order
                sorted_data = sorted(resp.data, key=lambda x: x.index)
                all_embeddings.extend([d.embedding for d in sorted_data])
                break
            except Exception as e:
                err_str = str(e)
                is_retryable = "429" in err_str or "5" in err_str[:50]
                if attempt < max_retries and is_retryable:
                    wait = 2**attempt * 1.0  # 1s, 2s, 4s, 8s, 16s
                    logger.warning(f"Embedding API retry {attempt+1}/{max_retries} after {wait:.0f}s: {e}")
                    await asyncio.sleep(wait)
                else:
                    raise
    return np.array(all_embeddings, dtype=np.float32)


def cosine_similarity_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a single query vector and corpus vectors."""
    query_norm = query / (np.linalg.norm(query, axis=-1, keepdims=True) + 1e-12)
    corpus_norms = np.linalg.norm(corpus, axis=-1, keepdims=True) + 1e-12
    corpus_norm = corpus / corpus_norms
    return (query_norm @ corpus_norm.T).flatten()


# ---------------------------------------------------------------------------
# RRF fusion (aligned with search.py)
# ---------------------------------------------------------------------------
_RRF_K = 60


def rrf_merge(
    vector_results: list[tuple[str, float]],
    keyword_results: list[tuple[str, float]],
    vector_weight: float = 0.7,
) -> list[tuple[str, float]]:
    """Fuse two ranked lists with RRF, matching search.py logic."""
    text_weight = 1.0 - vector_weight
    merged: dict[str, float] = {}

    for rank, (doc_id, _) in enumerate(vector_results, start=1):
        contrib = vector_weight / (_RRF_K + rank)
        merged[doc_id] = contrib

    for rank, (doc_id, _) in enumerate(keyword_results, start=1):
        contrib = text_weight / (_RRF_K + rank)
        merged[doc_id] = merged.get(doc_id, 0.0) + contrib

    ranked = sorted(merged.items(), key=lambda kv: -kv[1])
    return ranked


# ---------------------------------------------------------------------------
# Session text building
# ---------------------------------------------------------------------------
def session_to_text(messages: list[dict]) -> str:
    """Concatenate all messages in a session into a single text string."""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def load_config(config_path: str | None = None) -> dict:
    """Load and resolve evaluation config from YAML."""
    if config_path is None:
        config_path = str(Path(__file__).parent / "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        raw = f.read()

    def _expand(m):
        expr = m.group(1)
        if ":-" in expr:
            key, default = expr.split(":-", 1)
            return os.environ.get(key, default)
        return os.environ.get(expr, "")

    raw = re.sub(r"\$\{([^}]+)\}", _expand, raw)
    return yaml.safe_load(raw)


async def evaluate_one_item(
    item: dict,
    client: AsyncOpenAI,
    vector_weight: float = 0.7,
    candidate_multiplier: float = 3.0,
) -> dict:
    """Evaluate session-level retrieval recall for one item."""
    question = item["question"]
    question_date_str = item.get("question_date", "")
    answer_session_ids = set(item.get("answer_session_ids", []))

    # Parse question time
    if question_date_str:
        question_dt = parse_haystack_date(question_date_str)
    else:
        question_dt = None

    # Build session list with time filtering
    sessions = []
    for date_str, sid, msgs in zip(
        item["haystack_dates"],
        item["haystack_session_ids"],
        item["haystack_sessions"],
    ):
        session_dt = parse_haystack_date(date_str)
        # Only keep sessions before question time
        if question_dt and session_dt > question_dt:
            continue
        sessions.append((sid, session_dt, msgs))

    if not sessions:
        logger.warning(f"[{item['question_id']}] No sessions before question_date")
        return {
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "total_sessions": 0,
            "gt_sessions_before_q": 0,
            "skipped_reason": "no_sessions_before_question_date",
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
        }

    # Filter ground truth sessions to only those before question time
    gt_before_q = answer_session_ids & {sid for sid, _, _ in sessions}
    if not gt_before_q:
        logger.info(
            f"[{item['question_id']}] No ground-truth sessions before question_date, skipping",
        )
        return {
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "total_sessions": len(sessions),
            "gt_sessions_before_q": 0,
            "skipped_reason": "no_gt_sessions_before_question_date",
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
        }

    # Build session texts
    session_ids = [sid for sid, _, _ in sessions]
    session_texts = [session_to_text(msgs) for _, _, msgs in sessions]

    # ── BM25 retrieval ──
    bm25 = SimpleBM25()
    bm25.add_docs(list(zip(session_ids, session_texts)))
    candidates = min(200, max(1, int(10 * candidate_multiplier)))
    keyword_results = bm25.retrieve(question, limit=candidates)

    # ── Vector retrieval ──
    # Embed all sessions + query
    all_texts = session_texts + [question]
    embeddings = await batch_embed(client, all_texts)
    session_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1:]

    similarities = cosine_similarity_matrix(query_embedding, session_embeddings)
    # Sort by similarity descending
    vector_order = np.argsort(-similarities)
    vector_results = [
        (session_ids[int(i)], float(similarities[int(i)])) for i in vector_order if similarities[int(i)] > 0
    ][:candidates]

    # ── RRF fusion ──
    fused = rrf_merge(vector_results, keyword_results, vector_weight)

    # ── Compute recall ──
    top5_ids = {doc_id for doc_id, _ in fused[:5]}
    top10_ids = {doc_id for doc_id, _ in fused[:10]}

    recall_at_5 = len(gt_before_q & top5_ids) / len(gt_before_q)
    recall_at_10 = len(gt_before_q & top10_ids) / len(gt_before_q)

    return {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": question,
        "total_sessions": len(sessions),
        "gt_sessions_before_q": len(gt_before_q),
        "gt_session_ids": sorted(gt_before_q),
        "top5_ids": sorted(top5_ids),
        "top10_ids": sorted(top10_ids),
        "recall_at_5": recall_at_5,
        "recall_at_10": recall_at_10,
        "vector_hits": len(vector_results),
        "keyword_hits": len(keyword_results),
    }


async def main():
    """Run recall evaluation from CLI."""
    parser = argparse.ArgumentParser(description="Session-level retrieval recall evaluation")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--num_items", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to sleep between items (rate limit)")
    args = parser.parse_args()

    # Load config
    eval_config = load_config(args.config)
    dataset_cfg = eval_config["dataset"]

    dataset_path = _PROJECT_ROOT / dataset_cfg["path"]
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    start = args.start_index if args.start_index is not None else dataset_cfg.get("start_index", 0)
    num_items = args.num_items if args.num_items is not None else dataset_cfg.get("num_items", len(data))
    items = data[start : start + num_items]

    # Filter by question_type if specified in config
    question_types = dataset_cfg.get("question_types") or []
    if question_types:
        items = [item for item in items if item.get("question_type") in question_types]

    logger.info(f"Evaluating {len(items)} item(s) starting from index {start}")

    client = _get_embedding_client()

    results = []
    total = len(items)
    start_time = time.time()

    for i, item in enumerate(items):
        logger.info(f"[{i+1}/{total}] question_id={item['question_id']} type={item['question_type']}")
        try:
            result = await evaluate_one_item(item, client)
            results.append(result)
            logger.info(
                f"  sessions={result['total_sessions']} gt_before_q={result['gt_sessions_before_q']} "
                f"recall@5={result['recall_at_5']:.3f} recall@10={result['recall_at_10']:.3f}",
            )
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results.append(
                {
                    "question_id": item["question_id"],
                    "question_type": item["question_type"],
                    "error": str(e),
                    "recall_at_5": 0.0,
                    "recall_at_10": 0.0,
                },
            )

        # Progress every 10 items
        if (i + 1) % 10 == 0 or i + 1 == total:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
            print(
                f"[PROGRESS] {i+1}/{total} ({100*(i+1)/total:.1f}%) | "
                f"elapsed={elapsed/60:.1f}min | ETA={eta/60:.1f}min",
                flush=True,
            )

        # Rate-limit delay between items
        if args.delay > 0 and i + 1 < total:
            await asyncio.sleep(args.delay)

    # ── Save results ──
    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        variant = dataset_cfg.get("variant", "unknown")
        output_dir = _PROJECT_ROOT / "evaluation" / "longmemeval" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"recall_{variant}_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_path}")

    # ── Summary ──
    golden_qids = _load_golden_qids()
    _print_summary(results, golden_qids)


_GOLDEN_QIDS_PATH = _PROJECT_ROOT / "datasets" / "longmemeval" / "golden_session_yes_question_ids.json"


def _load_golden_qids() -> set[str]:
    """Load question IDs that have valid golden sessions."""
    if not _GOLDEN_QIDS_PATH.exists():
        logger.warning(f"Golden question IDs file not found: {_GOLDEN_QIDS_PATH}")
        return set()
    with open(_GOLDEN_QIDS_PATH, encoding="utf-8") as f:
        return set(json.load(f))


def _recall_stats(items: list[dict]) -> dict:
    """Compute count, avg R@5, avg R@10 and per-type breakdown for a list of valid results."""
    if not items:
        return {"n": 0, "r5": 0.0, "r10": 0.0, "per_type": {}}
    r5 = float(np.mean([r["recall_at_5"] for r in items]))
    r10 = float(np.mean([r["recall_at_10"] for r in items]))
    per_type: dict[str, dict] = {}
    for r in items:
        qtype = r["question_type"]
        if qtype not in per_type:
            per_type[qtype] = {"r5": [], "r10": []}
        per_type[qtype]["r5"].append(r["recall_at_5"])
        per_type[qtype]["r10"].append(r["recall_at_10"])
    per_type_summary = {}
    for qtype, s in per_type.items():
        per_type_summary[qtype] = {
            "n": len(s["r5"]),
            "r5": float(np.mean(s["r5"])),
            "r10": float(np.mean(s["r10"])),
        }
    return {"n": len(items), "r5": r5, "r10": r10, "per_type": per_type_summary}


def _print_group(label: str, stats: dict) -> None:
    """Print one group's global + per-type stats."""
    print(f"\n  ── {label} ──")
    print(f"  Cases: {stats['n']}")
    if stats["n"] == 0:
        print("  (no cases)")
        return
    print(f"  Recall@5:  {stats['r5']:.4f} ({stats['r5']*100:.1f}%)")
    print(f"  Recall@10: {stats['r10']:.4f} ({stats['r10']*100:.1f}%)")
    if stats["per_type"]:
        print("  Per-type:")
        for qtype in sorted(stats["per_type"]):
            s = stats["per_type"][qtype]
            print(f"    {qtype:30s}  n={s['n']:4d}  R@5={s['r5']:.4f}  R@10={s['r10']:.4f}")


def _print_summary(results: list[dict], golden_qids: set[str]) -> None:
    """Print aggregate recall statistics with 4 groups."""
    error_items = [r for r in results if "error" in r]
    empty_gt_items = [r for r in results if "error" not in r and r.get("gt_sessions_before_q", 0) == 0]
    valid = [r for r in results if "error" not in r and r.get("gt_sessions_before_q", 0) > 0]
    golden_valid = [r for r in valid if r["question_id"] in golden_qids]

    print("\n" + "=" * 70)
    print("SESSION-LEVEL RETRIEVAL RECALL RESULTS")
    print("=" * 70)
    print(f"  Total items:              {len(results)}")
    print(f"  Valid (gt > 0):           {len(valid)}")
    print(f"  Skipped (gt empty after time filter): {len(empty_gt_items)}")
    print(f"  Errors:                   {len(error_items)}")
    if golden_qids:
        print(f"  Golden QIDs loaded:       {len(golden_qids)}")
        print(f"  Golden ∩ Valid:           {len(golden_valid)}")

    # Per-type case count table: original → valid → golden∩valid
    all_qtypes = sorted(set(r["question_type"] for r in results))
    count_total: dict[str, int] = {}
    count_valid: dict[str, int] = {}
    count_golden: dict[str, int] = {}
    for r in results:
        qt = r["question_type"]
        count_total[qt] = count_total.get(qt, 0) + 1
    for r in valid:
        qt = r["question_type"]
        count_valid[qt] = count_valid.get(qt, 0) + 1
    for r in golden_valid:
        qt = r["question_type"]
        count_golden[qt] = count_golden.get(qt, 0) + 1

    print("\n  Per-type case counts:")
    print(f"    {'question_type':30s}  {'total':>5s}  {'valid':>5s}  {'golden':>6s}")
    print(f"    {'─'*30}  {'─'*5}  {'─'*5}  {'─'*6}")
    for qt in all_qtypes:
        t = count_total.get(qt, 0)
        v = count_valid.get(qt, 0)
        g = count_golden.get(qt, 0)
        print(f"    {qt:30s}  {t:5d}  {v:5d}  {g:6d}")
    print(f"    {'─'*30}  {'─'*5}  {'─'*5}  {'─'*6}")
    print(f"    {'TOTAL':30s}  {len(results):5d}  {len(valid):5d}  {len(golden_valid):6d}")

    if empty_gt_items:
        print(f"\n  [NOTE] {len(empty_gt_items)} item(s) had no ground-truth sessions")
        print("         before question_date and were excluded from recall stats:")
        for r in empty_gt_items:
            print(f"           - {r['question_id']} (type={r['question_type']})")

    if error_items:
        print(f"\n  [ERROR] {len(error_items)} item(s) failed:")
        for r in error_items:
            print(f"           - {r['question_id']}: {r['error'][:80]}")

    # Group 1+2: all valid (gt > 0)
    all_stats = _recall_stats(valid)
    _print_group("All Valid Cases (gt > 0)", all_stats)

    # Group 3+4: golden ∩ valid
    if golden_qids:
        golden_stats = _recall_stats(golden_valid)
        _print_group("Golden ∩ Valid Cases", golden_stats)

    print("\n" + "=" * 70)

    # Per-item detail
    print("\n  Per-item detail:")
    for r in results:
        if "error" in r:
            print(f"  [{r['question_id']}] ERROR: {r['error'][:80]}")
        elif r.get("gt_sessions_before_q", 0) == 0:
            reason = r.get("skipped_reason", "unknown")
            print(
                f"  [{r['question_id']}] type={r['question_type']} "
                f"sessions={r['total_sessions']} gt=0 [SKIPPED: {reason}]",
            )
        else:
            golden_tag = " ★" if r["question_id"] in golden_qids else ""
            print(
                f"  [{r['question_id']}] type={r['question_type']} "
                f"sessions={r['total_sessions']} gt={r['gt_sessions_before_q']} "
                f"R@5={r['recall_at_5']:.3f} R@10={r['recall_at_10']:.3f}{golden_tag}",
            )
    print()


if __name__ == "__main__":
    asyncio.run(main())
