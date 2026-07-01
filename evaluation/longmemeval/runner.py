"""Main evaluation runner for LongMemEval on ReMe.

Orchestrates the evaluation loop:
1. Load LongMemEval data
2. For each question:
   a. Create fresh workspace
   b. Input sessions with proper date handling
   c. Run dream when crossing 23:00 boundaries
   d. Search and generate answer
   e. Evaluate with EM, F1, and LLM-as-judge
   f. Clean up workspace
"""

import asyncio
import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .data_adapter import EvalItem, load_longmemeval_data, format_session_for_reme
from .dream_scheduler import find_next_23h_boundary, get_dream_date_for_boundary
from .metrics import evaluate_single, evaluate_batch, llm_as_judge_binary, llm_as_judge_score


class LLMClient:
    """Simple LLM client wrapper for answer generation and judging."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        # Normalize base_url: strip trailing /v1 if present
        base_url = base_url.rstrip('/')
        if base_url.endswith('/v1'):
            base_url = base_url[:-3]
        self.base_url = base_url
        self.model = model

    async def chat_completion(self, messages: list[dict], temperature: float = 0.7, model: str = None) -> dict:
        """Call LLM API for chat completion."""
        import httpx

        use_model = model or self.model
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": use_model,
                    "messages": messages,
                    "temperature": temperature,
                },
            )
            response.raise_for_status()
            return response.json()


class LongMemEvalRunner:
    """Runner for LongMemEval evaluation on ReMe."""

    def __init__(
        self,
        data_path: str,
        output_dir: str,
        llm_config: dict,
        limit: Optional[int] = None,
        workspace_base_dir: Optional[str] = None,
        session_limit: Optional[int] = None,
    ):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.session_limit = session_limit  # Max sessions per item (None = all)
        # Base directory for per-item workspaces (each item gets its own subdirectory)
        if workspace_base_dir:
            self.workspace_base_dir = Path(workspace_base_dir)
        else:
            self.workspace_base_dir = Path(output_dir).parent / "memory_workspaces"
        self.workspace_base_dir.mkdir(parents=True, exist_ok=True)

        # Parse LLM config
        self.memory_model = llm_config.get("memory base", "qwen-flash")
        self.answer_model = llm_config.get("generate answer", "qwen3-max")
        self.judge_model = llm_config.get("llm-as-judge", "qwen3-max")

        # LLM clients (will be initialized when needed)
        self._answer_client: Optional[LLMClient] = None
        self._judge_client: Optional[LLMClient] = None

    async def _get_answer_client(self) -> LLMClient:
        """Get or create LLM client for answer generation."""
        if self._answer_client is None:
            api_key = os.environ.get("LLM_API_KEY", "")
            base_url = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode")
            self._answer_client = LLMClient(api_key, base_url, self.answer_model)
        return self._answer_client

    async def _get_judge_client(self) -> LLMClient:
        """Get or create LLM client for LLM-as-judge."""
        if self._judge_client is None:
            api_key = os.environ.get("LLM_API_KEY", "")
            base_url = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode")
            self._judge_client = LLMClient(api_key, base_url, self.judge_model)
        return self._judge_client

    async def run(self) -> dict:
        """Run the full evaluation loop.

        Returns:
            Dict with aggregated metrics and per-question results
        """
        print(f"Loading LongMemEval data from {self.data_path}...")
        items = load_longmemeval_data(self.data_path, limit=self.limit)
        print(f"Loaded {len(items)} evaluation items")

        all_results = []

        for idx, item in enumerate(items):
            print(f"\n{'='*60}")
            print(f"Evaluating item {idx+1}/{len(items)}: {item.question.question_id}")
            print(f"Question: {item.question.question}")
            print(f"Sessions: {len(item.sessions)}")
            print(f"{'='*60}")

            try:
                result = await self._evaluate_single_item(item, idx)
                all_results.append(result)
                print(f"Result - EM: {result['em']}, F1: {result['f1']:.4f}")
                if 'llm_judge_binary' in result:
                    print(f"LLM Judge - Correct: {result['llm_judge_binary']['correct']}")
                if 'llm_judge_score' in result:
                    print(f"LLM Judge - Score: {result['llm_judge_score']['score']}/5")
            except Exception as e:
                print(f"Error evaluating item {item.question.question_id}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "question_id": item.question.question_id,
                    "error": str(e),
                    "em": 0.0,
                    "f1": 0.0,
                })

        # Aggregate metrics
        print(f"\n{'='*60}")
        print("Computing aggregate metrics...")
        print(f"{'='*60}")

        metrics = evaluate_batch(all_results)

        output = {
            "metrics": metrics,
            "results": all_results,
        }

        # Save results
        output_path = self.output_dir / "evaluation_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

        # Print summary
        print(f"\nFinal Metrics:")
        print(f"  Questions evaluated: {metrics['count']}")
        print(f"  Exact Match (EM): {metrics['em']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        if 'llm_judge_accuracy' in metrics:
            print(f"  LLM Judge Accuracy: {metrics['llm_judge_accuracy']:.4f}")
        if 'llm_judge_avg_score' in metrics:
            print(f"  LLM Judge Avg Score: {metrics['llm_judge_avg_score']:.2f}/5")

        return output

    async def _evaluate_single_item(self, item: EvalItem, item_idx: int) -> dict:
        """Evaluate a single question.

        Args:
            item: Evaluation item with question and sessions
            item_idx: Index of the item (for workspace naming)

        Returns:
            Dict with metrics for this question
        """
        from reme import Application
        from reme.config import resolve_app_config

        # Create per-item workspace directory
        workspace_dir = self.workspace_base_dir / f"item_{item_idx}"
        # Clean up any previous run for this item
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir, ignore_errors=True)
        workspace_path = workspace_dir / ".reme"
        workspace_path.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize ReMe application (with background jobs disabled for evaluation)
            cfg = resolve_app_config(
                workspace_dir=str(workspace_path),
                log_to_console=False,
                log_to_file=False,
                enable_logo=False,
            )
            # _deep_merge can't clear nested dicts; selectively remove
            # background/cron jobs while keeping base jobs (auto_memory, search, etc.)
            jobs = cfg.get("jobs", {})
            jobs_to_remove = [
                name for name, jcfg in jobs.items()
                if jcfg.get("backend") in ("background", "cron")
            ]
            for name in jobs_to_remove:
                del jobs[name]
            app = Application(**cfg)
            await app.start()

            # Apply session_limit if set
            sessions_to_process = item.sessions
            if self.session_limit:
                sessions_to_process = sessions_to_process[:self.session_limit]
                print(f"  Session limit: processing {len(sessions_to_process)} of {len(item.sessions)} sessions")

            # Process sessions with synchronous dream scheduling
            # Dream runs synchronously: wait for completion before next session
            prev_session_end = None
            dreamed_dates: set[str] = set()

            for sess_idx, session in enumerate(sessions_to_process):
                session_start = session.start_time
                print(f"  Processing session {sess_idx+1}/{len(sessions_to_process)}: {session.session_id} at {session_start}")

                # Check if we need to run dream before this session
                # Uses previous session's END time and current session's START time
                if prev_session_end and session_start:
                    # Check for 23:00 boundaries between sessions
                    boundary = find_next_23h_boundary(prev_session_end)
                    while boundary and boundary < session_start:
                        dream_date = get_dream_date_for_boundary(boundary)
                        if dream_date not in dreamed_dates:
                            print(f"    Running dream for date: {dream_date} (boundary at {boundary})")
                            await app.run_job("auto_dream", date=dream_date)
                            dreamed_dates.add(dream_date)
                            # DreamFinishStep now does incremental indexing of digest nodes
                        boundary = find_next_23h_boundary(boundary + timedelta(minutes=1))

                    # Also check if the date actually changed between sessions
                    # This handles sessions that span past 23:00:
                    # e.g., session ends at 23:05 on day A, next session starts at 09:00 on day B
                    # find_next_23h_boundary(23:05) returns 23:00 day B, which is > session_start,
                    # so the boundary loop above misses day A's dream.
                    prev_date = prev_session_end.strftime("%Y-%m-%d")
                    curr_date = session_start.strftime("%Y-%m-%d")
                    if prev_date != curr_date and prev_date not in dreamed_dates:
                        print(f"    Running dream for previous day: {prev_date} (date changed to {curr_date})")
                        await app.run_job("auto_dream", date=prev_date)
                        dreamed_dates.add(prev_date)
                        # DreamFinishStep now does incremental indexing of digest nodes

                # Convert session to ReMe format
                messages = format_session_for_reme(session)
                date_str = session.date_str

                # Call auto_memory with explicit date (synchronous - awaits completion)
                print(f"    Calling auto_memory with date={date_str}")
                await app.run_job(
                    "auto_memory",
                    messages=messages,
                    session_id=session.session_id,
                    date=date_str,
                )

                # Track session end time (last turn timestamp)
                if session.turns:
                    prev_session_end = session.turns[-1].timestamp

            # Final dream after all sessions: must dream the last day's notes
            if sessions_to_process:
                last_session = sessions_to_process[-1]
                if last_session.turns:
                    last_date = last_session.date_str
                    if last_date not in dreamed_dates:
                        print(f"  Running final dream for last session date: {last_date}")
                        await app.run_job("auto_dream", date=last_date)

            # Reindex to ensure all content is indexed
            print("  Running reindex...")
            await app.run_job("reindex")

            # Test search with multiple limit values
            search_limits = [1, 3, 5, 10]
            limit_results = {}
            answer_client = await self._get_answer_client()
            judge_client = await self._get_judge_client()

            for limit in search_limits:
                print(f"\n  --- Testing search limit={limit} ---")
                search_result = await app.run_job("search", query=item.question.question, limit=limit)
                search_results = search_result.metadata.get("results", [])
                counts = search_result.metadata.get("counts", {})
                vector_hits = counts.get("vector", 0)
                keyword_hits = counts.get("keyword", 0)

                context_parts = []
                for result in search_results:
                    text = result.get("text", "")
                    if text:
                        context_parts.append(text)
                context = "\n\n".join(context_parts) if context_parts else "No relevant memories found."
                print(f"  limit={limit}: search_hits={len(search_results)}, vector={vector_hits}, keyword={keyword_hits}, context_chars={len(context)}")

                # Generate answer
                answer_prompt = f"""Based on the following memories, answer the question.

Memories:
{context}

Question: {item.question.question}

Provide a concise, direct answer:"""
                answer_response = await answer_client.chat_completion(
                    messages=[{"role": "user", "content": answer_prompt}],
                    temperature=0.0,
                )
                generated_answer = answer_response["choices"][0]["message"]["content"].strip()
                print(f"  limit={limit} answer: {generated_answer[:100]}...")

                # Evaluate
                metrics_limit = evaluate_single(generated_answer, item.question.answer)

                # LLM-as-judge
                llm_binary = await llm_as_judge_binary(
                    item.question.question,
                    generated_answer,
                    item.question.answer,
                    judge_client,
                    model_name=self.judge_model,
                )
                llm_score = await llm_as_judge_score(
                    item.question.question,
                    generated_answer,
                    item.question.answer,
                    judge_client,
                    model_name=self.judge_model,
                )

                metrics_limit["llm_judge_binary"] = llm_binary
                metrics_limit["llm_judge_score"] = llm_score
                metrics_limit["search_limit"] = limit
                metrics_limit["search_hits"] = len(search_results)
                metrics_limit["prediction"] = generated_answer

                limit_results[str(limit)] = metrics_limit
                print(f"  limit={limit} -> EM={metrics_limit['em']}, F1={metrics_limit['f1']:.4f}, "
                      f"correct={llm_binary.get('correct')}, score={llm_score.get('score')}/5")

            # Use limit=10 as the primary result for backward compatibility
            primary = limit_results.get("10", limit_results.get(str(search_limits[-1]), {}))
            metrics = {
                "em": primary.get("em", 0.0),
                "f1": primary.get("f1", 0.0),
                "question_id": item.question.question_id,
                "question": item.question.question,
                "ground_truth": item.question.answer,
                "prediction": primary.get("prediction", ""),
                "search_hits": primary.get("search_hits", 0),
                "limit_results": limit_results,
            }

            # Print summary table for this item
            print(f"\n  {'='*50}")
            print(f"  Limit comparison summary for {item.question.question_id}:")
            print(f"  {'limit':>6} | {'EM':>5} | {'F1':>6} | {'correct':>7} | {'score':>5} | {'hits':>4}")
            print(f"  {'-'*6}-+-{'-'*5}-+-{'-'*6}-+-{'-'*7}-+-{'-'*5}-+-{'-'*4}")
            for lim in search_limits:
                lr = limit_results.get(str(lim), {})
                lb = lr.get("llm_judge_binary", {})
                ls = lr.get("llm_judge_score", {})
                print(f"  {lim:>6} | {lr.get('em', 0.0):>5.1f} | {lr.get('f1', 0.0):>6.4f} | "
                      f"{str(lb.get('correct', False)):>7} | {ls.get('score', 0):>5}/5 | {lr.get('search_hits', 0):>4}")
            print(f"  {'='*50}")

            await app.close()
            return metrics

        finally:
            # Keep workspace for inspection (do not delete)
            pass


async def main():
    """Main entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="LongMemEval evaluation for ReMe")
    parser.add_argument(
        "--data-path",
        type=str,
        default="datasets/longmemeval/data/longmemeval_s_cleaned.json",
        help="Path to LongMemEval data file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/longmemeval/output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="longmemeval_config.json",
        help="Path to LLM config file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of evaluation items (default: all)",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        llm_config = json.load(f).get("model", {})

    # Run evaluation
    runner = LongMemEvalRunner(
        data_path=args.data_path,
        output_dir=args.output_dir,
        llm_config=llm_config,
        limit=args.limit,
    )

    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
