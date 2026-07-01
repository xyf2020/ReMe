#!/usr/bin/env python3
"""Test script for LongMemEval evaluation.

Runs evaluation on the first 3 items from LongMemEval s_clean dataset.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from reme.utils import load_env
from evaluation.longmemeval.runner import LongMemEvalRunner


async def test_evaluation():
    """Run test evaluation with 3 items."""

    # Load environment variables
    load_env()

    # Paths
    data_path = PROJECT_ROOT / "datasets/longmemeval/data/longmemeval_s_cleaned.json"
    config_path = PROJECT_ROOT / "longmemeval_config.json"
    output_dir = PROJECT_ROOT / "evaluation/longmemeval/output_test"
    workspace_base = PROJECT_ROOT / "memory_workspaces"

    # Load config
    import json
    with open(config_path, 'r', encoding='utf-8') as f:
        llm_config = json.load(f).get("model", {})

    print(f"Data path: {data_path}")
    print(f"Config path: {config_path}")
    print(f"Output dir: {output_dir}")
    print(f"LLM config: {llm_config}")
    print()

    # Create runner with limit=1 (first item, all sessions)
    runner = LongMemEvalRunner(
        data_path=str(data_path),
        output_dir=str(output_dir),
        llm_config=llm_config,
        limit=1,
        workspace_base_dir=str(workspace_base),
    )

    # Run evaluation
    results = await runner.run()

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print(f"Metrics: {results['metrics']}")

    return results


if __name__ == "__main__":
    asyncio.run(test_evaluation())
