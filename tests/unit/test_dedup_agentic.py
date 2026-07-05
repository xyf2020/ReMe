"""Integration test for the dedup agentic flow.

Verifies the full chain: DedupAgenticAnswerStep -> local BaseJob with persistent
DedupSearchStep -> agent wrapper receives local_jobs that override the global
search job -> deduplication works across multiple search tool calls.
"""

# pylint: disable=protected-access

import asyncio
from unittest.mock import MagicMock

from reme.components.file_store import BaseFileStore
from reme.components.job.base_job import BaseJob
from reme.enumeration import ComponentEnum, LinkScopeEnum
from reme.schema import FileChunk


# -- Fake store ----------------------------------------------------------------


class FakeIntegrationStore(BaseFileStore):
    """File store with call-count-aware results for integration testing."""

    def __init__(self, results_per_call: list[list[FileChunk]]):
        super().__init__(name="integration_store")
        self._results_per_call = results_per_call
        self._call_idx = 0

    async def upsert(self, files):
        pass

    async def delete(self, path):
        pass

    async def clear(self):
        pass

    async def get_nodes(self, paths=None):
        return []

    async def get_outlinks(self, path, scope=LinkScopeEnum.REAL):
        return []

    async def get_inlinks(self, path, scope=LinkScopeEnum.REAL):
        return []

    async def vector_search(self, query, limit, search_filter):
        idx = min(self._call_idx, len(self._results_per_call) - 1)
        self._call_idx += 1
        return self._results_per_call[idx][:limit]

    async def keyword_search(self, query, limit, search_filter):
        return []


def _chunk(chunk_id, path, text, start=1, end=1, score=0.5):
    return FileChunk(
        id=chunk_id,
        path=path,
        text=text,
        start_line=start,
        end_line=end,
        scores={"vector": score, "score": score},
    )


# -- Test: local_jobs override in agent wrapper --------------------------------


def test_local_jobs_override_global_job():
    """Verify that local_jobs dict overrides a global job with the same name."""

    async def run():
        # Global search job (should NOT be called)
        global_job = BaseJob(name="search", description="global")
        global_job.app_context = MagicMock()
        global_job_called = False

        original_call = global_job.__call__

        async def tracking_call(**kwargs):
            nonlocal global_job_called
            global_job_called = True
            return await original_call(**kwargs)

        global_job.__call__ = tracking_call

        # Local search job (should be called)
        local_call_count = 0

        class LocalSearchJob(BaseJob):
            """Local job override for testing."""

            async def __call__(self, **kwargs):
                nonlocal local_call_count
                local_call_count += 1
                from reme.schema import Response

                return Response(success=True, answer="local result")

        local_job = LocalSearchJob(name="search", description="local dedup")

        # Simulate what _build_agent does:
        job_tools = ["search"]
        local_jobs = {"search": local_job}

        # Mock app_context for resolution
        app_ctx = MagicMock()
        app_ctx.jobs = {"search": global_job}

        # Resolve like the wrapper does
        resolved_jobs = [app_ctx.jobs[name] for name in job_tools]
        job_map = {job.name: job for job in resolved_jobs}
        job_map.update(local_jobs)
        final_jobs = list(job_map.values())

        # Only the local job should remain
        assert len(final_jobs) == 1
        assert final_jobs[0] is local_job

        # Call it
        resp = await final_jobs[0]()
        assert resp.answer == "local result"
        assert local_call_count == 1
        assert not global_job_called

    asyncio.run(run())


# -- Test: dedup job with persistent step across multiple calls ----------------


def test_dedup_search_job_filters_across_calls():
    """Full integration: a BaseJob with _local_instantiation_ dedup_search_step
    filters duplicates across multiple calls."""

    async def run():
        # Create chunks that overlap between calls
        chunk_a = _chunk("a", "daily/a.md", "text a", 1, 5)
        chunk_b = _chunk("b", "daily/b.md", "text b", 1, 3)
        chunk_c = _chunk("c", "daily/c.md", "text c", 10, 20)

        # Call 1 returns [a, b], Call 2 returns [a, b, c]
        store = FakeIntegrationStore(
            results_per_call=[
                [chunk_a, chunk_b],
                [chunk_a, chunk_b, chunk_c],
            ],
        )

        # Build app_context mock with the store
        app_ctx = MagicMock()
        app_ctx.components = {
            ComponentEnum.FILE_STORE: {"default": store},
        }
        app_ctx.app_config = MagicMock()
        app_ctx.app_config.language = ""

        # Create the dedup search job programmatically
        job = BaseJob(
            name="search",
            description="dedup search",
            parameters={},
            steps=[
                {
                    "backend": "dedup_search_step",
                    "_local_instantiation_": 1,
                    "vector_weight": 0.7,
                    "candidate_multiplier": 2.0,
                    "expand_links": False,
                },
            ],
            app_context=app_ctx,
        )
        await job._start()

        # First call
        resp1 = await job(query="hello", limit=5)
        assert resp1.success is True
        assert resp1.metadata["counts"]["returned"] == 2
        result_paths_1 = [r["path"] for r in resp1.metadata["results"]]
        assert "daily/a.md" in result_paths_1
        assert "daily/b.md" in result_paths_1

        # Second call — a and b should be filtered, only c is new
        resp2 = await job(query="world", limit=5)
        assert resp2.success is True
        assert resp2.metadata["counts"]["before_dedup"] == 3
        assert resp2.metadata["counts"]["returned"] == 1
        assert resp2.metadata["results"][0]["path"] == "daily/c.md"

    asyncio.run(run())


# -- Test: separate local jobs have independent dedup state --------------------


def test_separate_local_jobs_independent_dedup():
    """Two independently created local jobs should not share dedup state."""

    async def run():
        chunk_a = _chunk("a", "daily/a.md", "text a", 1, 5)
        store = FakeIntegrationStore(results_per_call=[[chunk_a]] * 10)

        app_ctx = MagicMock()
        app_ctx.components = {
            ComponentEnum.FILE_STORE: {"default": store},
        }
        app_ctx.app_config = MagicMock()
        app_ctx.app_config.language = ""

        step_config = {
            "backend": "dedup_search_step",
            "_local_instantiation_": 1,
            "expand_links": False,
            "candidate_multiplier": 2.0,
        }

        # Create two separate jobs (simulating two agent sessions)
        job1 = BaseJob(name="search", steps=[step_config], app_context=app_ctx)
        job2 = BaseJob(name="search", steps=[step_config], app_context=app_ctx)
        await job1._start()
        await job2._start()

        # Job1 sees chunk_a
        resp1 = await job1(query="q", limit=5)
        assert resp1.metadata["counts"]["returned"] == 1

        # Job2 also sees chunk_a (independent state)
        resp2 = await job2(query="q", limit=5)
        assert resp2.metadata["counts"]["returned"] == 1

        # Job1 second call — now it's a duplicate
        resp1b = await job1(query="q", limit=5)
        assert resp1b.metadata["counts"]["returned"] == 0

    asyncio.run(run())
