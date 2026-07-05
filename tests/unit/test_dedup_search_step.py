"""Unit tests for DedupSearchStep — cross-call chunk deduplication."""

# pylint: disable=protected-access

import asyncio

from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.enumeration import LinkScopeEnum
from reme.schema import FileChunk
from reme.steps.index import DedupSearchStep


# -- Fake store ----------------------------------------------------------------


class FakeDedupStore(BaseFileStore):
    """Minimal file_store returning configurable static results."""

    def __init__(self, vector_results=None, keyword_results=None):
        super().__init__(name="fake_dedup_store")
        self.vector_results = vector_results or []
        self.keyword_results = keyword_results or []

    async def upsert(self, files):
        raise NotImplementedError

    async def delete(self, path):
        raise NotImplementedError

    async def clear(self):
        raise NotImplementedError

    async def get_nodes(self, paths=None):
        return []

    async def get_outlinks(self, path, scope=LinkScopeEnum.REAL):
        return []

    async def get_inlinks(self, path, scope=LinkScopeEnum.REAL):
        return []

    async def vector_search(self, query, limit, search_filter):
        return self.vector_results[:limit]

    async def keyword_search(self, query, limit, search_filter):
        return self.keyword_results[:limit]


# -- Helpers -------------------------------------------------------------------


def _chunk(chunk_id, path, text, start_line=1, end_line=1, score=0.5):
    return FileChunk(
        id=chunk_id,
        path=path,
        text=text,
        start_line=start_line,
        end_line=end_line,
        scores={"vector": score, "score": score},
    )


# -- Tests ---------------------------------------------------------------------


def test_first_call_returns_all_results():
    """First call should return everything — nothing is seen yet."""

    async def run():
        chunks = [
            _chunk("a", "daily/a.md", "text a", 1, 5),
            _chunk("b", "daily/b.md", "text b", 1, 3),
        ]
        store = FakeDedupStore(vector_results=chunks, keyword_results=[])
        step = DedupSearchStep(file_store=store, expand_links=False, candidate_multiplier=2)
        ctx = RuntimeContext(query="hello", limit=5)

        resp = await step(ctx)

        assert resp.success is True
        assert len(resp.metadata["results"]) == 2
        assert resp.metadata["counts"]["returned"] == 2

    asyncio.run(run())


def test_second_call_filters_duplicates():
    """Second call with same results should return empty (all previously seen)."""

    async def run():
        chunks = [
            _chunk("a", "daily/a.md", "text a", 1, 5),
            _chunk("b", "daily/b.md", "text b", 1, 3),
        ]
        store = FakeDedupStore(vector_results=chunks, keyword_results=[])
        step = DedupSearchStep(file_store=store, expand_links=False, candidate_multiplier=2)

        # First call
        ctx1 = RuntimeContext(query="hello", limit=5)
        await step(ctx1)

        # Second call with same chunks
        ctx2 = RuntimeContext(query="hello again", limit=5)
        resp2 = await step(ctx2)

        assert resp2.metadata["counts"]["before_dedup"] == 2
        assert resp2.metadata["counts"]["returned"] == 0
        assert len(resp2.metadata["results"]) == 0

    asyncio.run(run())


def test_new_chunks_pass_through():
    """New chunks not previously seen should pass through."""

    async def run():
        chunks_call1 = [_chunk("a", "daily/a.md", "text a", 1, 5)]
        chunks_call2 = [
            _chunk("a", "daily/a.md", "text a", 1, 5),  # duplicate
            _chunk("c", "daily/c.md", "text c", 10, 20),  # new
        ]

        store = FakeDedupStore(vector_results=chunks_call1, keyword_results=[])
        step = DedupSearchStep(file_store=store, expand_links=False, candidate_multiplier=2)

        # First call
        ctx1 = RuntimeContext(query="q1", limit=5)
        await step(ctx1)

        # Second call with mixed old+new
        store.vector_results = chunks_call2
        ctx2 = RuntimeContext(query="q2", limit=5)
        resp2 = await step(ctx2)

        # Only chunk "c" should be returned.
        assert resp2.metadata["counts"]["returned"] == 1
        assert resp2.metadata["results"][0]["path"] == "daily/c.md"
        assert resp2.metadata["results"][0]["start_line"] == 10

    asyncio.run(run())


def test_dedup_key_is_path_start_end():
    """Chunks with same path but different line ranges are not deduplicated."""

    async def run():
        chunks = [
            _chunk("a1", "daily/a.md", "para1", 1, 5),
            _chunk("a2", "daily/a.md", "para2", 6, 10),
        ]
        store = FakeDedupStore(vector_results=chunks, keyword_results=[])
        step = DedupSearchStep(file_store=store, expand_links=False, candidate_multiplier=2)

        ctx1 = RuntimeContext(query="q1", limit=5)
        resp1 = await step(ctx1)
        assert resp1.metadata["counts"]["returned"] == 2

        # Call again — both should be filtered now.
        ctx2 = RuntimeContext(query="q2", limit=5)
        resp2 = await step(ctx2)
        assert resp2.metadata["counts"]["returned"] == 0

    asyncio.run(run())


def test_seen_accumulates_across_calls():
    """_seen grows incrementally across multiple calls."""

    async def run():
        store = FakeDedupStore(vector_results=[], keyword_results=[])
        step = DedupSearchStep(file_store=store, expand_links=False, candidate_multiplier=2)

        # Call 1: chunk a
        store.vector_results = [_chunk("a", "p/a.md", "a", 1, 1)]
        ctx = RuntimeContext(query="q", limit=5)
        await step(ctx)
        assert len(step._seen) == 1

        # Call 2: chunk b
        store.vector_results = [_chunk("b", "p/b.md", "b", 1, 1)]
        ctx = RuntimeContext(query="q", limit=5)
        await step(ctx)
        assert len(step._seen) == 2

        # Call 3: both a and b — nothing new
        store.vector_results = [
            _chunk("a", "p/a.md", "a", 1, 1),
            _chunk("b", "p/b.md", "b", 1, 1),
        ]
        ctx = RuntimeContext(query="q", limit=5)
        resp = await step(ctx)
        assert resp.metadata["counts"]["returned"] == 0
        assert len(step._seen) == 2

    asyncio.run(run())
