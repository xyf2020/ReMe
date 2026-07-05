"""Dedup search step — wraps SearchStep with cross-call chunk deduplication."""

from ...components import R
from ...utils import expand_links, render_expansion_lines
from .search import SearchStep


@R.register("dedup_search_step")
class DedupSearchStep(SearchStep):
    """Search with instance-level deduplication.

    When used with ``_local_instantiation_ > 0`` inside a job, the ``_seen``
    set persists across calls so the same chunk is never returned twice within
    one job lifetime (typically one agent session).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._seen: set[tuple[str, int, int]] = set()

    async def execute(self):
        # Run the standard hybrid search pipeline.
        await super().execute()
        assert self.context is not None

        # Filter out previously seen chunks.
        raw_results: list[dict] = self.context.response.metadata.get("results", [])
        new_results: list[dict] = []
        for r in raw_results:
            key = (r.get("path", ""), r.get("start_line", 0), r.get("end_line", 0))
            if key not in self._seen:
                self._seen.add(key)
                new_results.append(r)

        # Update counts metadata.
        counts = self.context.response.metadata.get("counts", {})
        counts["before_dedup"] = len(raw_results)
        counts["returned"] = len(new_results)
        self.context.response.metadata["counts"] = counts
        self.context.response.metadata["results"] = new_results

        # Rebuild answer text from the filtered results.
        hybrid = counts.get("hybrid", False)
        expand_links_enabled: bool = bool(self.kwargs.get("expand_links", True))
        max_links_per_direction: int = int(self.kwargs.get("max_links_per_direction", 10))

        unique_paths = list(dict.fromkeys(r.get("path", "") for r in new_results))
        link_expansion: dict[str, dict] = (
            await expand_links(self.file_store, unique_paths, max_links_per_direction) if expand_links_enabled else {}
        )
        self.context.response.metadata["link_expansion"] = link_expansion

        answer_lines: list[str] = []
        for r in new_results:
            scores = r.get("scores", {})
            score_str = self._format_scores(scores, hybrid)
            path = r.get("path", "")
            start_line = r.get("start_line", 0)
            end_line = r.get("end_line", 0)
            text = r.get("text", "")
            answer_lines.append(
                f"========== {path}:{start_line}-{end_line} " f"[{score_str}] ==========\n{text}",
            )
            answer_lines.extend(render_expansion_lines(link_expansion.get(path, {})))

        self.context.response.answer = "\n".join(answer_lines)
        return self.context.response
