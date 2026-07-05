"""Benchmark query step with dedup search — avoids returning duplicate chunks across tool calls."""

from .agentic_answer_step import AgenticAnswerStep
from ...components import R
from ...components.job.base_job import BaseJob


@R.register("dedup_agentic_answer_step")
class DedupAgenticAnswerStep(AgenticAnswerStep):
    """AgenticAnswerStep variant that uses a session-local dedup search job.

    On each execution a fresh BaseJob is created with a persistent
    ``dedup_search_step`` so the dedup state lives exactly as long as
    the current agent session — no leakage between sessions and no loss
    of history within a session.
    """

    async def execute(self):
        assert self.context is not None
        query: str = self.context.get("query", "")
        query_time: str | None = self.context.get("query_time")

        if not query:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty query"
            return self.context.response

        # Build system prompt with optional temporal context.
        sys_prompt = self.DEFAULT_SYS_PROMPT
        if query_time:
            sys_prompt += self.TEMPORAL_HINT.format(query_time=query_time)

        # --- Create a session-local dedup search job ---
        global_search = self.app_context.jobs.get("search")
        search_description = global_search.description if global_search else "Hybrid workspace search"
        search_parameters = global_search.parameters if global_search else {}

        # Inherit step config from the global search job's step_configs if possible,
        # or fall back to sensible defaults. Replace backend with dedup variant.
        step_config: dict = {
            "backend": "dedup_search_step",
            "_local_instantiation_": 1,
            "vector_weight": 0.7,
            "candidate_multiplier": 3.0,
            "expand_links": True,
            "max_links_per_direction": 10,
        }

        local_search_job = BaseJob(
            name="search",
            description=search_description,
            parameters=search_parameters,
            steps=[step_config],
            app_context=self.app_context,
        )
        await local_search_job.start()

        # --- Build wrapper kwargs with local job override ---
        wrapper_kwargs = {
            "system_prompt": sys_prompt,
            "job_tools": ["search"],
            "local_jobs": {"search": local_search_job},
            "react_config": {"max_iters": self.MAX_ITERATION},
        }

        if self.context.stream:
            text = await self._stream_reply(query, **wrapper_kwargs)
        else:
            result = await self.agent_wrapper.reply(query, **wrapper_kwargs)
            text = (result.get("result") or "").strip()

        self.logger.debug(f"[{self.name}] response: {text!r}")

        self.context.response.success = True
        self.context.response.answer = text
        self.context.response.metadata.update(
            {
                "query": query,
                "query_time": query_time,
                "sys_prompt": sys_prompt,
                "response": text,
                "dedup": True,
            },
        )
        return self.context.response
