"""Answer a question from retrieved memory context (LongMemEval prompted baseline)."""

from agentscope.message import Msg

from ...base_step import BaseStep
from ....components import R


@R.register("lme_context_answer_step")
class LmeContextAnswerStep(BaseStep):
    """Answer a question using retrieved memory context via direct LLM call.

    Replicates the LongMemEval "prompted" baseline: search for relevant chunks,
    then answer using a system + user message pair.  Tracks input/output tokens.

    Context inputs:
        question:         the question to answer
        query_time:       optional ISO timestamp for temporal awareness
        retrieved_context: pre-built search context (if empty, a search job is run)
    """

    async def execute(self):
        assert self.context is not None
        question: str = self.context.get("question", "")
        query_time: str = self.context.get("query_time", "")
        retrieved_context: str = self.context.get("retrieved_context", "")

        if not question:
            raise ValueError("lme_context_answer_step requires non-empty question")

        # ── Obtain retrieved context (run search if not pre-supplied) ──
        search_hit_count = 0
        if not retrieved_context:
            search_job = self.get_job("search")
            if search_job is not None:
                search_resp = await search_job(query=question, limit=15)
                retrieved_context = (search_resp.answer or "").strip()
                search_hit_count = (search_resp.metadata or {}).get("counts", {}).get("returned", 0)
        if not retrieved_context:
            retrieved_context = "(no search results found)"

        if self.as_llm is None:
            raise RuntimeError("lme_context_answer_step requires as_llm component")

        # ── Build messages ──
        system_text = self.prompt_format("system_prompt")
        if query_time:
            system_text += "\n\n" + self.prompt_format("temporal_hint", query_time=query_time)
        user_text = self.prompt_format("user_message", retrieved_context=retrieved_context, question=question)

        messages = [
            Msg(name="system", role="system", content=[{"type": "text", "text": system_text}]),
            Msg(name="user", role="user", content=[{"type": "text", "text": user_text}]),
        ]

        # ── Direct LLM call ──
        input_tokens = 0
        output_tokens = 0
        answer = ""
        try:
            resp = await self.as_llm(messages)
            raw_text = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    raw_text += block.text
            answer = raw_text.strip()
            if resp.usage is not None:
                input_tokens = resp.usage.input_tokens
                output_tokens = resp.usage.output_tokens
        except Exception as e:
            self.logger.warning(f"[{self.name}] LLM call failed: {e}")

        self.logger.info(
            f"[{self.name}] tokens: input={input_tokens} output={output_tokens} " f"search_hits={search_hit_count}",
        )

        self.context.response.success = True
        self.context.response.answer = answer
        self.context.response.metadata.update(
            {
                "question": question,
                "query_time": query_time,
                "retrieved_context_preview": retrieved_context,
                "search_hit_count": search_hit_count,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )
        return self.context.response
