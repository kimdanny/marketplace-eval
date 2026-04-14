from __future__ import annotations

import random
from abc import abstractmethod
from typing import Any

from marketplace_eval.agents.agent import Agent
from marketplace_eval.system.types import GenerationResult, JudgeFeedback, UserQuery
from marketplace_eval.utils.llm_client import (
    BaseLLMClient,
    LLMClientError,
    create_llm_client,
)

# ---------------------------------------------------------------------------
# Registry: maps judge_prompt config strings -> LLMJudgeAgent subclasses.
# Subclasses register themselves via the @register_judge decorator.
# ---------------------------------------------------------------------------
_JUDGE_REGISTRY: dict[str, type[LLMJudgeAgent]] = {}


def register_judge(key: str):
    """Class decorator that registers an LLMJudgeAgent subclass under *key*."""

    def wrapper(cls: type[LLMJudgeAgent]) -> type[LLMJudgeAgent]:
        _JUDGE_REGISTRY[key] = cls
        return cls

    return wrapper


def _get_judge_class(key: str) -> type[LLMJudgeAgent]:
    """Look up a registered judge class by its config key."""
    if key not in _JUDGE_REGISTRY:
        available = ", ".join(sorted(_JUDGE_REGISTRY)) or "(none)"
        raise ValueError(
            f"Unknown judge type '{key}'. Registered types: {available}. "
            "Register a custom judge with @register_judge."
        )
    return _JUDGE_REGISTRY[key]


def create_judge(node_id: str, *, judge_prompt: str, **kwargs) -> LLMJudgeAgent:
    """Factory: instantiate the right LLMJudgeAgent subclass from config."""
    cls = _get_judge_class(judge_prompt)
    return cls(node_id, **kwargs)


class LLMJudgeAgent(Agent):
    """Abstract base class for LLM-based judge agents.

    Subclasses must implement:
        - ``format_prompt``: build the evaluation prompt from a generation and
          (optionally) the original user query.
        - ``parse_llm_response``: extract a numeric score from the raw LLM
          response string.

    Subclasses may optionally override:
        - ``build_judge_feedback``: construct the full ``JudgeFeedback``
          object.  The default implementation returns a minimal feedback
          containing only the score.  Override this to attach a rationale,
          generator-specific feedback, retriever scores, etc.

    The core orchestration — calling the LLM and assembling the feedback —
    lives in ``invoke`` and is shared by all subclasses.
    """

    def __init__(
        self,
        node_id: str,
        *,
        cost: float = 0.0,
        model_id: str | None = None,
        name: str | None = None,
        provider: str | None = None,
        generation_parameters: dict[str, Any] | None = None,
        base_url: str | None = None,
        rng_seed: int | None = None,
    ):
        super().__init__(node_id, cost=cost, model_id=model_id, name=name)
        self.rng = random.Random(rng_seed)

        self._llm_client: BaseLLMClient | None = None
        if provider and model_id:
            client_config: dict[str, Any] = {
                "provider": provider,
                "model_id": model_id,
            }
            if generation_parameters:
                client_config["generation_parameters"] = generation_parameters
            if base_url:
                client_config["base_url"] = base_url
            try:
                self._llm_client = create_llm_client(client_config)
            except LLMClientError as exc:
                raise ValueError(
                    f"Failed to initialise LLM client for judge '{node_id}': {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def format_prompt(
        self,
        *,
        generation: GenerationResult,
        user_query: UserQuery | None = None,
    ) -> str:
        """Build the full evaluation prompt string for the LLM judge."""

    @abstractmethod
    def parse_llm_response(self, response: str) -> float:
        """Extract a numeric score from the raw LLM judge response."""

    # ------------------------------------------------------------------
    # Overridable hook — subclasses CAN override this
    # ------------------------------------------------------------------

    def build_judge_feedback(
        self,
        *,
        score: float,
        raw_response: str,
        generation: GenerationResult,
        user_query: UserQuery | None = None,
    ) -> JudgeFeedback:
        """Construct the full ``JudgeFeedback`` from the parsed score.

        The default implementation returns minimal feedback with only the
        score.  Override in a subclass to attach a rationale, retriever
        scores, generator-specific metadata, etc.
        """
        return JudgeFeedback(score=score, rationale="", retriever_scores={})

    # ------------------------------------------------------------------
    # Core invoke — shared by all subclasses
    # ------------------------------------------------------------------

    async def invoke(
        self,
        *,
        generation: GenerationResult,
        user_query: UserQuery | None = None,
        **kwargs,
    ) -> JudgeFeedback:
        """Evaluate a generated answer using the LLM judge.

        1. Calls ``format_prompt`` to build the prompt.
        2. Sends the prompt to the configured LLM.
        3. Calls ``parse_llm_response`` to extract the numeric score.
        4. Calls ``build_judge_feedback`` to assemble the full feedback.
        """
        if self._llm_client is None:
            raise RuntimeError(
                f"Judge '{self.node_id}' has no LLM client configured. "
                "Provide 'provider' and 'model_id' in judge params."
            )

        prompt = self.format_prompt(generation=generation, user_query=user_query)

        try:
            response = await self._llm_client.generate(prompt=prompt)
        except LLMClientError as exc:
            raise RuntimeError(
                f"Judge '{self.node_id}' failed to complete LLM request: {exc}"
            ) from exc

        score = self.parse_llm_response(response)

        return self.build_judge_feedback(
            score=score,
            raw_response=response,
            generation=generation,
            user_query=user_query,
        )
