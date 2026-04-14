from __future__ import annotations

import asyncio
import random
import re
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Set, Union

from marketplace_eval.agents.agent import Agent
from marketplace_eval.utils.llm_client import (
    BaseLLMClient,
    LLMClientError,
    create_llm_client,
)
from marketplace_eval.agents.retriever_agent import RetrieverAgent
from marketplace_eval.agents.router_agent import RouterAgent
from marketplace_eval.core.io_nodes import OutputNode
from marketplace_eval.system.types import GenerationResult, RetrievalCall, UserQuery

if TYPE_CHECKING:
    from marketplace_eval.system.system import System


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


_PLANNER_REGISTRY: dict[str, type["BaseRetrievalPlanner"]] = {}


def register_planner(key: str):
    """Class decorator that registers a BaseRetrievalPlanner subclass under *key*.

    Usage::

        @register_planner("my_strategy")
        class MyPlanner(BaseRetrievalPlanner):
            async def run(self, generator, question, system):
                ...

    Reference in YAML config as ``retrieval_strategy.type: my_strategy``.
    """

    def wrapper(cls: type["BaseRetrievalPlanner"]) -> type["BaseRetrievalPlanner"]:
        _PLANNER_REGISTRY[key] = cls
        return cls

    return wrapper


def _get_planner_class(key: str) -> type["BaseRetrievalPlanner"]:
    """Look up a registered planner class by its config key."""
    if key not in _PLANNER_REGISTRY:
        available = ", ".join(sorted(_PLANNER_REGISTRY)) or "(none)"
        raise ValueError(
            f"Unknown retrieval strategy '{key}'. Registered types: {available}. "
            "Register a custom planner with @register_planner."
        )
    return _PLANNER_REGISTRY[key]


class BaseRetrievalPlanner:
    """Interface for retrieval-planning behaviours."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = dict(config or {})

    async def run(
        self,
        generator: "GeneratorAgent",
        question: str,
        system: "System",
    ) -> List[RetrievalCall]:
        raise NotImplementedError


@register_planner("naive")
class NaiveRetrievalPlanner(BaseRetrievalPlanner):
    """Always queries all connected routers and retrievers once."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.router_top_k = self.config.get("router_top_k", 1)

    async def run(
        self,
        generator: "GeneratorAgent",
        question: str,
        system: "System",
    ) -> List[RetrievalCall]:
        tasks: List[Any] = []

        connected_routers = generator._connected_routers()
        connected_retrievers = generator._connected_retrievers()

        for router in connected_routers:
            tasks.append(
                generator._invoke_router(
                    router,
                    question,
                    system,
                    top_k=self.router_top_k,
                )
            )
        for retriever in connected_retrievers:
            tasks.append(
                generator._invoke_retriever(
                    retriever,
                    question,
                    router_id=None,
                )
            )

        if not tasks:
            return []

        results = await asyncio.gather(*tasks)
        retrieval_calls: List[RetrievalCall] = []
        for result in results:
            if isinstance(result, list):
                retrieval_calls.extend(result)
            elif isinstance(result, RetrievalCall):
                retrieval_calls.append(result)
        return retrieval_calls


@register_planner("agentic")
class AgenticRetrievalPlanner(BaseRetrievalPlanner):
    """Agentic strategy that conditionally performs multi-round retrieval."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.max_rounds = max(1, int(self.config.get("max_rounds", 2)))
        self.router_top_k = optional_int(self.config.get("router_top_k"))
        self.max_retrieval_calls = optional_int(self.config.get("max_retrieval_calls"))
        self.target_num_documents = optional_int(
            self.config.get("target_num_documents")
        )

    async def run(
        self,
        generator: "GeneratorAgent",
        question: str,
        system: "System",
    ) -> List[RetrievalCall]:
        if not await self._should_retrieve(generator, question):
            return []

        routers = generator._connected_routers()
        direct_retrievers = generator._connected_retrievers()

        # Combine routers and direct retrievers into a single list
        combined_nodes: List[Union[RouterAgent, RetrieverAgent]] = []
        combined_nodes.extend(routers)
        combined_nodes.extend(direct_retrievers)

        # Shuffle the combined list
        random.shuffle(combined_nodes)

        retrieval_calls: List[RetrievalCall] = []

        for round_index in range(self.max_rounds):
            if (
                self.max_retrieval_calls is not None
                and len(retrieval_calls) >= self.max_retrieval_calls
            ):
                break

            round_calls: List[RetrievalCall] = []

            # Loop through combined list and call appropriate function based on node type
            for node in combined_nodes:
                if (
                    self.max_retrieval_calls is not None
                    and len(retrieval_calls) + len(round_calls)
                    >= self.max_retrieval_calls
                ):
                    break

                if isinstance(node, RouterAgent):
                    # Handle router
                    router_calls = await generator._invoke_router(
                        node,
                        question,
                        system,
                        top_k=self.router_top_k,
                    )
                    round_calls.extend(router_calls)
                elif isinstance(node, RetrieverAgent):
                    # Handle direct retriever
                    call = await generator._invoke_retriever(
                        node,
                        question,
                        router_id=None,
                    )
                    round_calls.append(call)

            if not round_calls:
                break

            retrieval_calls.extend(round_calls)
            if not self._should_continue(round_index, retrieval_calls):
                break

        return retrieval_calls

    async def _should_retrieve(
        self, generator: "GeneratorAgent", question: str
    ) -> bool:
        """
        Determine if the question should trigger retrieval.
        """
        # Practitioners can override this method to implement custom retrieval logic.
        answer = await generator._llm_client.generate(
            prompt=f"Determine if the following question should trigger retrieval for accurate answer generation. Answer with only YES or NO.\nQuestion: {question}",
            system_prompt="You are a helpful assistant that determines if a question should trigger retrieval for more accurate and up-to-date answer generation.",
        )
        return answer.lower() == "yes"

    def _should_continue(
        self, round_index: int, retrieval_calls: List[RetrievalCall]
    ) -> bool:
        """
        Determine if the generator should continue retrieving.
        """
        # Practitioners can override this method to implement custom retrieval logic.
        if round_index + 1 >= self.max_rounds:
            return False
        if (
            self.max_retrieval_calls is not None
            and len(retrieval_calls) >= self.max_retrieval_calls
        ):
            return False
        if self.target_num_documents is not None:
            document_count = sum(len(call.documents) for call in retrieval_calls)
            if document_count >= self.target_num_documents:
                return False
        return True


class GeneratorAgent(Agent):
    def __init__(
        self,
        node_id: str,
        *,
        cost: float = 0.0,
        model_id: str | None = None,
        name: str | None = None,
        prompt_template: str | None = None,
        no_context_prompt_template: str | None = None,
        system_prompt: str | None = None,
        generation_parameters: Dict[str, Any] | None = None,
        model: Dict[str, Any] | None = None,
        retrieval_strategy: Dict[str, Any] | None = None,
    ):
        super().__init__(node_id, cost=cost, model_id=model_id, name=name)
        self.prompt_template = (
            prompt_template or "Answer to '{question}' using context: {context}"
        )
        self.no_context_prompt_template = (
            no_context_prompt_template or "Answer to '{question}'"
        )
        self.system_prompt = system_prompt
        self.generation_parameters = dict(generation_parameters or {})
        model_config = model or {}
        if model_config and not isinstance(model_config, dict):
            raise ValueError("'model' parameter must be a mapping when provided")
        if not model_config and model_id:
            model_config = {"provider": "huggingface", "model_id": model_id}
        self._llm_client: BaseLLMClient | None = None
        if model_config:
            model_level_params = model_config.get("generation_parameters")
            if isinstance(model_level_params, dict):
                self.generation_parameters = {
                    **model_level_params,
                    **self.generation_parameters,
                }
            try:
                self._llm_client = create_llm_client(model_config)
            except LLMClientError as exc:
                raise ValueError(
                    f"Failed to initialise LLM client for generator '{node_id}': {exc}"
                ) from exc

        strategy_config = dict(retrieval_strategy or {})
        strategy_type = strategy_config.get("type", "naive").lower()
        planner_cls = _get_planner_class(strategy_type)
        self._retrieval_planner: BaseRetrievalPlanner = planner_cls(strategy_config)

        self.documents_per_retriever = optional_int(
            strategy_config.get("documents_per_retriever", 3)
        )
        self.max_context_documents = optional_int(
            strategy_config.get("max_context_documents")
        )
        self.context_separator = str(strategy_config.get("context_separator", " \n"))

    async def generate_query(self, user_query: UserQuery) -> str:
        return user_query.raw_question

    async def invoke(
        self, user_query: UserQuery, system: "System", **kwargs
    ) -> GenerationResult:
        question = await self.generate_query(user_query)
        retrieval_calls = await self._retrieval_planner.run(self, question, system)

        context_snippets: List[str] = []
        for call in retrieval_calls:
            documents = call.documents
            if self.documents_per_retriever is not None:
                documents = documents[: self.documents_per_retriever]
            context_snippets.extend(documents)
            if (
                self.max_context_documents is not None
                and len(context_snippets) >= self.max_context_documents
            ):
                context_snippets = context_snippets[: self.max_context_documents]
                break
        context_text = self.context_separator.join(context_snippets).strip()

        if context_snippets:
            prompt = self.prompt_template.format(
                question=question,
                context=context_text,
            )
        else:
            prompt = self.no_context_prompt_template.format(question=question)

        if self._llm_client is not None:
            try:
                answer = await self._llm_client.generate(
                    prompt,
                    system_prompt=self.system_prompt,
                    **self.generation_parameters,
                )
            except LLMClientError as exc:
                raise RuntimeError(
                    f"Generator '{self.node_id}' failed to complete LLM request: {exc}"
                ) from exc
        else:
            answer = prompt

        await self._emit_output(
            {
                "user_id": user_query.profile.user_id,
                "generator_id": self.node_id,
                "question": question,
                "answer": answer,
            }
        )

        return GenerationResult(
            user_id=user_query.profile.user_id,
            generator_id=self.node_id,
            question=question,
            answer=answer,
            retrievals=retrieval_calls,
            metadata={
                "raw_question": user_query.raw_question,
            },
        )

    async def _invoke_router(
        self,
        router: "RouterAgent",
        question: str,
        system: "System",
        *,
        top_k: int | None = 1,
    ) -> List[RetrievalCall]:
        """Invoke the router and downstream retrievers."""

        retriever_ids = await router.invoke(
            query=question,
            generator_id=self.node_id,
            top_k=top_k,
            system=system,
        )
        calls: List[RetrievalCall] = []
        for retriever_id in retriever_ids:
            retriever_agent = system.agents.get(retriever_id)
            if not isinstance(retriever_agent, RetrieverAgent):
                raise ValueError(f"Node '{retriever_id}' is not a retriever agent")
            result = await retriever_agent.invoke(
                query=question, generator_id=self.node_id
            )
            calls.append(
                RetrievalCall(
                    retriever_id=retriever_agent.node_id,
                    documents=result["documents"],
                    scores=result["scores"],
                    router_id=router.node_id,
                )
            )
        return calls

    async def _invoke_retriever(
        self, retriever: "RetrieverAgent", question: str, router_id: str | None
    ) -> RetrievalCall:
        """Invoke a retriever agent to fetch documents for the question."""

        result = await retriever.invoke(query=question, generator_id=self.node_id)
        return RetrievalCall(
            retriever_id=retriever.node_id,
            documents=result["documents"],
            scores=result["scores"],
            router_id=router_id,
        )

    async def _emit_output(self, payload: dict):
        for links in self.output_links.values():
            for link in links:
                if isinstance(link.target_node, OutputNode):
                    await link.transmit(payload)

    def _connected_routers(self) -> List[RouterAgent]:
        routers: List[RouterAgent] = []
        seen: Set[str] = set()
        for links in self.output_links.values():
            for link in links:
                target = link.target_node
                if isinstance(target, RouterAgent) and target.node_id not in seen:
                    routers.append(target)
                    seen.add(target.node_id)
        return routers

    def _connected_retrievers(self) -> List[RetrieverAgent]:
        retrievers: List[RetrieverAgent] = []
        seen: Set[str] = set()
        for links in self.output_links.values():
            for link in links:
                target = link.target_node
                if isinstance(target, RetrieverAgent) and target.node_id not in seen:
                    retrievers.append(target)
                    seen.add(target.node_id)
        return retrievers
