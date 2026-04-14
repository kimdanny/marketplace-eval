from __future__ import annotations

import random
from typing import Dict, List, TYPE_CHECKING

from marketplace_eval.agents.agent import Agent
from marketplace_eval.agents.retriever_agent import RetrieverAgent

if TYPE_CHECKING:
    from marketplace_eval.system.system import System


_ROUTER_REGISTRY: dict[str, type["RouterAgent"]] = {}


def register_router(key: str):
    """Class decorator that registers a RouterAgent subclass under *key*.

    Usage::

        @register_router("my_strategy")
        class MyRouterAgent(RouterAgent):
            ...

    Reference in YAML config as ``router_strategy: my_strategy``.
    """

    def wrapper(cls: type["RouterAgent"]) -> type["RouterAgent"]:
        _ROUTER_REGISTRY[key] = cls
        return cls

    return wrapper


def _get_router_class(key: str) -> type["RouterAgent"]:
    """Look up a registered router class by its config key."""
    if key not in _ROUTER_REGISTRY:
        available = ", ".join(sorted(_ROUTER_REGISTRY)) or "(none)"
        raise ValueError(
            f"Unknown router strategy '{key}'. Registered types: {available}. "
            "Register a custom router with @register_router."
        )
    return _ROUTER_REGISTRY[key]


def create_router(
    node_id: str, *, router_strategy: str = "epsilon_greedy", **kwargs
) -> "RouterAgent":
    """Factory: instantiate the right RouterAgent subclass from config."""
    cls = _get_router_class(router_strategy)
    return cls(node_id, **kwargs)


@register_router("epsilon_greedy")
class RouterAgent(Agent):
    def __init__(
        self,
        node_id: str,
        *,
        cost: float = 0.0,
        model_id: str | None = None,
        name: str | None = None,
        exploration_prob: float = 0.1,
        rng_seed: int | None = None,
    ):
        super().__init__(node_id, cost=cost, model_id=model_id, name=name)
        self.retriever_scores: Dict[str, float] = {}
        self.retriever_counts: Dict[str, int] = {}
        self.exploration_prob = exploration_prob
        self.rng = random.Random(rng_seed)

    async def invoke(
        self,
        *,
        query: str,
        generator_id: str,
        top_k: int | None = 1,
        system: "System" | None = None,
    ) -> List[str]:
        retrievers = self._available_retrievers()
        if not retrievers:
            return []

        effective_top_k: int
        if top_k is None or top_k <= 0:
            effective_top_k = len(retrievers)
        else:
            effective_top_k = min(top_k, len(retrievers))

        if self.rng.random() < self.exploration_prob:
            self.rng.shuffle(retrievers)
            selected = retrievers[:effective_top_k]
        else:
            ranked = sorted(
                retrievers,
                key=lambda rid: self.retriever_scores.get(rid, 0.0)
                / (self.retriever_counts.get(rid, 1)),
                reverse=True,
            )
            selected = ranked[:effective_top_k]

        for retriever_id in selected:
            self.retriever_counts[retriever_id] = (
                self.retriever_counts.get(retriever_id, 0) + 1
            )
        return selected

    def update_feedback(self, retriever_id: str, score: float):
        self.retriever_scores[retriever_id] = (
            self.retriever_scores.get(retriever_id, 0.0) + score
        )

    def _available_retrievers(self) -> List[str]:
        retrievers: set[str] = set()
        for links in self.output_links.values():
            for link in links:
                if isinstance(link.target_node, RetrieverAgent):
                    retrievers.add(link.target_node.node_id)
        return list(retrievers)
