from __future__ import annotations

from abc import abstractmethod
from typing import Any

from marketplace_eval.core.node import Node


class Agent(Node):
    def __init__(
        self,
        node_id: str,
        *,
        cost: float = 0.0,
        model_id: str | None = None,
        name: str | None = None,
    ):
        super().__init__(node_id)
        self.cost = cost
        self.model_id = model_id
        self.name = name or node_id
        self.model_state: dict[str, Any] = {}

    def update(self, data: dict[str, Any]):
        """Update the internal state of the agent."""

        self.model_state.update(data)

    @abstractmethod
    async def invoke(self, *args, **kwargs):
        """Method to be implemented by subclasses to define agent behavior."""

        raise NotImplementedError
