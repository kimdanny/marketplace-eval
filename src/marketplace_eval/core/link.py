from __future__ import annotations

from typing import Any
import inspect

from marketplace_eval.core.node import Node


class Link:
    """Directed connection between two nodes in the simulation graph."""

    def __init__(
        self,
        source_node: Node,
        target_node: Node,
        metadata: dict[str, Any] | None = None,
    ):
        self.source_node = source_node
        self.target_node = target_node
        self.metadata = metadata or {}
        self.data: Any = None  # Placeholder for data that the link may carry

    async def transmit(self, data: Any, **kwargs):
        """Send data along the link to the target node."""

        self.data = data
        response = self.target_node.invoke(
            data, source=self.source_node, link=self, **kwargs
        )
        if inspect.isawaitable(response):
            return await response
        return response

    def receive(self) -> Any:
        """Return the last piece of data transmitted on the link."""

        return self.data
