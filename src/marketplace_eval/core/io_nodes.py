from __future__ import annotations

from typing import Any, List

from marketplace_eval.core.node import Node


class InputNode(Node):
    """Node representing an entry point for questions into the simulation graph."""

    def __init__(self, node_id: str):
        super().__init__(node_id)

    async def invoke(self, payload: Any, **kwargs) -> List[Any]:
        responses: List[Any] = []
        for links in self.output_links.values():
            for link in links:
                responses.append(await link.transmit(payload, **kwargs))
        return responses


class OutputNode(Node):
    """Node collecting results produced by the graph."""

    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.received_payloads: list[Any] = []

    async def invoke(self, payload: Any, **kwargs):
        self.received_payloads.append(payload)
        return payload
