from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class Node(ABC):
    """Base class for all nodes in the simulation graph."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        # a dictionary of node_id to list of links
        self.input_links: Dict[str, List["Link"]] = {}
        self.output_links: Dict[str, List["Link"]] = {}

    def add_input_link(self, link: "Link"):
        """Register an incoming link for the node."""

        self.input_links.setdefault(link.source_node.node_id, []).append(link)

    def add_output_link(self, link: "Link"):
        """Register an outgoing link for the node."""

        self.output_links.setdefault(link.target_node.node_id, []).append(link)

    @abstractmethod
    async def invoke(self, *args, **kwargs):
        """Method to be implemented by subclasses to define node behavior."""

        raise NotImplementedError
