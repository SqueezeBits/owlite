from typing import Optional

import torch
from torch.fx.node import Node

from owlite_core.logger import log

from ..utils import nodestr


# pylint: disable-next=too-few-public-methods
class Edge:
    """Representation of the edge connecting two adjacent nodes"""

    def __init__(self, parent_node: Node, child_node: Node) -> None:
        self.parent_node: Optional[Node] = None
        self.child_node: Optional[Node] = None
        graph_module = parent_node.graph.owning_module
        if graph_module is None:
            log.warning(
                f"({nodestr(parent_node)} does not belong to any graph module",
                stacklevel=1,
            )
            return
        if graph_module is not child_node.graph.owning_module:
            log.warning(
                f"({nodestr(parent_node)} and {nodestr(child_node)} belong to different graph modules",
                stacklevel=1,
            )
            return
        if parent_node not in child_node.all_input_nodes:
            log.warning(
                f"{nodestr(parent_node)} is not a parent node of {nodestr(child_node)}",
                stacklevel=1,
            )
            return
        self.parent_node = parent_node
        self.child_node = child_node

    def insert(self, name: str, module: Optional[torch.nn.Module]) -> bool:
        """Creates and inserts a call-module node, named `name`, whose target is `module` on this edge

        Args:
            name (str): the name of the new node to create and insert
            module (Optional[torch.nn.Module]): the module to be targeted by the new node

        Returns:
            bool: True if the node was created and inserted successfully. False otherwise.
        """
        if module is None or self.parent_node is None or self.child_node is None:
            return False
        graph = self.parent_node.graph
        graph_module = graph.owning_module
        if graph_module is None:
            return False
        if not graph_module.add_submodule(name, module):
            log.warning(
                f"Failed to add submodule {module} with name {name}",
                stacklevel=1,
            )
            return False
        with graph.inserting_before(self.child_node):
            new_node = graph.call_module(name, (self.parent_node,))
            self.child_node.replace_input_with(self.parent_node, new_node)
        graph.lint()
        self.parent_node = None
        self.child_node = None
        return True
