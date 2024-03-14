from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import torch
from torch.fx.node import Argument, Node, Target

from ...options.tensor_type import TensorType
from ..utils import camel_to_snake, nodestr
from .types import Op


@dataclass
class Edge(ABC):
    """Representation of an edge, the address of an input, of an FX node"""

    root: Node
    key: str
    tensor: Optional[TensorType] = field(default=None)

    def insert(self, op: Op, target: Target, *args: Argument, **kwargs: Argument) -> None:
        """Inserts a new FX node along this edge

        Args:
            op (Op): the op for the new FX node to be created
                (See [torch.fx.Node](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) for more details.)
            target (Target): the target for the new FX node to be created
                (See [torch.fx.Node](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) for more details.)

        Raises:
            TypeError: when the `op` and `target` are incompatible with each other
            ValueError: when `op` is invalid
        """
        graph = self.root.graph
        for existing_node in graph.nodes:
            if existing_node.op == op and existing_node.target == target:
                node = existing_node
                break
        else:
            with graph.inserting_before(self.root):
                match op:
                    case "call_function":
                        if not callable(target):
                            raise TypeError(
                                f"Expected a callable target to create a call_function node, but got {target}"
                            )
                        node = graph.call_function(target, (self.parent, *args), kwargs)
                    case "call_method":
                        if not isinstance(target, str):
                            raise TypeError(f"Expected a string target to create a call_method node, but got {target}")
                        node = graph.call_method(target, (self.parent, *args), kwargs)
                    case "call_module":
                        if not isinstance(target, str):
                            raise TypeError(f"Expected a string target to create a call_module node, but got {target}")
                        node = graph.call_module(target, (self.parent, *args), kwargs)
                    case "get_attr":
                        if not isinstance(target, str):
                            raise TypeError(f"Expected a string target to create a get_attr node, but got {target}")
                        node = graph.get_attr(target)
                    case _:
                        raise ValueError(f"Invalid FX node op: {op}")
        self.update(node)
        graph.lint()

    @abstractmethod
    def update(self, node: Node) -> None:
        """Updates the input source of the root node (`self.root`) to the `node`

        Args:
            node (Node): the new input node to replace the old input pointed by this edge
        """

    @property
    @abstractmethod
    def arg(self) -> Argument:
        """The argument pointed by this edge"""

    @cached_property
    def parent(self) -> Node:
        """The parent node of pointed by this edge. This is exactly same as `self.arg` if it is already a `Node` object.
        Otherwise it is the new node created by `op="call_function", target=torch.tensor, args=(self.arg,)`
        with appropriate device and dtype.
        """
        graph = self.root.graph
        graph_module = graph.owning_module
        if graph_module is None:
            raise RuntimeError(f"The owning module for {nodestr(self.root)} is not found")

        if isinstance(self.arg, Node):
            return self.arg

        device = graph_module.meta["canary_device_node"]
        with graph.inserting_before(self.root):
            return graph.call_function(torch.tensor, (self.arg,), {"dtype": torch.float32, "device": device})

    def __repr__(self) -> str:
        return f"{self.root.name}.{camel_to_snake(type(self).__name__)}['{self.key}']: {self.tensor}"


class EdgeWithIntegralKey(Edge, ABC):
    """The edge whose key is supposed be an integer index"""

    @property
    def index(self) -> int:
        """The key (forcefully) casted into integer index"""
        return int(self.key)

    def __repr__(self) -> str:
        return super().__repr__().replace(f"['{self.key}']", f"[{self.index}]")


class AllInputNodes(EdgeWithIntegralKey):
    """An edge accessed by `node.all_input_nodes`"""

    @property
    def arg(self) -> Node:
        return self.root.all_input_nodes[self.index]

    def update(self, node: Node) -> None:
        self.root.replace_input_with(self.parent, node)


class Args(EdgeWithIntegralKey):
    """An edge accessed by `node.args`"""

    @property
    def arg(self) -> Argument:
        return self.root.args[self.index]

    def update(self, node: Node) -> None:
        self.root.update_arg(self.index, node)


class Kwargs(Edge):
    """An edge accessed by `node.kwargs`"""

    @property
    def arg(self) -> Argument:
        return self.root.kwargs.get(self.key)

    def update(self, node: Node) -> None:
        self.root.update_kwarg(self.key, node)
