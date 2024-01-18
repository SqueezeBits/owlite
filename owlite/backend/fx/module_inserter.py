from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch.fx.node import Argument, Node

from owlite_core.logger import log

from ...enums import ModuleInsertionPoint
from ..utils import nodestr
from .edge import Edge


@dataclass
class ModuleInserter:
    """Inserts module using the predefined insertion key"""

    point: ModuleInsertionPoint
    index_or_key: Union[int, str]
    argument: Argument
    input_shape: Optional[tuple[int, ...]] = None

    @classmethod
    def create(cls, node: Node, insertion_key: str) -> Optional["ModuleInserter"]:
        """Creates a `ModuleInserter` object if the `insertion_key` is recognizable.

        Returns:
            Optional[ModuleInserter]: a `ModuleInserter` object if the `insertion_key` is recognizable, None otherwise.
        """
        invalid_key_message = f"Unrecognized key '{insertion_key}' found in field {node} in the config."
        try:
            input_index = int(insertion_key)
            if input_index < len(node.all_input_nodes):
                input_node = node.all_input_nodes[input_index]
                input_shape: Optional[tuple[int, ...]] = None
                if input_node.op == "get_attr" and isinstance(input_node.target, str):
                    input_shape = getattr(input_node.graph.owning_module, input_node.target).shape
                return ModuleInserter(ModuleInsertionPoint.INPUT_NODES, input_index, input_shape)

            log.warning(
                f"{invalid_key_message} {nodestr(node)} has {len(node.all_input_nodes)}"
                f"input nodes but got index {input_index}"
            )
            return None
        except ValueError:
            tokens = insertion_key.split(".")
            if len(tokens) != 2:
                log.debug_warning(
                    f"{invalid_key_message} You must implement a subclass of `NodeConfigurator` for handling this key"
                )
                return None

            container, index_or_key = tokens
            if container == "args":
                try:
                    index = int(index_or_key)
                    if index < len(node.args):
                        return ModuleInserter(ModuleInsertionPoint.ARGS, index, node.args[index])
                    log.warning(
                        f"{invalid_key_message} The node {nodestr(node)} has "
                        f"{len(node.args)} arguments but given index {index}"
                    )
                except ValueError:
                    log.warning(f"{invalid_key_message} Expected an integer value after 'args.' but got {index_or_key}")
            elif container == "kwargs":
                key = index_or_key
                if key in node.kwargs:
                    return ModuleInserter(ModuleInsertionPoint.KWARGS, key, node.kwargs[key])
                log.warning(f"{invalid_key_message} No such keyword argument {key} found in {nodestr(node)}")
            else:
                log.debug_warning(
                    f"{invalid_key_message} You must implement a subclass of `NodeConfigurator` for handling this key"
                )
            return None

    @property
    def index(self) -> int:
        """Forcefully casts `self.index_or_key` to `int`"""
        return int(self.index_or_key)

    @property
    def key(self) -> str:
        """Forcefully casts `self.index_or_key` to `str`"""
        return str(self.index_or_key)

    @property
    def insertion_key(self) -> str:
        """The insertion key that this object is originated from"""
        match self.point:
            case ModuleInsertionPoint.INPUT_NODES:
                return f"{self.index_or_key}"
            case ModuleInsertionPoint.ARGS:
                return f"args.{self.index_or_key}"
            case ModuleInsertionPoint.KWARGS:
                return f"kwargs.{self.index_or_key}"

    def insert(self, node: Node, module: Optional[torch.nn.Module]) -> bool:
        """Inserts `module` at the designated position with respect to `node`

        Args:
            node (Node): a node
            module (Optional[torch.nn.Module]): a module to insert

        Returns:
            bool: `True` if the module was successfully inserted, `False` otherwise.
        """
        if module is None:
            return True

        graph = node.graph
        graph_module = graph.owning_module
        if graph_module is None:
            return False

        def create_call_module_node(name: str, module: torch.nn.Module, argument: Argument) -> Optional[Node]:
            if not graph_module.add_submodule(name, module):
                log.warning(f"Failed to add the following submodule by name '{name}': {module}")
                return None
            device = graph_module.meta["canary_device_node"]
            with graph.inserting_before(node):
                input_tensor_node = graph.call_function(
                    torch.tensor, (argument,), {"dtype": torch.float32, "device": device}
                )
                return graph.call_module(name, (input_tensor_node,))

        name = f"{node.name}_{self.insertion_key}_quantizer"

        match self.point:
            case ModuleInsertionPoint.INPUT_NODES:
                input_node = node.all_input_nodes[self.index]
                return Edge(input_node, node).insert(name, module)
            case ModuleInsertionPoint.ARGS:
                new_node = create_call_module_node(name, module, self.argument)
                node.update_arg(self.index, new_node)
                graph.lint()
                return True
            case ModuleInsertionPoint.KWARGS:
                new_node = create_call_module_node(name, module, self.argument)
                node.update_kwarg(self.key, new_node)
                graph.lint()
                return True
