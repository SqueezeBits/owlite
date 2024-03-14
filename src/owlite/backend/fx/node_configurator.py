from itertools import product
from typing import Callable, Optional

import torch
from torch.fx.node import Node

from ...nn import FakeQuantizer, QLinear
from ...nn.modules import UnaryNeuralQModuleMixin, promote_to_qmodule
from ...options.compression_option import NodeCompressionOptions
from ...owlite_core.logger import log
from ..utils import nodestr
from .edge import AllInputNodes, Args, Kwargs
from .node import get_target_module, get_torch_target
from .types import TorchTarget


class NodeConfigurator:
    """Configures inter-nodal fake quantizers for a `torch.fx.Node` based on a `NodeQuantizationOptions`"""

    registry: set[type["NodeConfigurator"]] = set()
    torch_targets: Optional[set[TorchTarget]] = None

    class DuplicateRegistrationError(Exception):
        """Error indicating duplicate registration of a torch target to more than one NodeConfigurator subclasses"""

    @classmethod
    def register(cls, *torch_target: TorchTarget) -> Callable[..., type["NodeConfigurator"]]:
        """
        A class-method meant to be used as a decorator when declaring a subclass of `NodeConfigurator` to register it
        with the designated Torch targets. There are three types of Torch targets.
        1. a function from the either `torch` or `operator`
            e.g. torch.matmul, torch.nn.functional.linear, operator.add
        2. a native subclass of `torch.nn.Module`
            e.g. torch.nn.Conv2d, torch.nn.Linear
        3. a string representing the name of a method of the `torch.Tensor`
            e.g. "matmul", "add", "add_"
        """
        if cls is not NodeConfigurator:
            raise TypeError("register method must be called by the `NodeConfigurator` class itself")

        def registerer(subclass: type[NodeConfigurator]) -> type[NodeConfigurator]:
            if not (
                subclass is not NodeConfigurator
                and subclass not in NodeConfigurator.registry
                and issubclass(subclass, NodeConfigurator)
            ):
                raise TypeError(
                    "The class to register must be a subclass of `NodeConfigurator` that had never been registered."
                )
            for registered_class, target in product(NodeConfigurator.registry, torch_target):
                if registered_class.torch_targets is not None and target in registered_class.torch_targets:
                    raise NodeConfigurator.DuplicateRegistrationError(
                        f"Cannot register {target} for {subclass.__name__} "
                        f"as it is already registered in {registered_class}"
                    )
            subclass.torch_targets = set(torch_target)
            NodeConfigurator.registry.add(subclass)
            return subclass

        return registerer

    @classmethod
    def configure(
        cls,
        node: Node,
        option: NodeCompressionOptions,
    ) -> None:
        """Configures the `node` with the `options` if there is any registered subclass of `NodeConfigurator`
        eligible for configuring it.

        Args:
            node (Node): an FX node to apply options
            option (NodeCompressionOptions): the options required for compressing the model
        """
        if cls is not NodeConfigurator:
            raise TypeError("configure method must be called by the `NodeConfigurator` class itself")
        for configurator_class in NodeConfigurator.registry:
            configurator = configurator_class(node, option)
            if configurator.is_target_matched(node):
                log.debug(
                    f"{configurator_class.__name__} (targets={configurator_class.torch_targets}) "
                    f"-> {nodestr(node)} (target={get_torch_target(node)})"
                )
                configurator.apply()
                break
        else:
            # If the torch target of the node does not require special logics provided by one of registered subclasses,
            # just use the base class `NodeConfigurator`'s `apply` method.
            NodeConfigurator(node, option).apply()

    @classmethod
    def is_target_matched(cls, node: Node) -> bool:
        """Checks if `node.target` matches one of the torch targets associated with
        the `NodeConfigurator` subclass.
        """
        if cls.torch_targets is None:
            raise NotImplementedError(
                f"{cls.__name__}.torch_targets must be defined to be a proper subclass of NodeConfigurator. "
                "Use the `NodeConfigurator.register` method to register torch targets for the class."
            )
        return get_torch_target(node) in [*cls.torch_targets]

    def __init__(self, node: Node, option: NodeCompressionOptions) -> None:
        self.node = node
        self.option = option

    def apply(self) -> None:
        """Applies quantization to the `self.node` with `self.options`"""
        graph_module = self.node.graph.owning_module
        if graph_module is None:
            log.warning(f"{nodestr(self.node)} does not belong to a graph module.")
            return
        constants: list[Node] = graph_module.meta.get("constants", None)
        if constants is None:
            log.warning(
                f"The key 'constants' not found in the meta data of the owning graph module of {nodestr(self.node)}."
            )

        for container, edge_type in zip(
            (self.option.all_input_nodes, self.option.args, self.option.kwargs), (AllInputNodes, Args, Kwargs)
        ):
            for key, input_option in container.items():
                try:
                    if input_option.fake_quantizer_name is None:
                        continue
                    edge = edge_type(self.node, key, input_option.tensor)  # type: ignore[abstract]
                    edge.insert("call_module", input_option.fake_quantizer_name)
                    fake_quantizer = graph_module.get_submodule(input_option.fake_quantizer_name)
                    if not isinstance(fake_quantizer, FakeQuantizer):
                        continue
                except (IndexError, ValueError, KeyError, RuntimeError) as e:
                    log.error(
                        f"Failed to insert fake quantizer ({e})\nNote:\n"
                        f"node={nodestr(self.node)}\n"
                        f"key={key}\n"
                        f"input_option={input_option}"
                    )
                    continue


@NodeConfigurator.register(
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
)
class CallModuleConvNodeConfigurator(NodeConfigurator):
    """Further configures intra-nodal fake quantizers for a call-module node with one input node whose module is an
    instance of torch.nn.Conv*d
    """

    def apply(self) -> None:
        super().apply()
        if (
            (graph_module := self.node.graph.owning_module) is None
            or (module := get_target_module(self.node)) is None
            or (quantized_class := promote_to_qmodule(type(module))) is None
            or not isinstance(self.node.target, str)
        ):
            return

        if (weight_option := self.option.custom.get("weight")) is not None:
            module = quantized_class(module)  # type: ignore[arg-type]
            if (
                weight_option.fake_quantizer_name is not None
                and isinstance(
                    (weight_quantizer := graph_module.get_submodule(weight_option.fake_quantizer_name)), FakeQuantizer
                )
                and isinstance(module, UnaryNeuralQModuleMixin)
            ):
                module.weight_quantizer = weight_quantizer
                graph_module.add_submodule(self.node.target, module)

        if isinstance(
            (input_quantizer := get_target_module(self.node.all_input_nodes[0])), FakeQuantizer
        ) and isinstance(module, UnaryNeuralQModuleMixin):
            module.input_quantizer = input_quantizer


@NodeConfigurator.register(torch.nn.Linear)
class CallModuleLinearNodeConfigurator(CallModuleConvNodeConfigurator):
    """Further configures intra-nodal fake quantizers for a call-module node with one input node whose module is an
    instance of torch.nn.Linear
    """

    def apply(self) -> None:
        super().apply()
        if (graph_module := self.node.graph.owning_module) is None:
            return
        module = get_target_module(self.node)
        bias_option = self.option.custom.get("bias", None)
        hidden_input_option = self.option.custom.get("hidden_input", None)
        if not isinstance(module, QLinear):
            if bias_option is not None:
                log.warning(
                    f"The quantization options of {nodestr(self.node)} for bias "
                    f"will be ignored as its weight is not quantized: {bias_option}"
                )
            if hidden_input_option is not None:
                log.warning(
                    f"The quantization options of {nodestr(self.node)} for bias "
                    f"will be ignored as its weight is not quantized: {hidden_input_option}"
                )
            return

        if (
            bias_option is not None
            and bias_option.fake_quantizer_name is not None
            and isinstance(
                (bias_quantizer := graph_module.get_submodule(bias_option.fake_quantizer_name)),
                FakeQuantizer,
            )
        ):
            module.bias_quantizer = bias_quantizer

        if (
            hidden_input_option is not None
            and hidden_input_option.fake_quantizer_name is not None
            and isinstance(
                (hidden_input_quantizer := graph_module.get_submodule(hidden_input_option.fake_quantizer_name)),
                FakeQuantizer,
            )
        ):
            module.hidden_input_quantizer = hidden_input_quantizer


@NodeConfigurator.register(torch.nn.functional.linear)
class CallFunctionLinearNodeConfigurator(NodeConfigurator):
    """Further configures intra-nodal fake quantizers for a call-function node whose target is
    torch.nn.functional.linear
    """

    def apply(self) -> None:
        super().apply()
        if (graph_module := self.node.graph.owning_module) is None:
            return

        hidden_input_option = self.option.custom.get("hidden_input")
        if hidden_input_option is None or hidden_input_option.fake_quantizer_name is None:
            return

        hidden_input_quantizer = graph_module.get_submodule(hidden_input_option.fake_quantizer_name)
        if len(self.node.all_input_nodes) < 3:
            return

        bias_node = self.node.all_input_nodes[2]

        if len(self.node.args) > 2:
            self.node.update_arg(2, None)
        else:
            self.node.update_kwarg("bias", None)

        activation_node = self.node
        with self.node.graph.inserting_after(self.node):
            if isinstance(hidden_input_quantizer, FakeQuantizer) and hidden_input_option is not None:
                activation_node = self.node.graph.call_module(hidden_input_option.fake_quantizer_name)
            bias_add = self.node.graph.call_function(the_function=torch.add, args=(bias_node, activation_node))

        self.node.replace_all_uses_with(bias_add)
        bias_add.update_arg(1, self.node)
