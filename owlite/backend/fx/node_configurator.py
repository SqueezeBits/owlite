from itertools import product
from typing import Callable, Optional

import torch
from torch.fx.node import Node

from owlite_core.logger import log

from ...nn import FakeQuantizer, QLinear
from ...nn.modules import UnaryNeuralQModuleMixin, promote_to_qmodule
from ...options.fake_quantizer_options import FakeQuantizerOptions
from ...options.quantization_options import NodeQuantizationOptions
from ..utils import nodestr
from .edge import Edge
from .module_inserter import ModuleInserter
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
        options: NodeQuantizationOptions,
    ) -> None:
        """Configures the `node` with the `options` if there is any registered subclass of `NodeConfigurator`
        eligible for configuring it.

        Args:
            node (Node): an FX node to apply options
            options (NodeQuantizationOptions): a `NodeQuantizationOptions` instance containing the information on how
            the node is to be modified by one of subclasses of `NodeConfigurator`.
        """
        if cls is not NodeConfigurator:
            raise TypeError("configure method must be called by the `NodeConfigurator` class itself")
        for configurator_class in NodeConfigurator.registry:
            configurator = configurator_class(node, options)
            if configurator.is_target_matched(node):
                log.debug(
                    f"{configurator_class.__name__} (targets={configurator_class.torch_targets}) "
                    f"-> {nodestr(node)} (target={get_torch_target(node)})"
                )
                configurator.apply()
                return
        # If the torch target of the node does not require special logics provided by one of registered subclasses,
        # just use the base class `NodeConfigurator`'s `apply` method.
        NodeConfigurator(node, options).apply()

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

    def __init__(self, node: Node, options: NodeQuantizationOptions) -> None:
        self.node = node
        self.options = options

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

        for insertion_key, fake_quantizer_options in sorted(self.options.items(), key=lambda item: item[0]):
            module_inserter = ModuleInserter.create(self.node, insertion_key)
            if module_inserter is None:
                continue
            channel_size = None
            if fake_quantizer_options is not None and fake_quantizer_options.per_channel:
                if module_inserter.input_shape is not None and len(module_inserter.input_shape) > 0:
                    channel_size = module_inserter.input_shape[0]
                else:
                    log.warning(
                        f"Cannot per-channel quantize {self.node} along {module_inserter.insertion_key}, "
                        "as the channel size is unknown. Will fallback to per-tensor quantization for this node",
                        stacklevel=2,
                    )
                    log.debug_warning(f"{nodestr(self.node, True)}")
                    fake_quantizer_options.per_channel = False
            module_inserter.insert(self.node, FakeQuantizer.create(fake_quantizer_options, channel_size=channel_size))


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
        graph_module = self.node.graph.owning_module
        module = get_target_module(self.node)
        if graph_module is None or module is None:
            return
        quantized_class = promote_to_qmodule(type(module))
        if quantized_class is None:
            return

        input_options: Optional[FakeQuantizerOptions] = self.options.get("0")
        weight_options: Optional[FakeQuantizerOptions] = self.options.get("weight")

        if weight_options is not None:
            module = quantized_class(module, weight_options)  # type: ignore[arg-type]
            if not graph_module.add_submodule(self.node.target, module):
                log.warning(f"Failed to quantize the weight of {nodestr(self.node)}")

        input_quantizer = get_target_module(self.node.all_input_nodes[0])
        if isinstance(input_quantizer, FakeQuantizer):
            if input_quantizer.per_channel.item():
                input_quantizer.channel_size = (
                    module.in_channels if (input_options is not None and input_options.per_channel) else None
                )
            if isinstance(module, UnaryNeuralQModuleMixin):
                module.input_quantizer = input_quantizer


@NodeConfigurator.register(torch.nn.Linear)
class CallModuleLinearNodeConfigurator(CallModuleConvNodeConfigurator):
    """Further configures intra-nodal fake quantizers for a call-module node with one input node whose module is an
    instance of torch.nn.Linear
    """

    def apply(self) -> None:
        super().apply()
        module = get_target_module(self.node)
        bias_options = self.options.get("bias", None)
        hidden_input_options = self.options.get("hidden_input", None)
        if not isinstance(module, QLinear):
            if bias_options is not None:
                log.warning(
                    f"The quantization options of {nodestr(self.node)} for bias "
                    f"will be ignored as its weight is not quantized: {bias_options}"
                )
            if hidden_input_options is not None:
                log.warning(
                    f"The quantization options of {nodestr(self.node)} for bias "
                    f"will be ignored as its weight is not quantized: {hidden_input_options}"
                )
            return
        module.set_bias_quantizer(
            bias_options,
            hidden_input_options,
        )


@NodeConfigurator.register(torch.nn.functional.linear)
class CallFunctionLinearNodeConfigurator(NodeConfigurator):
    """Further configures intra-nodal fake quantizers for a call-function node whose target is
    torch.nn.functional.linear
    """

    def apply(self) -> None:
        super().apply()
        if len(self.node.all_input_nodes) < 3:
            return
        bias_node = self.node.all_input_nodes[2]
        hidden_input_quantization_options: Optional[FakeQuantizerOptions] = self.options.get("hidden_input", None)
        if hidden_input_quantization_options is None:
            return

        if hidden_input_quantization_options.per_channel:
            log.warning(
                "Per channel hidden activation quantization for torch.nn.functional.linear "
                f"is not supported yet. Will set per_channel=False for the node {nodestr(self.node)}"
            )
            hidden_input_quantization_options.per_channel = False

        if len(self.node.args) > 2:
            self.node.update_arg(2, None)
        else:
            self.node.update_kwarg("bias", None)

        with self.node.graph.inserting_after(self.node):
            bias_add = self.node.graph.call_function(the_function=torch.add, args=(bias_node, self.node))

        self.node.replace_all_uses_with(bias_add)
        bias_add.update_arg(1, self.node)

        _ = Edge(self.node, bias_add).insert(
            f"{self.node.name}_hidden_input_quantizer",
            FakeQuantizer.create(hidden_input_quantization_options),
        )
