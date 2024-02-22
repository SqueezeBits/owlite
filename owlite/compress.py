from torch.fx.graph_module import GraphModule
from torch.fx.node import Node

from owlite_core.logger import log

from .backend.fx.node import find_constant_nodes
from .backend.fx.node_configurator import NodeConfigurator
from .backend.fx.transforms import fuse_linear_bn_with_quantized_bias
from .enums import OwLiteStatus
from .nn import FakeQuantizer, enable_quantizers
from .options.compression_option import CompressionOptions, FakeQuantizerConfig


def compress(model: GraphModule, option: CompressionOptions) -> GraphModule:
    """Quantizes the model with the specification described in options.

    This function inserts quantizers with the quantization options specified in the options,
    substitutes them with the Quantized module, and performs post-processing. The linear module
    that quantizes the bias cannot fuse the batch norm after quantizing, so it proceeds to fuse
    the batch norm. Then, it fuses quantizers with the same quantization option that correspond
    to the same tensor in the original model.

    Args:
        model (GraphModule): The symbolic traced model to be compressed.
        option (CompressionOptions): The option required for compressing the model.

    Raises:
        TypeError: If model is not a instance of `GraphModule`.

    Returns:
        GraphModule: Compressed model.
    """

    if not isinstance(model, GraphModule):
        raise TypeError("Only GraphModule instance can be quantized with `owlite.quantize`")
    configure(model, option)
    fuse_linear_bn_with_quantized_bias(model)
    enable_quantizers(model)
    return model


def configure(graph_module: GraphModule, option: CompressionOptions) -> None:
    """Configures the input model to a quantized model based on the provided options.

    Args:
        graph_module (GraphModule): The model to be compressed.
        option (CompressionOptions): The option required for compressing the model.
    """
    constants_key = "constants"
    if constants_key not in graph_module.meta:
        graph_module.meta[constants_key] = find_constant_nodes(graph_module.graph)
    add_fake_quantizers(graph_module, option.fake_quantizers)
    nodes: list[Node] = [*graph_module.graph.nodes]
    for node in nodes:
        if node_compression_option := option.node_compression_config.get(node.name):
            NodeConfigurator.configure(node, node_compression_option)
    graph_module.graph.lint()
    graph_module.graph.eliminate_dead_code()
    graph_module.recompile()
    graph_module.meta["owlite_status"] = OwLiteStatus.COMPRESSED
    try:
        graph_module.to(next(graph_module.parameters()).device)
    except StopIteration:
        pass


def add_fake_quantizers(graph_module: GraphModule, fake_quantizer_config: FakeQuantizerConfig) -> None:
    """Adds necessary fake quantizer submodules to the graph module according to the fake quantizer config.

    Args:
        graph_module (GraphModule): the graph module where new fake quantizer submodules are to be added
        fake_quantizer_config (FakeQuantizerConfig): the configurations for the fake quantizer submodules
            to be added
    """
    for fake_quantizer_id, target, fake_quantizer_layout in fake_quantizer_config.named_items():
        fake_quantizer = FakeQuantizer.create(
            fake_quantizer_layout.option, fake_quantizer_layout.channel, identification=fake_quantizer_id
        )
        if fake_quantizer is None:
            log.debug_warning(f"Found vacuous layout: {fake_quantizer_layout}")
            continue
        if not graph_module.add_submodule(target, fake_quantizer):
            log.warning(f"Failed to add FakeQuantizer module: {target}")
