""" Model quantize """


from torch.fx.graph_module import GraphModule

from .backend.fx.node import find_constant_nodes
from .backend.fx.node_configurator import NodeConfigurator
from .backend.fx.transforms import (
    fuse_linear_bn_with_quantized_bias,
    fuse_redundant_quantizers,
)
from .enums import OwLiteStatus
from .logger import log
from .nn.fake_quantizer import enable_quantizers
from .options import GraphQuantizationOptions


def quantize(model: GraphModule, options: GraphQuantizationOptions) -> GraphModule:
    """Quantizes the model with the specification described in options.

    This function inserts quantizers with the quantization options specified in the options,
    substitutes them with the Quantized module, and performs post-processing. The linear module
    that quantizes the bias cannot fuse the batch norm after quantizing, so it proceeds to fuse
    the batch norm. Then, it fuses quantizers with the same quantization option that correspond
    to the same tensor in the original model.

    Args:
        model (GraphModule): The symbolic traced model to be quantized.
        options (GraphQuantizationOptions): Options specifying the quantization.

    Raises:
        TypeError: If model is not a instance of `GraphModule`.

    Returns:
        GraphModule: Quantized model.
    """

    if not isinstance(model, GraphModule):
        raise TypeError("Only GraphModule instance can be quantized with `owlite.quantize`")
    configure(model, options)
    fuse_linear_bn_with_quantized_bias(model)
    log.debug("Fusing the redundant quantizers.")
    fuse_redundant_quantizers(model)
    enable_quantizers(model, True)
    return model


def configure(graph_module: GraphModule, options: GraphQuantizationOptions) -> None:
    """Configures the input model to a quantized model based on the provided options.

    Args:
        graph_module(GraphModule): The model to be quantized.
        options(GraphQuantizationOptions): Options specifying the quantization.
    """
    constants_key = "constants"
    if constants_key not in graph_module.meta:
        graph_module.meta[constants_key] = find_constant_nodes(graph_module.graph)
    nodes = [*graph_module.graph.nodes]
    for node in nodes:
        node_option = options.get(node.name, None)
        if node_option is None:
            continue
        NodeConfigurator.configure(node, node_option)
        graph_module.graph.lint()
    graph_module.graph.eliminate_dead_code()
    graph_module.recompile()
    graph_module.meta["owlite_status"] = OwLiteStatus.COMPRESSED
    try:
        graph_module.to(next(graph_module.parameters()).device)
    except StopIteration:
        pass
