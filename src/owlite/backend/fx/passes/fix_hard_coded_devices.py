import torch
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassBase, PassResult

from ...utils import get_most_common_device


class FixHardCodedDevice(PassBase):
    """Fix hard coded devices to enable data parallel."""

    def call(self, graph_module: GraphModule) -> PassResult:
        """Fix hard coded devices to enable data parallel.

        Args:
            graph_module (GraphModule): the input graph module

        Returns:
            PassResult: the result of the pass
        """
        if hasattr(graph_module, "sqzb_module_device_canary"):
            return PassResult(graph_module, False)

        canary_tensor = torch.tensor(12.03, dtype=torch.float32, device=get_most_common_device(graph_module))
        graph_module.register_buffer("sqzb_module_device_canary", canary_tensor)

        graph = graph_module.graph
        with graph.inserting_before(next(iter(graph_module.graph.nodes))):
            canary = graph.get_attr("sqzb_module_device_canary")
            canary_device = graph.call_function(getattr, (canary, "device"))
            graph_module.meta["canary_device_node"] = canary_device

        for node in graph.nodes:
            if node.kwargs.get("device", None) is not None:
                kwargs = node.kwargs.copy()
                kwargs["device"] = canary_device
                node.kwargs = kwargs

            if (
                node.op == "call_method"
                and node.target == "to"
                and len(node.args) == 2
                and isinstance(node.args[1], str)
            ):
                args = (node.args[0], canary_device)
                node.args = args

        return PassResult(graph_module, True)
