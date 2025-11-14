from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class EliminateDummyOutput(PassBase):
    """Eliminate dummy output node inserted by `FixHardCodedDevices`."""

    def call(self, graph_module: GraphModule) -> PassResult:
        """Eliminate dummy output node inserted by `FixHardCodedDevices`.

        Args:
            graph_module (GraphModule): the input graph module

        Returns:
            PassResult: the result of the pass
        """
        nodes: list[Node] = [*graph_module.graph.nodes]
        modified = False
        for node in nodes:
            if not (node.op == "output" and graph_module.meta["canary_device_node"] in node.all_input_nodes):
                continue

            node.args = tuple(filter(lambda x: x != graph_module.meta["canary_device_node"], node.args))
            modified = True
            break

        return PassResult(graph_module, modified)
