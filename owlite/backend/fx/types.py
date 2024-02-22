from typing import Literal, Union

import torch
from torch.fx.graph_module import GraphModule
from torch.fx.node import Target
from torch.nn.parallel import DataParallel, DistributedDataParallel

TorchTarget = Union[Target, type[torch.nn.Module]]
Numeric = Union[float, int]
GraphModuleOrDataParallel = Union[GraphModule, DataParallel, DistributedDataParallel]
Op = Union[
    Literal["call_function"],
    Literal["call_method"],
    Literal["call_module"],
    Literal["get_attr"],
]
