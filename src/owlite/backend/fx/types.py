from typing import Literal

import torch
from torch.fx.graph_module import GraphModule
from torch.fx.node import Target
from torch.nn.parallel import DataParallel, DistributedDataParallel

TorchTarget = Target | type[torch.nn.Module]
GraphModuleOrDataParallel = GraphModule | DataParallel | DistributedDataParallel
Op = Literal["call_function", "call_method", "call_module", "get_attr"]
