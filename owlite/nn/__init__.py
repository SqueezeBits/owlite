"""owlite.quant modules

Basic quantization specifications and functions and qmodules that use them
"""
from .fake_quantizer import FakeQuantizer
from .modules import QConv1d, QConv2d, QConv3d, QLinear
