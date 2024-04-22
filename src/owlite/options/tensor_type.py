from typing import Literal

import numpy as np

from .options_mixin import OptionsMixin

DType = Literal[
    "float16",
    "float32",
    "float64",
    "uint8",
    "int8",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "uint64",
    "int64",
    "complex64",
    "complex128",
    "bool",
]


class TensorType(OptionsMixin):
    """The properties of a `torch.Tensor` object required for specifying its MLIR-style type."""

    shape: tuple[int, ...]
    dtype: DType
    is_constant: bool

    @property
    def dim(self) -> int:
        """The number of dimensions."""
        return len(self.shape)

    @property
    def numpy_dtype(self) -> np.dtype:
        """The numpy dtype corresponding to this `dtype` of this object."""
        return np.dtype(self.dtype)

    def __repr__(self) -> str:
        header = "Constant" if self.is_constant else "Variable"
        return f"{header}(shape={self.shape}, dtype={self.dtype})"
