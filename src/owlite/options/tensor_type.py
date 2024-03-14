from dataclasses import dataclass

from ..enums import DType
from .options_mixin import OptionsMixin


@dataclass
class TensorType(OptionsMixin):
    """The properties of a `torch.Tensor` object required for specifying its MLIR-style type"""

    shape: tuple[int, ...]
    dtype: DType
    is_constant: bool

    @property
    def dim(self) -> int:
        """The number of dimensions"""
        return len(self.shape)

    def __repr__(self) -> str:
        header = "Constant" if self.is_constant else "Variable"
        return f"{header}(shape={self.shape}, dtype={self.dtype.name.lower()})"
