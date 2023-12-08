"""The enumeration of available QAT backward pass implementations"""
from enum import Enum
from typing import Callable

from ..nn.functions import clq_function, clq_plus_function, ste_function


# pylint: disable=invalid-name
class QATBackwardType(Enum):
    """The enum for specifying available QAT backward functions"""

    ste = 0
    clq = 1
    clq_plus = 2

    @property
    def function(self) -> Callable:
        """The apply method of the `torch.autograd.Function` class corresponding to this enum value"""
        return {
            "clq": clq_function,
            "clq_plus": clq_plus_function,
            "ste": ste_function,
        }[self.name]

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name
