from enum import Enum
from typing import Callable


# pylint: disable=invalid-name
class QATBackwardType(Enum):
    """The enum for specifying available QAT backward functions"""

    ste = 0
    clq = 1
    clq_plus = 2

    @property
    def function(self) -> Callable:
        """The apply method of the `torch.autograd.Function` class corresponding to this enum value"""
        # pylint: disable-next=import-outside-toplevel
        from ..nn.functions import clq_function, clq_plus_function, ste_function

        return {
            "clq": clq_function,
            "clq_plus": clq_plus_function,
            "ste": ste_function,
        }[self.name]
