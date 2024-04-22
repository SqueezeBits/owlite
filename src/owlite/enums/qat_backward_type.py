from collections.abc import Callable
from enum import IntEnum


# pylint: disable=invalid-name
class QATBackwardType(IntEnum):
    """The enum for specifying available QAT backward functions."""

    ste = 0
    clq = 1

    @property
    def function(self) -> Callable:
        """The apply method of the `torch.autograd.Function` class corresponding to this enum value."""
        # pylint: disable-next=import-outside-toplevel
        from ..nn.functions import clq_function, fake_quantize_ste_function

        return {
            "clq": clq_function,
            "ste": fake_quantize_ste_function,
        }[self.name]
