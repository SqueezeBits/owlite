from collections.abc import Callable
from enum import IntEnum

from .target_dtype import TargetDType


# pylint: disable=invalid-name
class QATBackwardType(IntEnum):
    """The enum for specifying available QAT backward functions."""

    ste = 0
    clq = 1

    def function(self, target_dtype: TargetDType = TargetDType.int8) -> Callable:
        """Return the `torch.autograd.Function` class corresponding to this enum value with the specified target dtype.

        The supported `target_dtype` values are:
            - `TargetDType.int8` and `TargetDType.uint8` for integer quantization
            - `TargetDType.fp8_e4m3` for floating-point quantization

        The returned `torch.autograd.Function` class depends on the `target_dtype` and the `QATBackwardType`. For
        example, if `target_dtype` is `TargetDType.int8` and `QATBackwardType` is "clq", the method returns the
        `clq_function`.

        Raises:
            ValueError: If the `target_dtype` is not supported for the given `QATBackwardType`.

        Args:
            target_dtype (TargetDType, optional): The target data type for quantization. Defaults to `TargetDType.int8`.

        Returns:
            Callable: The `torch.autograd.Function` class for fake quantization.
        """
        # pylint: disable-next=import-outside-toplevel
        from ..nn.functions import clq_function, fake_fp_quantize_ste_function, fake_quantize_ste_function

        if target_dtype in (TargetDType.int8, TargetDType.uint8):
            return {
                "clq": clq_function,
                "ste": fake_quantize_ste_function,
            }[self.name]
        if target_dtype in (TargetDType.fp8_e4m3,):
            return {"ste": fake_fp_quantize_ste_function}[self.name]

        raise ValueError(f"Invalid QATBackwardType({self.name}) for target dtype({target_dtype})")
