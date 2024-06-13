from enum import IntEnum

import torch


# pylint: disable=invalid-name
class TargetDType(IntEnum):
    """The enum for specifying available target data type."""

    fp16 = 0
    int8 = 1
    uint8 = 2
    fp8_e4m3 = 3

    @property
    def unsigned(self) -> bool:
        """Return True if the data type is unsigned, False otherwise.

        Returns:
            bool: True if the data type is unsigned, False otherwise.
        """
        return self.name == "uint8"

    @property
    def precision(self) -> int:
        """Return the precision of the data type.

        Returns:
            int: The precision of the data type in bits.
        """
        if self.name in ("int8", "uint8", "fp8_e4m3"):
            return 8
        return 16

    @property
    def torch_dtype(self) -> torch.dtype:
        """Returns the corresponding PyTorch data type.

        This property maps the target data type to its corresponding PyTorch data type.

        Returns:
            torch.dtype: The corresponding PyTorch data type.
        """
        torch_dtypes: dict[str, torch.dtype] = {
            "fp16": torch.float32,
            "int8": torch.int8,
            "uint8": torch.uint8,
            "fp8_e4m3": torch.float8_e4m3fn,
        }
        return torch_dtypes[self.name]

    @classmethod
    def invert_signedness(cls, target_dtype: "TargetDType") -> "TargetDType | None":
        """Return the TargetDType with inverted signedness, if possible.

        This method inverts the signedness of the target data type, i.e., int8 becomes uint8 and vice versa.
        It returns None for fp16 and fp8_e4m3, as they only have a a corresponding unsigned type.

        Args:
            target_dtype (TargetDType): The target data type.

        Returns:
            TargetDType | None: The TargetDType with inverted signedness, or None if it's not possible.
        """
        if target_dtype == TargetDType.int8:
            return TargetDType.uint8
        if target_dtype == TargetDType.uint8:
            return TargetDType.int8
        return None
