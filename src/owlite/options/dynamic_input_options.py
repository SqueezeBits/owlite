# ruff: noqa: D205
from dataclasses import dataclass

from .options_dict import OptionsDict
from .options_mixin import OptionsMixin


class DynamicAxisOptions(dict[str, int]):
    """Key (str): the name of an input tensor
    Value (int): the axis to be dynamic.
    """


# pylint: disable=redefined-builtin
@dataclass
class DynamicRangeOptions(OptionsMixin):
    """Dynamic axis range setting for benchmark."""

    min: int
    """The minimum size along the dynamic axis"""
    max: int
    """The maximum size along the dynamic axis"""
    opt: int
    """The size along the dynamic axis for optimizing latency"""
    test: int
    """The size along the dynamic axis for running benchmark"""

    def check_min(self, min: int) -> bool:
        """Min must be positive integer."""
        return 0 < min

    def check_max(self, max: int) -> bool:
        """Max must be greater or equal to min."""
        return self.min <= max

    def check_opt(self, opt: int) -> bool:
        """Opt must be in between min and max(inclusive)."""
        return self.min <= opt <= self.max

    def check_test(self, test: int) -> bool:
        """Test must be in between min and max(inclusive)."""
        return self.min <= test <= self.max

    # pylint: disable-next=missing-function-docstring
    def to_list(self) -> list[int]:
        return [self.min, self.opt, self.max, self.test]


class DynamicInputOptions(OptionsDict[str, DynamicRangeOptions]):
    """Key (str): the name of an input tensor
    Value (DynamicSizeOptions): the dynamic size options for the input tensor when engine executes.
    """
