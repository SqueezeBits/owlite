from dataclasses import dataclass

from .options_dict import OptionsDict
from .options_mixin import OptionsMixin


@dataclass
class DynamicAxisOption(OptionsMixin):
    """Dynamic axis setting for a model input tensor."""

    axis: int


class DynamicAxisOptions(OptionsDict[str, DynamicAxisOption]):
    """
    Key (str): the name of an input tensor
    Value (DynamicAxisOptions): the dynamic axis option for the input tensor
    """


# pylint: disable=redefined-builtin
@dataclass
class DynamicSizeOptions(OptionsMixin):
    """Dynamic axis setting for TensorRT benchmark"""

    min: int
    """The minimum size along the dynamic axis"""
    max: int
    """The maximum size along the dynamic axis"""
    opt: int
    """The size along the dynamic axis for optimizing latency"""
    test: int
    """The size along the dynamic axis for running benchmark"""

    def check_min(self, min: int) -> bool:
        """min must be positive integer"""
        return 0 < min

    def check_max(self, max: int) -> bool:
        """max must be greater or equal to min"""
        return self.min <= max

    def check_opt(self, opt: int) -> bool:
        """opt must be in between min and max(inclusive)"""
        return self.min <= opt <= self.max

    def check_test(self, test: int) -> bool:
        """test must be in between min and max(inclusive)"""
        return self.min <= test <= self.max


class DynamicInputOptions(OptionsDict[str, DynamicSizeOptions]):
    """
    Key (str): the name of an input tensor
    Value (DynamicSizeOptions): the dynamic size options for the input tensor when TensorRT executes
    """
