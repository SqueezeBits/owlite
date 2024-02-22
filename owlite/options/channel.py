from dataclasses import dataclass

from .options_mixin import OptionsMixin


@dataclass
class Channel(OptionsMixin):
    """The channel axis and size of a tensor"""

    axis: int
    size: int
