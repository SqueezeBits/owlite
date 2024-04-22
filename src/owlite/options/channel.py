from .options_mixin import OptionsMixin


class Channel(OptionsMixin):
    """The channel axis and size of a tensor."""

    axis: int
    size: int
