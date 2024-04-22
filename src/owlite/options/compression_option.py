# ruff: noqa: D205
from collections.abc import Generator
from typing import ClassVar

from packaging.version import Version
from pydantic import Field
from typing_extensions import Self

from ..owlite_core.constants import FX_CONFIGURATION_FORMAT_VERSION
from ..owlite_core.logger import log
from .channel import Channel
from .fake_quantizer_options import FakeQuantizerOptions
from .options_dict import OptionsDict
from .options_mixin import OptionsMixin
from .tensor_type import TensorType


class EdgeCompressionOptions(OptionsMixin):
    """The properties required to compress input nodes (or constants) of an FX node."""

    fake_quantizer_id: str | None = Field(default=None)
    tensor: TensorType | None = Field(default=None)

    @property
    def fake_quantizer_name(self) -> str | None:
        """The automatically generated name of the fake quantizer module."""
        if self.fake_quantizer_id is None:
            return None
        return fake_quantizer_id_to_name(self.fake_quantizer_id)


class EdgeCompressionConfig(OptionsDict[str, EdgeCompressionOptions]):
    """The collection of edge compression options of a node.

    * Key (str): the index or key (e.g. "0", "1", "weight") for the input container
        (e.g. `node.args` or `node.kwargs`) of an FX node
    * Value (EdgeCompressionOptions): the corresponding input compression option.
    """


class NodeCompressionOptions(OptionsMixin):
    """The properties required to compress an FX node."""

    all_input_nodes: EdgeCompressionConfig = Field(default_factory=EdgeCompressionConfig)
    args: EdgeCompressionConfig = Field(default_factory=EdgeCompressionConfig)
    kwargs: EdgeCompressionConfig = Field(default_factory=EdgeCompressionConfig)
    custom: EdgeCompressionConfig = Field(default_factory=EdgeCompressionConfig)
    simulate_int32_bias: bool = Field(default=False, exclude=FX_CONFIGURATION_FORMAT_VERSION < Version("1.2"))


class NodeCompressionConfig(OptionsDict[str, NodeCompressionOptions]):
    """The collection of node compression options of a graph module.

    * Key (str): the name of an FX node
    * Value (NodeCompressionOptions): the node compression option for the FX node.
    """


class FakeQuantizerLayout(OptionsMixin):
    """The full information required for initializing a fake quantizer."""

    option: FakeQuantizerOptions
    channel: Channel | None = Field(default=None)

    @classmethod
    def create(cls, option: FakeQuantizerOptions, channel: Channel | None = None) -> Self:
        """Create a valid `FakeQuantizerLayout` object, depending on the value of `option.per_channel`.

        Note that if `option.per_channel` is `False`, the value provided for `channel` will be ignored.

        Args:
            option (FakeQuantizerOptions): fake quantizer options
            channel (Channel | None, optional): channel to be used only if `option.per_channel` is `True`.
                Defaults to None.

        Returns:
            Self: the valid `FakeQuantizerLayout` object
        """
        if option.per_channel:
            if channel is None:
                log.debug_warning(f"per_channel=True but channel is not found: {option}")
            return cls(option=option, channel=channel)
        return cls(option=option)


class FakeQuantizerConfig(OptionsDict[str, FakeQuantizerLayout]):
    """The collection of all fake quantizer layouts for a graph module.

    * Key (str): the ID of a fake quantizer
    * Value (FakeQuantizerOptions): the options for the fake quantizer.
    """

    def named_items(self) -> Generator[tuple[str, str, FakeQuantizerLayout], None, None]:
        """Similar to `self.items()` but yields triples (`id`: str, `name`: str, layout: `FakeQuantizerLayout`)."""
        for fake_quantizer_id, fake_quantizer_layout in self.items():
            yield fake_quantizer_id, fake_quantizer_id_to_name(fake_quantizer_id), fake_quantizer_layout


class CompressionOptions(OptionsMixin):
    """The properties required to compress a graph module."""

    __version__: ClassVar[Version] = FX_CONFIGURATION_FORMAT_VERSION
    node_compression_config: NodeCompressionConfig = Field(default_factory=NodeCompressionConfig)
    fake_quantizers: FakeQuantizerConfig = Field(default_factory=FakeQuantizerConfig)


def fake_quantizer_id_to_name(fake_quantizer_id: str) -> str:
    """Automatically generates the name of a fake quantizer module from its ID."""
    return f"fake_quantizer_{fake_quantizer_id}"
