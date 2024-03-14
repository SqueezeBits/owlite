from collections.abc import Generator
from dataclasses import dataclass, field
from typing import ClassVar, Optional

from typing_extensions import Self

from ..owlite_core.logger import log
from .channel import Channel
from .fake_quantizer_options import FakeQuantizerOptions
from .options_dict import OptionsDict
from .options_mixin import OptionsMixin
from .tensor_type import TensorType


@dataclass
class EdgeCompressionOptions(OptionsMixin):
    """The properties required to compress input nodes (or constants) of an FX node"""

    fake_quantizer_id: Optional[str] = field(default=None)
    tensor: Optional[TensorType] = field(default=None)

    @property
    def fake_quantizer_name(self) -> Optional[str]:
        """The automatically generated name of the fake quantizer module"""
        if self.fake_quantizer_id is None:
            return None
        return fake_quantizer_id_to_name(self.fake_quantizer_id)


class EdgeCompressionConfig(OptionsDict[str, EdgeCompressionOptions]):
    """
    * Key (str): the index or key (e.g. "0", "1", "weight") for the input container
        (e.g. `node.args` or `node.kwargs`) of an FX node
    * Value (EdgeCompressionOptions): the corresponding input compression option
    """


@dataclass
class NodeCompressionOptions(OptionsMixin):
    """The properties required to compress an FX node"""

    all_input_nodes: EdgeCompressionConfig = field(default_factory=EdgeCompressionConfig)
    args: EdgeCompressionConfig = field(default_factory=EdgeCompressionConfig)
    kwargs: EdgeCompressionConfig = field(default_factory=EdgeCompressionConfig)
    custom: EdgeCompressionConfig = field(default_factory=EdgeCompressionConfig)


class NodeCompressionConfig(OptionsDict[str, NodeCompressionOptions]):
    """
    * Key (str): the name of an FX node
    * Value (NodeCompressionOptions): the node compression option for the FX node
    """


@dataclass
class FakeQuantizerLayout(OptionsMixin):
    """The full information required for initializing a fake quantizer"""

    option: FakeQuantizerOptions
    channel: Optional[Channel] = field(default=None)

    @classmethod
    def create(cls, option: FakeQuantizerOptions, channel: Optional[Channel] = None) -> Self:
        """Creates a valid `FakeQuantizerLayout` object, depending on the value of `option.per_channel`.
        Note that if `option.per_channel` is `False`, the value provided for `channel` will be ignored.

        Args:
            option (FakeQuantizerOptions): fake quantizer options
            channel (Optional[Channel], optional): channel to be used only if `option.per_channel` is `True`.
                Defaults to None.

        Returns:
            Self: the valid `FakeQuantizerLayout` object
        """
        if option.per_channel:
            if channel is None:
                log.debug_warning(f"per_channel=True but channel is not found: {option}")
            return cls(option, channel)
        return cls(option)


class FakeQuantizerConfig(OptionsDict[str, FakeQuantizerLayout]):
    """
    * Key (str): the ID of a fake quantizer
    * Value (FakeQuantizerOptions): the options for the fake quantizer
    """

    def named_items(self) -> Generator[tuple[str, str, FakeQuantizerLayout], None, None]:
        """Similar to `self.items()` but yields triples (`id`: str, `name`: str, layout: `FakeQuantizerLayout`)"""
        for fake_quantizer_id, fake_quantizer_layout in self.items():
            yield fake_quantizer_id, fake_quantizer_id_to_name(fake_quantizer_id), fake_quantizer_layout


@dataclass
class CompressionOptions(OptionsMixin):
    """The properties required to compress a graph module"""

    __version__: ClassVar[str] = "1.1"
    node_compression_config: NodeCompressionConfig = field(default_factory=NodeCompressionConfig)
    fake_quantizers: FakeQuantizerConfig = field(default_factory=FakeQuantizerConfig)


def fake_quantizer_id_to_name(fake_quantizer_id: str) -> str:
    """Automatically generates the name of a fake quantizer module from its ID"""
    return f"fake_quantizer_{fake_quantizer_id}"
