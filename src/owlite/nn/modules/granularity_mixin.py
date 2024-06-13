from abc import ABC, abstractmethod

import torch

from ...options.channel import Channel


class GranularityMixin(ABC):
    """Abstract base class for mixin classes that represent granularity of FakeQuantization.

    This class provides a common interface for per-channel and per-tensor granularity.
    """

    @abstractmethod
    def init_quantization_param(self, channel: Channel | None, zero_point_dtype: torch.dtype = torch.int32) -> None:
        """Initialize the quantization parameters with the specified granularity.

        Args:
            channel (Channel | None): The channel to initialize. If `None`, use per-tensor quantization.
                Otherwise, use per-channel quantization.
            zero_point_dtype (torch.dtype) : The data type of zero point.  Defaults to `torch.int32`.
        """

    @property
    @abstractmethod
    def channel(self) -> Channel | None:
        """Get the channel associated with the granularity.

        Returns:
            Channel | None: The channel associated with the granularity, or `None` if per-tensor quantization is used.
        """

    @property
    def per_channel(self) -> bool:
        """Check if per-channel quantization is used.

        Equivalent to `self.channel is not None`.

        Returns:
            bool: `True` if per-channel quantization is used, `False` otherwise.
        """
        return self.channel is not None


class PerChannelMixin(GranularityMixin):
    """Mixin class for per-channel granularity of FakeQuantization.

    This class provides the implementation for per-channel granularity.
    """

    def init_quantization_param(self, channel: Channel | None, zero_point_dtype: torch.dtype = torch.int32) -> None:
        assert isinstance(self, torch.nn.Module)
        assert channel is not None
        self.register_buffer("_channel_axis", torch.tensor([channel.axis], dtype=torch.int32))
        self.register_buffer("_channel_size", torch.tensor([channel.size], dtype=torch.int32))
        self.step_size = torch.nn.Parameter(torch.ones(channel.size))
        self.zero_point = torch.nn.Parameter(
            torch.zeros(
                channel.size,
                dtype=zero_point_dtype,
            ),
            requires_grad=False,
        )

    @abstractmethod
    def as_per_tensor(self) -> "PerTensorMixin":
        """Convert the per-channel granularity to per-tensor granularity."""

    @property
    def channel(self) -> Channel:
        return Channel(axis=self._channel_axis.item(), size=self._channel_size.item())  # type: ignore


class PerTensorMixin(GranularityMixin):
    """Mixin class for per-tensor granularity of FakeQuantization.

    This class provides the implementation for per-tensor granularity, where all channels share
    the same quantization parameters.
    """

    def init_quantization_param(
        self, channel: Channel | None = None, zero_point_dtype: torch.dtype = torch.int32
    ) -> None:
        assert channel is None
        self.step_size = torch.nn.Parameter(torch.ones(1))
        self.zero_point = torch.nn.Parameter(
            torch.zeros(
                1,
                dtype=zero_point_dtype,
            ),
            requires_grad=False,
        )

    @abstractmethod
    def as_per_channel(self, channel: Channel) -> "PerChannelMixin":
        """Create a new fake per-channel quantizer with the same option (except for the `per_channel` value).

        The `step_size` and `zero_point` of the new fake per-channel quantizer is initialized with shape
        `(channel.size,)` filled with values in `self.step_size` and `self.zero_point`, respectively.
        """

    @property
    def channel(self) -> None:
        return None
