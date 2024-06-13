# pylint: disable=invalid-name, too-many-public-methods
# ruff: noqa: N801
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeGuard

import torch
from typing_extensions import Self

from ...calib import PercentileCalibrator
from ...enums import PTQCalibrationType, QATBackwardType, TargetDType
from ...nn.functions import FakeQuantizeSignature, clq_function
from ...options.channel import Channel
from ...options.fake_quantizer_options import FakeQuantizerOptions, map_precision_and_unsigned_to_dtype
from ...owlite_core.logger import log
from .granularity_mixin import PerChannelMixin, PerTensorMixin

if TYPE_CHECKING:
    from ...calib.calibrator import Calibrator


class parameter(property):
    """The special property hosted by private parameter."""

    def __init__(self, fget: Callable):
        name = fget.__name__

        def getter(instance: Any) -> Any:
            attr = getattr(instance, f"_{name}")
            return None if attr is None else attr.item()

        def setter(instance: Any, value: Any) -> None:
            if value is None:
                setattr(instance, f"_{name}", None)
                return
            p = getattr(instance, f"_{name}")
            if p is None:
                setattr(instance, f"_{name}", torch.nn.Parameter(torch.full((), value), requires_grad=False))
                return
            new_param = torch.nn.Parameter(torch.full_like(p, value), requires_grad=p.requires_grad)
            setattr(instance, f"_{name}", new_param)

        super().__init__(getter, setter)


# pylint: disable=too-many-instance-attributes
class FakeQuantizer(torch.nn.Module, ABC):
    """An implementation of fake quantization (a.k.a. quantization simulation).

    Attributes:
        step_size: The step size for the fake quantization. The larger the step size, the shorter the length of
            quantization intervals, and vice versa.
        zero_point: The zero point for the fake quantization. When simulating symmetric quantization
            (i.e. `self.symmetric` is `True`), this parameter will be fixed to 0 and remain unchanged.
    """

    def __init__(
        self,
        options: FakeQuantizerOptions,
        *,
        enable: bool = True,
        identification: str | None = None,
    ):
        """Initialize a `FakeQuantizer` instance.

        Args:
            options (FakeQuantizerOptions): Options for the fake quantizer.
            enable (bool, optional): Whether to enable the fake quantizer. Defaults to `True`.
            identification (str | None, optional): unique ID for the fake quantizer. Defaults to `None`.

        Raises:
            ValueError: if `options.ptq_calibration` is "percentile" but `options.percentile` is `None`.
        """
        super().__init__()
        self.target_dtype = options.dtype
        self.precision = self.target_dtype.precision
        self._symmetric = torch.nn.Parameter(torch.tensor(options.symmetric), requires_grad=False)
        self._learn_zero_point = torch.nn.Parameter(torch.tensor(options.learn_zero_point), requires_grad=False)
        self._grad_scale = torch.nn.Parameter(torch.tensor(options.grad_scale), requires_grad=False)
        self.id: str | None = identification
        self.is_enabled = enable
        self.qat_backward_type = options.qat_backward
        self.ptq_calibration = options.ptq_calibration
        calibrator_class = options.ptq_calibration.calibrator_class
        if options.ptq_calibration == PTQCalibrationType.percentile:
            if options.percentile is None:
                raise ValueError("percentile value is required for percentile PTQ calibrator")
            self.calibrator: Calibrator = PercentileCalibrator(self, options.percentile)
        else:
            self.calibrator = calibrator_class(self)
        self.step_size: torch.nn.Parameter
        self.zero_point: torch.nn.Parameter

    @classmethod
    def create(
        cls,
        options: FakeQuantizerOptions | None,
        channel: Channel | None = None,
        *,
        enable: bool = True,
        narrow_range: bool = False,
        identification: str | None = None,
    ) -> "FakePerTensorINTQuantizer|FakePerChannelINTQuantizer|FakePerChannelFPQuantizer|FakePerTensorFPQuantizer|None":
        """Create a `FakeQuantizer` instance based on the provided options.

        Args:
            options (FakeQuantizerOptions | None): Options for the fake quantizer.
            channel (Channel | None, optional): if `channel` is provided and `options.per_channel` is `True`, a fake
                per-channel quantizer will be created. Otherwise, a fake per-tensor quantizer will be created.
            enable (bool, optional): Whether to enable the fake quantizer. Defaults to `True`.
            narrow_range (bool, optional): Whether the fake quantizer should use narrow range. Defaults to `False`.
                This is only required by FakeINTQuantizer.
                e.g. quantized integer values will be clipped into range [-127,127] instead of [-128,127]
                when `options.dtype == int8` and `narrow_range` is `True`.
            identification (str | None, optional): unique ID for the fake quantizer. Defaults to `None`.

        Returns:
            FakePerTensorINTQuantizer | FakePerChannelINTQuantizer | None: If the `options` is valid for
                quantization returns created fake quantizer, `None` otherwise.
        """
        if options is None or options.dtype == TargetDType.fp16:
            return None
        name = "fake quantizer (id=unknown)" if identification is None else f"fake quantizer (id={identification})"
        # FakeFPQuantizer
        if options.dtype == TargetDType.fp8_e4m3:
            if options.per_channel:
                if channel is not None:
                    return FakePerChannelFPQuantizer(options, channel, enable=enable, identification=identification)
                log.warning(
                    f"Cannot initialize the {name} with `per_channel=True` as its channel is unknown. "
                    "It will be initialized with per_channel=False instead"
                )
            options.per_channel = False
            return FakePerTensorFPQuantizer(options, enable=enable, identification=identification)
        # FakeINTQuantizer
        if options.per_channel:
            if channel is not None:
                return FakePerChannelINTQuantizer(
                    options, channel, enable=enable, narrow_range=narrow_range, identification=identification
                )
            log.warning(
                f"Cannot initialize the {name} with `per_channel=True` as its channel is unknown. "
                "It will be initialized with per_channel=False instead"
            )
            options.per_channel = False
        return FakePerTensorINTQuantizer(
            options, enable=enable, narrow_range=narrow_range, identification=identification
        )

    @classmethod
    def check_if_enabled(cls, fake_quantizer: Self | None) -> TypeGuard[Self]:
        """Check if the given fake quantizer (possibly `None`) is enabled.

        Args:
            fake_quantizer(FakeQuantizer | None): either a fake quantizer to verify or `None`.

        Returns:
            TypeGuard[FakeQuantizer]: `True` if `fake_quantizer` is not `None` and it is enabled, `False` otherwise.
        """
        if fake_quantizer is not None and fake_quantizer.is_enabled:
            return True
        return False

    @parameter
    def symmetric(self) -> bool:  # type: ignore[empty-body]
        """Whether to simulate symmetric quantization."""

    @property
    def unsigned(self) -> bool:
        """Whether this fake quantizer's underlying integer type is unsigned."""
        return self.target_dtype.unsigned

    @parameter
    def grad_scale(self) -> float:  # type: ignore[empty-body]
        """The gradient scale of this fake quantizer for updating step size in QAT."""  # noqa: D401

    @property
    def learn_zero_point(self) -> bool:
        """Whether to update `self.zero_point` during QAT.

        Can be set to `True` only if `self.symmetric` is `False`
        """
        if (zero_point := self.zero_point) is not None:
            return zero_point.requires_grad
        return self._learn_zero_point.item()

    @learn_zero_point.setter
    def learn_zero_point(self, value: bool) -> None:
        self._learn_zero_point = torch.nn.Parameter(
            torch.full_like(self._learn_zero_point, value),
            requires_grad=self._learn_zero_point.requires_grad,
        )
        if (zero_point := self.zero_point) is not None:
            zero_point.requires_grad = value

    @property
    def qat_function(self) -> FakeQuantizeSignature:
        """The autograd function of this fake quantizer."""
        return self.qat_backward_type.function(self.target_dtype)

    @property
    @abstractmethod
    def quant_min(self) -> int | float:
        """The minimum value this fake quantizer can handle."""

    @property
    @abstractmethod
    def quant_max(self) -> int | float:
        """The maximum value this fake quantizer can handle."""

    @property
    def maxabs_bound(self) -> int | float:
        """The maximum absolute value that this fake quantizer can handle."""
        return max(abs(self.quant_min), abs(self.quant_max))

    @property
    def option(self) -> FakeQuantizerOptions:
        """The options that current FakeQuantizer instance represents."""
        percentile = getattr(self.calibrator, "percentile", None)

        return FakeQuantizerOptions(
            dtype=self.target_dtype,
            qat_backward=self.qat_backward_type,
            ptq_calibration=self.ptq_calibration,
            percentile=percentile,
            symmetric=self.symmetric,
            per_channel=self.per_channel,
            learn_zero_point=self.learn_zero_point,
            grad_scale=self.grad_scale,
        )

    def enable(self) -> Self:
        """Enable this fake quantizer."""
        self.is_enabled = True
        return self

    def disable(self) -> Self:
        """Disable this fake quantizer."""
        self.is_enabled = False
        return self

    # pylint: disable=protected-access
    def extra_repr(self) -> str:
        string = f"dtype: {self.target_dtype}"
        string += f", channel: {self.channel}"
        string += f", quant_range: ({self.quant_min}, {self.quant_max})"
        string += f", symmetric: {self.symmetric}"
        string += f", qat_backward: {self.qat_backward_type.name}"
        string += f", calib: {self.calibrator.__class__.__name__}"
        string += f", zero_point: {self.zero_point.item()}" if not self.per_channel else ""
        string += f", is_enabled: {self.is_enabled}"
        return string

    def state_dict(  # type: ignore[no-untyped-def, override]
        self, *args, **kwargs
    ) -> OrderedDict[Any, Any] | dict[str, Any]:
        state: OrderedDict = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix", "")
        extra_state = {}
        # add data type index
        extra_state[f"{prefix}_target_dtype"] = torch.tensor([self.target_dtype.value])
        # add qat_backward index
        extra_state[f"{prefix}_qat_backward"] = torch.tensor([self.qat_backward_type.value])
        # add ptq_calibration index
        extra_state[f"{prefix}_ptq_calibration"] = torch.tensor([self.ptq_calibration.value])
        if self.ptq_calibration == PTQCalibrationType.percentile:
            if not isinstance(self.calibrator, PercentileCalibrator):
                raise TypeError(
                    "calibrator must be instance of 'PercentileCalibrator' when ptq_calibrtion is 'percentile',"
                    f"but got {self.calibrator}"
                )
            extra_state[f"{prefix}_ptq_calibration_percentile"] = torch.tensor([self.calibrator.percentile])
        state.update(extra_state)
        return state

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # for handling deprecated statedict
        if precision := state_dict.pop("{prefix}._precision", None) or (
            unsigned := state_dict.pop("{prefix}._unsigned", None)
        ):
            if not (precision and unsigned):
                raise ValueError("unsigned field and precision field should always be used together")
            log.warning("This state_dict was saved from an earlier version of OwLite. It is probably incorrect.")
            self.target_dtype = TargetDType(map_precision_and_unsigned_to_dtype(precision.item(), unsigned.item()))
        else:
            self.target_dtype = TargetDType(state_dict.pop(f"{prefix}_target_dtype").item())
        self.qat_backward_type = QATBackwardType(state_dict.pop(f"{prefix}_qat_backward").item())
        self.ptq_calibration = PTQCalibrationType(state_dict.pop(f"{prefix}_ptq_calibration").item())
        calibrator_class = self.ptq_calibration.calibrator_class
        if self.ptq_calibration == PTQCalibrationType.percentile:
            self.calibrator = PercentileCalibrator(self, state_dict.pop(f"{prefix}_ptq_calibration_percentile").item())
        else:
            self.calibrator = calibrator_class(self)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization to the input tensor.

        Args:
            inputs (torch.Tensor): A tensor to fake-quantize.

        Returns:
            torch.Tensor: the fake-quantized tensor if this fake quantizer is enabled, the unchanged input tensor
                otherwise.
        """
        if not self.is_enabled:
            return inputs

        return self.qat_function(
            inputs,
            self.step_size.data,
            self.zero_point.data,
            self.grad_scale,
            self.quant_min,
            self.quant_max,
            self.channel.axis if self.channel is not None else None,
        )


class FakeINTQuantizer(FakeQuantizer):
    """Fake integer quantizer."""

    def __init__(
        self,
        options: FakeQuantizerOptions,
        *,
        enable: bool = True,
        narrow_range: bool = False,
        identification: str | None = None,
    ):
        if options.dtype not in (TargetDType.int8, TargetDType.uint8):
            raise TypeError(f"Attempting to create a FakeQuantizer with an invalid TargetDType({options.dtype.name})")
        super().__init__(options=options, enable=enable, identification=identification)
        if narrow_range and not (options.symmetric and not self.unsigned):
            log.warning(
                "narrow_range should only be used with symmetric signed quantization.\n"
                "(narrow_range, symmetric, unsigned) = "
                f"({narrow_range}, {options.symmetric}, {self.unsigned})"
            )
        self._narrow_range = torch.nn.Parameter(torch.tensor(narrow_range), requires_grad=False)

    def invert_signedness(self) -> Self:
        """Invert signedness of this fake quantizer's underlying integer type."""
        if (inverted_dtype := TargetDType.invert_signedness(self.target_dtype)) is None:
            raise TypeError(
                f"Attempting to create a FakeQuantizer with an invalid TargetDType({self.target_dtype.name})"
            )
        self.target_dtype = inverted_dtype
        return self

    @property
    def quant_min(self) -> int:
        if self.narrow_range:
            return -(1 << (self.precision - 1)) + 1
        return 0 if self.unsigned else -(1 << (self.precision - 1))

    @property
    def quant_max(self) -> int:
        if self.narrow_range:
            return (1 << self.precision) - 1 + self.quant_min - 1
        return (1 << self.precision) - 1 + self.quant_min

    @property
    def narrow_range(self) -> bool:
        """Whether to apply narrow range clipping for this fake quantizer underlying integer type.

        e.g. quantized integer values will be clipped into range [-127,127] instead of [-128,127]
        when `self.target_dtype == "int8"` and `self.narrow_range` is `True`.
        """
        if torch.jit.is_tracing():
            return False
        return self._narrow_range.item()  # type: ignore[return-value]

    @narrow_range.setter
    def narrow_range(self, value: bool) -> None:
        if torch.jit.is_tracing():
            log.warning("Cannot set narrow_range value during tracing")
            return
        self._narrow_range = torch.nn.Parameter(
            torch.full_like(self._narrow_range, value), requires_grad=self._narrow_range.requires_grad
        )
        if value and not (self.symmetric and not self.unsigned):
            log.warning(
                "narrow_range should only be used with symmetric signed quantization.\n"
                "(narrow_range, symmetric, unsigned) = "
                f"({value}, {self.symmetric}, {self.unsigned})"
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization to the input tensor.

        Args:
            inputs (torch.Tensor): A tensor to fake-quantize.

        Raises:
            ValueError: If the fake quantizer has a negative step size or its channel size mismatches with the `inputs`.

        Returns:
            torch.Tensor: the fake-quantized tensor if this fake quantizer is enabled,
                the unchanged input tensor otherwise.
        """
        if self.qat_function is not clq_function and not self.narrow_range and self.step_size.min() <= 0:
            log.error(
                f"Expected `step_size` to be positive, but got `step_size={self.step_size.data}`. "
                "Please try one of the suggestions below:\n"
                '   * select "clq" from the "qat_backward" field in the OwLite Web UI (https://owlite.ai/project);\n'
                "   * set the weight_decay of the fake quantizer's parameters to 0;\n"
                "   * reduce the learning rate for the fake quantizer's parameters; or\n"
                "   * reduce the grad_scale of the fake quantizer"
            )
            raise ValueError("Step_size must be positive")

        return super().forward(inputs)


class FakePerChannelINTQuantizer(FakeINTQuantizer, PerChannelMixin):
    """Fake quantizer that simulates per-channel quantization."""

    def __init__(
        self,
        options: FakeQuantizerOptions,
        channel: Channel,
        *,
        enable: bool = True,
        narrow_range: bool = False,
        identification: str | None = None,
    ):
        assert options.per_channel
        super().__init__(options, enable=enable, narrow_range=narrow_range, identification=identification)
        self.init_quantization_param(channel=channel)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.is_enabled and not (
            self.channel.axis < inputs.ndim and self.channel.size == inputs.shape[self.channel.axis]
        ):
            raise RuntimeError(
                "FakeQuantizer channel size mismatched:\n"
                f"id: {self.id}\n"
                f"inputs.shape: {inputs.shape}\n"
                f"channel: {self.channel}"
            )
        return super().forward(inputs)

    def as_per_tensor(self) -> "FakePerTensorINTQuantizer":
        """Create a new fake per-tensor quantizer with the same option (except for the `per_channel` value).

        The `step_size` of the new fake per-tensor quantizer is initialized as the maximum element of `self.step_size`.
        """
        option = self.option
        option.per_channel = False
        fake_per_tensor_quantizer = FakePerTensorINTQuantizer(
            option,
            enable=self.is_enabled,
            narrow_range=self.narrow_range,
            identification=f"{self.id}_as_per_tensor",
        )
        fake_per_tensor_quantizer.step_size.data = self.step_size.data.max().reshape(
            fake_per_tensor_quantizer.step_size.shape
        )
        return fake_per_tensor_quantizer.to(self.step_size.device)


class FakePerTensorINTQuantizer(FakeINTQuantizer, PerTensorMixin):
    """Fake integer quantizer that simulates per-tensor quantization."""

    def __init__(
        self,
        options: FakeQuantizerOptions,
        *,
        enable: bool = True,
        narrow_range: bool = False,
        identification: str | None = None,
    ):
        assert not options.per_channel
        super().__init__(options, enable=enable, narrow_range=narrow_range, identification=identification)
        self.init_quantization_param()

    def as_per_channel(self, channel: Channel) -> "FakePerChannelINTQuantizer":
        """Create a new fake per-channel quantizer with the same option (except for the `per_channel` value).

        The `step_size` and `zero_point` of the new fake per-channel quantizer is initialized with shape
        `(channel.size,)` filled with values in `self.step_size` and `self.zero_point`, respectively.
        """
        option = self.option
        option.per_channel = True
        fake_per_channel_quantizer = FakePerChannelINTQuantizer(
            option,
            channel,
            enable=self.is_enabled,
            narrow_range=self.narrow_range,
            identification=f"{self.id}_as_per_channel",
        )

        fake_per_channel_quantizer.step_size.data = self.step_size.data.broadcast_to(
            fake_per_channel_quantizer.step_size.shape
        ).clone()
        fake_per_channel_quantizer.zero_point.data = self.zero_point.data.broadcast_to(
            fake_per_channel_quantizer.zero_point.shape
        ).clone()

        return fake_per_channel_quantizer.to(self.step_size.device)


class FakeFPQuantizer(FakeQuantizer):
    """Fake FP quantizer."""

    def __init__(
        self,
        options: FakeQuantizerOptions,
        *,
        enable: bool = True,
        identification: str | None = None,
    ):
        if options.dtype not in (TargetDType.fp8_e4m3,):
            raise TypeError(f"Attempting to create a FakeQuantizer with an invalid TargetDType({options.dtype.name})")
        super().__init__(options=options, enable=enable, identification=identification)

    @property
    def quant_min(self) -> float:
        """The minimum value this fake quantizer can handle."""
        return torch.finfo(self.target_dtype.torch_dtype).min

    @property
    def quant_max(self) -> float:
        """The maximum value this fake quantizer can handle."""
        return torch.finfo(self.target_dtype.torch_dtype).max

    @property
    def narrow_range(self) -> bool:
        """Whether to apply narrow range clipping for this fake quantizer.

        Do not use narrow range for the FP fake quantizer.
        """
        return False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization to the input tensor.

        Args:
            inputs (torch.Tensor): A tensor to fake-quantize.

        Raises:
            ValueError: If the fake quantizer has a negative step size.

        Returns:
            torch.Tensor: the fake-quantized tensor if this fake quantizer is enabled, the unchanged input tensor
                otherwise.
        """
        if self.step_size.min() <= 0:
            log.error(
                f"Expected `step_size` to be positive, but got `step_size={self.step_size.data}`. "
                "Please try one of the suggestions below:\n"
                "   * set the weight_decay of the fake quantizer's parameters to 0;\n"
                "   * reduce the learning rate for the fake quantizer's parameters; or\n"
                "   * reduce the grad_scale of the fake quantizer"
            )
            raise ValueError("Step_size must be positive")

        return super().forward(inputs)


class FakePerChannelFPQuantizer(FakeFPQuantizer, PerChannelMixin):
    """Fake quantizer that simulates per-channel quantization."""

    def __init__(
        self,
        options: FakeQuantizerOptions,
        channel: Channel,
        *,
        enable: bool = True,
        identification: str | None = None,
    ):
        assert options.per_channel
        super().__init__(options, enable=enable, identification=identification)
        self.init_quantization_param(channel=channel, zero_point_dtype=torch.float32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.is_enabled and not (
            self.channel.axis < inputs.ndim and self.channel.size == inputs.shape[self.channel.axis]
        ):
            raise RuntimeError(
                "FakeQuantizer channel size mismatched:\n"
                f"id: {self.id}\n"
                f"inputs.shape: {inputs.shape}\n"
                f"channel: {self.channel}"
            )
        return super().forward(inputs)

    def as_per_tensor(self) -> "FakePerTensorFPQuantizer":
        """Create a new fake per-tensor fp quantizer with the same option (except for the `per_channel` value).

        The `step_size` of the new fake per-tensor quantizer is initialized as the maximum element of `self.step_size`.
        """
        option = self.option
        option.per_channel = False
        fake_per_tensor_quantizer = FakePerTensorFPQuantizer(
            option,
            enable=self.is_enabled,
            identification=f"{self.id}_as_per_tensor",
        )
        fake_per_tensor_quantizer.step_size.data = self.step_size.data.max().reshape(
            fake_per_tensor_quantizer.step_size.shape
        )
        return fake_per_tensor_quantizer.to(self.step_size.device)


class FakePerTensorFPQuantizer(FakeFPQuantizer, PerTensorMixin):
    """Fake integer quantizer that simulates per-tensor quantization."""

    def __init__(
        self,
        options: FakeQuantizerOptions,
        *,
        enable: bool = True,
        identification: str | None = None,
    ):
        assert not options.per_channel
        super().__init__(options, enable=enable, identification=identification)
        self.init_quantization_param(zero_point_dtype=torch.float32)

    def as_per_channel(self, channel: Channel) -> "FakePerChannelFPQuantizer":
        """Create a new fake per-channel quantizer with the same option (except for the `per_channel` value).

        The `step_size` and `zero_point` of the new fake per-channel quantizer is initialized with shape
        `(channel.size,)` filled with values in `self.step_size` and `self.zero_point`, respectively.
        """
        option = self.option
        option.per_channel = True
        fake_per_channel_quantizer = FakePerChannelFPQuantizer(
            option,
            channel,
            enable=self.is_enabled,
            identification=f"{self.id}_as_per_channel",
        )

        fake_per_channel_quantizer.step_size.data = self.step_size.data.broadcast_to(
            fake_per_channel_quantizer.step_size.shape
        ).clone()
        fake_per_channel_quantizer.zero_point.data = self.zero_point.data.broadcast_to(
            fake_per_channel_quantizer.zero_point.shape
        ).clone()

        return fake_per_channel_quantizer.to(self.step_size.device)
