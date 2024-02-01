from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from owlite_core.logger import log

from ..calib import PercentileCalibrator
from ..enums import PTQCalibrationType, QATBackwardType
from ..nn.functions import clq_function
from ..nn.functions.fake_quantize import FakeQuantFunc
from ..options.fake_quantizer_options import FakeQuantizerOptions

if TYPE_CHECKING:
    from ..calib.calibrator import Calibrator


# pylint: disable=too-many-instance-attributes
class FakeQuantizer(torch.nn.Module):
    """An implementation of fake quantization (a.k.a. quantization simulation)

    Attributes:
        step_size (torch.Tensor): The quantization scale, determining the magnitude of each quantization interval.
        zero_point (torch.Tensor): The quantization zero_point. It may be expressed as a float in the context
            of asymmetric quantization, while for symmetric quantization, it is fixed at zero tensor.
        precision (torch.IntTensor): The number of bits used for quantization.
        symmetric (torch.BoolTensor): Whether symmetric quantization is applied.
        unsigned (torch.BoolTensor): Whether unsigned quantization is applied
        per_channel (torch.BoolTensor): Whether per-channel quantization or per-tensor quantization is applied
        learn_zero_point (torch.BoolTensor):  whether the zero point is learnable.
        grad_scale (torch.FloatTensor): The gradient scaling factor of quantization parameters.
        _narrow_range (torch.BoolTensor): Whether a narrow range is used in quantization.
    """

    precision: torch.IntTensor
    symmetric: torch.BoolTensor
    unsigned: torch.BoolTensor
    per_channel: torch.BoolTensor
    learn_zero_point: torch.BoolTensor
    grad_scale: torch.FloatTensor
    _narrow_range: torch.BoolTensor

    @classmethod
    def create(
        cls,
        options: Optional[FakeQuantizerOptions],
        channel_size: Optional[int] = None,
        enable: bool = True,
        narrow_range: bool = False,
    ) -> Optional["FakeQuantizer"]:
        """Creates a `FakeQuantizer` instance if options is not `None`, otherwise returns `None`

        Args:
            options (Optional[FakeQuantizerOptions]): Options for fake quantizer to return. If `None`,
                dose notcreate fake quantizer.
            channel_size (Optional[int], optional): Channel size of per-channel quantization. Not used in
                per-tensor quantization. If `None`, no channel size is set. Defaults to `None`.
            enable (bool, optional): If true, returns the enabled quantzier. If false, returns the quantizer
                that was disabled. Defaults to `True`
            narrow_range (bool, optional): If true, returns the quantzier with a narrow range. If false, it
                does not have a narrow range. Defaults to `False`

        Returns:
            Optional[FakeQuantizer]: If the `options` is valid for quantization returns created fake quantizer.
                Otherwise return `None`.
        """
        if options is None or options.precision > 8:
            return None
        return FakeQuantizer(options, channel_size, enable, narrow_range)

    def __init__(
        self,
        options: FakeQuantizerOptions,
        channel_size: Optional[int] = None,
        enable: bool = True,
        narrow_range: bool = False,
    ):
        """Initializes a FakeQuantizer instance.

        Args:
            options (QuantizerOptions): options
            channel_size (Optional[int], optional): The channel size for per-channel quantization. Defaults to None.
                This value is required only when `options.per_channel` is `True`, otherwise has no effect.
                It can be set after the instantiation of the object, must be set before calling its `forward` method.
            enable (bool, optional): whether to enable this quantizer object as soon as it is initialized.
                Defaults to True.
            narrow_range (bool, optional): Use symmetric integer range for signed quantization
                eg) [-127,127] instead of [-128,127] for num_bits=8. Default False.

        Raises:
            ValueError: if `options.ptq_calibration` is "percentile" but `options.percentile` is `None`.
        """
        super().__init__()
        self.register_buffer("precision", torch.tensor(options.precision))
        self.register_buffer("symmetric", torch.tensor(options.symmetric))
        self.register_buffer("unsigned", torch.tensor(options.unsigned))
        self.register_buffer("per_channel", torch.tensor(options.per_channel))
        if not self.symmetric.item() and self.per_channel.item():
            raise RuntimeError("asymmetric per_channel quantization is not available")
        self.register_buffer("learn_zero_point", torch.tensor(options.learn_zero_point))
        self.register_buffer("grad_scale", torch.tensor(options.grad_scale))
        if narrow_range and not (self.symmetric.item() and not self.unsigned.item()):
            log.warning(
                "narrow_range should only be used with symmetric signed quantization.\n"
                "(narrow_range, symmetric, unsigned) = "
                f"({narrow_range}, {self.symmetric.item()}, {self.unsigned.item()})"
            )
        self.register_buffer("_narrow_range", torch.tensor(narrow_range))

        if self.per_channel:
            if channel_size is not None:
                self.channel_size = channel_size
        else:
            self.step_size: torch.nn.Parameter = torch.nn.Parameter(torch.ones(1))
            self.zero_point: torch.nn.Parameter = torch.nn.Parameter(
                torch.zeros(1, dtype=torch.int32),
                requires_grad=bool(not self.symmetric.item() and self.learn_zero_point.item()),
            )
        self._is_enabled = enable
        self.is_zero_point_folded = False
        self.qat_backward_type = options.qat_backward
        self.ptq_calibration = options.ptq_calibration
        calibrator_class = options.ptq_calibration.calibrator_class
        if options.ptq_calibration == PTQCalibrationType.percentile:
            if options.percentile is None:
                raise ValueError("percentile value is required for percentile PTQ calibrator")
            self.calibrator: Calibrator = calibrator_class(self, options.percentile)
        else:
            self.calibrator = calibrator_class(self)

    @property
    def qat_function(
        self,
    ) -> FakeQuantFunc:
        """The autograd function providing forward and backward methods of this fake quantizer
        for the quantization-aware training"""
        return self.qat_backward_type.function

    @property
    def channel_size(self) -> Optional[int]:
        """The channel size for the input tensor of this fake quantizer"""
        if not self.per_channel.item():
            return 1
        step_size = getattr(self, "step_size", None)
        zero_point = getattr(self, "zero_point", None)
        if not (
            isinstance(step_size, (torch.nn.Parameter, torch.Tensor))
            and isinstance(zero_point, (torch.nn.Parameter, torch.Tensor))
        ):
            return None
        if not (len(step_size.shape) == 1 and step_size.shape == zero_point.shape):
            log.error("step_size and zero_point have invalid shapes.")
            log.debug(f"self={self}\n" "self.step_size={step_size}\n" "self.zero_point={zero_point}\n")
            raise ValueError("step_size and zero_point have invalid shapes")
        return int(step_size.shape[0])

    @channel_size.setter
    def channel_size(self, value: Optional[int]) -> None:
        """Sets the channel size for the input tensor of this fake quantizer. Note that this property must be set at
        least (and exactly) once before calling this fake quantizer instance when `per_channel=True`
        """
        if not self.per_channel.item():
            log.warning(
                "Setting channel_size value will have no effect for per tensor weight quantization.",
                stacklevel=2,
            )
            return
        existing_channel_size = self.channel_size
        if existing_channel_size is not None:
            log.error(f"channel_size value was already set to {existing_channel_size}. It cannot be reset.")
            raise RuntimeError("channel_size cannot be reset.")
        if value is None:
            return
        self.step_size = torch.nn.Parameter(torch.ones(value))
        self.zero_point = torch.nn.Parameter(
            torch.zeros(value),
            requires_grad=bool(not self.symmetric.item() and self.learn_zero_point.item()),
        )

    @property
    def quant_min(self) -> int:
        """The minimum integer value this fake quantizer can handle"""
        if self.narrow:
            return int(-(1 << (int(self.precision.item()) - 1)) + 1)
        return 0 if self.unsigned.item() else int(-(1 << (int(self.precision.item()) - 1)))

    @property
    def quant_max(self) -> int:
        """The maximum integer value this fake quantizer can handle"""
        if self.narrow:
            return (1 << int(self.precision.item())) - 1 + self.quant_min - 1
        return (1 << int(self.precision.item())) - 1 + self.quant_min

    @property
    def narrow(self) -> bool:
        """Returns True in quantizer using narrow range and False otherwise."""
        if torch.jit.is_tracing():
            return False
        return bool(self._narrow_range.item() and not self.unsigned.item() and self.symmetric.item())

    @property
    def is_enabled(self) -> bool:
        """get quantizer mode"""
        return self._is_enabled

    def enable(self, mode: bool = True) -> None:
        """Sets Quantizer in quantization enabling mode

        Args:
            mode (bool, optional): If `True`, enable quantization. Otherwise, disable quantization. Defaults to `True`.
        """
        self._is_enabled = mode

    def disable(self) -> None:
        """Sets quantizer in quantization disabling mode"""
        self._is_enabled = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """The forward pass of fake quantizer

        Args:
            inputs (torch.Tensor): A tensor to fake-quantize.

        Raises
            ValueError: If fake quantizer has negative step_size or size of channel is invalid.

        Returns
            torch.Tensor: If fake quantizer is enabled, it returns a fake-quantized tensor.
                If fake quantizer is disable, it returns the value as entered.
        """
        if not self._is_enabled:
            return inputs

        if (self.per_channel) and isinstance(inputs, torch.Tensor) and (self.channel_size != inputs.shape[0]):
            if self.channel_size is None:
                raise ValueError("channel_size(=None) must be set for per channel weight quantization")
            raise ValueError(
                f"channel_size(={self.channel_size}) value must be the same as "
                f"the first dimension of the input tensor (={inputs.shape[0]})."
            )

        if self.qat_function is not clq_function and not self.narrow and self.step_size.min() <= 0:
            log.error(
                f"Expected step_size to be positive, but got step_size={self.step_size.data}. "
                "Please try one of the suggestions below:\n"
                '   * select "clq" from the "qat_backward" field in the OwLite Web UI (https://owlite.ai/project);\n'
                "   * set the weight_decay of the fake quantizer's parameters to 0;\n"
                "   * reduce the learning rate for the fake quantizer's parameters; or\n"
                "   * reduce the grad_scale of the fake quantizer"
            )
            raise ValueError("Step_size must be positive")

        return self.qat_function(
            inputs,
            self.step_size,
            self.zero_point,
            self.grad_scale,
            self.quant_min,
            self.quant_max,
            self.per_channel,
            not self.is_zero_point_folded,
        )

    def invert_signedness(self) -> None:
        """Inverts signedness of this fake quantizer"""
        self.unsigned.data = torch.logical_not(self.unsigned.data)

    # pylint: disable=protected-access
    def extra_repr(self) -> str:
        if self.precision.item() == 32:
            return f"precision: {self.precision.item()}"
        string = f"{self.qat_backward_type}(precision: {self.precision.item()}"
        string += ", per_tensor" if not self.per_channel.item() else ", per_channel"
        string += f", quant_min: {self.quant_min}, quant_max: {self.quant_max}"
        if not self.symmetric.item():
            string += ", asymmetric"
        string += (
            f", zero_point: {self.zero_point.item()}, is_zero_point_folded: {self.is_zero_point_folded}"
            if not self.per_channel.item()
            else ""
        )
        string += f", is_enabled: {self.is_enabled}"
        string += f", calib: {self.calibrator.__class__.__name__}"
        string += ")"
        return string

    @property
    def maxabs_bound(self) -> int:
        """The maximum absolute limit value of the quantized domain.

        Returns:
            int: A Maximum absolute bound value.
        """
        return max(abs(self.quant_min), abs(self.quant_max))

    @property
    def options(self) -> FakeQuantizerOptions:
        """The options that current FakeQuantizer instance represents."""
        percentile = getattr(self.calibrator, "percentile", None)
        zero_point = getattr(self, "zero_point", None)
        learn_zero_point = False if zero_point is None else zero_point.requires_grad

        return FakeQuantizerOptions(
            qat_backward=self.qat_backward_type,
            ptq_calibration=self.ptq_calibration,
            percentile=percentile,
            precision=int(self.precision.item()),
            symmetric=bool(self.symmetric.item()),
            unsigned=bool(self.unsigned.item()),
            per_channel=bool(self.per_channel.item()),
            learn_zero_point=learn_zero_point,
            grad_scale=self.grad_scale.item(),
        )

    def state_dict(  # type: ignore[no-untyped-def, override]
        self, *args, **kwargs
    ) -> Union[OrderedDict[Any, Any], dict[str, Any]]:
        """Stores the indices of ptq_calibration and qat_backward in addition to the torch state dict.

        Returns:
            dict:
                a dictionary containing a whole state of the module.
        """
        state: OrderedDict = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix")
        extra_state = {}
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
        self.qat_backward_type = QATBackwardType(state_dict.pop(f"{prefix}_qat_backward").item())
        self.ptq_calibration = PTQCalibrationType(state_dict.pop(f"{prefix}_ptq_calibration").item())
        calibrator_class = self.ptq_calibration.calibrator_class
        if self.ptq_calibration == PTQCalibrationType.percentile:
            self.calibrator = calibrator_class(self, state_dict.pop(f"{prefix}_ptq_calibration_percentile").item())
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


def enable_quantizers(net: torch.nn.Module, mod: bool = True) -> None:
    """Enables or disables fake quantizers within the specified module.

    Args:
        net (torch.nn.Module): The module containing fake quantizers to enable or disable.
        mod (bool, optional): If True, enables all fake quantizers in the module. If False, disables them.
    """
    for _, module in net.named_modules():
        if isinstance(module, FakeQuantizer):
            module.enable(mod)
