from typing import TYPE_CHECKING, Any

import torch
from torch.utils.hooks import RemovableHandle

from ..owlite_core.logger import log
from .calibrator import Calibrator

if TYPE_CHECKING:
    from ..nn import FakeQuantizer


class AbsmaxCalibrator(Calibrator):
    r"""Absmax Calibrator Class.

    The AbsMaxCalibration calibrator stores the **maximum absolute value** encountered in the passed data,
    utilizing this value as the quantization range. When the original data is represented by $$X$$, the `step_size`
    is calculated as:

    $$
        \text{step\_size}=\frac{\max_{x \\in X}(|x|)}{\text{quant\_max}-\text{quant\_min}}
    $$

    This approach eliminates clipping errors, but potentially introduces significant rounding errors.

    Attributes:
        absmax (`torch.Tensor`, `optional`): absolute maximum value of data passing through the quantizer.
    """

    def __init__(self, quantizer: "FakeQuantizer"):
        super().__init__(quantizer)
        self.absmax: torch.Tensor | None = None

    def prepare(self) -> RemovableHandle:
        """Prepare the forward hook function."""

        def absmax_forward_hook_func(module: "FakeQuantizer", inputs: tuple[Any, ...], output: Any) -> Any | None:
            """Forward hook function to get absmax value."""
            calibrator = module.calibrator
            assert isinstance(calibrator, AbsmaxCalibrator)

            if not calibrator.check_calib_ready():
                raise RuntimeError("Not all conditions for calibration were not met")
            if calibrator.absmax is None:
                raise RuntimeError("During calibration, an absmax should be initialized, but None was provided")

            if module.unsigned and (inputs[0].min() < 0):
                log.warning(
                    "The unsigned fake quantizer has a negative number as input. "
                    "It will automatically convert to a signed fake quantizer",
                    stacklevel=2,
                )
                module.invert_signedness()
            # pylint:disable=duplicate-code
            _input = inputs[0].clone()
            with torch.no_grad():
                if module.channel is not None:
                    axis = module.channel.axis
                    (other_dims := list(range(_input.dim()))).remove(axis)
                    _input = _input.permute(axis, *other_dims)  # make channel dim is 0
                    new_absmax = _input.reshape(_input.size()[0], -1).abs().max(dim=1).values.clone()
                else:
                    new_absmax = _input.abs().max().clone()
                calibrator.absmax.data = torch.maximum(new_absmax.to(calibrator.absmax.device), calibrator.absmax).data
            return output
            # pylint:enable=duplicate-code

        # ~define forward hook function
        self.absmax = torch.zeros_like(self.quantizer.step_size.data).to(self.quantizer.step_size.device)
        if not self.check_calib_ready():
            raise RuntimeError("Not all conditions for calibration were not met.")
        self.hook_handler = self.quantizer.register_forward_hook(absmax_forward_hook_func)
        return self.hook_handler

    def update(self) -> None:
        """Update `self.step_size` using `self.absmax`."""
        assert self.absmax is not None
        assert self.quantizer.step_size.data.shape == self.absmax.shape
        if not self.check_calib_ready():
            raise RuntimeError("Not all conditions for calibration were not met.")
        assert isinstance(self.hook_handler, RemovableHandle)

        self.update_fake_quantizer_param_with_max_min(self.absmax)

        # remove registered forward_hook
        self.hook_handler.remove()
        self.hook_handler = None
