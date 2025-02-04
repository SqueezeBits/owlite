from typing import TYPE_CHECKING, Any

import torch
from torch.utils.hooks import RemovableHandle

from ..core.logger import log
from .calibrator import Calibrator

if TYPE_CHECKING:
    from ..nn import FakeQuantizer


class MinmaxCalibrator(Calibrator):
    r"""Minmax Calibrator Class.

    The MinMaxCalibration calibrator stores the **maximum value** and **minimum value** encountered in the passed data,
    utilizing this value as the quantization range. When the original data is represented by $$X$$,
    the `step_size` and `zero_point` are caculated as:

    $$
        \text{step\_size}=\frac{\max_{x \\in X}(x) - \min_{x \\in X}(x) }{\text{quant\_max}-\text{quant\_min}} \\
        \text{zero\_point} = - \frac{\min_{x \\in X}(x)}{\text{step\_size}} + \text{quant\_min}
    $$

    For symmetric quantization:

    $$
        \text{step\_size}=\frac{\max_{x \\in X}(|x|)}{\text{quant\_max}-\text{quant\_min}}
        \text{zero\_point} = 0
    $$

    Attributes:
        max_value (`torch.Tensor`, `optional`): maximum value of data passing through the quantizer.
        min_value (`torch.Tensor`, `optional`): minimum value of data passing through the quantizer.
    """

    def __init__(self, quantizer: "FakeQuantizer"):
        super().__init__(quantizer)
        self.max_value: torch.Tensor | None = None
        self.min_value: torch.Tensor | None = None

    def prepare(self) -> RemovableHandle:
        # define forward hook function
        def minmax_forward_hook_func(module: "FakeQuantizer", inputs: tuple[Any, ...], output: Any) -> Any | None:
            """Forward hook function to get minmax value."""
            calibrator = module.calibrator
            assert isinstance(calibrator, MinmaxCalibrator)
            assert self.check_calib_ready()

            if calibrator.max_value is None or calibrator.min_value is None:
                raise ValueError(
                    "During calibration, calibration attributions should be initialized, but None was provided"
                )
            _input: torch.Tensor = inputs[0].clone()
            calibrator.input_dtype = _input.dtype
            _input = _input.float()
            if module.channel is not None:
                axis = module.channel.axis
                (other_dims := list(range(_input.dim()))).remove(axis)
                _input = _input.permute(axis, *other_dims)  # make channel dim is 0
                new_max = _input.reshape(_input.size()[0], -1).max(dim=1).values.clone()
                new_min = _input.reshape(_input.size()[0], -1).min(dim=1).values.clone()
            else:
                new_max = _input.max().clone()
                new_min = _input.min().clone()
            calibrator.max_value.data = torch.maximum(
                new_max.to(calibrator.max_value.device), calibrator.max_value
            ).data
            calibrator.min_value.data = torch.minimum(
                new_min.to(calibrator.min_value.device), calibrator.min_value
            ).data
            return output

        # ~define forward hook function

        if self.max_value is not None or self.min_value is not None:
            log.error(
                "The min-max attributions are already set before the calibration is prepared.\n"
                f"`max_value`: {self.max_value}\n`min_value`: {self.min_value}"
            )
            raise ValueError("The min-max attributions are already set before the calibration is prepared")

        self.max_value = (
            torch.ones_like(self.quantizer.step_size.data).to(self.quantizer.step_size.device) * torch.finfo().min
        )
        self.min_value = (
            torch.ones_like(self.quantizer.step_size.data).to(self.quantizer.step_size.device) * torch.finfo().max
        )
        self.hook_handler = self.quantizer.register_forward_hook(minmax_forward_hook_func)
        return self.hook_handler

    def update(self) -> None:
        assert self.check_calib_ready()
        assert isinstance(self.hook_handler, RemovableHandle)
        if self.max_value is None or self.min_value is None:
            log.error(f"`max_value` : {self.max_value}")
            log.error(f"`min_value` : {self.min_value}")
            raise ValueError(
                "During preparing calibration, calibration attributions should be initialized, but None was provided"
            )
        if self.quantizer.symmetric:
            self.update_fake_quantizer_param_with_max_min(torch.max(self.max_value.abs(), self.min_value.abs()))
        else:
            self.update_fake_quantizer_param_with_max_min(self.max_value, self.min_value)

        # set "min_value" and "max_value" attritbutions to `None`
        self.max_value, self.min_value = None, None

        # remove registered forward_hook
        self.hook_handler.remove()
        self.hook_handler = None
