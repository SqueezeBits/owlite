from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.utils.hooks import RemovableHandle

from ..owlite_core.logger import log
from .calibrator import Calibrator

if TYPE_CHECKING:
    from ..nn import FakeQuantizer


class MinmaxCalibrator(Calibrator):
    """Minmax Calibrator Class.

    Minmax calibration to set step_size and zero_point using min-max during calibration
    with asymmetric quantization.

    Attributes:
        max_value (`torch.Tensor`, `optional`): maximum value of data passing through the quantizer.
        min_value (`torch.Tensor`, `optional`): minimum value of data passing through the quantizer.
    """

    def __init__(self, quantizer: "FakeQuantizer"):
        super().__init__(quantizer)
        self.max_value: Optional[torch.Tensor] = None
        self.min_value: Optional[torch.Tensor] = None

    def check_calib_ready(self) -> bool:
        if self.quantizer.symmetric:
            log.error("MinMax Calibration only surpports aymmetric quantization")
            return False
        return super().check_calib_ready()

    def prepare(self) -> RemovableHandle:
        # define forward hook function
        def minmax_forward_hook_func(module: "FakeQuantizer", inputs: tuple[Any, ...], output: Any) -> Optional[Any]:
            """forward hook function to get minmax value"""

            calibrator = module.calibrator
            assert isinstance(calibrator, MinmaxCalibrator)
            assert self.check_calib_ready()

            if calibrator.max_value is None or calibrator.min_value is None:
                raise ValueError(
                    "During calibration, calibration attributions should be initialized, but None was provided"
                )

            _input = inputs[0].clone()
            with torch.no_grad():
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

        # update step_size using "min & max value"
        self.quantizer.step_size.data = (
            (self.max_value - self.min_value) / (self.quantizer.quant_max - self.quantizer.quant_min)
        ).detach()

        # update zero_point
        self.quantizer.zero_point.data = self.min_value - (self.quantizer.quant_min * self.quantizer.step_size.data)

        # set  "min_value" and "max_value" attritbutions to `None`
        self.max_value, self.min_value = None, None

        # remove registered forward_hook
        self.hook_handler.remove()
        self.hook_handler = None
