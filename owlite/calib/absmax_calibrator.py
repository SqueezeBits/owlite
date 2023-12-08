"""Absmax calibrator class

For symmetric quantization collect absolute max values and
set step size using that information
"""


import torch

from ..logger import log
from ._calibrator import _Calibrator


class AbsmaxCalibrator(_Calibrator):
    """Absmax calibrator."""

    def prepare(self):
        # define forward hook function
        def absmax_forward_hook_func(module, inputs, output):
            """forward hook function to get absmax value"""
            if module.is_enabled:
                raise RuntimeError(
                    "The quantizer should be disabled during calibration."
                )
            if module.unsigned.item() and (inputs[0].min() < 0):
                log.warning(
                    "The unsigned fake quantizer has a negative number as input. "
                    "It will automatically convert to a signed fake quantizer.",
                    stacklevel=2,
                )
                module.invert_signedness()

            _input = inputs[0].clone()
            with torch.no_grad():
                if module.per_channel.item():
                    # assume channel axis is 0
                    new_absmax = (
                        _input.reshape(_input.size()[0], -1)
                        .abs()
                        .max(dim=1)
                        .values.clone()
                    )
                else:
                    new_absmax = _input.abs().max().clone()
                module.absmax.data = torch.maximum(
                    new_absmax.to(module.absmax.device), module.absmax
                ).data
            return output

        # ~define forward hook function
        self.quantizer.absmax = torch.zeros_like(self.quantizer.step_size.data).to(
            self.quantizer.step_size.device
        )
        self.hook_handler = self.quantizer.register_forward_hook(
            absmax_forward_hook_func
        )
        return self.hook_handler

    def update(self):
        # update step_size using "abs max"
        self.quantizer.step_size.data = (
            self.quantizer.absmax / self.quantizer.maxabs_bound
        ).detach()
        # delete "absmax" attritbution from quantizer
        delattr(self.quantizer, "absmax")
        # remove registered forward_hook
        self.hook_handler.remove()
