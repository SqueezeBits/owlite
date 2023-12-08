"""Minmax calibrator class

For asymmetric quantization collect min & max values and
set step size using that information
"""

import torch

from ._calibrator import _Calibrator


class MinmaxCalibrator(_Calibrator):
    """Minmax calibrator

    Minmax calibration to set step_size and zero_point using min-max during calibration
    with asymmetric quantization
    """

    def prepare(self):
        # define forward hook function
        def minmax_forward_hook_func(module, inputs, output):
            """forward hook function to get minmax value"""
            if module.is_enabled:
                raise RuntimeError(
                    "The quantizer should be disabled during calibration."
                )
            if module.symmetric.item():
                raise RuntimeError(
                    "MinMax Calibration only surpports aymmetric quantization"
                )

            _input = inputs[0].clone()
            with torch.no_grad():
                if module.per_channel.item():
                    # assume channel axis is 0
                    new_max = (
                        _input.reshape(_input.size()[0], -1).max(dim=1).values.clone()
                    )
                    new_min = (
                        _input.reshape(_input.size()[0], -1).min(dim=1).values.clone()
                    )
                else:
                    new_max = _input.max().clone()
                    new_min = _input.min().clone()
                module.max_value.data = torch.maximum(
                    new_max.to(module.max_value.device), module.max_value
                ).data
                module.min_value.data = torch.minimum(
                    new_min.to(module.min_value.device), module.min_value
                ).data
            return output

        # ~define forward hook function

        self.quantizer.max_value = (
            torch.ones_like(self.quantizer.step_size.data).to(
                self.quantizer.step_size.device
            )
            * torch.finfo().min
        )
        self.quantizer.min_value = (
            torch.ones_like(self.quantizer.step_size.data).to(
                self.quantizer.step_size.device
            )
            * torch.finfo().max
        )
        self.hook_handler = self.quantizer.register_forward_hook(
            minmax_forward_hook_func
        )
        return self.hook_handler

    def update(self):
        # update step_size using "min & max value"
        self.quantizer.step_size.data = (
            (self.quantizer.max_value - self.quantizer.min_value)
            / (self.quantizer.quant_max - self.quantizer.quant_min)
        ).detach()

        # update zero_point
        self.quantizer.zero_point.data = self.quantizer.min_value - (
            self.quantizer.quant_min * self.quantizer.step_size.data
        )

        # delete "min_value" and "max_value" attritbutions from quantizer
        delattr(self.quantizer, "min_value")
        delattr(self.quantizer, "max_value")
        # remove registered forward_hook
        self.hook_handler.remove()
