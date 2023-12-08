"""Histogram calibrator class"""

import torch

from ..logger import log
from ._calibrator import _Calibrator


class _HistogramCalibrator(_Calibrator):
    """Histogram calibrator.

    Attributes:
        set_attr_list (Dict[str, torch.Tensor]): Initialized properties to register with the quantizer.
            'histogram': histogram count. Default [0, ..., 0], len = 2048.
            'bin_edges': histogram edges. Default [0, ..., 0], len = 2048.
            'histc_bins': integer. number of histogram bins. Default 2048.
    """

    def __init__(self, quantizer):
        """Initializes for histogram calibrator"""
        super().__init__(quantizer)
        self.set_attr_list = {}

    def update(self):
        raise NotImplementedError

    def prepare(self):
        # define forward hook function
        def histogram_forward_hook_func(module, inputs, output):
            """Forward hook function to get histogram value"""

            _input = inputs[0].clone()
            if module.is_enabled:
                raise RuntimeError(
                    "The quantizer should be disabled during calibration."
                )
            if (
                module.symmetric.item()
                and module.unsigned.item()
                and inputs[0].min() < 0
            ):
                log.warning(
                    "The unsigned fake quantizer has a negative number as input. "
                    "It will automatically convert to a signed fake quantizer.",
                    stacklevel=2,
                )
                module.invert_signedness()

            with torch.no_grad():
                new_input = []
                if module.per_channel:
                    _channel_axis = 0
                    _channel_size = _input.shape[_channel_axis]
                    for chn in range(_channel_size):
                        _input_chn = torch.select(_input, _channel_axis, chn)
                        new_input.append(_input_chn)
                else:
                    new_input.append(_input)

                # _histc_cuda does not have a deterministic implementation
                _deterministic_enable = torch.are_deterministic_algorithms_enabled()
                if _deterministic_enable:
                    torch.use_deterministic_algorithms(False)

                for i, val in enumerate(new_input):
                    local_max = val.abs().max().clone().to(module.bin_edges[i].device)
                    if (
                        module.histogram[i].data.sum() == 0
                        and module.bin_edges[i].data.sum() == 0
                    ):
                        module.histogram[i].data = torch.histc(
                            val.abs(),
                            bins=int(module.histc_bins[i].data),
                            min=0,
                            max=local_max,
                        ).to(module.histogram[i].device)
                        module.bin_edges[i].data = torch.linspace(
                            0, local_max, int(module.histc_bins[i].data) + 1
                        ).to(module.bin_edges[i].device)
                    else:
                        if module.per_channel:
                            break
                        if local_max > module.bin_edges[i].data[-1]:
                            interval = (
                                module.bin_edges[i].data[1]
                                - module.bin_edges[i].data[0]
                            )
                            module.histc_bins[i].data = torch.Tensor(
                                [int((local_max / interval).ceil().item())]
                            )
                            module.bin_edges[i].data = torch.arange(
                                0,
                                local_max + interval,
                                interval,
                                device=module.bin_edges[i].device,
                            )
                        local_hist = torch.histc(
                            val.abs(),
                            bins=int(module.histc_bins[i].data),
                            min=0,
                            max=module.bin_edges[i].data[-1],
                        ).to(module.bin_edges[i].device)
                        local_hist[
                            : module.histogram[i].data.numel()
                        ] += module.histogram[i].data
                        module.histogram[i].data = local_hist

                # allocate to original state
                if _deterministic_enable:
                    torch.use_deterministic_algorithms(True)

            return output

        # ~define forward hook function

        # set histogram, bin_edges attr and register forward hook
        _histogram_size = 2048
        if self.quantizer.per_channel:
            _channel_size = self.quantizer.step_size.shape[0]
        else:
            _channel_size = 1

        device = self.quantizer.step_size.device

        self.set_attr_list = {
            "histogram": [
                torch.zeros(_histogram_size).to(device) for _ch in range(_channel_size)
            ],
            "bin_edges": [
                torch.zeros(_histogram_size + 1).to(device)
                for _ch in range(_channel_size)
            ],
            "histc_bins": [
                torch.Tensor([_histogram_size]).to(device)
                for _ch in range(_channel_size)
            ],
        }

        for attr, default in self.set_attr_list.items():
            if hasattr(self.quantizer, attr):
                raise AttributeError(f"In Quantizer, {attr} attribution already exists")
            setattr(self.quantizer, attr, default)
        self.hook_handler = self.quantizer.register_forward_hook(
            histogram_forward_hook_func
        )
