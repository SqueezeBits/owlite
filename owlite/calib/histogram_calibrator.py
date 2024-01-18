from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.utils.hooks import RemovableHandle

from owlite_core.logger import log

from .calibrator import Calibrator

if TYPE_CHECKING:
    from ..nn.fake_quantizer import FakeQuantizer


class HistogramCalibrator(Calibrator, ABC):
    """Histogram calibrator.

    Attributes:
        histogram(list[torch.Tensor]): list of histogram counts. Each element defaults to [0, ..., 0], len = 2048.
        bin_edges(list[torch.Tensor]): histogram edges. Each element defaults to [0, ..., 0], len = 2048.
        histc_bins(list[torch.Tensor]): number of histogram bins. Each element defaults to 2048.
    """

    def __init__(self, quantizer: "FakeQuantizer"):
        """Initializes for histogram calibrator"""
        super().__init__(quantizer)
        self.histogram: list[torch.Tensor] = []
        self.bin_edges: list[torch.Tensor] = []
        self.histc_bins: list[torch.Tensor] = []

    def prepare(self) -> RemovableHandle:
        # define forward hook function
        def histogram_forward_hook_func(module: "FakeQuantizer", inputs: tuple[Any, ...], output: Any) -> Optional[Any]:
            """Forward hook function to get histogram value"""

            assert isinstance(module.calibrator, HistogramCalibrator)
            calibrator = module.calibrator
            assert self.check_calib_ready()

            if any(len(hist_attr) == 0 for hist_attr in (self.histogram, self.bin_edges, self.histc_bins)):
                log.error(
                    f"`histogram`: {calibrator.histogram}\n`bin_edge`: {calibrator.bin_edges}\n"
                    f"`histc_bins`: {calibrator.histc_bins}"
                )
                raise ValueError("During calibration, calibration attributions are not initialized")

            _input = inputs[0].clone()

            if module.symmetric.item() and module.unsigned.item() and inputs[0].min() < 0:
                log.warning(
                    "The unsigned fake quantizer has a negative number as input. "
                    "It will automatically convert to a signed",
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
                _deterministic_enable_status = torch.are_deterministic_algorithms_enabled()
                torch.use_deterministic_algorithms(False, warn_only=True)

                for i, val in enumerate(new_input):
                    local_max = val.abs().max().clone().to(calibrator.bin_edges[i].device)
                    if calibrator.histogram[i].data.sum() == 0 and calibrator.bin_edges[i].data.sum() == 0:
                        calibrator.histogram[i].data = torch.histc(
                            val.abs(),
                            bins=int(calibrator.histc_bins[i].data),
                            min=0,
                            max=float(local_max),
                        ).to(calibrator.histogram[i].device)
                        calibrator.bin_edges[i].data = torch.linspace(
                            0, float(local_max), int(calibrator.histc_bins[i].data) + 1
                        ).to(calibrator.bin_edges[i].device)
                    else:
                        if module.per_channel:
                            break
                        if local_max > calibrator.bin_edges[i].data[-1]:
                            interval = calibrator.bin_edges[i].data[1] - calibrator.bin_edges[i].data[0]
                            calibrator.histc_bins[i].data = torch.Tensor([int((local_max / interval).ceil().item())])
                            calibrator.bin_edges[i].data = torch.arange(
                                0,
                                float(local_max + interval),
                                float(interval),
                                device=calibrator.bin_edges[i].device,
                            )
                        local_hist = torch.histc(
                            val.abs(),
                            bins=int(calibrator.histc_bins[i].data),
                            min=0,
                            max=float(calibrator.bin_edges[i].data[-1]),
                        ).to(calibrator.bin_edges[i].device)
                        local_hist[: calibrator.histogram[i].data.numel()] += calibrator.histogram[i].data
                        calibrator.histogram[i].data = local_hist

                # allocate deterministic algorithms to original state
                torch.use_deterministic_algorithms(_deterministic_enable_status, warn_only=True)

            return output

        # ~define forward hook function

        # set histogram, bin_edges attr and register forward hook
        _histogram_size = 2048
        _channel_size = self.quantizer.channel_size
        assert _channel_size is not None
        device = self.quantizer.step_size.device

        if any(len(hist_attr) != 0 for hist_attr in (self.histogram, self.bin_edges, self.histc_bins)):
            log.error(
                "The histogram attributions are already set before the calibration is prepared.\n"
                f"`histogram`: {self.histogram}\n`bin_edges`: {self.bin_edges}\n`histc_bins`: {self.histc_bins}"
            )
            raise ValueError("The histogram attributions are already set before the calibration is prepared")

        self.histogram = [torch.zeros(_histogram_size).to(device) for _ in range(_channel_size)]
        self.bin_edges = [torch.zeros(_histogram_size + 1).to(device) for _ in range(_channel_size)]
        self.histc_bins = [torch.Tensor([_histogram_size]).to(device) for _ in range(_channel_size)]

        self.hook_handler = self.quantizer.register_forward_hook(histogram_forward_hook_func)
        return self.hook_handler

    @abstractmethod
    def update(self) -> None:
        assert self.check_calib_ready()
        if any(len(hist_attr) == 0 for hist_attr in (self.histogram, self.bin_edges, self.histc_bins)):
            log.error(f"`histogram`: {self.histogram}\n `bin_edge`: {self.bin_edges}\n `histc_bins`: {self.histc_bins}")
            raise ValueError("During calibration, calibration attributions are not initialized")

    def clear_attribution(self) -> None:
        """clear attributions of histogram(`histogram`, `bin_edges`, `histc_bins`) and registered forward_hook"""
        assert isinstance(self.hook_handler, RemovableHandle)
        # clear attributions about histogram from calibrator
        self.histogram.clear()
        self.bin_edges.clear()
        self.histc_bins.clear()

        # remove registered forward_hook
        self.hook_handler.remove()
        self.hook_handler = None
